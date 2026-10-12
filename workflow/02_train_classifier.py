#!/usr/bin/env python3
"""Step 2. Train the point-feature classifier on frozen DINO ViT-B/8 features.

For each CPCe point the 768-dimensional tokens in the 5 x 5 window around the point are
averaged. Features are standardized with the training mean and standard deviation, and the
checkpoint with the highest validation accuracy is kept (early stopping after 40 epochs
without improvement). No random seed was fixed in the study, so a rerun will not reproduce
the shipped weights exactly. The checkpoint used in the paper is models/dino_vitb8_point_classifier.pt.

Example
    python workflow/02_train_classifier.py --tiles-dir tiles --output-dir outputs/model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from densemap.classifier import PointFeatureClassifier
from densemap.constants import CLASS_NAMES
from densemap.dino import build_dino_vitb8, extract_feature_grid
from densemap.points import iter_images, read_points_txt


def point_feature(feats: torch.Tensor, x: int, y: int, orig_w: int, orig_h: int, radius: int) -> torch.Tensor:
    feat_h, feat_w, _ = feats.shape
    fx = min(max(int((x - 1) / max(orig_w, 1) * feat_w), 0), feat_w - 1)
    fy = min(max(int((y - 1) / max(orig_h, 1) * feat_h), 0), feat_h - 1)
    y0 = max(fy - radius, 0)
    y1 = min(fy + radius + 1, feat_h)
    x0 = max(fx - radius, 0)
    x1 = min(fx + radius + 1, feat_w)
    return feats[y0:y1, x0:x1, :].mean(dim=(0, 1)).float().cpu()


def collect_split_features(tiles_dir: Path, split: str, dino, device: torch.device, point_radius: int):
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    images_root = tiles_dir / "images"
    points_root = tiles_dir / "points" / split
    for image_path in tqdm(iter_images(images_root, split), desc=f"features {split}"):
        point_path = points_root / f"{image_path.stem}.txt"
        points = read_points_txt(point_path)
        if not points:
            continue
        feats, (orig_w, orig_h) = extract_feature_grid(image_path, dino, device)
        for point in points:
            xs.append(point_feature(feats, point.x, point.y, orig_w, orig_h, point_radius))
            ys.append(point.class_id)
    if not xs:
        raise RuntimeError(f"No point features collected for split: {split}")
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def evaluate(model, x: torch.Tensor, y_idx: torch.Tensor, batch_size: int, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in DataLoader(TensorDataset(x, y_idx), batch_size=batch_size):
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            losses.append(float(loss.item()) * len(yb))
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(len(yb))
    return sum(losses) / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DINO ViT-B/8 point-feature classifier.")
    parser.add_argument("--tiles-dir", type=Path, default=ROOT / "tiles", help="Folder with images/{split} and points/{split}.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/model")
    parser.add_argument("--dino-weights", type=Path, default=None, help="Local copy of the DINO ViT-B/8 weights (downloaded if omitted).")
    parser.add_argument("--point-radius", type=int, default=2, help="Half-width of the token window (2 gives 5 x 5).")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--checkpoint-name", default="dino_vitb8_point_classifier.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    dino = build_dino_vitb8(device, args.dino_weights)
    train_x_raw, train_y_ids = collect_split_features(args.tiles_dir, "train", dino, device, args.point_radius)
    val_x_raw, val_y_ids = collect_split_features(args.tiles_dir, "val", dino, device, args.point_radius)

    class_ids = sorted(int(v) for v in CLASS_NAMES)
    id_to_index = {class_id: idx for idx, class_id in enumerate(class_ids)}
    train_y = torch.tensor([id_to_index[int(v)] for v in train_y_ids.tolist()], dtype=torch.long)
    val_y = torch.tensor([id_to_index[int(v)] for v in val_y_ids.tolist()], dtype=torch.long)

    x_mean = train_x_raw.mean(dim=0)
    x_std = train_x_raw.std(dim=0).clamp_min(1e-6)
    train_x = (train_x_raw - x_mean) / x_std
    val_x = (val_x_raw - x_mean) / x_std

    model = PointFeatureClassifier(in_dim=train_x.shape[1], num_classes=len(class_ids)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_total = 0
        train_correct = 0
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(yb)
            train_total += int(len(yb))
            train_correct += int((logits.argmax(dim=1) == yb).sum().item())

        val_loss, val_acc = evaluate(model, val_x, val_y, args.batch_size, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_total, 1),
            "train_accuracy": train_correct / max(train_total, 1),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        history.append(row)
        print(json.dumps(row))

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            stale = 0
            checkpoint = {
                "classifier": model.state_dict(),
                "num_classes": len(class_ids),
                "class_ids": class_ids,
                "class_names": {str(k): CLASS_NAMES.get(k, str(k)) for k in class_ids},
                "x_mean": x_mean,
                "x_std": x_std,
                "point_radius": args.point_radius,
                "feature_backbone": "DINO ViT-B/8",
                "label_source": "pointcoral class crosswalk",
                "best_epoch": best_epoch,
                "best_val_accuracy": best_acc,
            }
            torch.save(checkpoint, args.output_dir / args.checkpoint_name)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}.")
                break

    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    pd.DataFrame(
        [{"class_id": cid, "class_name": CLASS_NAMES.get(cid, str(cid)), "class_index": idx} for idx, cid in enumerate(class_ids)]
    ).to_csv(args.output_dir / "class_index_map.csv", index=False)
    print(f"Saved best checkpoint to {args.output_dir / args.checkpoint_name}")


if __name__ == "__main__":
    main()
