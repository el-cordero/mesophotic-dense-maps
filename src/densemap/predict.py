"""Per-token inference and export of class and confidence rasters."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .constants import IMAGE_EXTENSIONS
from .dino import extract_feature_grid


@torch.no_grad()
def predict_image(
    image_path: Path,
    out_path: Path,
    conf_path: Path | None,
    dino_model,
    classifier,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    class_ids: list[int] | torch.Tensor,
    device: torch.device,
    smooth: bool = False,
) -> None:
    """Classify every token of one tile.

    The class grid is enlarged to the tile size by nearest-neighbor resampling and the
    maximum softmax probability (confidence) by bilinear resampling, saved as 8-bit PNG.
    Smoothing is off by default and was not used in the study.
    """
    feats, (orig_w, orig_h) = extract_feature_grid(image_path, dino_model, device)
    fh, fw, channels = feats.shape
    flat = feats.reshape(-1, channels).float().to(device)
    flat = (flat - x_mean.to(device)) / x_std.to(device)

    logits = classifier(flat)
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    class_id_tensor = torch.as_tensor(class_ids, device=pred.device, dtype=torch.long)
    pred = class_id_tensor[pred]

    pred_np = pred.reshape(fh, fw).cpu().numpy().astype(np.uint8)
    if smooth:
        pred_np = cv2.medianBlur(pred_np, 5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred_np, mode="L").resize((orig_w, orig_h), Image.NEAREST).save(out_path)

    if conf_path is not None:
        conf_np = conf.reshape(fh, fw).cpu().numpy()
        conf_img = Image.fromarray((conf_np * 255).clip(0, 255).astype(np.uint8), mode="L")
        conf_path.parent.mkdir(parents=True, exist_ok=True)
        conf_img.resize((orig_w, orig_h), Image.BILINEAR).save(conf_path)


def predict_split(
    images_root: Path,
    out_root: Path,
    conf_root: Path,
    split: str,
    dino_model,
    classifier,
    checkpoint: dict,
    device: torch.device,
    overwrite: bool = False,
    max_images: int | None = None,
    smooth: bool = False,
) -> None:
    image_dir = images_root / split
    images = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
    )
    if max_images is not None:
        images = images[:max_images]
    for image_path in tqdm(images, desc=f"predict {split}"):
        out_path = out_root / split / f"{image_path.stem}.png"
        conf_path = conf_root / split / f"{image_path.stem}_confidence.png"
        if out_path.exists() and conf_path.exists() and not overwrite:
            continue
        predict_image(
            image_path=image_path,
            out_path=out_path,
            conf_path=conf_path,
            dino_model=dino_model,
            classifier=classifier,
            x_mean=checkpoint["x_mean"],
            x_std=checkpoint["x_std"],
            class_ids=checkpoint.get("class_ids", list(range(checkpoint["num_classes"]))),
            device=device,
            smooth=smooth,
        )
