#!/usr/bin/env python3
"""Step 3. Classify every image token of every tile.

Writes one class raster (outputs/predictions/{split}/<tile>.png, pixel values are class IDs)
and one confidence raster (outputs/confidence/{split}/<tile>_confidence.png, maximum softmax
probability x 255) per tile.

Example
    python workflow/03_predict_tiles.py --tiles-dir tiles --checkpoint models/dino_vitb8_point_classifier.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from densemap.classifier import load_classifier_checkpoint
from densemap.dino import build_dino_vitb8
from densemap.predict import predict_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DINO ViT-B/8 point-supervised inference on tiles.")
    parser.add_argument("--tiles-dir", type=Path, default=ROOT / "tiles", help="Folder with images/{split}.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/dino_vitb8_point_classifier.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--dino-weights", type=Path, default=None, help="Local copy of the DINO ViT-B/8 weights (downloaded if omitted).")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    classifier, ckpt = load_classifier_checkpoint(args.checkpoint, device)
    dino = build_dino_vitb8(device, args.dino_weights)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        predict_split(
            images_root=args.tiles_dir / "images",
            out_root=args.output_dir / "predictions",
            conf_root=args.output_dir / "confidence",
            split=split,
            dino_model=dino,
            classifier=classifier,
            checkpoint=ckpt,
            device=device,
            overwrite=args.overwrite,
            max_images=args.max_images,
        )


if __name__ == "__main__":
    main()
