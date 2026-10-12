#!/usr/bin/env python3
"""Step 5. Read the saved prediction and confidence rasters at every CPCe point.

Writes
    data/point_predictions.csv.gz   one row per CPCe point with its label, the predicted class
                                    and the 8-bit confidence (0-255) read from the saved rasters
    data/photoquadrats.csv          survey (site, depth, date) and partition of each photoquadrat

Tiles are processed in the order of outputs/cover_tables/analysis_file_index.csv, written by
step 4, so that the rows come out in the same order as in the study. Point coordinates are
1-based tile pixel coordinates, so the raster is read at (x - 1, y - 1).

Example
    python workflow/05_point_predictions.py --tiles-dir tiles --output-dir outputs
"""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASS_IDS = [0, 1, 2, 4, 5, 6, 7, 8]


def read_points(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["x", "y", "class_id"])
    return pd.read_csv(path, sep=r"\s+", header=None, names=["x", "y", "class_id"])


def harmonize_site(site: str) -> str:
    text = str(site).replace("_", " ")
    return "Hoyo Terrace" if text == "Hoyo Terrrace" else text


def survey_date(photoquadrat: str) -> str:
    m = re.match(r"(?P<day>\d{2})_(?P<month>[A-Za-z]+)_(?P<year>\d{4})_", photoquadrat)
    return f"{m.group('day')}_{m.group('month')}_{m.group('year')}" if m else "UNKNOWN"


def write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    """Write a gzip CSV without a timestamp, so that identical tables give identical files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0, compresslevel=9) as gz:
        gz.write(df.to_csv(index=False).encode())


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample the saved rasters at the CPCe points.")
    parser.add_argument("--tiles-dir", type=Path, default=ROOT / "tiles", help="Folder with points/{split}.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs", help="Folder with predictions/, confidence/ and cover_tables/.")
    parser.add_argument("--tiles-csv", type=Path, default=ROOT / "data/tiles.csv")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    index = pd.read_csv(args.output_dir / "cover_tables/analysis_file_index.csv", usecols=["split", "stem", "base_stem"])
    tiles = pd.read_csv(args.tiles_csv)

    records = []
    skipped = {"label_not_retained": 0, "outside_raster": 0}
    for row in index.itertuples(index=False):
        pts = read_points(args.tiles_dir / "points" / row.split / f"{row.stem}.txt")
        if pts.empty:
            continue
        pred = np.array(Image.open(args.output_dir / "predictions" / row.split / f"{row.stem}.png"))
        conf = np.array(Image.open(args.output_dir / "confidence" / row.split / f"{row.stem}_confidence.png"))
        h, w = pred.shape[:2]
        for pt in pts.itertuples(index=False):
            x, y, cid = int(pt.x), int(pt.y), int(pt.class_id)
            if cid not in CLASS_IDS:
                skipped["label_not_retained"] += 1
                continue
            ix, iy = x - 1, y - 1
            if ix < 0 or iy < 0 or ix >= w or iy >= h:
                skipped["outside_raster"] += 1
                continue
            pred_id = int(pred[iy, ix])
            if conf.shape[:2] != (h, w):
                raise ValueError(f"Confidence raster does not match the class raster for {row.stem}")
            confidence_8bit = int(conf[iy, ix])
            records.append(
                {
                    "split": row.split,
                    "tile": row.stem,
                    "photoquadrat": row.base_stem,
                    "x": x,
                    "y": y,
                    "label_class_id": cid,
                    "predicted_class_id": pred_id,
                    "confidence_8bit": confidence_8bit,
                    "correct": pred_id == cid,
                    "tile_width": w,
                    "tile_height": h,
                }
            )
    points = pd.DataFrame(records)
    write_csv_gz(points, args.data_dir / "point_predictions.csv.gz")

    t = tiles.assign(
        site=tiles.site_folder.map(harmonize_site),
        depth_m=tiles.depth_folder.str.split().str[0].astype(int),
        date=tiles.photoquadrat.map(survey_date),
    )
    photo = (
        t.groupby("photoquadrat", as_index=False)
        .agg(
            split=("split", lambda s: ";".join(sorted(set(s)))),
            site=("site", "first"),
            depth_m=("depth_m", "first"),
            date=("date", "first"),
            n_tiles=("tile", "nunique"),
            n_points=("n_points_tile", "sum"),
        )
        .sort_values(["site", "depth_m", "date", "photoquadrat"])
    )
    photo.to_csv(args.data_dir / "photoquadrats.csv", index=False)

    print(f"points: {len(points)} written, skipped {skipped}")
    print(f"photoquadrats: {len(photo)}")


if __name__ == "__main__":
    main()
