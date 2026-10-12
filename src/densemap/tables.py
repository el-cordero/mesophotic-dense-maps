"""Loaders for the shared tables in data/ used by the analysis and figure scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CLASS_IDS = [0, 1, 2, 4, 5, 6, 7, 8]
CLASS_LABELS = {
    0: "Coral",
    1: "Gorgonians",
    2: "Sponges",
    4: "Macroalgae",
    5: "Peyssonneliaceae",
    6: "Calcareous green algae",
    7: "Coralline algae",
    8: "Sand, pavement, rubble",
}


def load_points(data_dir: Path) -> pd.DataFrame:
    """One row per CPCe point (step 5), with short internal names t (label), p (prediction) and conf (0-1)."""
    r = pd.read_csv(Path(data_dir) / "point_predictions.csv.gz")
    r["correct"] = r["correct"].astype(str).str.lower().eq("true")
    r["confidence"] = r.pop("confidence_8bit") / 255.0
    return r.rename(
        columns={
            "tile": "stem",
            "photoquadrat": "base_stem",
            "confidence": "conf",
            "label_class_id": "t",
            "predicted_class_id": "p",
            "tile_width": "image_width",
            "tile_height": "image_height",
        }
    )


def load_cover(data_dir: Path) -> pd.DataFrame:
    """Point-count and dense cover by photoquadrat, class and confidence threshold (step 4)."""
    a = pd.read_csv(Path(data_dir) / "photoquadrat_cover_by_threshold.csv.gz", float_precision="round_trip")
    a["threshold"] = a.threshold.round(1)
    return a


def load_photoquadrats(data_dir: Path) -> pd.DataFrame:
    """Site, depth, date and partition of each photoquadrat (step 5)."""
    return pd.read_csv(Path(data_dir) / "photoquadrats.csv").rename(columns={"photoquadrat": "base_stem"})
