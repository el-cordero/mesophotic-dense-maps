#!/usr/bin/env python3
"""Step 8. Texture of the saved class rasters of the test tiles.

Confirms that each raster is an exact nearest-neighbor enlargement of the 128 x 128 token grid,
and reports the share of grid cells whose class differs from all four neighbors (12.5% in the
paper) and the share a 5 x 5 median filter would change. Needs the prediction rasters from step 3.

Example
    python workflow/08_texture_check.py --output-dir outputs
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Texture check of the test class rasters.")
parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs", help="Folder with predictions/test.")
args = parser.parse_args()

iso = []
exact = []
changed = []
for f in sorted(glob.glob(str(args.output_dir / "predictions/test/*.png"))):
    m = np.array(Image.open(f))
    H, W = m.shape
    ys = ((np.arange(128) + 0.5) * H / 128).astype(int)
    xs = ((np.arange(128) + 0.5) * W / 128).astype(int)
    g = m[np.ix_(ys, xs)]
    exact.append((np.array(Image.fromarray(g).resize((W, H), Image.NEAREST)) == m).mean())
    c = g[1:-1, 1:-1]
    iso.append(((c != g[:-2, 1:-1]) & (c != g[2:, 1:-1]) & (c != g[1:-1, :-2]) & (c != g[1:-1, 2:])).mean())
    p = np.pad(g, 2, mode="edge")
    med = np.median(swv(p, (5, 5)).reshape(128, 128, 25), axis=2).astype(g.dtype)
    changed.append((med != g).mean())

result = {
    "tiles": len(iso),
    "exact_reconstruction_from_grid_min": float(min(exact)),
    "isolated_cells_mean": round(float(np.mean(iso)), 4),
    "changed_by_5x5_median_mean": round(float(np.mean(changed)), 4),
}
for k, v in result.items():
    print(k, v)
(ROOT / "results").mkdir(exist_ok=True)
json.dump(result, open(ROOT / "results/texture_check.json", "w"), indent=1)
