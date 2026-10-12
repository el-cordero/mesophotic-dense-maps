"""Step 7. Figures 2-10 of the paper.

Reads results/results.json (step 6) and the shared tables in data/, and writes each figure to
outputs/figures/ at 600 and 300 dpi. Figures 4, 5 and 8 show example tiles and need the tile
images, prediction rasters and confidence rasters (--tiles-dir and --output-dir). They are
skipped when those files are not available. Figure 1 (the study area) is not made here.

Example
    python workflow/07_figures.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from densemap.tables import CLASS_IDS as CLS  # noqa: E402
from densemap.tables import CLASS_LABELS as NAME  # noqa: E402
from densemap.tables import load_cover, load_points  # noqa: E402

COL = {0: "#ff4d4d", 1: "#ff944d", 2: "#3399ff", 4: "#33cc66", 5: "#cc3366", 6: "#7fbf3f", 7: "#8b0000", 8: "#d8c28a"}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2de"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
plt.rcParams.update(
    {
        "font.family": "Liberation Sans",
        "font.size": 8,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "savefig.dpi": 600,
        "figure.dpi": 150,
        "axes.grid": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    }
)
W = 6.7
parser = argparse.ArgumentParser(description="Draw Figures 2-10.")
parser.add_argument(
    "--tiles-dir", type=Path, default=ROOT / "tiles", help="Folder with images/test (Figures 4, 5 and 8 only)."
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=ROOT / "outputs",
    help="Folder with predictions/test and confidence/test. Figures are written to its figures/ folder.",
)
args = parser.parse_args()
TILES = args.tiles_dir
OUTPUTS = args.output_dir
FIGS = OUTPUTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
o = json.load(open(ROOT / "results/results.json"))
r = load_points(ROOT / "data")
a = load_cover(ROOT / "data")
IMG = str(TILES / "images/test/{}.jpg")
MSK = str(OUTPUTS / "predictions/test/{}.png")
CNF = str(OUTPUTS / "confidence/test/{}_confidence.png")
# Test tiles shown in the figures. Figure 4 uses sel[8], Figure 5 sel[0], sel[2] and sel[5], Figure 8 sel[4].
sel = [
    "01_Jun_2009_Hole_in_the_Wall_65_meters_HIW_215_E_D-12_tile_r1_c1",
    "04_Mar_2010_Hole_in_the_Wall_48_meters_HIW4_159_E_D-6_tile_r2_c2",
    "01_Jun_2009_Hole_in_the_Wall_65_meters_HIW_220_W_D-9_tile_r1_c2",
    "01_Jun_2009_Hole_in_the_Wall_65_meters_HIW_220_W_D-12_tile_r2_c2",
    "01_Jun_2009_Hole_in_the_Wall_65_meters_HIW_220_W_D-13_tile_r1_c1",
    "04_Mar_2010_Hoyo_Terrrace_64_meters_HT4_220_W_D-12_tile_r1_c1",
    "01_Jun_2009_Hole_in_the_Wall_65_meters_HIW_220_W_D-15_tile_r2_c1",
    "01_Jun_2009_Hole_in_the_Wall_65_meters_HIW_215_E_D-4_tile_r2_c1",
    "04_Mar_2010_Hoyo_Terrrace_64_meters_HT4_211_E_U-14_tile_r1_c3",
    "01_Jun_2009_Hole_in_the_Wall_65_meters_HIW_215_E_D-6_tile_r2_c1",
]
lut = np.ones((9, 3))
for c in CLS:
    lut[c] = to_rgb(COL[c])


def grid128(m):
    H, Wd = m.shape
    ys = ((np.arange(128) + 0.5) * H / 128).astype(int)
    xs = ((np.arange(128) + 0.5) * Wd / 128).astype(int)
    return m[np.ix_(ys, xs)]


def panel_label(ax, s, x=0.0, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def ptitle(ax, letter, text, fs=8, dx=0.085):
    ax.text(0, 1.035, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)
    ax.text(dx, 1.035, text, transform=ax.transAxes, fontsize=fs, va="bottom", ha="left", color=INK)


def class_legend(fig, loc, ncol=4, extra=None):
    hs = [Rectangle((0, 0), 1, 1, fc=COL[c], ec=INK2, lw=0.4) for c in CLS]
    ls = [NAME[c] for c in CLS]
    if extra:
        hs += [extra[0]]
        ls += [extra[1]]
    return fig.legend(
        hs, ls, loc=loc, ncol=ncol, fontsize=7.5, handlelength=1.1, handleheight=0.9, columnspacing=1.2, frameon=False
    )


def save(fig, name):
    fig.savefig(FIGS / f"{name}.png", dpi=600, facecolor="white")
    fig.savefig(FIGS / f"{name}_300.png", dpi=300, facecolor="white")
    plt.close(fig)


# ---------------- Figure 2. Workflow
def figure_02_workflow():
    fig = plt.figure(figsize=(W, 3.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    stages = [
        (
            "Data preparation",
            "R, pointcoral package",
            2,
            [
                ("Index images and CPCe files", "Tile, point file and parent\nphotoquadrat stay linked"),
                ("Harmonize labels", "Eight benthic classes,\nexcluded labels logged"),
                ("Audit coordinates and files", "Image-array convention,\nbounds and completeness"),
                ("Assign splits by survey", "Whole site, depth and\ndate units held out"),
            ],
        ),
        (
            "Dense prediction",
            "Python, PyTorch",
            35.3,
            [
                ("Frozen DINO ViT-B/8", "1024 × 1024 input,\n128 × 128 token grid"),
                ("Point features", "Mean of the 5 × 5 token\nwindow at each point"),
                ("Classifier", "Three hidden layers,\neight output classes"),
                ("Per-token inference", "Class and maximum softmax\nprobability for every token"),
            ],
        ),
        (
            "Evaluation",
            "R and Python",
            68.6,
            [
                ("Aggregate to sampling units", "Tiles to photoquadrats,\nphotoquadrats to surveys"),
                ("Point validation", "Maps sampled at held-out\nCPCe points"),
                ("Cover agreement", "Unfiltered dense cover\nagainst point-count cover"),
                ("Confidence filtering", "Retained area, denominator\nand class retention"),
            ],
        ),
    ]
    bw = 29.4
    for title, sub, x, items in stages:
        ax.add_patch(
            FancyBboxPatch((x, 1.5), bw, 53, boxstyle="round,pad=0,rounding_size=1.2", fc="#f6f5f2", ec=INK2, lw=0.7)
        )
        ax.text(x + 1.6, 51, title, fontsize=9.5, fontweight="bold", color=INK, va="center")
        ax.text(x + 1.6, 47.6, sub, fontsize=7.5, color=INK2, va="center")
        for i, (h, d) in enumerate(items):
            y = 34.6 - i * 10.8
            ax.add_patch(
                FancyBboxPatch(
                    (x + 1.3, y),
                    bw - 2.6,
                    9.7,
                    boxstyle="round,pad=0,rounding_size=0.8",
                    fc="white",
                    ec="#c9c7c0",
                    lw=0.6,
                )
            )
            ax.text(x + 2.3, y + 7.5, h, fontsize=8, fontweight="bold", color=INK, va="center")
            ax.text(x + 2.3, y + 3.5, d, fontsize=7, color=INK2, va="center", linespacing=1.25)
    for x in (31.4, 64.7):
        ax.annotate(
            "",
            xy=(x + 3.9, 28),
            xytext=(x, 28),
            arrowprops=dict(arrowstyle="-|>", lw=1.1, color=INK2, shrinkA=0, shrinkB=0, mutation_scale=9),
        )
    save(fig, "Figure_02_workflow")


# ---------------- Figure 4. Method on a test tile
def figure_04_method():
    s = sel[8]
    im = Image.open(IMG.format(s)).convert("RGB")
    m = np.array(Image.open(MSK.format(s)))
    cf = np.array(Image.open(CNF.format(s))) / 255.0
    d = r[r.stem == s]
    Wd, H = im.size
    cw = Wd / 128
    ch = H / 128
    # choose a point near the middle with room for the window
    d2 = d.assign(dist=(d.x - Wd * 0.62) ** 2 + (d.y - H * 0.35) ** 2).sort_values("dist")
    p = d2.iloc[0]
    fx = min(max(int((p.x - 1) / Wd * 128), 0), 127)
    fy = min(max(int((p.y - 1) / H * 128), 0), 127)
    fig = plt.figure(figsize=(W, 2.45))
    axs = [fig.add_axes([0.012 + i * 0.247, 0.245, 0.235, 0.645]) for i in range(4)]
    ax = axs[0]
    ax.imshow(im)
    ax.scatter(d.x, d.y, c=[COL[c] for c in d.t], s=20, edgecolors="white", linewidths=0.7, zorder=3)
    ax.add_patch(Rectangle(((fx - 2) * cw, (fy - 2) * ch), 5 * cw, 5 * ch, fill=False, ec="white", lw=1.4, zorder=4))
    ax.add_patch(Rectangle(((fx - 2) * cw, (fy - 2) * ch), 5 * cw, 5 * ch, fill=False, ec=INK, lw=0.6, zorder=5))
    ptitle(ax, "A", "Tile with CPCe points")
    ax = axs[1]
    z = 9
    x0 = (fx - z) * cw
    x1 = (fx + z + 1) * cw
    y0 = (fy - z) * ch
    y1 = (fy + z + 1) * ch
    ax.imshow(im)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    for i in range(fx - z, fx + z + 2):
        ax.axvline(i * cw, color="white", lw=0.25, alpha=0.7)
    for j in range(fy - z, fy + z + 2):
        ax.axhline(j * ch, color="white", lw=0.25, alpha=0.7)
    ax.add_patch(Rectangle(((fx - 2) * cw, (fy - 2) * ch), 5 * cw, 5 * ch, fc="white", alpha=0.28, ec="none"))
    ax.add_patch(Rectangle(((fx - 2) * cw, (fy - 2) * ch), 5 * cw, 5 * ch, fill=False, ec="white", lw=1.6))
    ax.add_patch(Rectangle(((fx - 2) * cw, (fy - 2) * ch), 5 * cw, 5 * ch, fill=False, ec=INK, lw=0.6))
    ax.scatter([p.x], [p.y], c=[COL[p.t]], s=42, edgecolors="white", linewidths=0.9, zorder=5)
    ptitle(ax, "B", "Token grid, 5 × 5 window")
    ax = axs[2]
    ax.imshow(lut[grid128(m)], interpolation="nearest")
    ptitle(ax, "C", "Predicted class per token")
    ax = axs[3]
    h = ax.imshow(grid128((cf * 255).astype(np.uint8)) / 255.0, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
    ptitle(ax, "D", "Confidence per token")
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        [sp.set_visible(True) or sp.set_linewidth(0.5) for sp in ax.spines.values()]
    class_legend(fig, (0.012, 0.02), ncol=4)
    cax = fig.add_axes([0.78, 0.15, 0.19, 0.03])
    cb = fig.colorbar(h, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.4)
    cb.set_label("Confidence", fontsize=7.5, labelpad=1)
    save(fig, "Figure_04_method")
    return dict(stem=s, acc=float(d.correct.mean()), n=len(d), w=Wd, h=H)


# ---------------- Figure 3. Survey split
def figure_03_survey_split():
    ss = o["survey_scale"]["0.1"]
    rows = [
        ("Hole in the Wall", 48, "Hole-in-the-Wall, 48 m"),
        ("Hole in the Wall", 65, "Hole-in-the-Wall, 65 m"),
        ("Hoyo Terrace", 48, "El Hoyo, 48 m"),
        ("Hoyo Terrace", 64, "El Hoyo, 64 m"),
    ]
    cols = [
        ("01_Jun_2009", "June 2009"),
        ("02_Sep_2009", "September 2009"),
        ("03_Dic_2009", "December 2009"),
        ("04_Mar_2010", "March 2010"),
    ]
    sty = {"train": ("#e9eef5", "Training"), "val": ("#cfeee2", "Validation"), "test": ("#f9d3c3", "Test")}
    fig = plt.figure(figsize=(W, 2.25))
    ax = fig.add_axes([0.21, 0.04, 0.78, 0.82])
    ax.set_xlim(0, 4)
    ax.set_ylim(4, 0)
    ax.axis("off")
    for i, (s, dp, lab) in enumerate(rows):
        fig.text(0.20, 0.04 + 0.82 * (1 - (i + 0.5) / 4), lab, ha="right", va="center", fontsize=8, color=INK)
        for j, (dt, dl) in enumerate(cols):
            x = [q for q in ss if q["site"] == s and q["depth"] == dp and q["date"] == dt][0]
            ax.add_patch(
                FancyBboxPatch(
                    (j + 0.04, i + 0.07),
                    0.92,
                    0.86,
                    boxstyle="round,pad=0,rounding_size=0.05",
                    fc=sty[x["split"]][0],
                    ec=INK2 if x["split"] != "train" else "#b9c2cf",
                    lw=0.9 if x["split"] != "train" else 0.5,
                )
            )
            ax.text(
                j + 0.5,
                i + 0.40,
                sty[x["split"]][1],
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold" if x["split"] != "train" else "normal",
                color=INK,
            )
            ax.text(j + 0.5, i + 0.68, f"{x['n_pq']} photoquadrats", ha="center", va="center", fontsize=7, color=INK2)
    for j, (dt, dl) in enumerate(cols):
        fig.text(0.21 + 0.78 * (j + 0.5) / 4, 0.89, dl, ha="center", va="bottom", fontsize=8, color=INK)
    save(fig, "Figure_03_survey_split")


# ---------------- Figure 5. Example test tiles
def figure_05_examples():
    picks = [(sel[0], "A"), (sel[2], "B"), (sel[5], "C")]
    info = []
    fig = plt.figure(figsize=(W, 5.75))
    for i, (s, lab) in enumerate(picks):
        im = Image.open(IMG.format(s)).convert("RGB")
        m = np.array(Image.open(MSK.format(s)))
        cf = np.array(Image.open(CNF.format(s)))
        d = r[r.stem == s]
        y = 0.69 - i * 0.295
        axs = [fig.add_axes([0.045 + k * 0.318, y, 0.305, 0.275]) for k in range(3)]
        axs[0].imshow(im)
        axs[0].scatter(d.x, d.y, c=[COL[c] for c in d.t], s=22, edgecolors="white", linewidths=0.7)
        axs[1].imshow(lut[grid128(m)], interpolation="nearest")
        h = axs[2].imshow(grid128(cf) / 255.0, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("auto")
            [sp.set_visible(True) or sp.set_linewidth(0.5) for sp in ax.spines.values()]
        fig.text(0.012, y + 0.1375, lab, fontsize=10, fontweight="bold", va="center", color=INK)
        if i == 0:
            for ax, tt in zip(axs, ["Image with CPCe points", "Predicted class", "Confidence"]):
                ax.set_title(tt, fontsize=8.5, fontweight="normal", pad=3)
        info.append(dict(stem=s, n=int(len(d)), correct=int(d.correct.sum()), acc=float(d.correct.mean())))
    class_legend(fig, (0.045, 0.015), ncol=4)
    cax = fig.add_axes([0.78, 0.058, 0.19, 0.013])
    cb = fig.colorbar(h, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.4)
    cb.set_label("Confidence", fontsize=7.5, labelpad=1)
    save(fig, "Figure_05_examples")
    return info


# ---------------- Figure 7. Calibration and selective accuracy
def figure_07_confidence():
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.75))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.90, wspace=0.26)
    ax = axs[0]
    rel = o["reliability_test"]
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.7, ls=(0, (4, 3)))
    x = [q["conf"] for q in rel]
    y = [q["acc"] for q in rel]
    n = np.array([q["n"] for q in rel])
    ax.plot(x, y, color=BLUE, lw=1.6, zorder=3)
    ax.scatter(x, y, s=10 + n / 22, color=BLUE, edgecolors="white", linewidths=0.9, zorder=4)
    ax.text(
        0.70,
        0.80,
        "Perfect calibration",
        fontsize=7,
        color=INK2,
        rotation=40,
        ha="center",
        va="center",
        rotation_mode="anchor",
    )
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Share of points classified correctly")
    ax.set_title("Reliability of confidence at test points", loc="left")
    panel_label(ax, "A", x=-0.13, y=1.03)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.text(
        0.985,
        0.05,
        f"Expected calibration error {o['ece_test']:.3f}\nMarker area scales with points per bin",
        transform=ax.transAxes,
        fontsize=7,
        color=INK2,
        ha="right",
        va="bottom",
    )
    ax = axs[1]
    se = o["selective_test"]
    th = [q["threshold"] for q in se]
    ax.plot(
        th,
        [q["accuracy"] for q in se],
        color=BLUE,
        lw=1.8,
        marker="o",
        ms=4.5,
        mec="white",
        mew=0.9,
        label="Accuracy among retained points",
    )
    ax.plot(
        th,
        [q["coverage"] for q in se],
        color=ORANGE,
        lw=1.8,
        marker="s",
        ms=4.3,
        mec="white",
        mew=0.9,
        label="Share of points retained",
    )
    ax.text(0.9, se[-1]["accuracy"] - 0.075, f"{se[-1]['accuracy']:.2f}", fontsize=7.5, color=INK, ha="center")
    ax.text(0.9, se[-1]["coverage"] + 0.045, f"{se[-1]['coverage']:.2f}", fontsize=7.5, color=INK, ha="center")
    ax.text(0.1, se[0]["accuracy"] - 0.075, f"{se[0]['accuracy']:.2f}", fontsize=7.5, color=INK, ha="center")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(th)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Proportion")
    ax.set_title("Accuracy and retention under filtering", loc="left")
    panel_label(ax, "B", x=-0.13, y=1.03)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02), handlelength=1.8)
    save(fig, "Figure_07_confidence")


# ---------------- Figure 8. Threshold sequence on one tile
def figure_08_threshold_sequence():
    s = sel[4]
    im = Image.open(IMG.format(s)).convert("RGB")
    m = grid128(np.array(Image.open(MSK.format(s))))
    cf = grid128(np.array(Image.open(CNF.format(s)))) / 255.0
    d = r[r.stem == s]
    ths = [None, 0.5, 0.7, 0.9]
    UN = np.array(to_rgb("#eceae4"))
    fig = plt.figure(figsize=(W, 1.98))
    axs = [fig.add_axes([0.008 + i * 0.1988, 0.24, 0.190, 0.645]) for i in range(5)]
    axs[0].imshow(im)
    axs[0].scatter(d.x, d.y, c=[COL[c] for c in d.t], s=16, edgecolors="white", linewidths=0.6)
    kept = []
    for ax, th in zip(axs[1:], ths):
        rgb = lut[m].copy()
        if th is not None:
            rgb[cf < th] = UN
            kept.append(float((cf >= th).mean()))
        ax.imshow(rgb, interpolation="nearest")
        ax._ttl = "Unfiltered" if th is None else f"Confidence ≥ {th:.2f}"
        if th is not None:
            ax.text(
                0.5,
                -0.045,
                f"{100 * (cf >= th).mean():.0f}% of tile retained",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=7,
                color=INK2,
            )
    for ax, l in zip(axs, "ABCDE"):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("auto")
        [sp.set_visible(True) or sp.set_linewidth(0.5) for sp in ax.spines.values()]
        ptitle(ax, l, getattr(ax, "_ttl", "Image with points"), fs=7.6, dx=0.10)
    class_legend(fig, (0.008, 0.0), ncol=5, extra=(Rectangle((0, 0), 1, 1, fc=UN, ec=INK2, lw=0.4), "Below threshold"))
    save(fig, "Figure_08_threshold_sequence")
    return dict(stem=s, kept=kept, acc=float(d.correct.mean()), n=len(d))


# ---------------- Figure 9. Response to thresholds
def figure_09_threshold_response():
    ct = o["cover_test"]
    th = [q["threshold"] for q in ct]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.8))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.155, top=0.90, wspace=0.24)
    ax = axs[0]
    at = (
        a[(a.split == "test") & (a.threshold.isin(th))]
        .drop_duplicates(["base_stem", "threshold"])
        .assign(ret=lambda d: 100 * d.confident_pixels / d.total_pixels)
    )
    g = at.groupby("threshold").ret
    lo = g.quantile(0.05)
    hi = g.quantile(0.95)
    ax.fill_between(th, lo.values, hi.values, color=BLUE, alpha=0.12, lw=0)
    ax.plot(th, [q["retained_area"] for q in ct], color=BLUE, lw=1.8, marker="o", ms=4.5, mec="white", mew=0.9)
    for t_ in (0.5, 0.7, 0.9):
        q = [x for x in ct if x["threshold"] == t_][0]
        ax.text(t_ + 0.012, q["retained_area"] + 4.5, f"{q['retained_area']:.0f}%", fontsize=7.5, color=INK, ha="left")
    ax.set_ylim(0, 105)
    ax.set_xticks(th)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Photoquadrat area retained (%)")
    ax.set_title("Area retained", loc="left")
    panel_label(ax, "A", x=-0.13, y=1.03)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.text(
        0.03,
        0.06,
        "Line shows the mean of 102 test photoquadrats\nBand spans the 5th to 95th percentile",
        transform=ax.transAxes,
        fontsize=7,
        color=INK2,
    )
    ax = axs[1]
    ax.fill_between(
        th, [q["mae_ret_ci"][0] for q in ct], [q["mae_ret_ci"][1] for q in ct], color=BLUE, alpha=0.14, lw=0
    )
    ax.fill_between(
        th, [q["mae_tot_ci"][0] for q in ct], [q["mae_tot_ci"][1] for q in ct], color=ORANGE, alpha=0.14, lw=0
    )
    ax.plot(
        th,
        [q["mae_ret"] for q in ct],
        color=BLUE,
        lw=1.8,
        marker="o",
        ms=4.5,
        mec="white",
        mew=0.9,
        label="Composition of retained pixels",
    )
    ax.plot(
        th,
        [q["mae_tot"] for q in ct],
        color=ORANGE,
        lw=1.8,
        marker="s",
        ms=4.3,
        mec="white",
        mew=0.9,
        label="Full photoquadrat denominator",
    )
    nf = o["noise_floor"]["expected"]
    ax.axhline(nf, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.text(0.1, nf - 0.25, "Expected from 100-point sampling alone", fontsize=7, color=INK2, va="top", ha="left")
    ax.text(
        0.935, ct[-1]["mae_ret"] - 0.15, f"{ct[-1]['mae_ret']:.1f}", fontsize=7.5, color=INK, ha="left", va="center"
    )
    ax.text(0.935, ct[-1]["mae_tot"], f"{ct[-1]['mae_tot']:.1f}", fontsize=7.5, color=INK, ha="left", va="center")
    ax.text(0.1, ct[0]["mae_ret"] + 0.62, f"{ct[0]['mae_ret']:.1f}", fontsize=7.5, color=INK, ha="center")
    ax.set_xlim(0.06, 1.0)
    ax.set_ylim(0, 12.5)
    ax.set_xticks(th)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Mean absolute difference (pp)")
    ax.set_title("Difference from point-count cover", loc="left")
    panel_label(ax, "B", x=-0.13, y=1.03)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", handlelength=1.8)
    save(fig, "Figure_09_threshold_response")


# ---------------- Figure 10. Class-specific response
def figure_10_class_response():
    ct = o["cover_test"]
    th = [q["threshold"] for q in ct]
    rp = o["retention_pixels_test"]
    A = np.array([[100 * rp[NAME[c]][str(t)] for t in th] for c in CLS])
    B = np.array([[q["bias_ret"][NAME[c]] for q in ct] for c in CLS])
    seq = LinearSegmentedColormap.from_list("s", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"])
    div = LinearSegmentedColormap.from_list("d", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a58f", "#c2402a"])
    fig = plt.figure(figsize=(W, 2.95))
    ax1 = fig.add_axes([0.205, 0.13, 0.372, 0.74])
    ax2 = fig.add_axes([0.622, 0.13, 0.372, 0.74])
    for ax, M, cm, vmin, vmax, fmt, ttl, lab in [
        (ax1, A, seq, 0, 100, "{:.0f}", "Predicted pixels retained (%)", "A"),
        (ax2, B, div, -18, 18, "{:+.1f}", "Difference from point cover (pp)", "B"),
    ]:
        ax.imshow(M, cmap=cm, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                dark = (v > 62) if ax is ax1 else (abs(v) > 11)
                ax.text(
                    j,
                    i,
                    fmt.format(v).replace("-", "−").replace("+0.0", "0.0").replace("−0.0", "0.0"),
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    color="white" if dark else INK,
                )
        ax.set_xticks(range(len(th)))
        ax.set_xticklabels([f"{t:.1f}" for t in th])
        ax.set_xlabel("Confidence threshold")
        ax.set_title(ttl, loc="left")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(th)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(CLS)), minor=True)
        ax.grid(which="minor", color="white", lw=1.4)
        ax.tick_params(which="minor", length=0)
        panel_label(ax, lab, x=-0.065 if ax is ax2 else -0.5, y=1.03)
    ax1.set_yticks(range(len(CLS)))
    ax1.set_yticklabels([NAME[c] for c in CLS], fontsize=7.8, color=INK)
    ax2.set_yticks([])
    save(fig, "Figure_10_class_response")


# ---------------- Figure 6. Cover scatter by class
def figure_06_cover_scatter():
    at = a[a.split == "test"]
    fig, axs = plt.subplots(2, 4, figsize=(W, 3.75))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.185, top=0.93, wspace=0.34, hspace=0.42)
    for ax, c in zip(axs.ravel(), CLS):
        k1 = at[(at.threshold == 0.1) & (at.class_id == c)]
        k9 = at[(at.threshold == 0.9) & (at.class_id == c)]
        y9 = 100 * k9.ml_pixel_count / k9.confident_pixels
        y1 = 100 * k1.ml_pixel_count / k1.confident_pixels
        mx = max(k1.point_cover_pct.max(), y1.max(), y9.max())
        mx = max(5, np.ceil(mx / 10) * 10) if mx > 10 else max(5, np.ceil(mx / 5) * 5)
        ax.plot([0, mx], [0, mx], color=INK2, lw=0.6, ls=(0, (4, 3)), zorder=1)
        ax.scatter(
            k9.point_cover_pct,
            y9,
            s=9,
            color=ORANGE,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.3,
            zorder=2,
            marker="s",
        )
        ax.scatter(k1.point_cover_pct, y1, s=10, color=BLUE, alpha=0.8, edgecolors="white", linewidths=0.3, zorder=3)
        ax.set_xlim(-mx * 0.03, mx * 1.03)
        ax.set_ylim(-mx * 0.03, mx * 1.03)
        ax.set_aspect("equal")
        ax.set_title(NAME[c], fontsize=7.8, loc="left", pad=3)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    fig.text(0.53, 0.095, "Point-count cover (%)", ha="center", fontsize=8.5, color=INK)
    fig.text(0.012, 0.56, "Dense-prediction cover (%)", va="center", rotation=90, fontsize=8.5, color=INK)
    hs = [
        plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=5, mec="white", mew=0.5),
        plt.Line2D([], [], marker="s", ls="", color=ORANGE, ms=5, mec="white", mew=0.5),
        plt.Line2D([], [], color=INK2, lw=0.7, ls=(0, (4, 3))),
    ]
    fig.legend(
        hs,
        ["Unfiltered", "Retained pixels at confidence ≥ 0.90", "Equal cover"],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.53, 0.0),
        handlelength=1.8,
        columnspacing=2,
    )
    save(fig, "Figure_06_cover_scatter")


def tiles_available(stems):
    return all(os.path.exists(p.format(s)) for s in stems for p in (IMG, MSK, CNF))


figure_02_workflow()
figure_03_survey_split()
figure_06_cover_scatter()
figure_07_confidence()
figure_09_threshold_response()
figure_10_class_response()
if tiles_available([sel[0], sel[2], sel[4], sel[5], sel[8]]):
    figure_04_method()
    figure_05_examples()
    figure_08_threshold_sequence()
else:
    print("Tile images or rasters not found, so Figures 4, 5 and 8 were skipped.")
print("Figures written to", FIGS)
