"""Step 6. Every number reported in the paper.

Reads the shared tables in data/ and the training history in models/, and writes
results/results.json and Tables 1-9 of the paper as CSV files in results/tables/.
Runs in under a minute on a laptop and needs no images or GPU.

Example
    python workflow/06_analysis.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from densemap.tables import CLASS_IDS as CLS  # noqa: E402
from densemap.tables import CLASS_LABELS as NAME  # noqa: E402
from densemap.tables import load_cover, load_photoquadrats, load_points  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"


def rounded(value, digits):
    """Round as printed in the paper, without negative zero."""
    return round(value, digits) + 0.0


def write_tables(out, folder):
    """Tables 1-9 of the paper, rounded as printed, from the results dictionary."""

    folder.mkdir(parents=True, exist_ok=True)
    names = [NAME[c] for c in CLS]
    parts = [("train", "Training"), ("val", "Validation"), ("test", "Test")]
    pm = out["point_metrics"]
    ds = out["dataset"]

    # Table 1. Retained CPCe points by class and partition
    t1 = pd.DataFrame({lab: [ds["points_by_split_class"][s][n] for n in names] for s, lab in parts}, index=names)
    t1["Total"] = t1.sum(axis=1)
    t1.loc["All classes"] = t1.sum()
    t1.rename_axis("Class").to_csv(folder / "table_01_points_by_class.csv")

    # Table 2. Agreement with point labels by partition
    surveys = pd.Series([x["split"] for x in out["survey_scale"]["0.1"]]).value_counts()
    ci = out["point_test_ci"]
    rows = []
    for s, lab in parts:
        rows.append(
            {
                "Partition": lab,
                "Surveys": int(surveys[s]),
                "Photoquadrats": int(ds["pq_by_split"][s]),
                "Points": pm[s]["n"],
                "Accuracy": rounded(pm[s]["accuracy"], 3),
                "Accuracy CI low": rounded(ci["acc"][0], 3) if s == "test" else None,
                "Accuracy CI high": rounded(ci["acc"][1], 3) if s == "test" else None,
                "Macro-F1": rounded(pm[s]["macro_f1"], 3),
                "Macro-F1 CI low": rounded(ci["macro_f1"][0], 3) if s == "test" else None,
                "Macro-F1 CI high": rounded(ci["macro_f1"][1], 3) if s == "test" else None,
            }
        )
    pd.DataFrame(rows).to_csv(folder / "table_02_agreement_by_partition.csv", index=False)

    # Table 3. Further point-level metrics
    keys = [
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_precision", "Macro precision"),
        ("macro_recall", "Macro recall"),
        ("weighted_f1", "Weighted F1"),
        ("kappa", "Cohen's kappa"),
    ]
    t3 = pd.DataFrame([{"Partition": lab, **{k2: rounded(pm[s][k], 3) for k, k2 in keys}} for s, lab in parts])
    t3.to_csv(folder / "table_03_point_metrics.csv", index=False)

    # Table 4. Class-specific agreement on the test surveys
    pc = pm["test"]["per_class"]
    t4 = pd.DataFrame(
        [
            {
                "Class": n,
                "Test points": pc[n]["support"],
                "Precision": rounded(pc[n]["precision"], 3),
                "Recall": rounded(pc[n]["recall"], 3),
                "F1": rounded(pc[n]["f1"], 3),
            }
            for n in names
        ]
    )
    t4.to_csv(folder / "table_04_class_agreement_test.csv", index=False)

    # Table 5. Confusion matrix of the test surveys (rows are analyst labels)
    abbr = ["C", "G", "S", "MA", "PEY", "CG", "CA", "SPR"]
    t5 = pd.DataFrame(out["confusion_test"], index=names, columns=abbr)
    t5["Total"] = t5.sum(axis=1)
    t5.loc["Total"] = t5.sum()
    t5.rename_axis("Analyst label").to_csv(folder / "table_05_confusion_test.csv")

    # Table 6. Point-count and unfiltered dense cover of the test photoquadrats
    c0 = out["cover_test"][0]
    nf = out["noise_floor"]
    rows = []
    for n in names:
        r = c0["r_class"][n]
        rows.append(
            {
                "Class": n,
                "Point-count cover (%)": rounded(c0["mean_point"][n], 1),
                "Dense cover (%)": rounded(c0["mean_ret"][n], 1),
                "Mean difference (pp)": rounded(c0["bias_ret"][n], 1),
                "Mean absolute difference (pp)": rounded(c0["mae_ret_class"][n], 1),
                "Expected from sampling (pp)": rounded(nf["per_class"][n]["expected"], 1),
                "Pearson r": None if r is None else rounded(r, 2),
            }
        )
    rows.append(
        {
            "Class": "All classes",
            "Mean absolute difference (pp)": rounded(c0["mae_ret"], 2),
            "Expected from sampling (pp)": rounded(nf["expected"], 2),
        }
    )
    pd.DataFrame(rows).to_csv(folder / "table_06_cover_by_class_test.csv", index=False)

    # Table 7. Survey-scale cover of the three test surveys
    rows = []
    for x in [x for x in out["survey_scale"]["0.1"] if x["split"] == "test"]:
        survey = f"{x['site']} {x['depth']} m {x['date']}"
        for n in names:
            rows.append(
                {
                    "Survey": survey,
                    "Class": n,
                    "Point-count cover (%)": rounded(x["point"][n], 1),
                    "Dense cover (%)": rounded(x["dense"][n], 1),
                }
            )
        rows.append(
            {"Survey": survey, "Class": "Mean absolute difference (pp)", "Dense cover (%)": rounded(x["mae"], 2)}
        )
    pd.DataFrame(rows).to_csv(folder / "table_07_survey_cover_test.csv", index=False)

    # Table 8. Effect of confidence thresholds on the test surveys
    sel = {q["threshold"]: q for q in out["selective_test"]}
    rows = []
    for q in out["cover_test"]:
        th = q["threshold"]
        rows.append(
            {
                "Threshold": th,
                "Area retained (%)": rounded(q["retained_area"], 1),
                "Points retained (%)": rounded(100 * sel[th]["coverage"], 1),
                "Accuracy of retained points": rounded(sel[th]["accuracy"], 3),
                "Difference, retained-pixel denominator (pp)": rounded(q["mae_ret"], 2),
                "CI low": rounded(q["mae_ret_ci"][0], 2),
                "CI high": rounded(q["mae_ret_ci"][1], 2),
                "Difference, full-photoquadrat denominator (pp)": rounded(q["mae_tot"], 2),
                "Dominant class agreement (%)": rounded(100 * q["dom_agree"], 1),
            }
        )
    pd.DataFrame(rows).to_csv(folder / "table_08_thresholds_test.csv", index=False)

    # Table 9. Four-group scheme on the test surveys
    groups = ["Coral", "Other invertebrates", "Algae", "Abiotic substrate"]
    rows = []
    for q in out["coarse"]["thresholds"]:
        if q["threshold"] in (0.1, 0.3, 0.5, 0.7, 0.9):
            rows.append(
                {
                    "Threshold": q["threshold"],
                    "Accuracy of retained points": rounded(q["sel_acc"], 3),
                    "Mean absolute difference (pp)": rounded(q["mae"], 2),
                    **{f"{g} (pp)": rounded(q["bias"][g], 1) for g in groups},
                }
            )
    pd.DataFrame(rows).to_csv(folder / "table_09_four_groups_test.csv", index=False)


# All random draws (bootstrap and sampling simulations) come from this one generator, in this order.
rng = np.random.default_rng(20261002)
out = {}
r = load_points(DATA)
a = load_cover(DATA)
me = load_photoquadrats(DATA)
# Confidence thresholds. ml_ret is cover among retained pixels, ml_tot is cover over the whole photoquadrat.
TH = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
a = a[a.threshold.isin(TH)].copy()
a["ret_pct"] = 100 * a.confident_pixels / a.total_pixels
a["ml_ret"] = np.where(a.confident_pixels > 0, 100 * a.ml_pixel_count / a.confident_pixels.replace(0, np.nan), np.nan)
a["ml_tot"] = 100 * a.ml_pixel_count / a.total_pixels


# ---------- Point-level agreement by partition (Tables 2-5)
def metrics(d):
    acc = d.correct.mean()
    n = len(d)
    P = []
    Rc = []
    F = []
    S = []
    for c in CLS:
        tp = ((d.t == c) & (d.p == c)).sum()
        fp = ((d.t != c) & (d.p == c)).sum()
        fn = ((d.t == c) & (d.p != c)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * rc / (p + rc) if p + rc else 0.0
        P.append(p)
        Rc.append(rc)
        F.append(f)
        S.append(int((d.t == c).sum()))
    S = np.array(S)
    w = S / S.sum()
    pe = sum(((d.t == c).mean()) * ((d.p == c).mean()) for c in CLS)
    kappa = (acc - pe) / (1 - pe)
    return dict(
        n=n,
        accuracy=acc,
        balanced_accuracy=float(np.mean(Rc)),
        macro_precision=float(np.mean(P)),
        macro_recall=float(np.mean(Rc)),
        macro_f1=float(np.mean(F)),
        weighted_f1=float((w * np.array(F)).sum()),
        kappa=kappa,
        per_class={NAME[c]: dict(support=int(S[i]), precision=P[i], recall=Rc[i], f1=F[i]) for i, c in enumerate(CLS)},
    )


out["point_metrics"] = {s: metrics(r[r.split == s]) for s in ["train", "val", "test"]}
t = r[r.split == "test"]
# 95% bootstrap intervals for test accuracy and macro-F1, resampling photoquadrats
stems = t.base_stem.unique()
grp = {k: v for k, v in t.groupby("base_stem")}
accs = []
f1s = []
for _ in range(1000):
    d = pd.concat([grp[s] for s in rng.choice(stems, len(stems))])
    m = metrics(d)
    accs.append(m["accuracy"])
    f1s.append(m["macro_f1"])
out["point_test_ci"] = dict(
    acc=[float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
    macro_f1=[float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
)
cm = pd.crosstab(t.t, t.p).reindex(index=CLS, columns=CLS, fill_value=0)
out["confusion_test"] = cm.values.tolist()
# Most frequent confusions
conf = []
for i in CLS:
    for j in CLS:
        if i != j:
            conf.append((int(cm.loc[i, j]), NAME[i], NAME[j], float(cm.loc[i, j] / cm.loc[i].sum())))
out["top_confusions"] = sorted(conf, reverse=True)[:8]
# ---------- Selective accuracy and calibration (Figure 7, Table 8)
sel = []
for th in TH:
    k = t[t.conf >= th]
    sel.append(dict(threshold=th, coverage=len(k) / len(t), accuracy=float(k.correct.mean()), n=len(k)))
out["selective_test"] = sel


def ece(d, nb=10):
    b = np.linspace(0, 1, nb + 1)
    e = 0
    rows = []
    for lo, hi in zip(b[:-1], b[1:]):
        k = d[(d.conf > lo) & (d.conf <= hi)]
        if len(k):
            e += len(k) / len(d) * abs(k.conf.mean() - k.correct.mean())
            rows.append(dict(lo=lo, hi=hi, n=len(k), conf=float(k.conf.mean()), acc=float(k.correct.mean())))
    return float(e), rows


out["ece_test"], out["reliability_test"] = ece(t)
out["ece_val"], _ = ece(r[r.split == "val"])
out["conf_test"] = dict(
    mean_correct=float(t[t.correct].conf.mean()),
    mean_wrong=float(t[~t.correct].conf.mean()),
    median=float(t.conf.median()),
)
# ---------- Class-specific retention of points and pixels (Figure 10A)
ret = {}
for c in CLS:
    tt = t[t.t == c]
    pp = t[t.p == c]
    ret[NAME[c]] = dict(
        n_true=len(tt),
        n_pred=len(pp),
        true={str(th): float((tt.conf >= th).mean()) if len(tt) else None for th in TH},
        pred={str(th): float((pp.conf >= th).mean()) if len(pp) else None for th in TH},
        prec={str(th): (float(pp[pp.conf >= th].correct.mean()) if (pp.conf >= th).sum() > 0 else None) for th in TH},
        nret={str(th): int((pp.conf >= th).sum()) for th in TH},
    )
out["retention_points_test"] = ret
at = a[a.split == "test"]
px = at.groupby(["class_id", "threshold"]).ml_pixel_count.sum().unstack()
out["retention_pixels_test"] = {NAME[c]: {str(th): float(px.loc[c, th] / px.loc[c, 0.1]) for th in TH} for c in CLS}
out["pixel_share_pred_test"] = {NAME[c]: float(px.loc[c, 0.1] / px[0.1].sum()) for c in CLS}


# ---------- Cover agreement by threshold with both denominators (Tables 6 and 8, Figures 9 and 10B)
def cov_summary(d, nboot=2000):
    res = []
    for th in TH:
        k = d[d.threshold == th]
        P = k.pivot(index="base_stem", columns="class_id", values="point_cover_pct")[CLS].values
        Rt = k.pivot(index="base_stem", columns="class_id", values="ml_ret")[CLS].values
        Tt = k.pivot(index="base_stem", columns="class_id", values="ml_tot")[CLS].values
        ra = k.groupby("base_stem").ret_pct.first().values
        ok = ~np.isnan(Rt).any(1)
        e_r = np.abs(Rt - P).mean(1)
        e_t = np.abs(Tt - P).mean(1)
        idx = rng.integers(0, len(P), (nboot, len(P)))
        ci_r = np.nanpercentile(np.nanmean(e_r[idx], 1), [2.5, 97.5])
        ci_t = np.percentile(e_t[idx].mean(1), [2.5, 97.5])
        dom_p = P.argmax(1)
        dom_r = np.where(ok, np.nan_to_num(Rt).argmax(1), -1)
        res.append(
            dict(
                threshold=th,
                n_pq=len(P),
                n_pq_with_retained=int(ok.sum()),
                retained_area=float(ra.mean()),
                retained_area_min=float(ra.min()),
                mae_ret=float(np.nanmean(e_r)),
                mae_ret_ci=[float(ci_r[0]), float(ci_r[1])],
                rmse_ret=float(np.sqrt(np.nanmean((Rt - P) ** 2))),
                mae_tot=float(e_t.mean()),
                mae_tot_ci=[float(ci_t[0]), float(ci_t[1])],
                dom_agree=float((dom_p[ok] == dom_r[ok]).mean()),
                pearson_ret=float(np.corrcoef(Rt[ok].ravel(), P[ok].ravel())[0, 1]),
                bias_ret={NAME[c]: float(np.nanmean(Rt[:, i] - P[:, i])) for i, c in enumerate(CLS)},
                mae_ret_class={NAME[c]: float(np.nanmean(np.abs(Rt[:, i] - P[:, i]))) for i, c in enumerate(CLS)},
                bias_tot={NAME[c]: float(np.mean(Tt[:, i] - P[:, i])) for i, c in enumerate(CLS)},
                mean_ret={NAME[c]: float(np.nanmean(Rt[:, i])) for i, c in enumerate(CLS)},
                mean_point={NAME[c]: float(P[:, i].mean()) for i, c in enumerate(CLS)},
                r_class={
                    NAME[c]: (
                        float(np.corrcoef(Rt[ok, i], P[ok, i])[0, 1])
                        if np.nanstd(Rt[ok, i]) > 0 and P[ok, i].std() > 0
                        else None
                    )
                    for i, c in enumerate(CLS)
                },
            )
        )
    return res


out["cover_test"] = cov_summary(at)
out["cover_val"] = cov_summary(a[a.split == "val"], 200)
out["cover_train"] = cov_summary(a[a.split == "train"], 200)
out["cover_all"] = cov_summary(a, 200)
# ---------- Difference expected from 100-point sampling alone, by multinomial simulation (Table 6)
k = at[at.threshold == 0.1]
P = k.pivot(index="base_stem", columns="class_id", values="point_cover_pct")[CLS].values / 100
W = k.pivot(index="base_stem", columns="class_id", values="ml_tot")[CLS].values / 100
W = W / W.sum(1, keepdims=True)
N = (
    k.groupby("base_stem")
    .point_total.first()
    .reindex(k.pivot(index="base_stem", columns="class_id", values="ml_tot").index)
    .values.astype(int)
)
sims = []
simc = []
for _ in range(5000):
    S = np.vstack([rng.multinomial(n, w) / n for n, w in zip(N, W)])
    d = np.abs(S - W)
    sims.append(d.mean() * 100)
    simc.append(d.mean(0) * 100)
simc = np.array(simc)
out["noise_floor"] = dict(
    observed=float(np.abs(W - P).mean() * 100),
    expected=float(np.mean(sims)),
    ci=[float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))],
    mean_points=float(N.mean()),
    min_points=int(N.min()),
    per_class={
        NAME[c]: dict(observed=float(np.abs(W[:, i] - P[:, i]).mean() * 100), expected=float(simc[:, i].mean()))
        for i, c in enumerate(CLS)
    },
)
# ---------- Survey-scale cover (Table 7)
me2 = me.set_index("base_stem")


def survey_scale(d, th):
    kk = d[d.threshold == th]
    # pool all points and all retained pixels of a survey
    kk = kk.join(me2[["site", "depth_m", "date"]], on="base_stem")
    rows = []
    for (s, dp, dt), g in kk.groupby(["site", "depth_m", "date"]):
        pc = g.groupby("class_id").point_count.sum()
        pt = pc.sum()
        ml = g.groupby("class_id").ml_pixel_count.sum()
        mt = ml.sum()
        pcov = (100 * pc / pt).reindex(CLS).values
        mcov = (100 * ml / mt).reindex(CLS).values
        rows.append(
            dict(
                site=s,
                depth=int(dp),
                date=dt,
                split=g.split.iloc[0],
                n_pq=int(g.base_stem.nunique()),
                n_points=int(pt),
                mae=float(np.abs(mcov - pcov).mean()),
                max_abs=float(np.abs(mcov - pcov).max()),
                point={NAME[c]: float(pcov[i]) for i, c in enumerate(CLS)},
                dense={NAME[c]: float(mcov[i]) for i, c in enumerate(CLS)},
                retained=float(
                    100
                    * g.groupby("base_stem").confident_pixels.first().sum()
                    / g.groupby("base_stem").total_pixels.first().sum()
                ),
            )
        )
    return rows


out["survey_scale"] = {str(th): survey_scale(a, th) for th in [0.1, 0.5, 0.7, 0.9]}
# Sampling-only difference at the survey scale (about 3400 points per test survey)
ss = [x for x in out["survey_scale"]["0.1"] if x["split"] == "test"]
nf = []
for x in ss:
    w = np.array([x["dense"][NAME[c]] for c in CLS]) / 100
    n = x["n_points"]
    nf.append(
        float(np.mean([np.abs(rng.multinomial(n, w / w.sum()) / n - w / w.sum()).mean() * 100 for _ in range(4000)]))
    )
out["survey_noise_floor"] = nf
# ---------- Four-group class scheme (Table 9)
G = {
    0: "Coral",
    1: "Other invertebrates",
    2: "Other invertebrates",
    4: "Algae",
    5: "Algae",
    6: "Algae",
    7: "Algae",
    8: "Abiotic substrate",
}
GN = ["Coral", "Other invertebrates", "Algae", "Abiotic substrate"]
tg = t.assign(tg=t.t.map(G), pg=t.p.map(G))
accg = float((tg.tg == tg.pg).mean())
f = []
for g in GN:
    tp = ((tg.tg == g) & (tg.pg == g)).sum()
    fp = ((tg.tg != g) & (tg.pg == g)).sum()
    fn = ((tg.tg == g) & (tg.pg != g)).sum()
    f.append(2 * tp / (2 * tp + fp + fn))
coarse = dict(accuracy=accg, macro_f1=float(np.mean(f)), f1=dict(zip(GN, map(float, f))), thresholds=[])
ag = at.assign(g=at.class_id.map(G))
for th in TH:
    kk = ag[ag.threshold == th]
    s = kk.groupby(["base_stem", "g"]).agg(pc=("point_count", "sum"), ml=("ml_pixel_count", "sum")).reset_index()
    tot = kk.groupby("base_stem").agg(pt=("point_total", "first"), cp=("confident_pixels", "first"))
    s = s.join(tot, on="base_stem")
    s["p"] = 100 * s.pc / s.pt
    s["m"] = 100 * s.ml / s.cp
    Pg = s.pivot(index="base_stem", columns="g", values="p")[GN].values
    Mg = s.pivot(index="base_stem", columns="g", values="m")[GN].values
    coarse["thresholds"].append(
        dict(
            threshold=th,
            mae=float(np.nanmean(np.abs(Mg - Pg))),
            bias={g: float(np.nanmean(Mg[:, i] - Pg[:, i])) for i, g in enumerate(GN)},
            sel_acc=float((tg[tg.conf >= th].tg == tg[tg.conf >= th].pg).mean()),
        )
    )
out["coarse"] = coarse
# ---------- Data set and training facts (Table 1, Methods)
man = pd.read_csv(DATA / "tiles.csv")
out["dataset"] = dict(
    tiles=man.groupby("split").size().to_dict(),
    orig_w=man.original_width.value_counts().to_dict(),
    orig_h=man.original_height.value_counts().to_dict(),
    tile_w=[int(man.tile_width.min()), int(man.tile_width.median()), int(man.tile_width.max())],
    tile_h=[int(man.tile_height.min()), int(man.tile_height.median()), int(man.tile_height.max())],
    points_per_tile=[int(man.n_points_tile.min()), float(man.n_points_tile.mean()), int(man.n_points_tile.max())],
    points_by_split_class={
        s: {NAME[c]: int(((r.split == s) & (r.t == c)).sum()) for c in CLS} for s in ["train", "val", "test"]
    },
    pq_by_split=a.groupby("split").base_stem.nunique().to_dict(),
    points_per_pq=[
        int(a.point_total.min()),
        float(a.groupby("base_stem").point_total.first().mean()),
        int(a.point_total.max()),
    ],
)
h = pd.read_csv(ROOT / "models/training_history.csv")
b = h.loc[h.val_accuracy.idxmax()]
out["training"] = dict(
    best_epoch=int(b.epoch),
    best_val_acc=float(b.val_accuracy),
    train_acc_at_best=float(b.train_accuracy),
    last_epoch=int(h.epoch.max()),
    last_train_acc=float(h.train_accuracy.iloc[-1]),
    last_val_acc=float(h.val_accuracy.iloc[-1]),
    val_loss_best=float(b.val_loss),
    val_loss_last=float(h.val_loss.iloc[-1]),
)
RESULTS.mkdir(exist_ok=True)
json.dump(out, open(RESULTS / "results.json", "w"), indent=1, default=float)
write_tables(out, RESULTS / "tables")
print("wrote", RESULTS / "results.json")
