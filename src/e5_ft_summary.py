"""Item 10 comparison: base Any4D vs the fine-tuned variants. numpy + matplotlib only.

  e5_ft_summary.py QUEUE_RESULTS_DIR OUT_DIR TASK_ID

Reads QUEUE_RESULTS_DIR/ft_eval_<variant>/ (kubric_<ds>.json, <real>_wide.csv, probeP_v3float_results.json).
Statistics are per scene and paired by scene name:
  value  = mean over scenes (slope, sep_ratio, mask IoU/AUC, geometry errors) with a bootstrap 95% CI over scenes;
           EPEs are per-scene medians (round-1 definition), summarised by their mean over scenes with the same CI
           (the median over scenes is also listed in the CSV);
  delta  = variant minus reference (base, or ctrl for the head-specificity test), mean paired difference over scenes,
           bootstrap 95% CI (2000 resamples of scenes, seed 0) and a two-sided paired sign test.
"""
import csv, glob, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QR, OUT, TID = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
VARS = ["base", "heads", "sf", "heads_sf", "ctrl", "gain"]
LABEL = {"base": "base", "heads": "(a) heads-LoRA", "sf": "(b) sf-decoder", "heads_sf": "(c) heads+sf",
         "ctrl": "(d) control heads", "gain": "(e) head gains"}
KUB = ["v2", "v3float", "v3far"]
REAL = ["kitti15_rgb_2v_wide", "kitti15_rgb_mv4_wide", "sintel_final_2v_wide", "sintel_final_mv4_wide", "sintel_clean_2v_wide"]
KM = ["slope", "epe_static", "epe_moving", "sep_ratio", "mask_iou", "mask_auc", "depth_absrel_med", "depth_absrel",
      "rot_err_deg", "tdir_err_deg", "trans_relerr_med"]
RM = ["slope", "epe_static", "epe_moving", "sep_ratio"]
LOWER_BETTER = {"epe_static", "epe_moving", "depth_absrel_med", "depth_absrel", "rot_err_deg", "tdir_err_deg", "trans_relerr_med"}


def load(v):
    MAP = dict(kv.split("=") for kv in os.environ.get("FT_MAP", "").split(",") if "=" in kv)
    d = os.path.join(QR, f"ft_eval_{MAP.get(v, v)}"); D = {}
    if not os.path.isdir(d):
        return D
    for ds in KUB:
        p = os.path.join(d, f"kubric_{ds}.json")
        if os.path.exists(p):
            rows = [r for r in json.load(open(p)) if "error" not in r]
            D[ds] = {r["scene"]: r for r in rows}
            D[ds + "_epi"] = {r["scene"]: r for r in rows if r["split"] == "epi_aligned"}
    for nm in REAL:
        p = os.path.join(d, nm + ".csv")
        if os.path.exists(p):
            D[nm] = {r["scene"]: {k: (float(x) if x not in ("", None, "nan") and k != "scene" else (np.nan if k != "scene" else x))
                                  for k, x in r.items() if k in RM + ["scene"]} for r in csv.DictReader(open(p))}
    p = os.path.join(d, "probeP_v3float_results.json")
    if os.path.exists(p):
        P = json.load(open(p))["P"]["rows"]
        for l in sorted({r["layer"] for r in P}):
            for key in ("motion_auc_v0_objects", "motion_auc_v0"):
                for tag, filt in (("", lambda r: True), ("_epi", lambda r: r["split"] == "epi_aligned")):
                    D.setdefault(f"probe{tag}", {})
                    for r in P:
                        if r["layer"] == l and filt(r):
                            D[f"probe{tag}"].setdefault(r["scene"], {})[f"{key}_L{l}"] = r.get(key, np.nan)
    return D


R = {v: load(v) for v in VARS}
rng = np.random.default_rng(0)


def boot_ci(x, n=2000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    m = rng.integers(0, len(x), (n, len(x)))
    b = x[m].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def sign_p(d):
    d = np.asarray(d, float); d = d[np.isfinite(d) & (d != 0)]
    n = len(d); k = int((d > 0).sum())
    if n == 0:
        return np.nan
    t = min(k, n - k)
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(t + 1)) / 2 ** n))


def stat(v, ds, m, ref="base"):
    A = R.get(v, {}).get(ds, {}); B = R.get(ref, {}).get(ds, {})
    x = np.array([A[s].get(m, np.nan) for s in A], float)
    out = dict(variant=v, dataset=ds, metric=m, n=int(np.isfinite(x).sum()), mean=float(np.nanmean(x)) if len(x) else np.nan,
               median=float(np.nanmedian(x)) if len(x) else np.nan)
    out["ci_lo"], out["ci_hi"] = boot_ci(x)
    common = sorted(set(A) & set(B))
    if v != ref and common:
        d = np.array([A[s].get(m, np.nan) - B[s].get(m, np.nan) for s in common], float)
        out.update(ref=ref, n_paired=int(np.isfinite(d).sum()), delta=float(np.nanmean(d)), sign_p=sign_p(d),
                   frac_up=float(np.mean(d[np.isfinite(d)] > 0)) if np.isfinite(d).any() else np.nan)
        out["delta_lo"], out["delta_hi"] = boot_ci(d)
    return out


rows = []
present = [v for v in VARS if R[v]]
for v in present:
    for ds in KUB + [k + "_epi" for k in KUB]:
        for m in KM:
            rows.append(stat(v, ds, m))
    for ds in REAL:
        for m in RM:
            rows.append(stat(v, ds, m))
    for tag in ("probe", "probe_epi"):
        keys = sorted({k for s in R[v].get(tag, {}).values() for k in s})
        for m in keys:
            rows.append(stat(v, tag, m))
for ds in KUB + [k + "_epi" for k in KUB] + REAL:   # head specificity: (a) vs (d) and (c) vs (d)
    for m in (KM if ds in KUB or ds.endswith("_epi") else RM):
        for v in ("heads", "heads_sf"):
            if R.get(v) and R.get("ctrl"):
                rows.append(dict(stat(v, ds, m, ref="ctrl"), contrast=f"{v}_vs_ctrl"))
keys = ["contrast", "variant", "dataset", "metric", "n", "mean", "ci_lo", "ci_hi", "median", "ref", "n_paired", "delta",
        "delta_lo", "delta_hi", "sign_p", "frac_up"]
with open(os.path.join(OUT, "ft_comparison_long.csv"), "w", newline="") as fh:
    wr = csv.DictWriter(fh, keys, extrasaction="ignore"); wr.writeheader()
    for r in rows:
        wr.writerow({k: (f"{r[k]:.5g}" if isinstance(r.get(k), float) else r.get(k, "")) for k in keys})
json.dump(rows, open(os.path.join(OUT, "ft_comparison_long.json"), "w"), indent=1)
S = {(r.get("contrast", ""), r["variant"], r["dataset"], r["metric"]): r for r in rows}


def cell(v, ds, m, contrast=""):
    r = S.get((contrast, v, ds, m))
    if not r or not np.isfinite(r["mean"]):
        return "-"
    s = f"{r['mean']:.3f}" if abs(r["mean"]) < 10 else f"{r['mean']:.1f}"
    if "delta" in r and np.isfinite(r.get("delta", np.nan)):
        f = (lambda x: f"{x:+.3f}") if abs(r["delta"]) < 10 else (lambda x: f"{x:+.1f}")
        s += f" ({f(r['delta'])} [{f(r['delta_lo'])}, {f(r['delta_hi'])}], p={r['sign_p']:.2g})"
    return s


MAIN = [("v2", "slope"), ("v2", "epe_static"), ("v2", "epe_moving"), ("v2", "sep_ratio"), ("v2", "mask_iou"), ("v2_epi", "mask_iou"),
        ("v3float", "slope"), ("v3float", "epe_static"), ("v3float", "epe_moving"), ("v3float", "sep_ratio"), ("v3float", "mask_iou"),
        ("v3float_epi", "mask_iou"), ("v3float_epi", "mask_auc"), ("v3far", "slope"), ("v3far", "epe_moving"), ("v3far", "sep_ratio"),
        ("v2", "depth_absrel_med"), ("v2", "rot_err_deg"), ("v2", "trans_relerr_med"),
        ("kitti15_rgb_2v_wide", "slope"), ("kitti15_rgb_2v_wide", "epe_static"), ("kitti15_rgb_2v_wide", "epe_moving"),
        ("kitti15_rgb_mv4_wide", "slope"), ("sintel_final_2v_wide", "slope"), ("sintel_final_2v_wide", "epe_moving"),
        ("sintel_final_2v_wide", "sep_ratio"), ("sintel_final_mv4_wide", "slope"), ("sintel_clean_2v_wide", "slope"),
        ("probe_epi", "motion_auc_v0_objects_L24"), ("probe", "motion_auc_v0_objects_L24")]
L = [f"# {TID}: item 10, head-targeted and decoder-side fine-tuning of Any4D", "",
     "Each cell: mean over scenes (EPE: mean of per-scene medians, px), then in brackets the paired change vs base "
     "with a bootstrap 95% CI over scenes and the two-sided paired sign-test p. Lower is better for EPE and geometry errors.", "",
     "| dataset | metric | " + " | ".join(LABEL[v] for v in present) + " |", "|---|---|" + "---|" * len(present)]
for ds, m in MAIN:
    L.append(f"| {ds} | {m} | " + " | ".join(cell(v, ds, m) for v in present) + " |")
L += ["", "## Are the gains specific to the five heads? (paired vs the matched control heads, variant d)", "",
      "| dataset | metric | (a) heads-LoRA vs (d) | (c) heads+sf vs (d) |", "|---|---|---|---|"]
for ds, m in MAIN[:16] + MAIN[19:28]:
    L.append(f"| {ds} | {m} | {cell('heads', ds, m, 'heads_vs_ctrl')} | {cell('heads_sf', ds, m, 'heads_sf_vs_ctrl')} |")


def g(v, ds, m, k="mean"):
    r = S.get(("", v, ds, m)); return r[k] if r and k in r else np.nan


L += ["", "## Questions", ""]
best = max([v for v in present if v != "base"], key=lambda v: g(v, "v2", "slope") if np.isfinite(g(v, "v2", "slope")) else -9, default=None)
if best:
    L.append(f"- Largest Kubric v2 slope: {LABEL[best]} ({g(best, 'v2', 'slope'):.3f} vs base {g('base', 'v2', 'slope'):.3f}); "
             f"moving EPE {g(best, 'v2', 'epe_moving'):.1f} vs {g('base', 'v2', 'epe_moving'):.1f} px; static EPE {g(best, 'v2', 'epe_static'):.2f} vs {g('base', 'v2', 'epe_static'):.2f} px.")
for v in [x for x in present if x != "base"]:
    L.append(f"- {LABEL[v]}: KITTI 2-view slope change {g(v, 'kitti15_rgb_2v_wide', 'slope', 'delta'):+.3f} "
             f"(p={g(v, 'kitti15_rgb_2v_wide', 'slope', 'sign_p'):.2g}), Sintel final 2-view {g(v, 'sintel_final_2v_wide', 'slope', 'delta'):+.3f} "
             f"(p={g(v, 'sintel_final_2v_wide', 'slope', 'sign_p'):.2g}); floating epi mask IoU {g(v, 'v3float_epi', 'mask_iou'):.3f} "
             f"(base {g('base', 'v3float_epi', 'mask_iou'):.3f}), layer-24 probe AUC on the same scenes {g(v, 'probe_epi', 'motion_auc_v0_objects_L24'):.3f}.")
L += ["", "Automatic text above only reports numbers; interpretation is left to the write-up. Full statistics: ft_comparison_long.csv."]
open(os.path.join(OUT, f"{TID}_summary.md"), "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- figure
cols = {"base": "#7f7f7f", "heads": "#1f77b4", "sf": "#ff7f0e", "heads_sf": "#2ca02c", "ctrl": "#9467bd", "gain": "#8c564b"}
panels = [("slope", ["v2", "v3float", "v3far", "kitti15_rgb_2v_wide", "sintel_final_2v_wide", "sintel_clean_2v_wide"], "slope (1 = full object motion)"),
          ("epe_moving", ["v2", "v3float", "v3far", "kitti15_rgb_2v_wide", "sintel_final_2v_wide"], "EPE moving (mean of scene medians, px)"),
          ("epe_static", ["v2", "v3float", "v3far", "kitti15_rgb_2v_wide", "sintel_final_2v_wide"], "EPE static (px)")]
short = {"v2": "Kubric v2", "v3float": "floating", "v3far": "far-glued", "kitti15_rgb_2v_wide": "KITTI 2v",
         "sintel_final_2v_wide": "Sintel final 2v", "sintel_clean_2v_wide": "Sintel clean 2v"}
fig, axs = plt.subplots(1, 4, figsize=(17, 3.8), gridspec_kw=dict(width_ratios=[1.3, 1.1, 1.1, 0.9]))
for ax, (m, dss, title) in zip(axs[:3], panels):
    wdt = 0.8 / max(1, len(present))
    for j, v in enumerate(present):
        y = [g(v, ds, m) for ds in dss]; lo = [g(v, ds, m, "ci_lo") for ds in dss]; hi = [g(v, ds, m, "ci_hi") for ds in dss]
        x = np.arange(len(dss)) + (j - (len(present) - 1) / 2) * wdt
        ax.bar(x, y, wdt, color=cols[v], label=LABEL[v])
        ax.errorbar(x, y, yerr=[np.array(y) - np.array(lo), np.array(hi) - np.array(y)], fmt="none", ecolor="k", lw=0.7, capsize=1.5)
    if m == "slope":
        ax.axhline(1, color="0.4", lw=0.8, ls="--")
    ax.set_xticks(range(len(dss))); ax.set_xticklabels([short[d] for d in dss], rotation=25, ha="right"); ax.set_title(title)
ax = axs[3]
for v in present:
    xo, yo = g(v, "probe_epi", "motion_auc_v0_objects_L24"), g(v, "v3float_epi", "mask_auc")
    ax.scatter([xo], [yo], color=cols[v], s=40, label=LABEL[v], zorder=3)
ax.plot([0.5, 1], [0.5, 1], color="0.5", lw=0.8, ls="--")
ax.set_xlabel("internal: layer-24 probe motion AUC"); ax.set_ylabel("output: |scene flow| motion AUC")
ax.set_title("floating epipolar-aligned scenes"); ax.set_xlim(0.5, 1.01); ax.set_ylim(0.5, 1.01)
ax.legend(frameon=False, fontsize=7, loc="lower right")
fig.suptitle(f"{TID}: base Any4D vs fine-tuned variants (bars: mean over scenes, 95% bootstrap CI)", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "ft_comparison.png"), dpi=120); plt.close(fig)
print("rows", len(rows), "variants", present)
