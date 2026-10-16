"""Round 2, review item 1 (W2): analysis of the baseline comparison, the alpha/beta decomposition, the boundary test
and the dose-response, from the per-scene outputs of e5_w2_eval.py (Kubric) and e5_w2_realdata.py (KITTI, Sintel).

  python e5_w2_analysis.py MODE --kubric <dir with w2_eval_<model>_<ds>.json> --real <dir with w2_real_*.json> --out <dir>
  MODE: outputs (baseline output metrics), realout, decomp, dose, summary.

Statistics: every interval is a percentile bootstrap 95% CI over scenes (10000 resamples, seed 0). Paired comparisons
(model vs Any4D, alpha vs beta, beta vs 1, interior vs edge) use the two-sided sign test on per-scene differences
(ties dropped). The dose-response uses per-object slopes (objects with more than 20 moving query points with
|d_true| > 2 px), winsorised to [-0.5, 1.5]; bin means and regression coefficients get scene-cluster bootstrap CIs
(2000 resamples: scenes are resampled with all their objects).
"""
import argparse, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

MODELS = [("any4d", "Any4D"), ("vggt", "VGGT-1B tracker"), ("cotracker3_24f", "CoTracker3, 24 fr"),
          ("cotracker3_7f", "CoTracker3, 7 fr"), ("raft_direct", "RAFT, 0 to 23"), ("raft_chain7", "RAFT chain, 7 fr"),
          ("raft_chain24", "RAFT chain, 24 fr")]
MLAB = dict(MODELS)
COL = {"any4d": "#000000", "vggt": "#7f7f7f", "cotracker3_24f": "#0072B2", "cotracker3_7f": "#56B4E9",
       "raft_direct": "#D55E00", "raft_chain7": "#E69F00", "raft_chain24": "#CC79A7", "raft": "#009E73"}
KDS = [("v2", "Kubric v2 grounded (100)"), ("v3float", "Kubric floating (70)")]
RDS = [("kitti15_rgb", "KITTI 2015, 2 views"), ("sintel_clean", "Sintel clean, 2 views"), ("sintel_final", "Sintel final, 2 views")]
FACTORS = [("speed_px", "object speed |x_true - x_static| (px)", True), ("area", "object image area (fraction)", True),
           ("texture", "texture (mean gradient, grey levels/px)", False), ("occlusion", "occluded fraction", False),
           ("cam_rot_deg", "camera rotation 0 to 23 (deg)", False), ("n_motions", "number of independent motions", False),
           ("cam_disp_px", "camera-induced displacement (px)", True)]
NB = 10000
COND_MAX = 10.0   # condition number of [A B] above which alpha and beta are treated as not identifiable
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False})


def boot(v, n=NB, seed=0, stat=np.mean):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n=0)
    rs = np.random.default_rng(seed); bs = stat(v[rs.integers(0, len(v), (n, len(v)))], axis=1)
    return dict(mean=float(stat(v)), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=int(len(v)),
                median=float(np.median(v)))


def sign_test(d):
    d = np.asarray(d, float); d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return dict(n_pos=0, n_neg=0, p=np.nan)
    k = int((d > 0).sum()); return dict(n_pos=k, n_neg=int(len(d) - k), p=float(binomtest(k, len(d), 0.5).pvalue))


def fmt(b, nd=2):
    return "n/a" if not np.isfinite(b.get("mean", np.nan)) else f"{b['mean']:.{nd}f} [{b['lo']:.{nd}f}, {b['hi']:.{nd}f}]"


def fp(p):
    return "n/a" if not np.isfinite(p) else (f"{p:.1e}" if p < 1e-3 else f"{p:.3f}")


def load_kubric(d):
    out = {}
    for m, _ in MODELS:
        for ds, _ in KDS:
            f = os.path.join(d, f"w2_eval_{m}_{ds}.json")
            if os.path.exists(f):
                z = json.load(open(f)); rows = [r for r in z["rows"] if "error" not in r]
                out[(m, ds)] = dict(rows=rows, objects=z["objects"], n_err=len(z["rows"]) - len(rows))
    return out


def load_real(d):
    out = {}
    for m in ("any4d", "raft"):
        for ds, _ in RDS:
            f = os.path.join(d, f"w2_real_{m}_{ds}.json")
            if os.path.exists(f):
                z = json.load(open(f)); out[(m, ds)] = dict(rows=z["rows"], objects=[], n_err=0)
    return out


def col(rows, k):
    return np.array([r.get(k, np.nan) if r.get(k) is not None else np.nan for r in rows], float)


def paired(ra, rb, k):
    a = {r["scene"]: r.get(k, np.nan) for r in ra}; b = {r["scene"]: r.get(k, np.nan) for r in rb}
    s = sorted(set(a) & set(b)); return np.array([b[x] for x in s], float) - np.array([a[x] for x in s], float)


def write(path, txt):
    open(path + ".tmp", "w").write(txt); os.replace(path + ".tmp", path)


# ======================================================================= outputs
def metric_table(D, keys, ref="any4d", slope_key="slope2d_q"):
    lines = ["| model | data | n | " + " | ".join(keys) + f" | paired {slope_key} diff vs Any4D [sign test p] |",
             "|---|---|---|" + "---|" * len(keys) + "---|"]
    stats = {}
    for (m, ds), z in D.items():
        rows = z["rows"]; st = {k: boot(col(rows, k)) for k in keys}
        if m != ref and (ref, ds) in D:
            d = paired(D[(ref, ds)]["rows"], rows, slope_key); st["diff_vs_ref"] = boot(d); st["sign_vs_ref"] = sign_test(d)
            extra = f"{fmt(st['diff_vs_ref'])} [p {fp(st['sign_vs_ref']['p'])}, {st['sign_vs_ref']['n_pos']}+/{st['sign_vs_ref']['n_neg']}-]"
        else:
            extra = "reference"
        stats[f"{m}|{ds}"] = st
        lines.append(f"| {MLAB.get(m, m)} | {ds} | {len(rows)} | " + " | ".join(fmt(st[k]) for k in keys) + f" | {extra} |")
    return "\n".join(lines), stats


def fig_metrics(D, keys, titles, dsl, path, ref_lines):
    fig, axs = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 3.2), squeeze=False)
    models = [m for m, _ in MODELS if any((m, ds) in D for ds, _ in dsl)] + (["raft"] if any((("raft", ds) in D) for ds, _ in dsl) else [])
    for j, k in enumerate(keys):
        ax = axs[0, j]
        for i, (ds, dl) in enumerate(dsl):
            for mi, m in enumerate(models):
                if (m, ds) not in D:
                    continue
                b = boot(col(D[(m, ds)]["rows"], k)); x = i + (mi - (len(models) - 1) / 2) * 0.8 / max(len(models), 1)
                ax.errorbar(x, b["mean"], yerr=[[b["mean"] - b["lo"]], [b["hi"] - b["mean"]]], fmt="o", ms=4,
                            color=COL.get(m, "C0"), label=MLAB.get(m, "RAFT") if (i == 0 and j == 0) else None, capsize=2)
        for y in ref_lines.get(k, []):
            ax.axhline(y, color="0.6", lw=0.8, ls="--")
        ax.set_xticks(range(len(dsl))); ax.set_xticklabels([dl.replace(" (", "\n(").replace(", ", "\n") for _, dl in dsl], fontsize=7.5)
        ax.set_title(titles[j])
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 7), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(path, dpi=130); plt.close(fig)


def mode_outputs(a):
    D = load_kubric(a.kubric)
    keys = ["slope2d_q", "epe_static_q", "epe_moving_q", "sep_ratio_q"]
    tab, stats = metric_table(D, keys)
    fig_metrics(D, keys, ["object-motion slope (query set)", "static EPE (px)", "moving EPE (px)", "separation ratio"],
                KDS, os.path.join(a.out, "fig_w2_baselines_kubric.png"), {"slope2d_q": [1.0], "sep_ratio_q": [1.0]})
    errs = {f"{m}|{ds}": z["n_err"] for (m, ds), z in D.items()}
    json.dump(dict(stats=stats, errors=errs), open(os.path.join(a.out, "w2_baselines_kubric_stats.json"), "w"), indent=1)
    write(os.path.join(a.out, f"{a.task}_summary.md"),
          f"# {a.task}: non-reconstruction baselines on Kubric (review item 1, W2)\n\n"
          "Same scenes and query sets as e5_xm_eval.py (slope2d_q, epe_*_q, sep_ratio_q). Mean over scenes with bootstrap "
          "95% CI; last column is the paired per-scene difference (model minus Any4D) with a two-sided sign test.\n\n"
          + tab + f"\n\nScenes with evaluation errors: {errs}\n\n![baselines](fig_w2_baselines_kubric.png)\n")
    print(tab)


def mode_realout(a):
    D = load_real(a.real)
    keys = ["slope", "epe_static", "epe_moving", "sep_ratio"]
    tab, stats = metric_table(D, keys, slope_key="slope")
    fig_metrics(D, ["slope", "epe_static", "epe_moving", "sep_ratio"], ["object-motion slope", "static EPE (px)", "moving EPE (px)",
                "separation ratio"], RDS, os.path.join(a.out, "fig_w2_baselines_real.png"), {"slope": [1.0], "sep_ratio": [1.0]})
    json.dump(stats, open(os.path.join(a.out, "w2_baselines_real_stats.json"), "w"), indent=1)
    write(os.path.join(a.out, f"{a.task}_summary.md"),
          f"# {a.task}: RAFT vs Any4D on KITTI 2015 and Sintel (2 views)\n\n"
          "RAFT: torchvision raft_large, C_T_V2 weights (FlyingChairs + FlyingThings; no KITTI or Sintel training), native "
          "resolution. Any4D: round-1 2-view wide-input predictions (cached). Metrics: e5_realdata_eval.evaluate on the same "
          "evaluation points (the alpha/beta decomposition is in w2_decomp). Bootstrap 95% CIs over scenes; paired sign test vs Any4D.\n\n"
          + tab + "\n\n![real](fig_w2_baselines_real.png)\n")
    print(tab)


# ======================================================================= decomposition
DB = [("slope_d0_3", "0-3"), ("slope_d3_6", "3-6"), ("slope_d6_12", "6-12"), ("slope_d12_24", "12-24"), ("slope_d24_999", ">24")]


def decomp_stats(D, well=0.9):
    S = {}
    for (m, ds), z in D.items():
        r = z["rows"]
        if not r:
            continue
        al0, be0 = col(r, "alpha"), col(r, "beta"); cs = np.abs(col(r, "cos_AB"))
        # identifiability guard: alpha and beta are only separable when [A B] is well conditioned and both the object's
        # own motion and the camera-induced motion are visible (median above 2 px); other scenes are excluded from alpha/beta
        ok = (col(r, "cond_AB") <= COND_MAX) & (col(r, "median_A_px") >= 2) & (col(r, "median_B_px") >= 2)
        al, be = np.where(ok, al0, np.nan), np.where(ok, be0, np.nan)
        s = dict(slope=boot(col(r, "slope2d_q") if "slope2d_q" in r[0] else col(r, "slope")), alpha=boot(al), beta=boot(be),
                 beta_static=boot(col(r, "beta_static")), alpha_minus_beta=boot(al - be), sign_alpha_beta=sign_test(al - be),
                 sign_beta_1=sign_test(be - 1), sign_alpha_1=sign_test(al - 1), n_identifiable=int(ok.sum()), n_scenes=len(r),
                 alpha_all_median=float(np.nanmedian(al0)), beta_all_median=float(np.nanmedian(be0)), median_abs_cos=float(np.nanmedian(cs)),
                 sep_ratio=boot(col(r, "sep_ratio_q") if "sep_ratio_q" in r[0] else col(r, "sep_ratio")))
        if "slope_edge" in r[0]:
            e, i = col(r, "slope_edge"), col(r, "slope_interior")
            s.update(slope_edge=boot(e), slope_interior=boot(i), interior_minus_edge=boot(i - e), sign_interior_edge=sign_test(i - e),
                     bins={lab: boot(col(r, k)) for k, lab in DB},
                     alpha_edge=boot(col(r, "alpha_edge")), alpha_interior=boot(col(r, "alpha_interior")),
                     beta_edge=boot(col(r, "beta_edge")), beta_interior=boot(col(r, "beta_interior")))
        S[f"{m}|{ds}"] = s
    return S


def classify(s):
    """Rule-based reading of one model x dataset (stated in the summary)."""
    sl, al, be = s["slope"], s["alpha"], s["beta"]
    if not np.isfinite(sl["mean"]):
        return "undefined"
    if sl["lo"] > 0.9:
        kind = "no pull (slope CI above 0.9)"
    elif sl["hi"] < 0.9:
        kind = "pull (slope CI below 0.9)"
    else:
        kind = "weak or unclear pull (slope CI includes 0.9)"
    if np.isfinite(be["mean"]):
        if be["hi"] < 0.95 and not (s["sign_alpha_beta"]["p"] < 0.05):
            kind += "; shrinkage toward zero image motion (beta below 1 and alpha not different from beta)"
        elif be["lo"] > 0.95 or (be["lo"] <= 1 <= be["hi"]):
            kind += "; camera motion kept (beta near 1)" + ("" if al["lo"] > 0.9 else ", own motion under-recovered (alpha < 1): camera-motion pull")
        else:
            kind += "; mixed (beta below 1 and alpha different from beta)"
    if "sign_interior_edge" in s and s["sign_interior_edge"]["p"] < 0.05 and s["interior_minus_edge"]["mean"] > 0:
        kind += "; slope lower at object boundaries (spatial smoothing contributes)"
    return kind


def mode_decomp(a):
    D = {**load_kubric(a.kubric), **load_real(a.real)}
    S = decomp_stats(D)
    L = ["| model | data | n | slope | alpha | beta | beta_static | alpha - beta [sign p] | beta vs 1 [sign p] | median abs cos(A,B) | median alpha, beta over all scenes (no guard) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in S.items():
        m, ds = k.split("|")
        L.append(f"| {MLAB.get(m, 'RAFT' if m == 'raft' else m)} | {ds} | {s['n_identifiable']} of {s['n_scenes']} | {fmt(s['slope'])} | {fmt(s['alpha'])} | {fmt(s['beta'])} | "
                 f"{fmt(s['beta_static'])} | {fmt(s['alpha_minus_beta'])} [{fp(s['sign_alpha_beta']['p'])}] | "
                 f"{s['sign_beta_1']['n_pos']}+/{s['sign_beta_1']['n_neg']}- [{fp(s['sign_beta_1']['p'])}] | {s['median_abs_cos']:.2f} | "
                 f"{s['alpha_all_median']:.2f}, {s['beta_all_median']:.2f} |")
    B = ["| model | data | " + " | ".join(f"slope, d {lab} px" for _, lab in DB) + " | edge (d <= 4) | interior (d >= 14) | interior - edge [sign p] |",
         "|---|---|" + "---|" * (len(DB) + 3)]
    for k, s in S.items():
        if "bins" not in s:
            continue
        m, ds = k.split("|")
        B.append(f"| {MLAB.get(m, m)} | {ds} | " + " | ".join(fmt(s["bins"][lab]) for _, lab in DB) +
                 f" | {fmt(s['slope_edge'])} | {fmt(s['slope_interior'])} | {fmt(s['interior_minus_edge'])} [{fp(s['sign_interior_edge']['p'])}] |")
    C = [f"- {MLAB.get(k.split('|')[0], k.split('|')[0])}, {k.split('|')[1]}: {classify(s)}" for k, s in S.items()]
    json.dump(S, open(os.path.join(a.out, "w2_decomp_stats.json"), "w"), indent=1)
    # ---- figure 1: alpha vs beta
    dsl = [d for d in KDS + RDS if any(k.endswith("|" + d[0]) for k in S)]
    fig, axs = plt.subplots(1, len(dsl), figsize=(2.9 * len(dsl), 3.1), squeeze=False)
    for j, (ds, dl) in enumerate(dsl):
        ax = axs[0, j]; ax.plot([0, 1.3], [0, 1.3], color="0.75", lw=0.8, ls=":"); ax.axhline(1, color="0.6", lw=0.8, ls="--"); ax.axvline(1, color="0.85", lw=0.8)
        for k, s in S.items():
            m, d2 = k.split("|")
            if d2 != ds:
                continue
            al, be = s["alpha"], s["beta"]
            ax.errorbar(al["mean"], be["mean"], xerr=[[al["mean"] - al["lo"]], [al["hi"] - al["mean"]]],
                        yerr=[[be["mean"] - be["lo"]], [be["hi"] - be["mean"]]], fmt="o", ms=4, color=COL.get(m, "C0"),
                        capsize=2, label=MLAB.get(m, "RAFT" if m == "raft" else m))
        ax.set_xlabel("alpha (own object motion)"); ax.set_title(dl)
        if j == 0:
            ax.set_ylabel("beta (camera-induced motion)")
        ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.text(0.01, 0.01, "dotted: alpha = beta (shrinkage toward zero image motion); dashed: beta = 1 (camera motion kept)", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(a.out, "fig_w2_alpha_beta.png"), dpi=130); plt.close(fig)
    # ---- figure 2: slope vs boundary distance
    fig, axs = plt.subplots(1, len(KDS), figsize=(4.2 * len(KDS), 3.2), squeeze=False)
    for j, (ds, dl) in enumerate(KDS):
        ax = axs[0, j]
        for m, ml in MODELS:
            s = S.get(f"{m}|{ds}")
            if not s or "bins" not in s:
                continue
            y = [s["bins"][lab]["mean"] for _, lab in DB]; lo = [s["bins"][lab]["lo"] for _, lab in DB]; hi = [s["bins"][lab]["hi"] for _, lab in DB]
            x = np.arange(len(DB)); ax.plot(x, y, "-o", ms=3, color=COL[m], label=ml, lw=1.6 if m == "any4d" else 1)
            ax.fill_between(x, lo, hi, color=COL[m], alpha=0.12, lw=0)
        ax.axhline(1, color="0.6", lw=0.8, ls="--"); ax.set_xticks(range(len(DB))); ax.set_xticklabels([lab for _, lab in DB])
        ax.set_xlabel("distance to object boundary in frame 0 (px)"); ax.set_ylabel("object-motion slope"); ax.set_title(dl)
    axs[0, -1].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_w2_boundary.png"), dpi=130); plt.close(fig)
    write(os.path.join(a.out, f"{a.task}_summary.md"),
          f"# {a.task}: pull toward camera motion vs shrinkage vs smoothing (review item 1, W2)\n\n"
          "Per scene, on moving query points with |x_true - x_static| > 2 px: x_pred - x1 = alpha (x_true - x_static) + "
          "beta (x_static - x1) + e (least squares over both coordinates). Pure camera-motion pull: beta near 1, alpha < 1. "
          "Shrinkage toward zero image motion: alpha near beta, both < 1. beta_static is the same fit on static points "
          "(only beta). alpha and beta are only separable when the object's motion is not parallel to the camera-induced "
          "motion and both are visible: alpha, beta and their tests use only scenes where the condition number of [A B] is at "
          "most 10 and the median |A| and |B| are at least 2 px (column n: identifiable of all scenes). Epipolar-aligned "
          "scenes, most KITTI cars and Sintel shots with a near-static camera fail this guard. slope and beta_static use "
          "all scenes. Mean over scenes [bootstrap 95% CI]; sign tests are two-sided.\n\n"
          + "\n".join(L) + "\n\n## Slope vs distance to the object boundary (Kubric)\n\n"
          "Spatial smoothing toward the neighbouring background predicts a lower slope near the boundary and a slope near "
          "1 in the interior. A pull that is the same at all distances is not explained by smoothing.\n\n" + "\n".join(B) +
          "\n\n## Rule-based reading (thresholds: slope CI vs 0.9; beta CI vs 0.95 and 1; sign tests at 0.05)\n\n" + "\n".join(C) +
          "\n\n![alpha beta](fig_w2_alpha_beta.png)\n\n![boundary](fig_w2_boundary.png)\n")
    print("\n".join(L)); print("\n".join(C))


# ======================================================================= dose-response
def objects_df(z, conf_ok):
    O = [o for o in z["objects"] if np.isfinite(o.get("slope", np.nan))]
    sc = np.array([o["scene"] for o in O]); y = np.clip(np.array([o["slope"] for o in O], float), -0.5, 1.5)
    F = {k: np.array([o.get(k, np.nan) for o in O], float) for k, _, _ in FACTORS}
    if conf_ok:
        F["conf"] = np.array([o.get("conf", np.nan) for o in O], float)
    return sc, y, F


def cluster_boot(scenes, fn, n=2000, seed=0):
    u = np.unique(scenes); idx = {s: np.where(scenes == s)[0] for s in u}; rs = np.random.default_rng(seed); out = []
    for _ in range(n):
        pick = rs.choice(u, len(u), replace=True); ii = np.concatenate([idx[s] for s in pick]); out.append(fn(ii))
    return np.array(out)


def ols_std(X, y):
    Xs = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1); Xs = np.c_[np.ones(len(y)), Xs]
    return np.linalg.lstsq(Xs, y, rcond=None)[0][1:]


def mode_dose(a):
    D = load_kubric(a.kubric)
    R = {}
    for ds, dl in KDS:
        allo = [o for m, _ in MODELS if (m, ds) in D for o in D[(m, ds)]["objects"]]
        if not allo:
            continue
        edges = {}
        for k, _, logx in FACTORS:
            v = np.array([o.get(k, np.nan) for o in D[("any4d", ds)]["objects"]] if ("any4d", ds) in D else [o.get(k, np.nan) for o in allo], float)
            v = v[np.isfinite(v)]
            edges[k] = np.unique(v) if k == "n_motions" else np.quantile(v, np.linspace(0, 1, 6))
        fig, axs = plt.subplots(2, 4, figsize=(13, 6.2)); axs = axs.ravel()
        for m, ml in MODELS:
            if (m, ds) not in D:
                continue
            has_conf = np.isfinite([o.get("conf", np.nan) for o in D[(m, ds)]["objects"]]).any()
            sc, y, F = objects_df(D[(m, ds)], has_conf)
            res = dict(n_objects=int(len(y)), n_scenes=int(len(np.unique(sc))), bins={}, spearman={})
            for fi, (k, lab, logx) in enumerate(FACTORS + ([("conf", "model confidence (quintiles within model)", False)] if has_conf else [])):
                x = F[k]; e = edges.get(k, np.quantile(x[np.isfinite(x)], np.linspace(0, 1, 6)))
                if k == "n_motions":
                    lab_b = [np.isclose(x, v) for v in e]; xc = list(e)
                else:
                    lab_b = [(x >= lo) & ((x < hi) if j < len(e) - 2 else (x <= hi)) for j, (lo, hi) in enumerate(zip(e[:-1], e[1:]))]
                    xc = [float(np.nanmedian(x[b])) if b.any() else np.nan for b in lab_b]
                mu, lo_, hi_, nn = [], [], [], []
                for b in lab_b:
                    ii = np.where(b)[0]
                    if len(ii) < 5:
                        mu.append(np.nan); lo_.append(np.nan); hi_.append(np.nan); nn.append(int(len(ii))); continue
                    bs = cluster_boot(sc[ii], lambda jj, yy=y[ii]: yy[jj].mean() if len(jj) else np.nan, n=1000)
                    mu.append(float(y[ii].mean())); lo_.append(float(np.nanpercentile(bs, 2.5))); hi_.append(float(np.nanpercentile(bs, 97.5))); nn.append(int(len(ii)))
                res["bins"][k] = dict(x=xc, mean=mu, lo=lo_, hi=hi_, n=nn, edges=[float(t) for t in e])
                from scipy.stats import spearmanr
                ok = np.isfinite(x) & np.isfinite(y); res["spearman"][k] = float(spearmanr(x[ok], y[ok])[0]) if ok.sum() > 5 else np.nan
                ax = axs[fi if k != "conf" else 7]
                ax.errorbar(xc, mu, yerr=[np.array(mu) - np.array(lo_), np.array(hi_) - np.array(mu)], fmt="-o", ms=3,
                            color=COL[m], lw=1.6 if m == "any4d" else 0.9, capsize=1.5, label=ml)
                if k != "conf":
                    ax.set_xscale("log" if logx else "linear"); ax.set_xlabel(lab)
                else:
                    ax.set_xlabel("confidence bin (quintiles within each model)")
                ax.set_ylabel("per-object slope")
            # regression of per-object slope on standardised factors (confidence excluded, so models are comparable)
            keys = [k for k, _, _ in FACTORS]
            X = np.c_[[np.log(F[k]) if lg else F[k] for k, _, lg in FACTORS]].T
            ok = np.isfinite(X).all(1) & np.isfinite(y)
            if ok.sum() > int(os.environ.get('W2_REG_MIN', 3 * len(keys))):
                b0 = ols_std(X[ok], y[ok]); bs = cluster_boot(sc[ok], lambda jj, XX=X[ok], yy=y[ok]: ols_std(XX[jj], yy[jj]))
                res["regression"] = {k: dict(coef=float(b0[i]), lo=float(np.percentile(bs[:, i], 2.5)), hi=float(np.percentile(bs[:, i], 97.5)))
                                     for i, k in enumerate(keys)}
                res["regression_n"] = int(ok.sum())
            if has_conf:
                Xc = np.c_[X, F["conf"]]; okc = np.isfinite(Xc).all(1) & np.isfinite(y)
                if okc.sum() > int(os.environ.get('W2_REG_MIN', 3 * (len(keys) + 1))):
                    b0 = ols_std(Xc[okc], y[okc]); bs = cluster_boot(sc[okc], lambda jj, XX=Xc[okc], yy=y[okc]: ols_std(XX[jj], yy[jj]))
                    res["regression_with_conf"] = {k: dict(coef=float(b0[i]), lo=float(np.percentile(bs[:, i], 2.5)), hi=float(np.percentile(bs[:, i], 97.5)))
                                                   for i, k in enumerate(keys + ["conf"])}
            R[f"{m}|{ds}"] = res
        for ax in axs:
            ax.axhline(1, color="0.6", lw=0.8, ls="--")
        axs[0].legend(frameon=False, fontsize=6.5)
        fig.suptitle(f"Dose-response of the object-motion slope, {dl} (mean per-object slope, scene-cluster bootstrap 95% CI)", fontsize=9, x=0.01, ha="left")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"fig_w2_dose_{ds}.png"), dpi=120); plt.close(fig)
    # forest plot of standardised coefficients
    fig, axs = plt.subplots(1, len(KDS), figsize=(5.2 * len(KDS), 3.6), squeeze=False)
    keys = [k for k, _, _ in FACTORS]
    for j, (ds, dl) in enumerate(KDS):
        ax = axs[0, j]; ms = [m for m, _ in MODELS if f"{m}|{ds}" in R and "regression" in R[f"{m}|{ds}"]]
        for mi, m in enumerate(ms):
            rg = R[f"{m}|{ds}"]["regression"]; yy = np.arange(len(keys)) + (mi - (len(ms) - 1) / 2) * 0.11
            c = np.array([rg[k]["coef"] for k in keys]); lo = np.array([rg[k]["lo"] for k in keys]); hi = np.array([rg[k]["hi"] for k in keys])
            ax.errorbar(c, yy, xerr=[c - lo, hi - c], fmt="o", ms=3, color=COL[m], capsize=1.5, label=MLAB[m])
        ax.axvline(0, color="0.6", lw=0.8); ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(["log speed", "log area", "texture", "occlusion", "camera rotation", "n motions", "log camera disp."])
        ax.invert_yaxis(); ax.set_xlabel("change in per-object slope per 1 SD of the factor"); ax.set_title(dl)
    axs[0, -1].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_w2_dose_regression.png"), dpi=130); plt.close(fig)
    json.dump(R, open(os.path.join(a.out, "w2_dose_stats.json"), "w"), indent=1, default=float)
    L = []
    for k, r in R.items():
        if "regression" not in r:
            continue
        m, ds = k.split("|")
        L.append(f"| {MLAB[m]} | {ds} | {r['regression_n']} | " + " | ".join(f"{r['regression'][f]['coef']:+.3f} [{r['regression'][f]['lo']:+.3f}, {r['regression'][f]['hi']:+.3f}]" for f in keys) +
                 (f" | {r['regression_with_conf']['conf']['coef']:+.3f} [{r['regression_with_conf']['conf']['lo']:+.3f}, {r['regression_with_conf']['conf']['hi']:+.3f}]" if "regression_with_conf" in r else " | n/a") + " |")
    hdr = "| model | data | objects | " + " | ".join(["log speed", "log area", "texture", "occlusion", "cam rotation", "n motions", "log cam disp"]) + " | confidence (added to the same fit) |\n|---|---|---|" + "---|" * 8
    write(os.path.join(a.out, f"{a.task}_summary.md"),
          f"# {a.task}: dose-response of the object-motion slope (review item 1, W2)\n\n"
          "Per-object slope (objects with more than 20 moving query points), winsorised to [-0.5, 1.5]. Bins are quintiles of "
          "the Any4D objects (number of motions: its values). Regression: OLS of per-object slope on all factors, standardised "
          "(coefficient = change in slope per 1 SD), scene-cluster bootstrap 95% CI. Confidence is model-specific and is added "
          "in a separate fit.\n\n"
          "What each pattern implies:\n"
          "- Regression shrinkage (the prediction moves toward the training-set mean under uncertainty) predicts a LOWER slope "
          "at low texture, small area, high occlusion and low confidence, so positive coefficients for texture, area and "
          "confidence and a negative one for occlusion.\n"
          "- A structural pull toward the static (camera-only) hypothesis that does not depend on how well the object is "
          "seen predicts flat curves for texture, area, occlusion and confidence, with dependence on the geometry instead "
          "(camera rotation, camera-induced displacement, number of motions).\n"
          "- Speed: a lower slope at high speed is expected under both readings (large motions are rarer and harder), so it "
          "does not separate them; a lower slope at LOW speed points to a fixed offset or a detection threshold.\n\n"
          + hdr + "\n" + "\n".join(L) + "\n\n![v2](fig_w2_dose_v2.png)\n\n![floating](fig_w2_dose_v3float.png)\n\n![regression](fig_w2_dose_regression.png)\n")
    print(hdr); print("\n".join(L))


# ======================================================================= summary
def mode_summary(a):
    Dk, Dr = load_kubric(a.kubric), load_real(a.real)
    D = {**Dk, **Dr}; S = decomp_stats(D)
    dose = json.load(open(a.dose)) if a.dose and os.path.exists(a.dose) else {}
    L = ["| model | data | n scenes | slope | alpha (n identifiable) | beta | beta_static | sep_ratio | slope vs Any4D [sign p] |", "|---|---|---|---|---|---|---|---|---|"]
    for k, s in S.items():
        m, ds = k.split("|"); key = "slope2d_q" if (m, ds) in Dk else "slope"
        if m != "any4d" and ("any4d", ds) in D:
            d = paired(D[("any4d", ds)]["rows"], D[(m, ds)]["rows"], key); st = sign_test(d); b = boot(d)
            ex = f"{b['mean']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}] [p {fp(st['p'])}]"
        else:
            ex = "reference"
        L.append(f"| {MLAB.get(m, 'RAFT' if m == 'raft' else m)} | {ds} | {s['slope']['n']} | {fmt(s['slope'])} | {fmt(s['alpha'])} ({s['n_identifiable']}) | {fmt(s['beta'])} | "
                 f"{fmt(s['beta_static'])} | {fmt(s['sep_ratio'])} | {ex} |")
    json.dump(dict(decomp=S), open(os.path.join(a.out, "w2_summary_stats.json"), "w"), indent=1)
    # figure: 4 panels slope, alpha, beta, sep_ratio
    dsl = [d for d in KDS + RDS if any(k.endswith("|" + d[0]) for k in S)]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    names = ["slope", "alpha", "beta", "sep_ratio"]; ttl = ["object-motion slope", "alpha (own motion)", "beta (camera-induced motion)", "separation ratio"]
    models = [m for m, _ in MODELS] + ["raft"]
    for j, nm in enumerate(names):
        ax = axs[j]
        for i, (ds, dl) in enumerate(dsl):
            ms = [m for m in models if f"{m}|{ds}" in S]
            for mi, m in enumerate(ms):
                b = S[f"{m}|{ds}"][nm]
                if not np.isfinite(b["mean"]):
                    continue
                x = i + (mi - (len(ms) - 1) / 2) * 0.11
                ax.errorbar(x, b["mean"], yerr=[[b["mean"] - b["lo"]], [b["hi"] - b["mean"]]], fmt="o", ms=4, color=COL[m], capsize=2,
                            label=(MLAB.get(m, "RAFT (2-view)") if m != "raft" else "RAFT (2-view)"))
        ax.axhline(1, color="0.6", lw=0.8, ls="--"); ax.set_xticks(range(len(dsl))); ax.set_xticklabels([dl.replace(" (", "\n(").replace(", ", "\n") for _, dl in dsl], fontsize=7)
        ax.set_title(ttl[j])
    h, l = [], []
    for ax in axs:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(a.out, "fig_w2_summary.png"), dpi=130); plt.close(fig)
    # rule-based verdict
    def g(m, ds):
        return S.get(f"{m}|{ds}")
    V = []
    for ds, dl in KDS:
        a4 = g("any4d", ds); ct = g("cotracker3_24f", ds) or g("cotracker3_7f", ds); rf = [g(m, ds) for m in ("raft_direct", "raft_chain7", "raft_chain24") if g(m, ds)]
        if not a4:
            continue
        bl = [x for x in [ct] + rf if x]
        best = max(bl, key=lambda s: s["slope"]["mean"]) if bl else None
        msg = f"{dl}: Any4D slope {fmt(a4['slope'])}"
        if g("vggt", ds):
            msg += f", VGGT tracker {fmt(g('vggt', ds)['slope'])}"
        if ct:
            msg += f", CoTracker3 {fmt(ct['slope'])}"
        if rf:
            msg += ", RAFT " + "; ".join(fmt(s["slope"]) for s in rf)
        if best is not None:
            gap = best["slope"]["lo"] > a4["slope"]["hi"]
            msg += (". The best non-reconstruction baseline recovers clearly more object motion than Any4D (CIs do not overlap)" if gap
                    else ". The baselines do not recover clearly more object motion than Any4D (CIs overlap)")
        V.append(msg + ".")
    for ds, dl in RDS:
        if g("any4d", ds) and g("raft", ds):
            V.append(f"{dl}: Any4D slope {fmt(g('any4d', ds)['slope'])}, RAFT {fmt(g('raft', ds)['slope'])}.")
    C = [f"- {MLAB.get(k.split('|')[0], 'RAFT' if k.split('|')[0] == 'raft' else k.split('|')[0])}, {k.split('|')[1]}: {classify(s)}" for k, s in S.items()]
    DZ = []
    for k, r in dose.items():
        if k.startswith("any4d|") and "regression" in r:
            sig = [f"{f} {r['regression'][f]['coef']:+.2f}" for f in r["regression"] if r["regression"][f]["lo"] > 0 or r["regression"][f]["hi"] < 0]
            DZ.append(f"- Any4D, {k.split('|')[1]}: factors whose standardised coefficient CI excludes 0: " + (", ".join(sig) if sig else "none"))
            if "regression_with_conf" in r:
                cc = r["regression_with_conf"]["conf"]; DZ.append(f"  confidence (added): {cc['coef']:+.2f} [{cc['lo']:+.2f}, {cc['hi']:+.2f}]")
    write(os.path.join(a.out, f"{a.task}_summary.md"),
          f"# {a.task}: is the camera-motion pull regression shrinkage or a 4D-model property? (review item 1, W2)\n\n"
          "## All models, same scenes and query sets\n\nMean over scenes [bootstrap 95% CI]. Kubric slope is slope2d_q and "
          "sep_ratio is sep_ratio_q (e5_xm_eval.py definitions, frame 0 to 23); real data use e5_realdata_eval.py. alpha "
          "and beta: x_pred - x1 = alpha (x_true - x_static) + beta (x_static - x1) on moving points, only on scenes where they "
          "are identifiable (condition number of [A B] at most 10, median |A| and |B| at least 2 px). Last column: paired "
          "per-scene difference to Any4D with a two-sided sign test.\n\n" + "\n".join(L) +
          "\n\n![summary](fig_w2_summary.png)\n\n## Rule-based reading per model\n\n"
          "Thresholds: pull if the slope CI is below 0.9, no pull if it is above 0.9; shrinkage toward zero image motion if "
          "the beta CI is below 0.95 and alpha is not different from beta (sign test p >= 0.05); camera-motion pull if "
          "beta is near 1 and alpha below 1; smoothing if the interior slope is higher than the edge slope (sign test p < 0.05).\n\n"
          + "\n".join(C) + "\n\n## Dose-response (Any4D)\n\n" + ("\n".join(DZ) if DZ else "not available") +
          "\n\nSee w2_dose summary for all models and the figures fig_w2_dose_v2.png, fig_w2_dose_v3float.png, fig_w2_dose_regression.png.\n\n"
          "## Verdict (generated from the numbers above; to be checked and edited by hand)\n\n" + " ".join(V) +
          "\nIf the dedicated trackers (CoTracker3, RAFT) recover close to all object motion on the same scenes while Any4D and "
          "the VGGT tracker head do not, the pull is a property of the reconstruction models and not a generic property of "
          "dense correspondence under uncertainty. If the baselines show the same deficit, with alpha close to beta, it is "
          "generic shrinkage. The decomposition rows above say which of the two readings each model follows.\n")
    print("\n".join(L)); print("\n".join(C)); print(" ".join(V))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["outputs", "realout", "decomp", "dose", "summary"])
    ap.add_argument("--kubric", default=""); ap.add_argument("--real", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--dose", default=""); ap.add_argument("--task", default=os.environ.get("TASK_ID", "w2"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    dict(outputs=mode_outputs, realout=mode_realout, decomp=mode_decomp, dose=mode_dose, summary=mode_summary)[a.mode](a)
