"""E5 v4 (review item W3) summary: view-count test on epi_accel vs epi_const, and the size cue (epi_sizerand,
epi_sizematch vs v3float epi_aligned). Pure numpy + matplotlib (runs in the post venv).

Usage: python e5_v4_summary.py <result_dir> <combined_out_dir> <task_id> [views, default 2,3,4,7]
Reads  vs_pool_<n>v_results.json  probes, 5-fold CV over v3float (70) + v4epi pooled, paired scenes share a fold
       vs_xfer_<n>v_results.json  probes trained on v3float scenes only (round-1 folds), applied to every scene
       outm_<n>v.json             output slope / EPE / mask IoU per target view (round-1 definitions)
Writes <task_id>_summary.md, v4_stats.json, fig_v4_accel_viewcount.png, fig_v4_sizecue.png

Pre-specified read-outs (fixed before the scaled data existed):
  probe   motion_auc_v0_objects (view-0 token, object patches only, moving vs static), per scene;
          primary = mean over layers 8, 12 and 16 (below the last-layer ceiling); layer 24 is secondary.
  output  mask IoU and slope for the frame-23 target.
  accel   primary test: difference in differences per seed, [AUC(n) - AUC(2)]_accel - [AUC(n) - AUC(2)]_const,
          n = 3, 4, 7, pooled probe; sign test, Holm over the three n.
  size    primary test: AUC(epi_sizerand) - AUC(v3float epi_aligned), transfer probe, n = 2, 3, 4, 7 (unpaired:
          bootstrap CI of the difference of means, Mann-Whitney p, Holm over the four n).
"""
import json, math, os, sys
import numpy as np

RD, OUT, TID = sys.argv[1], sys.argv[2], sys.argv[3]
VIEWS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "2,3,4,7").split(",")]
MID = (8, 12, 16)
KEY = "motion_auc_v0_objects"
REF = "epi_aligned"
SPL = ["epi_accel", "epi_const", REF, "epi_sizerand", "epi_sizematch"]
LBL = {"epi_accel": "accelerated (v4)", "epi_const": "constant velocity (v4, paired)", REF: "v3float epi_aligned",
       "epi_sizerand": "size randomised (v4)", "epi_sizematch": "size matched to static reading (v4)"}
COL = {"epi_accel": "#d95f02", "epi_const": "#1b9e77", REF: "#555555", "epi_sizerand": "#7570b3", "epi_sizematch": "#e7298a"}
B = 10000


def seed(n):
    return int(n.split("_s")[-1])


def split_of(n):
    return n.rsplit("_s", 1)[0]


def boot_ci(x, rs=None):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return dict(mean=None, lo=None, hi=None, n=0)
    rs = rs or np.random.default_rng(0)
    m = x[rs.integers(0, len(x), (B, len(x)))].mean(1)
    return dict(mean=float(x.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), n=int(len(x)))


def boot_diff(x, y):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float); y = np.asarray([v for v in y if v is not None and np.isfinite(v)], float)
    if len(x) == 0 or len(y) == 0:
        return dict(mean=None, lo=None, hi=None, nx=len(x), ny=len(y))
    rs = np.random.default_rng(1)
    m = x[rs.integers(0, len(x), (B, len(x)))].mean(1) - y[rs.integers(0, len(y), (B, len(y)))].mean(1)
    return dict(mean=float(x.mean() - y.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), nx=len(x), ny=len(y))


def sign_test(d):
    d = [v for v in d if v is not None and np.isfinite(v)]
    pos = sum(v > 1e-9 for v in d); neg = sum(v < -1e-9 for v in d); n = pos + neg
    if n == 0:
        return dict(pos=pos, neg=neg, ties=len(d), p=None)
    k = min(pos, neg)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return dict(pos=pos, neg=neg, ties=len(d) - n, p=float(p))


def mwu(x, y):
    x = [v for v in x if v is not None and np.isfinite(v)]; y = [v for v in y if v is not None and np.isfinite(v)]
    if not x or not y:
        return None
    a = np.r_[x, y]; order = np.argsort(a, kind="mergesort"); r = np.empty(len(a)); i = 0
    s = a[order]
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    n1, n2 = len(x), len(y); U = r[:n1].sum() - n1 * (n1 + 1) / 2
    _, cnt = np.unique(a, return_counts=True)
    var = n1 * n2 / 12 * ((n1 + n2 + 1) - (cnt ** 3 - cnt).sum() / ((n1 + n2) * (n1 + n2 - 1)))
    if var <= 0:
        return None
    z = (U - n1 * n2 / 2) / math.sqrt(var)
    return float(math.erfc(abs(z) / math.sqrt(2)))


def spearman(x, y, perms=5000):
    m = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    if len(m) < 5:
        return dict(rho=None, p=None, n=len(m))
    x, y = np.array(m).T
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)  # noqa: E731
    rx, ry = rk(x), rk(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rs = np.random.default_rng(2)
    null = np.array([np.corrcoef(rx, rs.permutation(ry))[0, 1] for _ in range(perms)])
    return dict(rho=rho, p=float((np.abs(null) >= abs(rho)).mean()), n=len(m))


def holm(ps):
    idx = [i for i, p in enumerate(ps) if p is not None]
    out = [None] * len(ps); srt = sorted(idx, key=lambda i: ps[i]); run = 0.0
    for r, i in enumerate(srt):
        run = max(run, min(1.0, (len(idx) - r) * ps[i])); out[i] = run
    return out


def load_probe(var, nv):
    p = os.path.join(RD, f"vs_{var}_{nv}v_results.json")
    if not os.path.exists(p):
        return None
    A = {}
    for r in json.load(open(p))["P"]["rows"]:
        v = r.get(KEY)
        A.setdefault(r["scene"], {})[int(r["layer"])] = float(v) if v is not None and np.isfinite(v) else np.nan
    return A


def load_out(nv):
    p = os.path.join(RD, f"outm_{nv}v.json")
    if not os.path.exists(p):
        return None
    O = {}
    for r in json.load(open(p))["rows"]:
        if "error" in r:
            continue
        O[r["scene"]] = {t["target_frame"]: t for t in r["targets"]}
    return O


PR = {var: {nv: load_probe(var, nv) for nv in VIEWS} for var in ("pool", "xfer")}
OM = {nv: load_out(nv) for nv in VIEWS}
LAYERS = sorted({l for var in PR for nv in VIEWS if PR[var][nv] for s in PR[var][nv].values() for l in s})
sizes = {}
for n in os.listdir(OUT):
    mp = os.path.join(OUT, n, "e5_meta.json")
    if os.path.exists(mp):
        mv = json.load(open(mp))["motions"].get("1", {})
        if "mov_size" in mv:
            sizes[n] = dict(size=mv["mov_size"], implied=mv["implied_static_size"], k=mv["k"])


def mid_auc(A, n):
    v = [A[n].get(l, np.nan) for l in MID] if A and n in A else []
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def readout(var, nv, n, what):
    A = PR[var][nv]
    if what == "mid":
        return mid_auc(A, n)
    if what.startswith("L"):
        return A[n].get(int(what[1:]), np.nan) if A and n in A else np.nan
    O = OM[nv]
    if not O or n not in O:
        return np.nan
    t = O[n].get(23) or O[n][max(O[n])]
    v = t.get(what)
    return float(v) if v is not None else np.nan


def scenes_of(sp, var="pool", nv=None):
    pool = set()
    for v in VIEWS:
        if PR[var][v]:
            pool |= set(PR[var][v])
        if OM[v]:
            pool |= set(OM[v])
    return sorted(n for n in pool if split_of(n) == sp)


S = {"by_split": {}, "accel": {}, "size": {}, "notes": []}
WHATS = ["mid", "L24", "mask_iou", "slope"]
for var in ("pool", "xfer"):
    for sp in SPL:
        for nv in VIEWS:
            for w in WHATS:
                if w in ("mask_iou", "slope") and var == "xfer":
                    continue
                S["by_split"].setdefault(var, {}).setdefault(sp, {}).setdefault(str(nv), {})[w] = boot_ci([readout(var, nv, n, w) for n in scenes_of(sp, var)])

# ---- accel vs const (paired by seed)
acc = {seed(n): n for n in scenes_of("epi_accel")}; con = {seed(n): n for n in scenes_of("epi_const")}
pairs = sorted(set(acc) & set(con))
S["accel"]["n_pairs"] = len(pairs)
for var in ("pool", "xfer"):
    for w in WHATS:
        if w in ("mask_iou", "slope") and var == "xfer":
            continue
        T = S["accel"].setdefault(var, {}).setdefault(w, {})
        if 2 not in VIEWS:
            continue
        base_a = {s: readout(var, 2, acc[s], w) for s in pairs}; base_c = {s: readout(var, 2, con[s], w) for s in pairs}
        T["accel_minus_const_2v_maxabs"] = float(np.nanmax(np.abs([base_a[s] - base_c[s] for s in pairs]))) if pairs else None
        for nv in VIEWS:
            if nv == 2:
                continue
            da = [readout(var, nv, acc[s], w) - base_a[s] for s in pairs]
            dc = [readout(var, nv, con[s], w) - base_c[s] for s in pairs]
            did = [a - c for a, c in zip(da, dc)]
            ac = [readout(var, nv, acc[s], w) - readout(var, nv, con[s], w) for s in pairs]
            T[str(nv)] = dict(delta_accel=boot_ci(da), sign_accel=sign_test(da), delta_const=boot_ci(dc), sign_const=sign_test(dc),
                              did=boot_ci(did), sign_did=sign_test(did), accel_minus_const=boot_ci(ac), sign_ac=sign_test(ac))
        ps = [T[str(nv)]["sign_did"]["p"] for nv in VIEWS if nv != 2]
        for nv, ph in zip([v for v in VIEWS if v != 2], holm(ps)):
            T[str(nv)]["sign_did"]["p_holm"] = ph
# slope at the frame-12 target (where a frame-12 view exists)
for nv in VIEWS:
    O = OM[nv]
    if not O:
        continue
    x = [(O[acc[s]].get(12, {}).get("slope"), O[con[s]].get(12, {}).get("slope")) for s in pairs if acc[s] in O and con[s] in O]
    x = [(a, c) for a, c in x if a is not None and c is not None]
    if x:
        S["accel"].setdefault("slope_target12", {})[str(nv)] = dict(accel=boot_ci([a for a, _ in x]), const=boot_ci([c for _, c in x]),
                                                                    diff=boot_ci([a - c for a, c in x]), sign=sign_test([a - c for a, c in x]))

# ---- size cue (unpaired vs reference)
for var in ("pool", "xfer"):
    for w in WHATS:
        if w in ("mask_iou", "slope") and var == "xfer":
            continue
        T = S["size"].setdefault(var, {}).setdefault(w, {})
        for nv in VIEWS:
            ref = [readout(var, nv, n, w) for n in scenes_of(REF, var)]
            sr = [readout(var, nv, n, w) for n in scenes_of("epi_sizerand", var)]
            sm = [readout(var, nv, n, w) for n in scenes_of("epi_sizematch", var)]
            cv = [readout(var, nv, n, w) for n in scenes_of("epi_const", var)]
            T[str(nv)] = dict(sizerand_minus_ref=dict(boot_diff(sr, ref), p_mwu=mwu(sr, ref)),
                              sizerand_minus_const=dict(boot_diff(sr, cv), p_mwu=mwu(sr, cv)),
                              sizematch_minus_const=dict(boot_diff(sm, cv), p_mwu=mwu(sm, cv)),
                              sizematch_minus_ref=dict(boot_diff(sm, ref), p_mwu=mwu(sm, ref)),
                              sizematch_minus_sizerand=dict(boot_diff(sm, sr), p_mwu=mwu(sm, sr)))
            for sp in ("epi_sizerand", "epi_sizematch"):
                ns = [n for n in scenes_of(sp, var) if n in sizes]
                T[str(nv)][f"spearman_{sp}_vs_implied_size"] = spearman([readout(var, nv, n, w) for n in ns], [sizes[n]["implied"] for n in ns])
                T[str(nv)][f"spearman_{sp}_vs_true_size"] = spearman([readout(var, nv, n, w) for n in ns], [sizes[n]["size"] for n in ns])
        ps = [T[str(nv)]["sizerand_minus_ref"]["p_mwu"] for nv in VIEWS]
        for nv, ph in zip(VIEWS, holm(ps)):
            T[str(nv)]["sizerand_minus_ref"]["p_holm"] = ph
json.dump(S, open(os.path.join(RD, "v4_stats.json"), "w"), indent=1)


# ---------------------------------------------------------------- figures
def curve(var, nv, sp):
    A = PR[var][nv]
    if not A:
        return None
    ns = [n for n in scenes_of(sp, var) if n in A]
    if not ns:
        return None
    M = np.array([[A[n].get(l, np.nan) for l in LAYERS] for n in ns])
    return np.nanmean(M, 0)


import matplotlib; matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def errbar(ax, x, c, color, label, off=0.0, marker="o"):
    if c and c.get("mean") is not None:
        ax.errorbar([x + off], [c["mean"]], yerr=[[c["mean"] - c["lo"]], [c["hi"] - c["mean"]]], fmt=marker, color=color,
                    capsize=3, label=label, ms=5)


fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
for nv, ls in ((2, ":"), (VIEWS[-1], "-")):
    for sp in ("epi_accel", "epi_const"):
        cv = curve("pool", nv, sp)
        if cv is not None:
            ax[0].plot(LAYERS, cv, ls, marker="o", ms=3, color=COL[sp], label=f"{LBL[sp]}, {nv} views")
ax[0].axhline(0.5, color="0.7", lw=0.8); ax[0].set_xlabel("layer"); ax[0].set_ylabel("motion AUC, view-0 object patches")
ax[0].set_title("(a) probe AUC by layer (pooled probe)", fontsize=9); ax[0].legend(fontsize=7)
for k, sp in enumerate(("epi_accel", "epi_const")):
    for nv in VIEWS:
        errbar(ax[1], nv, S["by_split"]["pool"][sp][str(nv)]["mid"], COL[sp], LBL[sp] if nv == VIEWS[0] else None, off=(k - 0.5) * 0.15)
ax[1].set_xticks(VIEWS); ax[1].set_xlabel("input views"); ax[1].set_ylabel("motion AUC, mean of layers 8, 12, 16")
ax[1].set_title("(b) mid-depth AUC vs views (95% CI)", fontsize=9); ax[1].legend(fontsize=7)
for nv in VIEWS[1:]:
    c = S["accel"]["pool"]["mid"].get(str(nv), {}).get("did")
    errbar(ax[2], nv, c, "#d95f02", "probe, mid-depth AUC" if nv == VIEWS[1] else None, off=-0.08)
    c2 = S["accel"]["pool"]["mask_iou"].get(str(nv), {}).get("did")
    errbar(ax[2], nv, c2, "#1f78b4", "output mask IoU" if nv == VIEWS[1] else None, off=0.08, marker="s")
ax[2].axhline(0, color="0.5", lw=0.8); ax[2].set_xticks(VIEWS[1:]); ax[2].set_xlabel("input views (n)")
ax[2].set_ylabel("[X(n) - X(2)] accel  minus  [X(n) - X(2)] const"); ax[2].set_title("(c) difference in differences, paired by seed", fontsize=9)
ax[2].legend(fontsize=7)
for k, sp in enumerate(("epi_accel", "epi_const")):
    for nv in VIEWS:
        errbar(ax[3], nv, S["by_split"]["pool"][sp][str(nv)]["mask_iou"], COL[sp], LBL[sp] if nv == VIEWS[0] else None, off=(k - 0.5) * 0.15)
        errbar(ax[3], nv, S["by_split"]["pool"][sp][str(nv)]["slope"], COL[sp], None, off=(k - 0.5) * 0.15 + 0.04, marker="^")
ax[3].plot([], [], "ko", label="mask IoU"); ax[3].plot([], [], "k^", label="slope (frame 23)")
ax[3].set_xticks(VIEWS); ax[3].set_xlabel("input views"); ax[3].set_ylabel("output metric"); ax[3].legend(fontsize=7)
ax[3].set_title("(d) Any4D outputs (95% CI)", fontsize=9)
fig.suptitle(f"View-count test with accelerated epipolar motion (n = {len(pairs)} seed pairs)", fontsize=10, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(os.path.join(RD, "fig_v4_accel_viewcount.png"), dpi=110); plt.close(fig)

fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
for sp in (REF, "epi_sizerand", "epi_sizematch"):
    for nv, ls in ((2, ":"), (VIEWS[-1], "-")):
        cv = curve("xfer", nv, sp)
        if cv is not None:
            ax[0].plot(LAYERS, cv, ls, marker="o", ms=3, color=COL[sp], label=f"{LBL[sp]}, {nv} views")
ax[0].axhline(0.5, color="0.7", lw=0.8); ax[0].set_xlabel("layer"); ax[0].set_ylabel("motion AUC, view-0 object patches")
ax[0].set_title("(a) AUC by layer (probe trained on v3float only)", fontsize=9); ax[0].legend(fontsize=6)
for k, sp in enumerate((REF, "epi_sizerand", "epi_sizematch")):
    for nv in VIEWS:
        errbar(ax[1], nv, S["by_split"]["xfer"][sp][str(nv)]["mid"], COL[sp], LBL[sp] if nv == VIEWS[0] else None, off=(k - 1) * 0.15)
ax[1].set_xticks(VIEWS); ax[1].set_xlabel("input views"); ax[1].set_ylabel("motion AUC, mean of layers 8, 12, 16")
ax[1].set_title("(b) mid-depth AUC vs views, transfer probe (95% CI)", fontsize=9); ax[1].legend(fontsize=7)
for sp in ("epi_sizerand", "epi_sizematch"):
    ns = [n for n in scenes_of(sp, "xfer") if n in sizes]
    ax[2].plot([sizes[n]["implied"] for n in ns], [readout("xfer", 2, n, "mid") for n in ns], "o", color=COL[sp], label=LBL[sp], alpha=0.8)
ax[2].axvspan(1.4, 2.2, color="0.92", zorder=0, label="static-object sizes")
ax[2].set_xlabel("size of the moving object if read as static (m)"); ax[2].set_ylabel("mid-depth AUC, 2 views, transfer probe")
ax[2].set_title("(c) dose-response on the size cue", fontsize=9); ax[2].legend(fontsize=7)
for k, sp in enumerate((REF, "epi_sizerand", "epi_sizematch")):
    for nv in VIEWS:
        errbar(ax[3], nv, S["by_split"]["pool"][sp][str(nv)]["mask_iou"], COL[sp], LBL[sp] if nv == VIEWS[0] else None, off=(k - 1) * 0.15)
ax[3].set_xticks(VIEWS); ax[3].set_xlabel("input views"); ax[3].set_ylabel("output motion-mask IoU"); ax[3].legend(fontsize=7)
ax[3].set_title("(d) Any4D motion mask (95% CI)", fontsize=9)
fig.suptitle("Monocular size cue: moving-object size decoupled from depth", fontsize=10, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(os.path.join(RD, "fig_v4_sizecue.png"), dpi=110); plt.close(fig)


# ---------------------------------------------------------------- markdown
def f(c, d=3):
    if not c or c.get("mean") is None:
        return "n/a"
    return f"{c['mean']:.{d}f} [{c['lo']:.{d}f}, {c['hi']:.{d}f}]"


def fp(p):
    return "n/a" if p is None else (f"{p:.1e}" if p < 1e-3 else f"{p:.3f}")


L = [f"# {TID}: view-count test with accelerated epipolar motion, and the monocular size cue (review item W3)", "",
     "Data: v4epi (epi_accel and epi_const paired by seed, epi_sizerand, epi_sizematch; floating, shadowless, v3float settings) "
     "pooled with the 70 v3float scenes (reference: its 30 epi_aligned scenes). Any4D, frames 2 views = 0,23; 3 = 0,12,23; 4 = 0,8,16,23; 7 = 0,4,8,12,16,20,23.", "",
     "Metric definitions are those of round 1. Probe = ridge moving-vs-static probe on the view-0 token (object patches), 5-fold CV by scene.",
     "Pooled probe: trained on all scenes, paired accel and const scenes share a fold. Transfer probe: trained only on v3float scenes (round-1 folds), so it has never seen an accelerated or size-randomised mover.",
     "CIs are bootstrap 95% over scenes (10000 resamples). Paired tests are two-sided sign tests; unpaired tests are Mann-Whitney.", "",
     "Pre-specified primary read-outs: probe AUC averaged over layers 8, 12 and 16 (layer 24 is near ceiling in round 1), output mask IoU and slope at the frame-23 target.", "",
     "## 1. Accelerated motion: does a third view help?", "",
     f"Seed pairs: {len(pairs)}. With 2 views the accel and const inputs are the same images, so any difference there is a bug: largest |accel - const| at 2 views, pooled mid-depth AUC = "
     f"{S['accel']['pool']['mid'].get('accel_minus_const_2v_maxabs')}.", "",
     "| views | AUC accel | AUC const | delta accel vs 2 views | delta const vs 2 views | difference in differences | sign test (+/-), p, Holm p |", "|---|---|---|---|---|---|---|"]
for nv in VIEWS:
    a = S["by_split"]["pool"]["epi_accel"][str(nv)]["mid"]; c = S["by_split"]["pool"]["epi_const"][str(nv)]["mid"]
    if nv == 2:
        L.append(f"| {nv} | {f(a)} | {f(c)} | | | | |"); continue
    T = S["accel"]["pool"]["mid"][str(nv)]
    L.append(f"| {nv} | {f(a)} | {f(c)} | {f(T['delta_accel'])} | {f(T['delta_const'])} | {f(T['did'])} | "
             f"{T['sign_did']['pos']}/{T['sign_did']['neg']}, {fp(T['sign_did']['p'])}, {fp(T['sign_did'].get('p_holm'))} |")
L += ["", "Same tests for the output (frame-23 target) and for the probe at layer 24:", "",
      "| read-out | views | accel | const | difference in differences | sign test (+/-), p |", "|---|---|---|---|---|---|"]
for w, var in (("mask_iou", "pool"), ("slope", "pool"), ("L24", "pool"), ("mid", "xfer")):
    for nv in VIEWS[1:]:
        T = S["accel"][var][w].get(str(nv))
        if not T:
            continue
        L.append(f"| {w if var == 'pool' else w + ' (transfer probe)'} | {nv} | {f(S['by_split'][var]['epi_accel'][str(nv)][w])} | "
                 f"{f(S['by_split'][var]['epi_const'][str(nv)][w])} | {f(T['did'])} | {T['sign_did']['pos']}/{T['sign_did']['neg']}, {fp(T['sign_did']['p'])} |")
if "slope_target12" in S["accel"]:
    L += ["", "Slope for the frame-12 target (only where frame 12 is an input view):", ""]
    for nv, T in S["accel"]["slope_target12"].items():
        L.append(f"- {nv} views: accel {f(T['accel'])}, const {f(T['const'])}, accel minus const {f(T['diff'])}, sign {T['sign']['pos']}/{T['sign']['neg']} p = {fp(T['sign']['p'])}.")
prim = [S["accel"]["pool"]["mid"].get(str(nv), {}) for nv in VIEWS[1:]]
sup = [t for t in prim if t and t["did"]["mean"] is not None and t["did"]["lo"] > 0 and (t["sign_did"].get("p_holm") or 1) < 0.05]
L += ["", f"Verdict (pre-specified rule: DiD > 0 with CI above 0 and Holm p < 0.05): {len(sup)} of {len(prim)} view counts meet it. "
      + ("The model's internal motion signal for epipolar-aligned movers improves when a third view makes the motion geometrically resolvable, as a geometric account predicts."
         if len(sup) == len(prim) and prim else
         "The prediction is not met at every view count; see the table for which effects are present." if sup else
         "No view count meets it: adding views that make the motion geometrically resolvable does not raise decodability beyond the constant-velocity control."), ""]
L += ["![accel view-count](fig_v4_accel_viewcount.png)", "", "## 2. Size cue: is apparent size what reveals epipolar-aligned motion?", "",
      "| views | v3float epi_aligned | sizerand | sizematch | sizerand minus ref, MWU p, Holm p | sizematch minus ref, MWU p |", "|---|---|---|---|---|---|"]
for nv in VIEWS:
    T = S["size"]["xfer"]["mid"][str(nv)]; B_ = S["by_split"]["xfer"]
    L.append(f"| {nv} | {f(B_[REF][str(nv)]['mid'])} | {f(B_['epi_sizerand'][str(nv)]['mid'])} | {f(B_['epi_sizematch'][str(nv)]['mid'])} | "
             f"{f(T['sizerand_minus_ref'])}, {fp(T['sizerand_minus_ref']['p_mwu'])}, {fp(T['sizerand_minus_ref'].get('p_holm'))} | "
             f"{f(T['sizematch_minus_ref'])}, {fp(T['sizematch_minus_ref']['p_mwu'])} |")
L += ["", "Matched reference: epi_const is generated with the same v4 constraints as the size splits (no image overlap between the mover and static objects), "
      "while v3float epi_aligned is the round-1 reference requested in the plan. Transfer probe, mid-depth AUC:", "",
      "| views | epi_const | sizerand minus const, MWU p | sizematch minus const, MWU p |", "|---|---|---|---|"]
for nv in VIEWS:
    T = S["size"]["xfer"]["mid"][str(nv)]
    L.append(f"| {nv} | {f(S['by_split']['xfer']['epi_const'][str(nv)]['mid'])} | {f(T['sizerand_minus_const'])}, {fp(T['sizerand_minus_const']['p_mwu'])} | "
             f"{f(T['sizematch_minus_const'])}, {fp(T['sizematch_minus_const']['p_mwu'])} |")
L += ["", "Primary table uses the transfer probe, mid-depth AUC. Secondary read-outs:", "",
      "| read-out | views | ref | sizerand | sizematch | sizerand minus ref, MWU p | Spearman(AUC, size if read as static) in sizerand |", "|---|---|---|---|---|---|---|"]
for w, var in (("mid", "pool"), ("L24", "xfer"), ("mask_iou", "pool"), ("slope", "pool")):
    for nv in VIEWS:
        T = S["size"][var][w][str(nv)]; B_ = S["by_split"][var]; sp_ = T["spearman_epi_sizerand_vs_implied_size"]
        L.append(f"| {w} ({'pooled' if var == 'pool' else 'transfer'}) | {nv} | {f(B_[REF][str(nv)][w])} | {f(B_['epi_sizerand'][str(nv)][w])} | {f(B_['epi_sizematch'][str(nv)][w])} | "
                 f"{f(T['sizerand_minus_ref'])}, {fp(T['sizerand_minus_ref']['p_mwu'])} | "
                 f"{'n/a' if sp_['rho'] is None else format(sp_['rho'], '.2f')} (p {fp(sp_['p'])}, n {sp_['n']}) |")
prim = [S["size"]["xfer"]["mid"][str(nv)]["sizerand_minus_ref"] for nv in VIEWS]
sup = [t for t in prim if t["mean"] is not None and t["hi"] < 0 and (t.get("p_holm") or 1) < 0.05]
L += ["", f"Verdict (pre-specified rule: sizerand minus ref < 0 with CI below 0 and Holm p < 0.05): {len(sup)} of {len(prim)} view counts meet it. "
      + ("Breaking the size cue lowers decodability, so apparent size contributes to how the model detects epipolar-aligned motion." if sup else
         "Breaking the size cue does not lower decodability, so apparent size is not what the model relies on (other monocular cues remain possible)."), "",
      "![size cue](fig_v4_sizecue.png)", "", "## Caveats", "",
      "- Probe AUC is linear decodability, not use (see review item W4).",
      "- The reference (v3float epi_aligned) and the size splits are different scenes, so the size comparison is unpaired.",
      "- v4 scenes keep static objects from overlapping the mover in the image (to avoid occlusion-order cues and hidden movers); v3float epi_aligned does not, which is why epi_const is also reported as a matched reference.",
      "- epi_sizematch makes the static reading a typical size, but that static copy also floats higher (it is scaled about the camera), so it is not a perfect look-alike of a static object.",
      "- Files: v4_stats.json (all numbers), outm_<n>v.json, vs_pool_<n>v_results.json, vs_xfer_<n>v_results.json, step3_v4epi_<n>v.json."]
open(os.path.join(RD, f"{TID}_summary.md"), "w").write("\n".join(L) + "\n")
print("summary written", len(pairs), "pairs")
