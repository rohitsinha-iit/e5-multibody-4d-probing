
import json, os, glob, csv, math, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.expanduser("~/e5_multibody/queue/results"); OUT = "figs"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans", "axes.linewidth": 0.7})
BLUE, ORANGE, GREEN, PINK, SKY, VERM, GREY, YEL = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#7f7f7f", "#F0E442"
J = lambda p: json.load(open(os.path.join(R, p)))
fin = lambda v: isinstance(v, (int, float)) and np.isfinite(v)
def Prows(folder):
    rows = []
    for f in sorted(glob.glob(os.path.join(R, folder, "partP*_results.json"))):
        rows += json.load(open(f))["P"]["rows"]
    return rows
def curve(rows, key, cond=lambda r: True, B=1000):
    out = {}
    for l in sorted({r["layer"] for r in rows}):
        d = collections.defaultdict(list)
        for r in rows:
            if r["layer"] == l and cond(r) and fin(r.get(key)): d[r["scene"]].append(r[key])
        v = np.array([np.mean(x) for x in d.values()])
        if len(v) > 1:
            rng = np.random.default_rng(0); bs = rng.choice(v, (B, len(v))).mean(1)
            out[l] = (v.mean(), *np.percentile(bs, [2.5, 97.5]), len(v))
    return out
def letter(ax, s): ax.text(-0.16, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
def plot_curve(ax, c, color, label, ls="-", x_scale=1.0):
    L = sorted(c); x = np.array(L) * x_scale
    m = np.array([c[l][0] for l in L]); lo = np.array([c[l][1] for l in L]); hi = np.array([c[l][2] for l in L])
    ax.plot(x, m, ls, color=color, lw=1.5, label=label, marker="o", ms=2.2); ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
epi = lambda r: r["split"] == "epi_aligned"; nonepi = lambda r: r["split"] != "epi_aligned"
summary = {}

# ---------------------------------------------------------------- Figure 1: C4 (motion beyond epipolar geometry)
v2 = Prows("v3_pipeline_float/v2_120"); fl = Prows("v3_pipeline_float/v3float")
fig, axs = plt.subplots(1, 3, figsize=(10.2, 2.9), gridspec_kw=dict(wspace=0.38))
ax = axs[0]
c_v2e = curve(v2, "motion_auc_v0_objects", epi); c_fle = curve(fl, "motion_auc_v0_objects", epi)
c_fln = curve(fl, "motion_auc_v0_objects", nonepi)
plot_curve(ax, c_v2e, GREY, f"grounded, epipolar-aligned (n={c_v2e[0][3]})")
plot_curve(ax, c_fle, BLUE, f"floating, no shadows, epipolar-aligned (n={c_fle[0][3]})")
plot_curve(ax, c_fln, SKY, f"floating, other splits (n={c_fln[0][3]})", ls="--")
ax.axhline(0.5, color="black", lw=0.6, ls=":"); ax.set_ylim(0.25, 1.03); ax.set_xlabel("layer (residual stream)"); ax.set_ylabel("moving vs static object AUC (view-0 token)")
ax.set_title("Motion is computed inside the transformer", loc="left"); ax.legend(frameon=False, loc="upper left", fontsize=6)
summary["c4_curves"] = {"grounded_epi": {str(k): v[0] for k, v in c_v2e.items()}, "float_epi": {str(k): v[0] for k, v in c_fle.items()}, "float_nonepi": {str(k): v[0] for k, v in c_fln.items()}}
ax = axs[1]; cols = {2: VERM, 3: ORANGE, 4: GREEN, 7: BLUE}
summary["viewsweep"] = {}
for nv in (2, 3, 4, 7):
    rows = J(f"v3_viewsweep/vs_v3float_{nv}v_results.json")["P"]["rows"]; c = curve(rows, "motion_auc_v0_objects", epi)
    plot_curve(ax, c, cols[nv], f"{nv} views")
    s3 = J(f"v3_viewsweep/step3_v3float_{nv}v.json"); last = [r for r in s3 if r["target_frame"] == 23]
    summary["viewsweep"][nv] = dict(auc_epi={str(k): v[0] for k, v in c.items()},
        iou_epi=float(np.nanmean([r["mask_iou"] for r in last if epi(r) and fin(r.get("mask_iou"))])),
        iou_non=float(np.nanmean([r["mask_iou"] for r in last if not epi(r) and fin(r.get("mask_iou"))])))
ax.axhline(0.5, color="black", lw=0.6, ls=":"); ax.set_ylim(0.25, 1.03); ax.set_xlabel("layer (residual stream)")
ax.set_ylabel("moving vs static AUC, epipolar-aligned"); ax.set_title("Two views already suffice (floating set)", loc="left"); ax.legend(frameon=False, loc="lower right", title="input views", title_fontsize=6.5)
ax = axs[2]
grp = []
for nm, folder in [("grounded", "v3_pipeline_float/v2_120"), ("floating", "v3_pipeline_float/v3float")]:
    s3 = J(folder + "/step3_any4d_7v.json"); last = [r for r in s3 if r["target_frame"] == 23 and fin(r.get("mask_iou"))]
    grp.append((nm, [r["mask_iou"] for r in last if not epi(r)], [r["mask_iou"] for r in last if epi(r)]))
x = np.arange(len(grp)); w = 0.36
for i, (nm, a, b) in enumerate(grp):
    for j, (v, col, lab) in enumerate([(a, SKY, "other splits"), (b, BLUE, "epipolar-aligned")]):
        m = np.mean(v); rng = np.random.default_rng(0); bs = rng.choice(v, (1000, len(v))).mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
        ax.bar(i + (j - 0.5) * w, m, w * 0.9, color=col, label=lab if i == 0 else None)
        ax.errorbar(i + (j - 0.5) * w, m, yerr=[[m - lo], [hi - m]], color="black", lw=0.8, capsize=2)
        ax.text(i + (j - 0.5) * w, hi + 0.02, f"{m:.2f}\nn={len(v)}", ha="center", fontsize=6)
summary["mask_iou"] = {nm: dict(non=float(np.mean(a)), epi=float(np.mean(b)), n_non=len(a), n_epi=len(b)) for nm, a, b in grp}
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in grp]); ax.set_ylim(0, 0.85); ax.set_ylabel("output motion-mask IoU (7 views)")
ax.set_title("...but the decoder uses it only partly", loc="left"); ax.legend(frameon=False, loc="upper right")
for a_, l in zip(axs, "abc"): letter(a_, l)
fig.savefig(f"{OUT}/fig_c4_beyond_epipolar.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Figure 2: C3 far-glued control
adj = J("v3_pipeline_far/v3far/adjacency_control.json")
def ptype(r): return {frozenset({0, 1}): "AB", frozenset({0, 2}): "AC", frozenset({1, 2}): "BC"}.get(frozenset({r["inst_a"], r["inst_b"]}))
lab = {"AB": "A-B: rigidly coupled, far apart, different look", "AC": "A-C: identical look, different motion", "BC": "B-C: different look, different motion"}
col = {"AB": GREEN, "AC": VERM, "BC": GREY}
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw=dict(wspace=0.35))
summary["farglued"] = {}
for ax, key, ttl in [(axs[0], "probe_join", "Supervised pair probe"), (axs[1], "km_join", "Unsupervised k-means")]:
    for t in ("AB", "AC", "BC"):
        c = {}
        for l in sorted({r["layer"] for r in adj}):
            v = np.array([r[key] for r in adj if ptype(r) == t and r["layer"] == l])
            rng = np.random.default_rng(0); bs = rng.choice(v, (1000, len(v))).mean(1); c[l] = (v.mean(), *np.percentile(bs, [2.5, 97.5]), len(v))
        plot_curve(ax, c, col[t], f"{lab[t]} (n={c[0][3]})")
        summary["farglued"].setdefault(key, {})[t] = {str(k): v[0] for k, v in c.items()}
    ax.set_xlabel("layer (residual stream)"); ax.set_ylabel("fraction of patch pairs joined"); ax.set_title(ttl, loc="left"); ax.set_ylim(0, 1.05)
axs[1].legend(frameon=False, loc="upper right", fontsize=6)
for a_, l in zip(axs, "ab"): letter(a_, l)
fig.savefig(f"{OUT}/fig_c3_farglued.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Figure 3: C2 behaviour across data and models
def rd_csv(p): return list(csv.DictReader(open(os.path.join(R, p))))
def fnum(x):
    try: v = float(x); return v if np.isfinite(v) else None
    except Exception: return None
groups = []
for nm, folder in [("Kubric v2\n(grounded)", "v3_pipeline_float/v2_120"), ("Kubric\nfloating", "v3_pipeline_float/v3float"), ("Kubric\nfar-glued", "v3_pipeline_far/v3far")]:
    o = {}
    for r in Prows(folder):
        if fin(r.get("output_slope")): o[r["scene"]] = r["output_slope"]
    groups.append((nm, list(o.values()), BLUE))
for nm, p, cl in [("KITTI 2015\n(2 views, wide)", "t8_realdata_kitti/kitti15_rgb_2v_wide.csv", GREEN), ("Sintel clean\n(2 views, wide)", "t8_realdata_sintel/sintel_clean_2v_wide.csv", ORANGE), ("Sintel final\n(2 views, wide)", "t8_realdata_sintel/sintel_final_2v_wide.csv", ORANGE)]:
    v = [fnum(r["slope"]) for r in rd_csv(p)]; groups.append((nm, [x for x in v if x is not None], cl))
fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.0), gridspec_kw=dict(width_ratios=[2.3, 1], wspace=0.28))
ax = axs[0]; summary["slopes"] = {}
for i, (nm, v, cl) in enumerate(groups):
    v = np.array(v); vc = np.clip(v, -0.5, 2.0)
    parts = ax.violinplot([vc], positions=[i], widths=0.75, showextrema=False)
    for b in parts["bodies"]: b.set_facecolor(cl); b.set_alpha(0.35); b.set_edgecolor("none")
    rng = np.random.default_rng(i); ax.scatter(i + rng.uniform(-0.12, 0.12, len(vc)), vc, s=2.5, color=cl, alpha=0.5, lw=0)
    m = v.mean(); bs = rng.choice(v, (1000, len(v))).mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color="black", lw=1.0, capsize=3, marker="s", ms=3.5)
    frac = float((v < 1).mean()); ax.text(i, 2.12, f"mean {m:.2f}\n{frac*100:.0f}% < 1\nn={len(v)}", ha="center", fontsize=6)
    summary["slopes"][nm.replace("\n", " ")] = dict(mean=float(m), lo=float(lo), hi=float(hi), frac_below1=frac, n=len(v))
ax.axhline(1.0, color="black", lw=0.8, ls="--"); ax.text(len(groups) - 0.5, 1.02, "full object motion", fontsize=6, ha="right", va="bottom")
ax.axhline(0.0, color="black", lw=0.6, ls=":"); ax.text(len(groups) - 0.5, 0.02, "object treated as static", fontsize=6, ha="right", va="bottom")
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups]); ax.set_ylim(-0.55, 2.45)
ax.set_ylabel("object-motion slope (fraction recovered)"); ax.set_title("Any4D under-recovers object motion on synthetic and real footage", loc="left")
ax = axs[1]
xm = {}
for m, k, ks in [("Any4D", "t7_any4d_v2/xm_output_any4d.json", ("slope2d_q", "sep_ratio_q")), ("VGGT-1B", "t7_vggt_v2/xm_output_vggt.json", ("slope2d_q", "sep_ratio_q"))]:
    o = json.load(open(os.path.join(R, k))); rows = [r for r in (o["rows"] if isinstance(o, dict) else o) if not r.get("error") and int(r["scene"].split("_s")[-1]) < 700]
    xm[m] = {kk: np.array([r[kk] for r in rows if fin(r.get(kk))]) for kk in ks}
pos = 0; summary["xm_output"] = {}
for kk, nm in [("slope2d_q", "object-motion slope"), ("sep_ratio_q", "separation ratio")]:
    for j, (m, cl) in enumerate([("Any4D", BLUE), ("VGGT-1B", PINK)]):
        v = xm[m][kk]; mu = v.mean(); rng = np.random.default_rng(0); bs = rng.choice(v, (1000, len(v))).mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
        ax.bar(pos, mu, 0.8, color=cl, label=m if kk == "slope2d_q" else None); ax.errorbar(pos, mu, yerr=[[mu - lo], [hi - mu]], color="black", lw=0.8, capsize=2)
        ax.text(pos, hi + 0.04, f"{mu:.2f}\nn={len(v)}", ha="center", fontsize=6, bbox=dict(fc="white", ec="none", pad=0.5)); pos += 1
        summary["xm_output"][f"{m} {kk}"] = dict(mean=float(mu), lo=float(lo), hi=float(hi), n=len(v), frac_below1=float((v < 1).mean()))
    pos += 0.6
ax.axhline(1.0, color="black", lw=0.8, ls="--"); ax.set_xticks([0.5, 3.1]); ax.set_xticklabels(["object-motion\nslope", "separation\nratio"])
ax.set_ylim(0, 1.4); ax.set_title("Same pull in a static-trained model", loc="left"); ax.legend(frameon=False, loc="upper right", ncol=2)
for a_, l in zip(axs, "ab"): letter(a_, l)
fig.savefig(f"{OUT}/fig_c2_pull_everywhere.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Figure 4: cross-model internals
models = [("any4d", "Any4D (dynamic)", BLUE, 24), ("mapanything", "MapAnything (static base of Any4D)", SKY, 24), ("vggt", "VGGT-1B (static)", PINK, 48), ("page4d", "PAGE-4D (dynamic, VGGT base)", VERM, 48)]
fig, axs = plt.subplots(1, 3, figsize=(10.2, 2.8), gridspec_kw=dict(wspace=0.35)); summary["xm_internal"] = {}
keep = lambda r: int(r["scene"].split("_s")[-1]) < 700
for m, nm, cl, nl in models:
    rows = [r for r in json.load(open(os.path.join(R, f"t7_{m}_v2/xm_internal_{m}_results.json")))["P"]["rows"] if keep(r)]
    for ax, key, cond in [(axs[0], "motion_auc_v0_objects", lambda r: True), (axs[1], "pair_auc_obj", lambda r: True), (axs[2], "copies_separated", lambda r: True)]:
        c = curve(rows, key, cond); plot_curve(ax, c, cl, nm, x_scale=1.0 / nl)
        summary["xm_internal"].setdefault(m, {})[key] = {str(k): v[0] for k, v in c.items()}
for ax, yl, ttl in [(axs[0], "moving vs static object AUC", "Motion decodable in all models"), (axs[1], "same-rigid-body pair AUC (objects)", "Rigid-body grouping"), (axs[2], "same-asset copies separated", "Splitting identical copies")]:
    ax.set_xlabel("relative depth in the transformer"); ax.set_ylabel(yl); ax.set_title(ttl, loc="left")
axs[0].axhline(0.5, color="black", lw=0.6, ls=":"); axs[2].legend(frameon=False, loc="upper left", fontsize=6)
for a_, l in zip(axs, "abc"): letter(a_, l)
fig.savefig(f"{OUT}/fig_xmodel_internals.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# overlap check of text in all figures is done visually; write the summary
json.dump(summary, open(f"{OUT}/final_summary.json", "w"), indent=1, default=float)
print("ok", sorted(os.listdir(OUT)))

# cross-model output table on the 100 original scenes
rows_out = {}
for m in ["any4d", "mapanything", "vggt", "page4d"]:
    o = json.load(open(os.path.join(R, f"t7_{m}_v2/xm_output_{m}.json"))); rr = [r for r in (o["rows"] if isinstance(o, dict) else o) if not r.get("error") and int(r["scene"].split("_s")[-1]) < 700]
    d = {"n": len(rr)}
    for k in ["slope3d", "slope3d_track", "slope2d_q", "sep_ratio_q", "epe_static_q", "epe_moving_q"]:
        v = np.array([r[k] for r in rr if fin(r.get(k))])
        if len(v): d[k] = dict(mean=float(v.mean()), median=float(np.median(v)), frac_below1=float((v < 1).mean()), n=len(v))
    rows_out[m] = d
json.dump(rows_out, open(f"{OUT}/xm_output_table.json", "w"), indent=1)
print(json.dumps(rows_out)[:3000])
