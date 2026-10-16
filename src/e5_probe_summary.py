"""Round 2, review W4: aggregate probe controls and draw the per-layer figure.
Usage: python e5_probe_summary.py <probe_root> <out_dir>
<probe_root>/<ds>/<source>/lin_<mode>_s<seed>_results.json  (part P of e5_internal_analysis.py, seeded / shuffled)
<probe_root>/<ds>/<source>/mlp_<mode>_s<seed>.json          (e5_probe_mlp.py, arch mlp and lin)
Probe families: "ridge/bilinear" = round-1 linear probes (ridge motion probe, low-rank bilinear pair probe),
"MLP" = 2-layer MLP, "linear (MLP recipe)" = linear probe trained with the MLP recipe.
For each (dataset, source, probe, label mode, layer, metric):
  per seed: mean over scenes, then mean +/- sd over seeds;
  per scene: mean over seeds, then the mean over scenes with a 95% bootstrap CI (2000 resamples over scenes).
Paired comparisons use the per-scene seed means: bootstrap 95% CI of the mean difference and a two-sided sign test.
Selectivity = real labels minus control labels (patch shuffle or instance shuffle), same source, probe and layer.
"""
import glob, json, math, os, re, sys
from collections import defaultdict
import numpy as np

root, outd = sys.argv[1], sys.argv[2]
os.makedirs(outd, exist_ok=True)
METRICS = ["motion_auc_v0_objects", "motion_auc_v0", "pair_auc_obj", "pair_auc", "copies_separated", "k_correct"]
SRC_LAB = {"trained": "Any4D trained", "randall": "random init (all)", "randtf": "DINOv2 + random transformer",
           "enc_any4d": "Any4D encoder tokens (no view PE)", "enc_dino": "public DINOv2 tokens (no view PE)"}
PROBE_LAB = {"ridge": "linear (round-1 ridge / bilinear)", "mlp": "MLP", "lin": "linear (MLP recipe)"}
# vals[(ds, src, probe, mode, layer, metric)][seed][scene] = value
vals = defaultdict(lambda: defaultdict(dict))
for f in glob.glob(os.path.join(root, "*", "*", "lin_*_results.json")):
    ds, src = f.split(os.sep)[-3], f.split(os.sep)[-2]
    m = re.match(r"lin_(\w+?)_s(\d+)_results\.json", os.path.basename(f)); mode, seed = m.group(1), int(m.group(2))
    for r in json.load(open(f))["P"]["rows"]:
        r = dict(r); r["k_correct"] = float(r["k_hat"] == r["k_true"]) if "k_hat" in r else float("nan")
        for met in METRICS:
            v = r.get(met)
            if v is not None and np.isfinite(v):
                vals[(ds, src, "ridge", mode, r["layer"], met)][seed][r["scene"]] = float(v)
for f in glob.glob(os.path.join(root, "*", "*", "mlp_*.json")):
    if f.endswith(".partial"):
        continue
    ds, src = f.split(os.sep)[-3], f.split(os.sep)[-2]
    m = re.match(r"mlp_(\w+?)_s(\d+)\.json", os.path.basename(f)); mode, seed = m.group(1), int(m.group(2))
    for r in json.load(open(f))["rows"]:
        for met in METRICS[:4]:
            v = r.get(met)
            if v is not None and np.isfinite(v):
                vals[(ds, src, r["arch"], mode, r["layer"], met)][seed][r["scene"]] = float(v)
rng = np.random.default_rng(0)


def scene_means(d):
    sc = sorted(set().union(*[set(x) for x in d.values()]))
    return {s: float(np.mean([d[k][s] for k in d if s in d[k]])) for s in sc}


def boot(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan"), float("nan")
    bs = rng.choice(x, (2000, len(x))).mean(1)
    return tuple(map(float, np.percentile(bs, [2.5, 97.5])))


def sign_p(d):
    d = np.asarray(d, float); d = d[np.isfinite(d) & (d != 0)]; n = len(d); k = int((d < 0).sum())
    if n == 0:
        return float("nan")
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)


agg = {}
for key, d in vals.items():
    per_seed = [np.mean(list(v.values())) for v in d.values() if v]
    sm = scene_means(d)
    lo, hi = boot(list(sm.values()))
    agg[key] = dict(mean=float(np.mean(list(sm.values()))), seed_mean=float(np.mean(per_seed)),
                    seed_sd=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else float("nan"),
                    n_seeds=len(per_seed), n_scenes=len(sm), ci=[lo, hi], scenes=sm)


def paired(k1, k2):
    if k1 not in agg or k2 not in agg:
        return None
    a, b = agg[k1]["scenes"], agg[k2]["scenes"]; sc = sorted(set(a) & set(b))
    if len(sc) < 3:
        return None
    d = np.array([a[s] - b[s] for s in sc]); lo, hi = boot(d)
    return dict(diff=float(d.mean()), ci=[lo, hi], sign_p=sign_p(d), n=len(sc))


rows = []
for key, a in sorted(agg.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
    ds, src, probe, mode, layer, met = key
    rows.append(dict(dataset=ds, source=src, probe=probe, labels=mode, layer=layer, metric=met, mean=a["mean"], ci_lo=a["ci"][0],
                     ci_hi=a["ci"][1], seed_mean=a["seed_mean"], seed_sd=a["seed_sd"], n_seeds=a["n_seeds"], n_scenes=a["n_scenes"]))
comps = []
for (ds, src, probe, mode, layer, met) in list(agg):
    if mode != "real":
        continue
    for ctrl in ("patch", "instance"):
        r = paired((ds, src, probe, "real", layer, met), (ds, src, probe, ctrl, layer, met))
        if r:
            comps.append(dict(dataset=ds, comparison=f"selectivity real - {ctrl}", source=src, probe=probe, layer=layer, metric=met, **r))
    if src == "trained":
        for other in ("randall", "randtf"):
            r = paired((ds, "trained", probe, "real", layer, met), (ds, other, probe, "real", layer, met))
            if r:
                comps.append(dict(dataset=ds, comparison=f"trained - {other}", source="trained", probe=probe, layer=layer, metric=met, **r))
        if layer == 0:
            for other in ("enc_any4d", "enc_dino"):
                r = paired((ds, "trained", probe, "real", 0, met), (ds, other, probe, "real", 0, met))
                if r:
                    comps.append(dict(dataset=ds, comparison=f"trained L0 - {other}", source="trained", probe=probe, layer=0, metric=met, **r))
    if probe == "mlp" and layer in (0, 24):
        r = paired((ds, src, "mlp", "real", layer, met), (ds, src, "ridge", "real", layer, met))
        if r:
            comps.append(dict(dataset=ds, comparison="MLP - ridge/bilinear", source=src, probe="mlp", layer=layer, metric=met, **r))
    if layer == 24:
        r = paired((ds, src, probe, "real", 24, met), (ds, src, probe, "real", 0, met))
        if r:
            comps.append(dict(dataset=ds, comparison="L24 - L0", source=src, probe=probe, layer=24, metric=met, **r))


def wcsv(path, rr):
    if not rr:
        return
    ks = list(dict.fromkeys(k for r in rr for k in r))
    with open(path, "w") as fh:
        fh.write(",".join(ks) + "\n")
        for r in rr:
            fh.write(",".join((f"{r[k]:.5g}" if isinstance(r.get(k), float) else str(r.get(k, ""))) if not isinstance(r.get(k), list) else
                              f"{r[k][0]:.5g};{r[k][1]:.5g}" for k in ks) + "\n")


wcsv(os.path.join(outd, "probe_controls_per_layer.csv"), rows)
wcsv(os.path.join(outd, "probe_controls_comparisons.csv"), comps)
json.dump(dict(rows=rows, comparisons=comps), open(os.path.join(outd, "probe_controls.json"), "w"), indent=1)

# ------------------------------------------------------------------ figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
dss = sorted({k[0] for k in agg})
COL = {"trained": "#1f4e79", "randall": "#c0392b", "randtf": "#e67e22", "enc_any4d": "#1f4e79", "enc_dino": "#e67e22"}
for metset, fname in [(["motion_auc_v0_objects", "pair_auc_obj"], "fig_probe_controls.png"), (["motion_auc_v0", "pair_auc"], "fig_probe_controls_allpatches.png")]:
    cols = [(ds, pr) for ds in dss for pr in ("ridge", "mlp")]
    fig, axs = plt.subplots(len(metset), len(cols), figsize=(2.9 * len(cols), 2.7 * len(metset)), squeeze=False, sharey="row")
    for i, met in enumerate(metset):
        for j, (ds, pr) in enumerate(cols):
            ax = axs[i, j]
            for src in ("trained", "randall", "randtf"):
                for mode, ls, lab in (("real", "-", ""), ("patch", ":", " shuffled (patch)"), ("instance", "--", " shuffled (instance)")):
                    if mode != "real" and src != "trained":
                        continue
                    ks = sorted([k for k in agg if k[:4] == (ds, src, pr, mode) and k[5] == met], key=lambda k: k[4])
                    if not ks:
                        continue
                    x = [k[4] for k in ks]; y = np.array([agg[k]["seed_mean"] for k in ks]); s = np.array([agg[k]["seed_sd"] for k in ks])
                    color = COL[src] if mode == "real" else "0.45"
                    ax.plot(x, y, ls, color=color, marker="o", ms=2.5, lw=1.3, label=SRC_LAB[src] + lab)
                    if np.isfinite(s).all():
                        ax.fill_between(x, y - s, y + s, color=color, alpha=0.18, lw=0)
            for src, mk in (("enc_any4d", "s"), ("enc_dino", "D")):
                k = (ds, src, pr, "real", 0, met)
                if k in agg:
                    ax.errorbar([-0.8 if src == "enc_any4d" else -1.6], [agg[k]["seed_mean"]], yerr=[[agg[k]["seed_mean"] - agg[k]["ci"][0]], [agg[k]["ci"][1] - agg[k]["seed_mean"]]],
                                fmt=mk, color=COL[src], mfc="white", ms=4, label=SRC_LAB[src])
            ax.axhline(0.5, color="k", lw=0.5, ls=":")
            ax.set_title(f"{ds}, {PROBE_LAB[pr]}", fontsize=8)
            ax.set_xlabel("layer (0 = transformer input)")
            if j == 0:
                ax.set_ylabel(met.replace("_", " "))
            ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    hl = {}
    for ax in axs.ravel():
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(l_, h_)
    fig.legend(list(hl.values()), list(hl.keys()), loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.suptitle("Probe controls: line = mean over probe seeds, band = sd over seeds;\nmarkers left of layer 0 = raw encoder tokens without view PE (95% CI over scenes)", fontsize=8)
    fig.tight_layout(rect=(0, 0.16, 1, 0.93)); fig.savefig(os.path.join(outd, fname), dpi=150); plt.close(fig)

# ------------------------------------------------------------------ markdown
def fmt(k):
    a = agg.get(k)
    if not a:
        return "n/a"
    sd = f" +/- {a['seed_sd']:.3f}" if np.isfinite(a["seed_sd"]) else ""
    return f"{a['seed_mean']:.3f}{sd} [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}]"


L = ["# Probe controls (review W4, plan item 6)\n",
     "Cells: mean over seeds +/- sd over seeds [95% bootstrap CI over scenes of the seed-averaged per-scene value]. "
     "Chance is 0.5 for all AUCs. Folds are grouped by scene (5 folds, as round 1). Sources: " +
     "; ".join(f"{k} = {v}" for k, v in SRC_LAB.items()) + ".\n"]
for ds in dss:
    for met in ["motion_auc_v0_objects", "pair_auc_obj", "motion_auc_v0", "pair_auc"]:
        L.append(f"\n## {ds}: {met}\n")
        layers = sorted({k[4] for k in agg if k[0] == ds and k[5] == met})
        L.append("| source | probe | labels | " + " | ".join(f"L{l}" for l in layers) + " |")
        L.append("|---|---|---|" + "---|" * len(layers))
        for src in ("trained", "randall", "randtf", "enc_any4d", "enc_dino"):
            for pr in ("ridge", "lin", "mlp"):
                for mode in ("real", "patch", "instance"):
                    if not any(k[:4] == (ds, src, pr, mode) and k[5] == met for k in agg):
                        continue
                    L.append(f"| {src} | {PROBE_LAB[pr]} | {mode} | " + " | ".join(fmt((ds, src, pr, mode, l, met)) for l in layers) + " |")
L.append("\n## Key paired comparisons (per-scene seed means; mean difference [95% CI], sign-test p, n scenes)\n")
L.append("| dataset | comparison | source | probe | layer | metric | difference [95% CI] | sign p | n |")
L.append("|---|---|---|---|---|---|---|---|---|")
for c in comps:
    if c["metric"] in ("motion_auc_v0_objects", "pair_auc_obj") and (c["layer"] in (0, 24)):
        L.append(f"| {c['dataset']} | {c['comparison']} | {c['source']} | {PROBE_LAB[c['probe']]} | {c['layer']} | {c['metric']} | "
                 f"{c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] | {c['sign_p']:.2g} | {c['n']} |")
L.append("\nFull per-layer tables: probe_controls_per_layer.csv; all comparisons (every layer and metric): probe_controls_comparisons.csv.\n")
open(os.path.join(outd, "probe_controls_tables.md"), "w").write("\n".join(L) + "\n")
# compact version for the task summary: headline metrics (object patches) and the key comparisons first
K = [L[1], "\n## Key paired comparisons, layers 0 and 24 (per-scene seed means; mean difference [95% CI], sign-test p, n scenes)\n"]
K += [l for l in L[L.index("| dataset | comparison | source | probe | layer | metric | difference [95% CI] | sign p | n |"):] if l.startswith("|")]
for ds in dss:
    for met in ["motion_auc_v0_objects", "pair_auc_obj"]:
        i0 = L.index(f"\n## {ds}: {met}\n")
        j0 = i0 + 1
        while j0 < len(L) and L[j0].startswith("|"):
            j0 += 1
        K += L[i0:j0]
K.append("\nAll metrics (including all-patch AUCs) per layer: probe_controls_tables.md, probe_controls_per_layer.csv, probe_controls_comparisons.csv.\n")
open(os.path.join(outd, "probe_controls_key.md"), "w").write("\n".join(K) + "\n")
print("rows", len(rows), "comparisons", len(comps))
