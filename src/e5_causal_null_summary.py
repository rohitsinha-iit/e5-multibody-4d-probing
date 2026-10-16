"""Round 2, review W5: null distribution of matched random 5-head sets for the five supply heads.
Usage: python e5_causal_null_summary.py causal_null_v2.json <out_prefix>
Writes <out_prefix>_null.csv, <out_prefix>_null.json, <out_prefix>_summary.md, fig_heads_null.png, fig_heads_resample.png.
Per scene, d = condition - base (same scene). Set-level statistic = mean over scenes of d (median also reported).
Null families: mean ablation (rand5_r*), resample ablation (resrand_r*), amplification (amprand_a*_r*).
z = (five - null mean) / null sd; robust z = (five - null median) / (1.4826 MAD).
Empirical p (one-sided in the direction of the five's effect) = (1 + #null sets at least as extreme) / (N + 1);
for static EPE also a two-sided p around the null median. Per-scene paired tests: five vs base, and five vs the
per-scene median over the null sets (bootstrap 95% CI over scenes, 2000 resamples, and a two-sided sign test).
"""
import json, math, re, sys
import numpy as np

src, outp = sys.argv[1], sys.argv[2]
J = json.load(open(src)); rows = J["rows"]
METRICS = ["slope", "epe_static", "epe_moving", "sep_ratio"]
scenes = sorted({r["scene"] for r in rows if r["condition"] == "base"})
conds = list(dict.fromkeys(r["condition"] for r in rows))
V = {c: {m: np.full(len(scenes), np.nan) for m in METRICS} for c in conds}
si = {s: i for i, s in enumerate(scenes)}
for r in rows:
    if r["scene"] in si:
        for m in METRICS:
            V[r["condition"]][m][si[r["scene"]]] = r[m]
D = {c: {m: V[c][m] - V["base"][m] for m in METRICS} for c in conds if c != "base"}
rng = np.random.default_rng(0)


def sign_p(d):
    d = d[np.isfinite(d) & (d != 0)]; n = len(d); k = int((d < 0).sum())
    if n == 0:
        return float("nan")
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)


def boot(d):
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return float("nan"), float("nan")
    bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return tuple(map(float, np.percentile(bs, [2.5, 97.5])))


fam = {"mean": ("abl5", sorted([c for c in D if re.fullmatch(r"rand5_r\d+", c)], key=lambda c: int(c.split("_r")[1]))),
       "resample": ("res5", sorted([c for c in D if c.startswith("resrand_r")], key=lambda c: int(c.split("_r")[1]))),
       "amp": (next((c for c in D if c.startswith("amp5_a")), None),
               sorted([c for c in D if c.startswith("amprand_a")], key=lambda c: int(c.split("_r")[1])))}
out = {"n_scenes": len(scenes), "families": {}}
csv = ["family,metric,five_cond,n_null,five_mean,five_median,null_mean,null_sd,null_median,null_mad,z,robust_z,p_emp_one_sided,p_emp_two_sided,percentile_of_five,five_ci_lo,five_ci_hi,five_vs_base_sign_p,five_minus_nullmed_mean,five_minus_nullmed_lo,five_minus_nullmed_hi,five_vs_nullmed_sign_p,n_null_broken"]
for fname, (five, nulls) in fam.items():
    if five is None or five not in D or not nulls:
        continue
    fr = {"five": five, "null_sets": nulls, "metrics": {}}
    for m in METRICS:
        nm = np.array([np.nanmean(D[c][m]) for c in nulls]); nmed = np.array([np.nanmedian(D[c][m]) for c in nulls])
        fm = float(np.nanmean(D[five][m])); fmed = float(np.nanmedian(D[five][m]))
        mu, sd = float(np.mean(nm)), float(np.std(nm, ddof=1)) if len(nm) > 1 else float("nan")
        med = float(np.median(nm)); mad = float(np.median(np.abs(nm - med)))
        z = (fm - mu) / sd if sd > 0 else float("nan"); rz = (fm - med) / (1.4826 * mad) if mad > 0 else float("nan")
        if fm <= med:
            p1 = (1 + int((nm <= fm).sum())) / (len(nm) + 1)
        else:
            p1 = (1 + int((nm >= fm).sum())) / (len(nm) + 1)
        p2 = (1 + int((np.abs(nm - med) >= abs(fm - med)).sum())) / (len(nm) + 1)
        pct = float((nm < fm).mean() * 100)
        lo, hi = boot(D[five][m])
        per_scene_null_med = np.nanmedian(np.stack([D[c][m] for c in nulls]), 0)
        dd = D[five][m] - per_scene_null_med
        dlo, dhi = boot(dd)
        broken = int((np.array([np.nanmean(D[c]["epe_static"]) for c in nulls]) > 10).sum())
        rec = dict(five_mean=fm, five_median=fmed, null_mean=mu, null_sd=sd, null_median=med, null_mad=mad, z=z, robust_z=rz,
                   p_emp_one_sided=p1, p_emp_two_sided=p2, percentile_of_five=pct, five_ci=[lo, hi],
                   five_vs_base_sign_p=sign_p(D[five][m]), five_minus_nullmed_mean=float(np.nanmean(dd)),
                   five_minus_nullmed_ci=[dlo, dhi], five_vs_nullmed_sign_p=sign_p(dd), null_set_means=nm.tolist(),
                   null_set_medians=nmed.tolist(), n_null=len(nm), n_null_broken_static_gt10px=broken)
        fr["metrics"][m] = rec
        csv.append(",".join(str(x) for x in [fname, m, five, len(nm), fm, fmed, mu, sd, med, mad, z, rz, p1, p2, pct, lo, hi,
                                              rec["five_vs_base_sign_p"], rec["five_minus_nullmed_mean"], dlo, dhi,
                                              rec["five_vs_nullmed_sign_p"], broken]))
    out["families"][fname] = fr
# singles (resample) and per-condition table
single = {}
for c in [c for c in D if c.startswith("res1_")] + [c for c in ["res5", "abl5"] if c in D] + [c for c in D if c.startswith("amp5_")]:
    single[c] = {m: dict(mean=float(np.nanmean(D[c][m])), ci=boot(D[c][m]), sign_p=sign_p(D[c][m])) for m in METRICS}
out["conditions"] = single
out["base"] = {m: float(np.nanmean(V["base"][m])) for m in METRICS}
json.dump(out, open(outp + "_null.json", "w"), indent=1)
open(outp + "_null.csv", "w").write("\n".join(csv) + "\n")

# ------------------------------------------------------------------ figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
famlab = {"mean": "Mean ablation", "resample": "Resample ablation", "amp": "Amplification x" + str(J.get("amp", 1.5))}
mlab = {"slope": "change in slope", "epe_static": "change in static EPE (px)", "epe_moving": "change in moving EPE (px)"}
fams = [f for f in ["mean", "resample", "amp"] if f in out["families"]]
fig, axs = plt.subplots(3, len(fams), figsize=(3.4 * len(fams), 7.2), squeeze=False)
for j, f in enumerate(fams):
    fr = out["families"][f]
    for i, m in enumerate(["slope", "epe_static", "epe_moving"]):
        ax = axs[i, j]; rec = fr["metrics"][m]; nm = np.array(rec["null_set_means"]); fv = rec["five_mean"]
        logx = m != "slope" and max(np.nanmax(np.abs(nm)), abs(fv)) > 20
        tr = (lambda v: np.sign(v) * np.log10(1 + np.abs(v))) if logx else (lambda v: np.asarray(v, float))
        ax.hist(tr(nm), bins=24, color="0.65", edgecolor="white", label=f"{len(nm)} random matched sets")
        ax.axvline(tr(fv), color="#c0392b", lw=2, label="the five heads")
        ax.axvline(0, color="k", lw=0.6, ls=":")
        if logx:
            tk = [t for t in [-100, -10, -1, 0, 1, 10, 100] if tr(np.nanmin(nm)) - 0.3 <= tr(t) <= max(tr(np.nanmax(nm)), tr(fv)) + 0.3]
            ax.set_xticks([float(tr(t)) for t in tk]); ax.set_xticklabels([str(t) for t in tk])
        ax.set_xlabel(mlab[m] + (" (log scale)" if logx else "")); ax.set_ylabel("number of sets" if j == 0 else "")
        ax.set_title(f"{famlab[f]}\nz = {rec['z']:.1f}, robust z = {rec['robust_z']:.1f}, p = {rec['p_emp_one_sided']:.3f}", fontsize=8)
        if i == 0 and j == 0:
            ax.legend(frameon=False, fontsize=7)
fig.suptitle(f"Five supply heads vs matched random 5-head sets (v2, {len(scenes)} scenes; mean over scenes of the change vs base)", fontsize=9)
fig.tight_layout(); fig.savefig("fig_heads_null.png", dpi=150); plt.close(fig)
if single:
    names_ = [c for c in single if c.startswith("res")]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, m in zip(axs, ["slope", "epe_static", "epe_moving"]):
        mu = [single[c][m]["mean"] for c in names_]; lo = [single[c][m]["ci"][0] for c in names_]; hi = [single[c][m]["ci"][1] for c in names_]
        ax.bar(range(len(names_)), mu, color=["#c0392b" if c == "res5" else "#7f8c8d" for c in names_])
        ax.errorbar(range(len(names_)), mu, yerr=[np.array(mu) - np.array(lo), np.array(hi) - np.array(mu)], fmt="none", ecolor="k", lw=0.8)
        if "resample" in out["families"]:
            nm = np.array(out["families"]["resample"]["metrics"][m]["null_set_means"])
            ax.axhspan(np.percentile(nm, 25), np.percentile(nm, 75), color="0.85", zorder=0, label="random 5-head sets, interquartile range")
            ax.axhline(np.median(nm), color="0.5", lw=0.8, ls="--", zorder=0)
        ax.set_xticks(range(len(names_))); ax.set_xticklabels([c.replace("res1_", "") for c in names_], rotation=45, ha="right")
        ax.set_ylabel(mlab[m]); ax.axhline(0, color="k", lw=0.6)
    axs[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Resample ablation from a donor scene: five heads together and each alone (mean, 95% bootstrap CI over scenes)", fontsize=9)
    fig.tight_layout(); fig.savefig("fig_heads_resample.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ markdown
L = [f"# Causal specificity of the five supply heads (review W5), v2 out100\n",
     f"Scenes: {len(scenes)}. Heads (block:head): {', '.join(f'{b}:{h}' for b, h in J['five'])}. "
     f"Base: slope {out['base']['slope']:.3f}, static EPE {out['base']['epe_static']:.2f} px, moving EPE {out['base']['epe_moving']:.2f} px, "
     f"separation ratio {out['base']['sep_ratio']:.3f}.\n",
     "Each null set has 4 global-attention heads and 1 frame-attention head, drawn without the five (seeded; the first three are the round-1 sets). "
     "Statistic: mean over scenes of (condition minus base). z uses the null mean and sd; robust z uses the null median and MAD; "
     "p is the empirical one-sided p, (1 + number of null sets at least as extreme) / (N + 1).\n"]
for f in fams:
    fr = out["families"][f]
    L.append(f"\n## {famlab[f]} ({fr['five']} vs {len(fr['null_sets'])} random sets)\n")
    L.append("| metric | five heads, mean [95% CI] | null mean (sd) | null median | z | robust z | p (one-sided) | p (two-sided) | percentile | five vs per-scene null median [95% CI] (sign p) | null sets with static EPE > +10 px |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in METRICS:
        r = fr["metrics"][m]
        L.append(f"| {m} | {r['five_mean']:+.3f} [{r['five_ci'][0]:+.3f}, {r['five_ci'][1]:+.3f}] | {r['null_mean']:+.3f} ({r['null_sd']:.3f}) | {r['null_median']:+.3f} | "
                 f"{r['z']:.2f} | {r['robust_z']:.2f} | {r['p_emp_one_sided']:.3f} | {r['p_emp_two_sided']:.3f} | {r['percentile_of_five']:.0f} | "
                 f"{r['five_minus_nullmed_mean']:+.3f} [{r['five_minus_nullmed_ci'][0]:+.3f}, {r['five_minus_nullmed_ci'][1]:+.3f}] ({r['five_vs_nullmed_sign_p']:.2g}) | {r['n_null_broken_static_gt10px']} |")
if single:
    L.append("\n## Single conditions (change vs base, mean [95% CI], sign-test p)\n")
    L.append("| condition | slope | static EPE (px) | moving EPE (px) | sep ratio |")
    L.append("|---|---|---|---|---|")
    for c, r in single.items():
        L.append(f"| {c} | " + " | ".join(f"{r[m]['mean']:+.3f} [{r[m]['ci'][0]:+.3f}, {r[m]['ci'][1]:+.3f}] ({r[m]['sign_p']:.2g})" for m in METRICS) + " |")
# plain statements, generated from the numbers
L.append("\n## Plain statements (generated from the numbers above)\n")
for f in fams:
    r = out["families"][f]["metrics"]["epe_static"]
    inside = r["p_emp_two_sided"] > 0.05
    L.append(f"- {famlab[f]}: the five heads' change in static EPE ({r['five_mean']:+.2f} px) is "
             + ("within" if inside else "outside") + f" the null of random sets (two-sided empirical p = {r['p_emp_two_sided']:.3f}, percentile {r['percentile_of_five']:.0f}).")
    rs = out["families"][f]["metrics"]["slope"]
    L.append(f"- {famlab[f]}: change in slope {rs['five_mean']:+.3f} vs null median {rs['null_median']:+.3f} (z = {rs['z']:.1f}, empirical p = {rs['p_emp_one_sided']:.3f}).")
if "amp" in out["families"]:
    a = out["families"]["amp"]["metrics"]
    if a["slope"]["five_mean"] > 0 and a["epe_moving"]["five_mean"] > 0:
        L.append(f"- Amplification fixes magnitude, not accuracy: the slope rises by {a['slope']['five_mean']:+.3f} while the moving-object EPE also rises by {a['epe_moving']['five_mean']:+.2f} px.")
    elif a["slope"]["five_mean"] > 0:
        L.append(f"- Amplification raises the slope by {a['slope']['five_mean']:+.3f} and changes moving-object EPE by {a['epe_moving']['five_mean']:+.2f} px, so here it does not worsen accuracy.")
open(outp + "_summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
