"""Item 8 summary: per (dataset, pass, views, resize) statistics over scenes.
Usage: python e5_realdata_summary.py out_prefix file1.csv [file2.csv ...]
Writes <out_prefix>.md and <out_prefix>.csv: n, mean, median, 95% bootstrap CI of the mean (2000 resamples,
seed 0), fraction of scenes < 1 and its two-sided sign-test p against 0.5, for slope, slope_inst_median and
sep_ratio; median epe_static / epe_moving."""
import csv, math, sys
import numpy as np


def sign_p(k, n):
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def stats(x, rng):
    x = np.array([float(v) for v in x if v not in ("", None)], float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0)
    bs = [rng.choice(x, len(x)).mean() for _ in range(2000)] if len(x) > 2 else [np.nan, np.nan]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    k = int((x < 1).sum())
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)), lo=float(lo), hi=float(hi),
                frac_lt1=k / len(x), p_sign=sign_p(k, len(x)))


out = sys.argv[1]; rows = []
for f in sys.argv[2:]:
    rows += list(csv.DictReader(open(f)))
groups = {}
for r in rows:
    groups.setdefault((r["dataset"], r["pass_"], r["n_views"], r["offsets"], r["resize"], r["model"]), []).append(r)
rng = np.random.default_rng(0); recs = []
for key, rs in sorted(groups.items()):
    rec = dict(dataset=key[0], pass_=key[1], n_views=key[2], offsets=key[3], resize=key[4], model=key[5], n_scenes=len(rs))
    for m in ("slope", "slope_inst_median", "sep_ratio"):
        for k, v in stats([r[m] for r in rs], rng).items():
            rec[f"{m}_{k}"] = v
    for m in ("epe_static", "epe_moving", "frac_moving_in_view"):
        x = np.array([float(r[m]) for r in rs if r[m] not in ("", None)], float); x = x[np.isfinite(x)]
        rec[f"{m}_median"] = float(np.median(x)) if len(x) else float("nan")
    recs.append(rec)
keys = list(dict.fromkeys(k for r in recs for k in r))
with open(out + ".csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, keys); w.writeheader()
    for r in recs:
        w.writerow({k: (f"{r[k]:.5g}" if isinstance(r.get(k), float) else r.get(k)) for k in keys})


def cell(r, m):
    if not r.get(f"{m}_n"):
        return "n/a"
    return (f"{r[f'{m}_mean']:.3f} [{r[f'{m}_lo']:.3f}, {r[f'{m}_hi']:.3f}], med {r[f'{m}_median']:.3f}, "
            f"<1: {r[f'{m}_frac_lt1']:.2f} of {r[f'{m}_n']} (p={r[f'{m}_p_sign']:.2g})")


with open(out + ".md", "w") as fh:
    fh.write("# Item 8: real-data output-level metrics\n\n")
    fh.write("slope = fraction of GT object motion recovered (1 = full motion, 0 = treated as static); "
             "sep_ratio = predicted / GT multibody separation. Cells: mean [95% bootstrap CI], median, "
             "fraction of scenes below 1 of n scenes with a defined value (two-sided sign test vs 0.5).\n\n")
    fh.write("| dataset | pass | views (offsets) | resize | scenes | slope | slope (per-instance median) | sep_ratio | "
             "EPE static px | EPE moving px | moving px in view |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in recs:
        fh.write(f"| {r['dataset']} | {r['pass_']} | {r['n_views']} ({r['offsets']}) | {r['resize']} | {r['n_scenes']} | "
                 f"{cell(r, 'slope')} | {cell(r, 'slope_inst_median')} | {cell(r, 'sep_ratio')} | "
                 f"{r['epe_static_median']:.2f} | {r['epe_moving_median']:.2f} | {r['frac_moving_in_view_median']:.2f} |\n")
print(open(out + ".md").read())
