"""Summarise per-scene intervention rows: change vs the 'base' condition, per condition.
Usage: python e5_summarize.py rows.json out_prefix [metric1,metric2,...]
Writes <out_prefix>_summary.csv and <out_prefix>_summary.md (mean delta, 95% bootstrap CI, sign-test p, n)."""
import json, math, sys
import numpy as np
rows = json.load(open(sys.argv[1]))["rows"]; outp = sys.argv[2]
metrics = sys.argv[3].split(",") if len(sys.argv) > 3 else [k for k, v in rows[0].items() if isinstance(v, float)]
base = {r["scene"]: r for r in rows if r["condition"] == "base"}
conds = []
for r in rows:
    if r["condition"] not in conds:
        conds.append(r["condition"])
def sign_p(k, n):
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)
rng = np.random.default_rng(0); out = []
for c in conds:
    sub = [r for r in rows if r["condition"] == c and r["scene"] in base]
    rec = dict(condition=c, n=len(sub))
    for m in metrics:
        vals = np.array([r[m] for r in sub], float); b = np.array([base[r["scene"]][m] for r in sub], float)
        rec[f"{m}_base"] = float(np.nanmean(b)) if len(b) else float("nan")
        rec[m] = float(np.nanmean(vals)) if len(vals) else float("nan")
        d = vals - b; d = d[np.isfinite(d)]
        if c != "base" and len(d) > 2:
            bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            rec[f"d_{m}"] = float(d.mean()); rec[f"d_{m}_lo"], rec[f"d_{m}_hi"] = map(float, np.percentile(bs, [2.5, 97.5]))
            rec[f"d_{m}_p"] = sign_p(int((d < 0).sum()), int((d != 0).sum()))
    out.append(rec)
keys = list(dict.fromkeys(k for r in out for k in r))
with open(outp + "_summary.csv", "w") as fh:
    fh.write(",".join(keys) + "\n")
    for r in out:
        fh.write(",".join("" if r.get(k) is None else (f"{r[k]:.5g}" if isinstance(r.get(k), float) else str(r[k])) for k in keys) + "\n")
with open(outp + "_summary.md", "w") as fh:
    fh.write("| condition | n | " + " | ".join(f"d {m} [95% CI] (p)" for m in metrics) + " |\n")
    fh.write("|---|---|" + "---|" * len(metrics) + "\n")
    for r in out:
        cells = []
        for m in metrics:
            if f"d_{m}" in r:
                cells.append(f"{r[f'd_{m}']:+.3f} [{r[f'd_{m}_lo']:+.3f}, {r[f'd_{m}_hi']:+.3f}] (p={r[f'd_{m}_p']:.2g})")
            else:
                cells.append(f"{r.get(m, float('nan')):.3f}")
        fh.write(f"| {r['condition']} | {r['n']} | " + " | ".join(cells) + " |\n")
print(open(outp + "_summary.md").read())
