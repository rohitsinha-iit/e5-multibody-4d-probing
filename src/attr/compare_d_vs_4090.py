
"""Compare A6000 re-rendered D datasets with the 4090 per-scene records (scene_summary.json from e5_post_driver)."""
import json, sys, os, numpy as np
R = "/data2/DATA/rohit_sinha/e5/repo"; B = "/data2/DATA/rohit_sinha/e5/big"
ref = {"v2": R + "/results/queue/v3_scale_float/v2/scene_summary.json", "v3float": R + "/results/queue/v3_scale_float/v3float/scene_summary.json",
       "v4epi": R + "/results/queue/v4_scale_epi/scene_summary.json"}
out = {}
for ds, rp in ref.items():
    np_ = B + f"/data/{ds}/scene_summary.json"
    if not os.path.exists(np_):
        continue
    a = {r["scene"]: r for r in json.load(open(rp))}; b = {r["scene"]: r for r in json.load(open(np_)) if "scene" in r}
    common = sorted(set(a) & set(b)); rows = []
    for s in common:
        x, y = a[s], b[s]
        rows.append(dict(scene=s, d_median_flow=abs(x.get("median_flow_px", np.nan) - y.get("median_flow_px", np.nan)),
                         d_sep_px=abs(x.get("sep_px", np.nan) - y.get("sep_px", np.nan)) if x.get("sep_px") is not None and y.get("sep_px") is not None else None,
                         gt_motions_equal=x.get("gt_motions") == y.get("gt_motions"), n_groups_equal=x.get("n_groups_visible") == y.get("n_groups_visible"),
                         flow_epe_new=y.get("flow_epe_px")))
    dm = [r["d_median_flow"] for r in rows]
    out[ds] = dict(n_ref=len(a), n_new=len(b), n_common=len(common), max_abs_d_median_flow_px=float(np.nanmax(dm)) if dm else None,
                   n_median_flow_within_0p01px=int(sum(d < 0.01 for d in dm)), all_gt_motions_equal=all(r["gt_motions_equal"] for r in rows),
                   all_n_groups_equal=all(r["n_groups_equal"] for r in rows), rows=rows)
    print(ds, {k: v for k, v in out[ds].items() if k != "rows"})
os.makedirs(B + "/results/a6000_data", exist_ok=True)
json.dump(out, open(B + "/results/a6000_data/d_vs_4090.json", "w"), indent=1, default=float)
