"""Item 10: round-1 output metrics plus geometry errors for one model's Kubric predictions.

  python e5_ft_eval_kubric.py --data <dataset>/out --pred <pred_dir> --out <rows.json> --src $SRC

pred_dir holds <scene>.npz from e5_any4d_infer.py (frames 0,4,8,12,16,20,23; view 6 = frame 23 is the target).
Per scene (si = index in the sorted scene list of --data, identical for every model):
  slope, epe_static, epe_moving, sep_gt, sep_pred, sep_ratio  (as e5_causal.py),  mask_iou (as e5_step3_analysis.py),
  depth_absrel (metric) / depth_absrel_med (median-scaled), rot_err_deg, tdir_err_deg, trans_relerr(_med).
Resumable through <out>.partial.
"""
import argparse, glob, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--pred", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--src", required=True); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, a.src)
import e5_kubric_post as kp  # noqa: E402
import e5_ft_lib as L  # noqa: E402

scenes = sorted(d for d in glob.glob(os.path.join(a.data, "*")) if os.path.exists(os.path.join(d, "e5_meta.json")))
part = a.out + ".partial"
rows = json.load(open(part)) if os.path.exists(part) else []
done = {r["scene"] for r in rows}
n_new = 0
for si, d in enumerate(scenes):
    name = os.path.basename(d); pz = os.path.join(a.pred, name + ".npz")
    if name in done or not os.path.exists(pz):
        continue
    if a.limit and n_new >= a.limit:
        break
    meta, gt, _ = kp.load_scene(d)
    P = dict(np.load(pz)); frames = [int(f) for f in P["frames"]]
    row = dict(scene=name, split=meta["split"], si=si)
    try:
        row.update(L.kubric_metrics(meta, gt, P, si, frames))
        row.update(L.geometry_errors(meta, gt, P, frames))
    except Exception as e:  # keep going; the summary counts errors
        row["error"] = repr(e)
    rows.append(row); n_new += 1
    json.dump(rows, open(part + ".tmp", "w")); os.replace(part + ".tmp", part)
json.dump(rows, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)
ok = [r for r in rows if "error" not in r]
print(f"{a.pred}: {len(ok)} scenes ok, {len(rows) - len(ok)} errors; mean slope "
      f"{np.nanmean([r['slope'] for r in ok]) if ok else float('nan'):.3f}")
