"""Per-sample GT statistics (numpy only). Usage: python e5_realdata_gtstats.py GT_DIR OUT.csv"""
import csv, glob, json, os, sys
import numpy as np
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz"))):
    if f.endswith(".tmp.npz"):
        continue
    z = np.load(f); m = json.loads(str(z["meta"])); g = z["group"]
    inst = m.get("instances", [])
    rows.append(dict(name=m["name"], H=m["H"], W=m["W"], tau_px=m.get("tau_px"), n_inst=len(inst),
                     n_moving_inst=int(len(set(np.unique(g).tolist()) - {-1, 0})),
                     frac_valid=float((g >= 0).mean()), frac_moving=float((g > 0).mean()),
                     bg_d2_median=m.get("bg_d2_median", m.get("static_d2_median")),
                     bg_d2_p90=m.get("bg_d2_p90", m.get("static_d2_p90")),
                     ego_rot_deg=m.get("ego_rot_deg"), ego_trans=m.get("ego_trans_m", m.get("cam_trans")),
                     ransac_inlier_frac=m.get("ransac_inlier_frac"), convention=m.get("convention")))
with open(sys.argv[2], "w", newline="") as fh:
    w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
a = np.array([r["n_moving_inst"] for r in rows])
print("samples", len(rows), "with moving instances", int((a > 0).sum()), "median bg residual px",
      np.nanmedian([r["bg_d2_median"] for r in rows]))
