"""Item 7: per-model GT patch files for e5_internal_analysis.py. The GT fields (body, inst, purity, vis, c518, x2_true,
x2_static) are copied from the v2 files (they do not depend on the model); x2_pred is replaced by the model's own
prediction for the patch centres in the last view (x2_patch from the runner, 518 grid), or NaN when the model has
no 2D motion output (then the analysis reports no output gain and no output slope).
Usage: python e5_xm_gtpatch.py <v2_gt_patches> <pred_dir> <out_dir>"""
import glob, os, sys
import numpy as np
gdir, pdir, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
n_done = 0
for pz in sorted(glob.glob(os.path.join(pdir, "*.npz"))):
    n = os.path.basename(pz)[:-4]
    if n.endswith(".tmp"):
        continue
    g = dict(np.load(os.path.join(gdir, f"{n}_gt.npz"), allow_pickle=True))
    P = np.load(pz)
    g["x2_pred"] = P["x2_patch"].astype(float) if "x2_patch" in P else np.full_like(g["x2_pred"], np.nan)
    tmp = os.path.join(out, f"{n}_gt.tmp.npz"); np.savez_compressed(tmp, **g); os.replace(tmp, os.path.join(out, f"{n}_gt.npz"))
    n_done += 1
print("gt patches", n_done)
