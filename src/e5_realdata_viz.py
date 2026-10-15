"""Visual GT check: reference image, motion groups, |x2_true - x2_static| for N samples.
Usage: python e5_realdata_viz.py GT_DIR PASS OUT.png N"""
import glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
fs = [f for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz"))) if not f.endswith(".tmp.npz")]
n = int(sys.argv[4]); sel = []
for f in fs:
    z = np.load(f)
    if (z["group"] > 0).sum() > 2000:
        sel.append(f)
sel = sel[:: max(1, len(sel) // n)][:n]
fig, ax = plt.subplots(len(sel), 3, figsize=(15, 2.2 * len(sel)), squeeze=False)
for i, f in enumerate(sel):
    z = np.load(f); m = json.loads(str(z["meta"]))
    img = np.asarray(Image.open(m["frames"][sys.argv[2]]["0"]).convert("RGB"))
    d = np.linalg.norm(z["x2_true"] - z["x2_static"], axis=-1); d[~z["valid"]] = np.nan
    ax[i, 0].imshow(img); ax[i, 0].set_title(m["name"], fontsize=8)
    g = z["group"].astype(float); g[g < 0] = np.nan
    ax[i, 1].imshow(g, cmap="tab10", interpolation="nearest"); ax[i, 1].set_title("group (0 static, k moving)", fontsize=8)
    im = ax[i, 2].imshow(d, vmin=0, vmax=np.nanpercentile(d, 99)); ax[i, 2].set_title("|x2_true - x2_static| px", fontsize=8)
    fig.colorbar(im, ax=ax[i, 2], fraction=0.02)
    for a in ax[i]:
        a.axis("off")
fig.tight_layout(); fig.savefig(sys.argv[3], dpi=70)
print("wrote", sys.argv[3], len(sel))
