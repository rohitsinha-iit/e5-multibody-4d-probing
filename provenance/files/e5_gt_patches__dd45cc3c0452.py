"""Per-patch GT for E5 Steps 4-6 (37x37 view-0 patch grid at 518 px; target = frame 23).

Saves <out>/<scene>_gt.npz:
  body (T,)      motion group of the patch (majority over its pixels; 0 = static world)
  inst (T,)      object instance (-1 = dome/background, k = object index)  -> semantic / instance control
  purity (T,)    fraction of the patch's pixels with the majority body AND majority instance
  vis (T,)       centre pixel has a visible GT correspondence in frame 23
  c518 (T,2)     patch centre in 518 px coordinates (pixel-index convention)
  x2_true (T,2)  GT location of the centre pixel in frame 23 (518 px)
  x2_static (T,2) location if the point were static (camera motion only) (518 px)
  x2_pred (T,2)  Any4D output correspondence (from pred_any4d_7v) (518 px)
"""
import glob, json, os, sys
import numpy as np
import e5_kubric_post as kp

data, preds, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
G, S518 = 37, 518
import e5_step3_analysis as s3

for d in sorted(glob.glob(os.path.join(data, "*"))):
    if not os.path.exists(os.path.join(d, "e5_meta.json")):
        continue
    name = os.path.basename(d)
    meta, gt, _ = kp.load_scene(d)
    W = meta["resolution"][0]; s = S518 / W
    b = len(meta["frames"]) - 1
    c = kp.correspondences(meta, gt, 0, b)
    seg0 = gt["segmentation"][0, ..., 0]
    s2g = kp.seg_to_group(meta)
    body_px = np.zeros_like(seg0, int); inst_px = -np.ones_like(seg0, int)
    for sid, (g, k) in s2g.items():
        body_px[seg0 == sid] = g; inst_px[seg0 == sid] = k
    # static hypothesis: back-project frame 0, keep world point fixed, project into frame b
    X = kp.backproject(meta, gt["depth"][0, ..., 0], 0)
    xs, _ = kp.project(meta, X, b)
    u, v = np.meshgrid(np.arange(W), np.arange(W))
    pr = s3.pred_correspondences(dict(np.load(os.path.join(preds, f"{name}.npz"))), 6, 0.0)  # 518 grid, view 6 = frame 23
    P = dict(np.load(os.path.join(preds, f"{name}.npz")))
    off = np.array(s3.pixel_offset(P)); pr = pr - off
    T = G * G
    body = np.zeros(T, int); inst = np.zeros(T, int); pur = np.zeros(T); vis = np.zeros(T, bool)
    c518 = np.zeros((T, 2)); x2t = np.full((T, 2), np.nan); x2s = np.full((T, 2), np.nan); x2p = np.full((T, 2), np.nan)
    for r in range(G):
        for q in range(G):
            t = r * G + q
            # pixels of this patch in the 512 grid
            m = ((u + 0.5) * s >= 14 * q) & ((u + 0.5) * s < 14 * (q + 1)) & ((v + 0.5) * s >= 14 * r) & ((v + 0.5) * s < 14 * (r + 1))
            bb = body_px[m]; ii = inst_px[m]
            vals, cnt = np.unique(np.c_[bb, ii], axis=0, return_counts=True)
            j = cnt.argmax(); body[t], inst[t] = vals[j]; pur[t] = cnt[j] / cnt.sum()
            cx, cy = 14 * q + 6.5, 14 * r + 6.5  # 518 pixel-index centre
            c518[t] = (cx, cy)
            ux = int(np.clip(np.round((cx + 0.5) / s - 0.5), 0, W - 1)); vy = int(np.clip(np.round((cy + 0.5) / s - 0.5), 0, W - 1))
            if c["dense_valid"][vy, ux]:
                vis[t] = True
                x2t[t] = (c["dense_x2"][vy, ux] + 0.5) * s - 0.5
            x2s[t] = (xs[vy, ux] + 0.5) * s - 0.5
            x2p[t] = pr[int(round(cy)), int(round(cx))]
    np.savez_compressed(os.path.join(out, f"{name}_gt.npz"), body=body, inst=inst, purity=pur, vis=vis, c518=c518,
                        x2_true=x2t, x2_static=x2s, x2_pred=x2p, split=meta["split"],
                        n_bodies=1 + len(meta["motions"]))
    print(name, "moving patches", int(((body > 0) & (pur > 0.8)).sum()), flush=True)
