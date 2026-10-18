"""E5 v4: output-level metrics of Any4D predictions for every target view (round-1 definitions).

Usage: python e5_v4_outmetrics.py <data_dir> <pred_dir> <out_json> [split_prefix]
Per scene and target view t (frame 0 = reference, 512 px GT frame):
  slope      sum(d_pred . d_true) / sum(|d_true|^2) over moving pixels with |d_true| > 2 px, d = x - x_static,
             x_static = where the point would land in frame t if it were static (same code as e5_e8_knockout.py)
  epe_static / epe_moving   median endpoint error (px)
  mask_iou   IoU of the Any4D motion mask (|scene flow| > 0.1 m) with GT moving pixels (same code as step 3)
Resumable: finished scenes are kept in <out_json>.partial.
"""
import glob, json, os, sys
import numpy as np
import e5_kubric_post as kp
import e5_step3_analysis as s3

data, preds, outp = sys.argv[1], sys.argv[2], sys.argv[3]
prefix = sys.argv[4] if len(sys.argv) > 4 else ""
rows = json.load(open(outp + ".partial"))["rows"] if os.path.exists(outp + ".partial") else []
done = {r["scene"] for r in rows}
for pz in sorted(glob.glob(os.path.join(preds, "*.npz"))):
    n = os.path.basename(pz)[:-4]
    if n in done or not n.startswith(prefix) or not os.path.exists(os.path.join(data, n, "e5_meta.json")):
        continue
    try:
        meta, gt, _ = kp.load_scene(os.path.join(data, n))
        P = dict(np.load(pz)); frames = [int(f) for f in P["frames"]]
        W = meta["resolution"][0]; Hm = P["pts3d_0"].shape[0]; s = Hm / W
        off = np.array(s3.pixel_offset(P))
        X = kp.backproject(meta, gt["depth"][0, ..., 0], 0)
        out = []
        for ti in range(1, len(frames)):
            b = frames[ti]
            c = kp.correspondences(meta, gt, 0, b)
            xs_dense, _ = kp.project(meta, X, b)
            x1 = c["x1"]; grp = c["group"]
            xs = xs_dense[x1[:, 1].astype(int), x1[:, 0].astype(int)]
            u = np.clip(np.round((x1[:, 0] + 0.5) * s - 0.5).astype(int), 0, Hm - 1)
            v = np.clip(np.round((x1[:, 1] + 0.5) * s - 0.5).astype(int), 0, Hm - 1)
            xp = s3.pred_correspondences(P, ti, off)[v, u]; xp = (xp + 0.5) / s - 0.5
            err = np.linalg.norm(xp - c["x2"], axis=1)
            mv = grp > 0
            dt = c["x2"] - xs; dp = xp - xs
            big = mv & (np.linalg.norm(dt, axis=1) > 2)
            sf = np.linalg.norm(P["scene_flow"][ti], axis=-1)
            gm = np.zeros((Hm, Hm), bool)
            yy, xx = np.nonzero(c["dense_group"] > 0)
            gm[np.clip(np.round((yy + 0.5) * s - 0.5).astype(int), 0, Hm - 1),
               np.clip(np.round((xx + 0.5) * s - 0.5).astype(int), 0, Hm - 1)] = True
            pm = sf > s3.FLOW_THR
            inter, union = (pm & gm).sum(), (pm | gm).sum()
            out.append(dict(target_frame=b, view_index=ti, n_big=int(big.sum()),
                            slope=float((dp[big] * dt[big]).sum() / (dt[big] ** 2).sum()) if big.sum() > 20 else None,
                            epe_static=float(np.median(err[~mv])), epe_moving=float(np.median(err[mv])) if mv.any() else None,
                            mask_iou=float(inter / union) if union else None,
                            d_true_med_px=float(np.median(np.linalg.norm(dt[big], axis=1))) if big.any() else None))
        rows.append(dict(scene=n, split=meta["split"], n_views=len(frames), frames=frames, targets=out))
    except Exception as e:
        import traceback
        rows.append(dict(scene=n, error=repr(e), tb=traceback.format_exc()[-800:]))
    json.dump(dict(rows=rows), open(outp + ".partial.tmp", "w")); os.replace(outp + ".partial.tmp", outp + ".partial")
    print(n, flush=True)
json.dump(dict(rows=rows), open(outp + ".tmp", "w")); os.replace(outp + ".tmp", outp)
print("done", len(rows))
