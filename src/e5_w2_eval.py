"""Round 2, review item 1 (W2): per-query scoring of any model on E5 Kubric scenes, plus the inputs for the
alpha/beta decomposition, the boundary-distance test and the dose-response analysis.

Input: a prediction dir in the e5_xm_eval.py format (<scene>.npz). Two kinds are understood, exactly as in
e5_xm_eval.eval_scene:
  dense Any4D format (pts3d_0, scene_flow, cam_pose, K): correspondences rebuilt as in e5_causal.py, read at the
      518-grid pixel of each query (confidence = conf0 at that pixel);
  track format (x2_q on the 518 grid): VGGT tracker head, CoTracker3, RAFT (confidence = conf_q where present).
The query set is e5_xm_eval.query_set(meta, gt, si) with si the index in the sorted scene list (with the same +/-20
retry as e5_xm_eval.py), so slope2d_q, sep_ratio_q, epe_static_q and epe_moving_q are the round-1 numbers.

Per scene (all on the query set, native 512 px):
  slope2d_q, epe_static_q, epe_moving_q, sep_pred_q, sep_ratio_q, sep_gt   (round-1 definitions)
  decomposition on moving queries with |x_true - x_static| > 2 px (the slope selection):
      x_pred - x1 = alpha * (x_true - x_static) + beta * (x_static - x1) + e   (least squares, both coordinates)
      A = x_true - x_static (object's own motion), B = x_static - x1 (camera-induced displacement).
      cos_AB = <A,B> / (|A||B|) over the stacked vectors and cond_AB (condition number of [A B]) say how well alpha and
      beta are separable in the scene (epipolar-aligned motion makes A parallel to B).
  beta_static: x_pred - x1 = beta_s * B on static queries with |B| > 2 px (does the model shrink camera motion too?)
  slope by distance to the object boundary in frame 0 (px, Euclidean distance transform of the object's own mask):
      bins [0,3), [3,6), [6,12), [12,24), [24,inf); slope_edge (d <= 4 px), slope_interior (d >= 14 px, beyond one
      Any4D patch)
Per moving object (one row per segmentation id with a moving group): slope, alpha, beta on its queries, speed
  (median |x_true - x_static| over its dense visible correspondences, px), camera-induced displacement (median
  |x_static - x1|), area (fraction of the frame-0 image), texture (mean gradient magnitude of the frame-0 grey image,
  0..255 units per px, inside the mask eroded by 3 px), occlusion (1 - visible-in-frame-23 / frame-0 pixels),
  edge fraction of its queries, mean model confidence; scene-level camera rotation angle (deg), camera translation,
  number of independent motions.
Outputs: <out>/<scene>.json (row + objects), <out>/<scene>_q.npz (per-query arrays), and with --combine a JSON with all
rows and objects. Resumable per scene; --procs parallel workers.
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e5_xm_eval as xe  # noqa: E402
import e5_kubric_post as kp  # noqa: E402
import e5_step3_analysis as s3  # noqa: E402

DBINS = [0, 3, 6, 12, 24, 1e9]
EDGE_PX, INTERIOR_PX = 4.0, 14.0


def fit_ab(A, B, y, min_n=20):
    """Least squares y = alpha A + beta B over 2D vectors. Returns alpha, beta, cos_AB, cond_AB, r2, n."""
    n = len(A)
    if n <= min_n:
        return dict(alpha=np.nan, beta=np.nan, cos_AB=np.nan, cond_AB=np.nan, r2_ab=np.nan, n_ab=int(n))
    X = np.c_[A.reshape(-1), B.reshape(-1)]; Y = y.reshape(-1)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    sv = np.linalg.svd(X, compute_uv=False)
    res = Y - X @ coef; r2 = 1 - (res ** 2).sum() / max((Y ** 2).sum(), 1e-12)
    cos = float((A * B).sum() / np.sqrt((A ** 2).sum() * (B ** 2).sum() + 1e-12))
    return dict(alpha=float(coef[0]), beta=float(coef[1]), cos_AB=cos, cond_AB=float(sv[0] / max(sv[-1], 1e-12)),
                r2_ab=float(r2), n_ab=int(n))


def pslope(dp, dt, m, min_n=20):
    return float((dp[m] * dt[m]).sum() / (dt[m] ** 2).sum()) if m.sum() > min_n else float("nan")


def cam_motion(meta):
    f0, fb = str(meta["frames"][0]), str(meta["frames"][-1])
    C0 = np.array(meta["camera_matrix_world"][f0]); Cb = np.array(meta["camera_matrix_world"][fb])
    R = C0[:3, :3].T @ Cb[:3, :3]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return ang, float(np.linalg.norm(Cb[:3, 3] - C0[:3, 3]))


def query_pred(P, meta, gt, si):
    """Query set (with the e5_xm_eval +/-20 retry) and the predicted native-px location of each query in the last frame."""
    s = xe.M518 / meta["resolution"][0]
    q, q_xy, c, idx_sep = xe.query_set(meta, gt, si)
    if "q_idx" in P and not np.array_equal(P["q_idx"], q):
        for alt in (si + 20, si - 20):
            q2, q_xy2, c2, i2 = xe.query_set(meta, gt, alt)
            if np.array_equal(P["q_idx"], q2):
                q, q_xy, c, idx_sep = q2, q_xy2, c2, i2; break
        assert np.array_equal(P["q_idx"], q), "query set mismatch"
    W = meta["resolution"][0]
    u1, v1 = xe.to518(c["x1"][q, 0], W), xe.to518(c["x1"][q, 1], W)
    conf = vis = None
    if "pts3d_0" in P and "scene_flow" in P:
        off = np.array(s3.pixel_offset(P)); TV = P["scene_flow"].shape[0] - 1
        xd = s3.pred_correspondences(P, TV, off)
        xp = (xd[v1, u1] + 0.5) / s - 0.5
        if "conf0" in P:
            conf = P["conf0"][v1, u1].astype(np.float64)
    elif "x2_q" in P:
        xp = (P["x2_q"].astype(np.float64) + 0.5) / s - 0.5
        if "conf_q" in P:
            conf = P["conf_q"].astype(np.float64)
        if "vis_q" in P:
            vis = P["vis_q"].astype(np.float64)
    else:
        raise ValueError("prediction has neither dense Any4D outputs nor x2_q tracks")
    return q, c, idx_sep, xp, conf, vis, np.c_[u1, v1]


def eval_scene(d, P, si):
    from scipy import ndimage
    meta, gt, rgb = kp.load_scene(d)
    W, H = meta["resolution"]; b = len(meta["frames"]) - 1; s = xe.M518 / W
    q, c, idx_sep, xp, conf, vis, uv518 = query_pred(P, meta, gt, si)
    x1, x2, grp, segc = c["x1"], c["x2"], c["group"], c["seg"]
    xi = x1.astype(int)
    X = kp.backproject(meta, gt["depth"][0, ..., 0], 0); xs_dense, _ = kp.project(meta, X, b)
    xs = xs_dense[xi[:, 1], xi[:, 0]]
    x1q, x2q, xsq, gq, sq = x1[q], x2[q], xs[q], grp[q], segc[q]
    fin = np.isfinite(xp).all(1)
    row = dict(scene=os.path.basename(d), split=meta["split"], n_query=int(len(q)), n_nonfinite=int((~fin).sum()))
    # ------------- round-1 metrics on the query set (e5_xm_eval.score2d with tag "_q")
    err = np.linalg.norm(xp - x2q, axis=1); mv = gq > 0
    dt2 = x2q - xsq; dp2 = xp - xsq; big = mv & (np.linalg.norm(dt2, axis=1) > 2)
    row["slope2d_q"] = xe.slope(dp2, dt2, big)
    row["epe_static_q"] = float(np.median(err[~mv])) if (~mv).any() else float("nan")
    row["epe_moving_q"] = float(np.median(err[mv])) if mv.any() else float("nan")
    x1p = (np.c_[xe.to518(x1[:, 0], W), xe.to518(x1[:, 1], W)] + 0.5) / s - 0.5
    sep_gt, _ = s3.sep_and_noise(x1[idx_sep], x2[idx_sep], grp[idx_sep]); row["sep_gt"] = float(sep_gt)
    pos = {j: i for i, j in enumerate(q)}
    if len(idx_sep) and all(j in pos for j in idx_sep) and fin[[pos[j] for j in idx_sep]].all():
        ii = np.array([pos[j] for j in idx_sep])
        sp, _ = s3.sep_and_noise(x1p[idx_sep], xp[ii], grp[idx_sep])
        row["sep_pred_q"] = float(sp); row["sep_ratio_q"] = float(sp / sep_gt) if sep_gt > 1 else float("nan")
    # ------------- decomposition
    A = x2q - xsq; Bc = xsq - x1q; y = xp - x1q
    m = big & fin
    row.update(fit_ab(A[m], Bc[m], y[m]))
    row["n_slope"] = int(m.sum())
    row["median_A_px"] = float(np.median(np.linalg.norm(A[m], axis=1))) if m.any() else float("nan")
    row["median_B_px"] = float(np.median(np.linalg.norm(Bc[m], axis=1))) if m.any() else float("nan")
    st = (~mv) & fin & (np.linalg.norm(Bc, axis=1) > 2)
    row["beta_static"] = pslope(y, Bc, st); row["n_static_b"] = int(st.sum())
    row["median_B_static_px"] = float(np.median(np.linalg.norm(Bc[st], axis=1))) if st.any() else float("nan")
    # ------------- distance to the object boundary (frame 0, own mask)
    seg0 = gt["segmentation"][0, ..., 0]; s2g = kp.seg_to_group(meta)
    dist = np.full(len(q), np.nan)
    edt = {}
    for sid, (g, k) in s2g.items():
        if k >= 0:
            msk = seg0 == sid
            if msk.any():
                edt[sid] = ndimage.distance_transform_edt(msk)
    for sid, D in edt.items():
        mm = sq == sid
        if mm.any():
            dist[mm] = D[xi[q][mm, 1], xi[q][mm, 0]]
    for lo, hi in zip(DBINS[:-1], DBINS[1:]):
        mb = m & (dist >= lo) & (dist < hi)
        row[f"slope_d{lo}_{int(min(hi, 999))}"] = pslope(dp2, dt2, mb); row[f"n_d{lo}_{int(min(hi, 999))}"] = int(mb.sum())
    me, mi = m & (dist <= EDGE_PX), m & (dist >= INTERIOR_PX)
    row["slope_edge"], row["n_edge"] = pslope(dp2, dt2, me), int(me.sum())
    row["slope_interior"], row["n_interior"] = pslope(dp2, dt2, mi), int(mi.sum())
    ab_e, ab_i = fit_ab(A[me], Bc[me], y[me]), fit_ab(A[mi], Bc[mi], y[mi])
    row.update(alpha_edge=ab_e["alpha"], beta_edge=ab_e["beta"], alpha_interior=ab_i["alpha"], beta_interior=ab_i["beta"])
    # ------------- scene-level factors
    ang, ctr = cam_motion(meta)
    row.update(cam_rot_deg=ang, cam_trans=ctr, n_motions=int(len(meta["motions"])),
               n_moving_objects=int(sum(1 for sid, (g, k) in s2g.items() if k >= 0 and g > 0)))
    if conf is not None:
        row["conf_moving_mean"] = float(np.nanmean(conf[m])) if m.any() else float("nan")
        row["conf_static_mean"] = float(np.nanmean(conf[~mv])) if (~mv).any() else float("nan")
    # ------------- per moving object
    grey = rgb[0].astype(np.float64).mean(-1); gy, gx = np.gradient(grey); gmag = np.hypot(gx, gy)
    objs = []
    for sid, (g, k) in s2g.items():
        if k < 0 or g <= 0 or sid not in edt:
            continue
        msk = seg0 == sid; npx = int(msk.sum())
        md = segc == sid; nvis = int(md.sum())
        er = ndimage.binary_erosion(msk, iterations=3)
        tex = float(gmag[er].mean()) if er.sum() >= 20 else float(gmag[msk].mean())
        oq = sq == sid; mo = m & oq
        o = dict(scene=row["scene"], split=row["split"], seg_id=int(sid), obj_index=int(k), group=int(g),
                 area=npx / float(W * H), n_px=npx, occlusion=1.0 - nvis / max(npx, 1), texture=tex,
                 speed_px=float(np.median(np.linalg.norm(x2[md] - xs[md], axis=1))) if nvis else float("nan"),
                 cam_disp_px=float(np.median(np.linalg.norm(xs[md] - x1[md], axis=1))) if nvis else float("nan"),
                 n_q=int(oq.sum()), n_slope=int(mo.sum()), slope=pslope(dp2, dt2, mo),
                 edge_frac=float(np.mean(dist[oq] <= EDGE_PX)) if oq.any() else float("nan"),
                 cam_rot_deg=ang, cam_trans=ctr, n_motions=row["n_motions"])
        ab = fit_ab(A[mo], Bc[mo], y[mo]); o.update(alpha=ab["alpha"], beta=ab["beta"], cos_AB=ab["cos_AB"])
        o["conf"] = float(np.nanmean(conf[mo])) if (conf is not None and mo.any()) else float("nan")
        o["vis"] = float(np.nanmean(vis[mo])) if (vis is not None and mo.any()) else float("nan")
        objs.append(o)
    arrays = dict(q_idx=q, x1=x1q.astype(np.float32), x2_true=x2q.astype(np.float32), x2_static=xsq.astype(np.float32),
                  x2_pred=xp.astype(np.float32), group=gq.astype(np.int16), seg=sq.astype(np.int16),
                  dist=dist.astype(np.float32))
    if conf is not None:
        arrays["conf"] = conf.astype(np.float32)
    if vis is not None:
        arrays["vis"] = vis.astype(np.float32)
    return row, objs, arrays


def work(args):
    si, d, pred, out, model = args
    n = os.path.basename(d); oj = os.path.join(out, n + ".json"); pz = os.path.join(pred, n + ".npz")
    if os.path.exists(oj) or not os.path.exists(pz):
        return n, "skip"
    try:
        row, objs, arr = eval_scene(d, dict(np.load(pz)), si)
        row["model"] = model
        tmp = os.path.join(out, n + "_q.tmp.npz"); np.savez_compressed(tmp, **arr); os.replace(tmp, os.path.join(out, n + "_q.npz"))
        res = dict(row=row, objects=objs)
    except Exception as e:
        import traceback
        res = dict(row=dict(scene=n, model=model, error=repr(e), tb=traceback.format_exc()[-1500:]), objects=[])
    json.dump(res, open(oj + ".tmp", "w"), default=float); os.replace(oj + ".tmp", oj)
    return n, res["row"].get("error", "ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--pred", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scenes", default=""); ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--combine", default="", help="write all rows + objects to this JSON")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    want = set(a.scenes.split(",")) if a.scenes else None
    jobs = [(si, d, a.pred, a.out, a.model) for si, d in enumerate(xe.scene_dirs(a.data, a.limit))
            if not want or os.path.basename(d) in want]
    if a.procs > 1:
        from multiprocessing import Pool
        with Pool(a.procs) as pool:
            res = list(pool.imap_unordered(work, jobs))
    else:
        res = [work(j) for j in jobs]
    bad = [r for r in res if r[1] not in ("ok", "skip")]
    print(a.model, "scenes", len(jobs), "new", sum(r[1] == "ok" for r in res), "errors", len(bad), bad[:3], flush=True)
    if a.combine:
        rows, objs = [], []
        for _, d, *_ in jobs:
            oj = os.path.join(a.out, os.path.basename(d) + ".json")
            if os.path.exists(oj):
                z = json.load(open(oj)); rows.append(z["row"]); objs += z["objects"]
        json.dump(dict(model=a.model, data=a.data, pred=a.pred, rows=rows, objects=objs), open(a.combine + ".tmp", "w"), indent=0)
        os.replace(a.combine + ".tmp", a.combine)
        v = np.array([r.get("slope2d_q", np.nan) for r in rows], float)
        print(a.model, "rows", len(rows), "objects", len(objs), "mean slope2d_q", round(float(np.nanmean(v)), 4) if len(v) else None)


if __name__ == "__main__":
    main()
