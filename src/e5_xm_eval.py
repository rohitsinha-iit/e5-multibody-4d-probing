"""Item 7 (cross-model): model-agnostic output metrics on E5 Kubric scenes (frame 0 -> last view = frame 23).

Every model runner writes <pred_dir>/<scene>.npz in one format (all image coordinates on the 518 x 518 model grid,
pixel-index convention, view 0 = frame 0, last view = frame 23):
  pts0, ptsL       (H,W,3)  pointmaps of view 0 and of the last view in the model's world frame
  ptsL_head etc.   optional alternative pointmaps (e.g. VGGT point head); evaluated as extra variants
  sfL              optional (H,W,3) scene flow of view-0 pixels to the last view (Any4D)
  pts3d_0, scene_flow, cam_pose, K  optional (Any4D) to rebuild dense 2D correspondences exactly as in e5_causal.py
  q_idx (N,)       indices into kp.correspondences(meta, gt, 0, last)["x1"] of the query set (see query_set)
  q_xy  (N,2)      the same query pixels on the 518 grid
  x2_q  (N,2)      optional predicted location of the query pixels in the last view (518 grid; tracks)
  x2_patch (T,2)   optional predicted location of the 37x37 view-0 patch centres in the last view (518 grid)

3D object-motion slope (defined for every model):
  GT: X(p) = back-projected frame-0 GT depth; X'(p) = X(p) moved by the object's world transform from frame 0 to 23
      (static points stay); dt = X'(p) - X(p).
  Prediction: a similarity transform (Umeyama, two trimming rounds keeping 80 %) maps the predicted world frame to the
      GT world frame; it is fit on static, non-dome-far points of view 0 and the last view.
      dp = s R (X_last(q) - X_0(p)), with q = GT correspondence of p in frame 23 and nearest 518-grid pixels.
  slope3d = sum(dp . dt) / sum(|dt|^2) over moving pixels with |dt| > 0.1 (scene units, metres in Kubric).
  Variants: slope3d_flow (dp = s R sfL(p)), slope3d_track (q replaced by the predicted 2D location of p),
            slope3d_head (alternative pointmap head).  epe3d_static = median |dp| on static pixels (should be 0),
            epe3d_moving = median |dp - dt| on moving pixels, align_rmse = trimmed static alignment residual.
2D metrics (only for models with scene flow or tracks), same definitions as e5_causal.py:
  slope2d = sum(d_pred . d_true) / sum(|d_true|^2) over moving pixels with |d_true| > 2 px, d = x - x_static;
  epe_static, epe_moving (median px, 512 grid); sep_ratio = sep(pred) / sep(GT) with 300 points per motion group
  (the same random sample as e5_causal.py: rng = default_rng(scene index)).
  Dense models are scored on all pixels ("full") and on the query set ("q"); track models on the query set only.
  The query set is the sep sample plus up to 1500 random moving and 500 random static correspondences.
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e5_kubric_post as kp  # noqa: E402
import e5_step3_analysis as s3  # noqa: E402

M518 = 518
THR3 = 0.1


def scene_dirs(data, limit=0):
    sc = sorted(d for d in glob.glob(os.path.join(data, "*")) if os.path.exists(os.path.join(d, "e5_meta.json")))
    return sc[:limit] if limit else sc


def to518(x512, W=512):
    s = M518 / W
    return np.clip(np.round((x512 + 0.5) * s - 0.5).astype(int), 0, M518 - 1)


def query_set(meta, gt, si):
    """Deterministic query set for scene index si (index in the sorted list of all scenes of the dataset)."""
    b = len(meta["frames"]) - 1
    c = kp.correspondences(meta, gt, 0, b)
    grp = c["group"]
    rs = np.random.default_rng(si); idx_sep = s3.sample_by_group(grp, 300, rs)   # identical to e5_causal.py
    r2 = np.random.default_rng(10000 + si)
    mv = np.where(grp > 0)[0]; st = np.where(grp == 0)[0]
    extra = [r2.choice(mv, min(1500, len(mv)), replace=False) if len(mv) else np.array([], int),
             r2.choice(st, min(500, len(st)), replace=False) if len(st) else np.array([], int)]
    q = np.unique(np.concatenate([idx_sep] + extra)).astype(int)
    u = to518(c["x1"][q, 0], meta["resolution"][0]); v = to518(c["x1"][q, 1], meta["resolution"][0])
    return q, np.c_[u, v].astype(np.float32), c, idx_sep


def patch_centres(G=37, P=14):
    r, q = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    return np.c_[(P * q + 6.5).ravel(), (P * r + 6.5).ravel()].astype(np.float32)   # (col,row), t = r*G + q


def true_disp3d(meta, gt, a, b):
    """Dense frame-a GT world points X and their position at frame b (object rigid motion), (H,W,3) each."""
    depth = gt["depth"][..., 0]; seg = gt["segmentation"][..., 0]
    X = kp.backproject(meta, depth[a], a); Xb = X.copy()
    fa, fb = str(meta["frames"][a]), str(meta["frames"][b])
    for sid, (g, k) in kp.seg_to_group(meta).items():
        msk = seg[a] == sid
        if k >= 0 and msk.any():
            Ma = np.array(meta["objects"][k]["matrix_world"][fa]); Mb = np.array(meta["objects"][k]["matrix_world"][fb])
            T = Mb @ np.linalg.inv(Ma)
            Xb[msk] = X[msk] @ T[:3, :3].T + T[:3, 3]
    return X, Xb


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0); A, B = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(B.T @ A / len(src))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; s = np.trace(np.diag(S) @ D) / (A ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def fit_sim3(src, dst, rounds=2, keep=0.8):
    ok = np.isfinite(src).all(1) & np.isfinite(dst).all(1); src, dst = src[ok], dst[ok]
    s, R, t = umeyama(src, dst)
    for _ in range(rounds):
        r = np.linalg.norm(src @ R.T * s + t - dst, axis=1); m = r <= np.quantile(r, keep)
        s, R, t = umeyama(src[m], dst[m])
    r = np.linalg.norm(src @ R.T * s + t - dst, axis=1)
    return s, R, t, float(np.sqrt(np.mean(np.sort(r)[: int(keep * len(r))] ** 2)))


def static_mask(meta, gt, f):
    """Static, finite, not-far pixels of frame f (dome/sky far points are excluded by a depth cap)."""
    seg = gt["segmentation"][f, ..., 0]; depth = gt["depth"][f, ..., 0]
    s2g = kp.seg_to_group(meta); grp = np.zeros(seg.shape, int); obj = np.zeros(seg.shape, bool)
    for sid, (g, k) in s2g.items():
        grp[seg == sid] = g; obj |= (seg == sid) & (k >= 0)
    cap = 2.0 * np.percentile(depth[obj], 95) if obj.any() else np.percentile(depth, 50)
    return (grp == 0) & np.isfinite(depth) & (depth < cap)


def slope(dp, dt, m, min_n=20):
    return float((dp[m] * dt[m]).sum() / (dt[m] ** 2).sum()) if m.sum() > min_n else float("nan")


def eval_scene(d, P, si):
    meta, gt, _ = kp.load_scene(d)
    W = meta["resolution"][0]; b = len(meta["frames"]) - 1; s = M518 / W
    q, q_xy, c, idx_sep = query_set(meta, gt, si)
    if "q_idx" in P:
        assert np.array_equal(P["q_idx"], q), "query set mismatch"
    x1, x2, grp = c["x1"], c["x2"], c["group"]
    row = dict(scene=os.path.basename(d), split=meta["split"], n_corr=int(len(x1)), n_query=int(len(q)))
    # ---------------- similarity alignment of the predicted world frame on static points
    X0, X0b = true_disp3d(meta, gt, 0, b)
    XL = kp.backproject(meta, gt["depth"][b, ..., 0], b)
    rs = np.random.default_rng(1)
    samp = []
    for f, Xg in ((0, X0), (b, XL)):
        m = static_mask(meta, gt, f); vv, uu = np.nonzero(m)
        k = rs.choice(len(vv), min(15000, len(vv)), replace=False); samp.append((vv[k], uu[k], Xg[vv[k], uu[k]]))

    def align(p0, pL):
        src = np.concatenate([Pm[to518(vv.astype(float), W), to518(uu.astype(float), W)] for Pm, (vv, uu, _) in zip((p0, pL), samp)])
        return fit_sim3(src.astype(np.float64), np.concatenate([g for _, _, g in samp]))

    sc, R, t, rmse = align(P["pts0"], P["ptsL"])
    row.update(align_scale=float(sc), align_rmse=rmse)
    # ---------------- 3D displacements at all GT correspondences
    xi = x1.astype(int); dt = (X0b - X0)[xi[:, 1], xi[:, 0]]
    u1, v1 = to518(x1[:, 0], W), to518(x1[:, 1], W); u2, v2 = to518(x2[:, 0], W), to518(x2[:, 1], W)
    big3 = (grp > 0) & (np.linalg.norm(dt, axis=1) > THR3); stat = grp == 0
    row["n_moving3d"] = int(big3.sum()); row["median_true_disp3d"] = float(np.median(np.linalg.norm(dt[big3], axis=1))) if big3.any() else float("nan")

    def score3d(tag, dp):
        row[f"slope3d{tag}"] = slope(dp, dt, big3)
        row[f"epe3d_static{tag}"] = float(np.median(np.linalg.norm(dp[stat], axis=1))) if stat.any() else float("nan")
        row[f"epe3d_moving{tag}"] = float(np.median(np.linalg.norm((dp - dt)[big3], axis=1))) if big3.any() else float("nan")

    dpw = P["ptsL"][v2, u2].astype(np.float64) - P["pts0"][v1, u1].astype(np.float64)
    score3d("", dpw @ R.T * sc)
    if "ptsL_head" in P and "pts0_head" in P:
        s_h, R_h, _, rm_h = align(P["pts0_head"], P["ptsL_head"]); row["align_rmse_head"] = rm_h
        dph = P["ptsL_head"][v2, u2].astype(np.float64) - P["pts0_head"][v1, u1].astype(np.float64)
        score3d("_head", dph @ R_h.T * s_h)
    if "sfL" in P:
        score3d("_flow", P["sfL"][v1, u1].astype(np.float64) @ R.T * sc)
    # ---------------- 2D, e5_causal.py definitions
    X = kp.backproject(meta, gt["depth"][0, ..., 0], 0); xs_dense, _ = kp.project(meta, X, b)
    xs = xs_dense[xi[:, 1], xi[:, 0]]
    x1p = (np.c_[u1, v1] + 0.5) / s - 0.5
    sep_gt, _ = s3.sep_and_noise(x1[idx_sep], x2[idx_sep], grp[idx_sep])
    row["sep_gt"] = float(sep_gt)

    def score2d(tag, xp, sel):
        err = np.linalg.norm(xp - x2[sel], axis=1); g = grp[sel]; mv = g > 0
        dt2 = x2[sel] - xs[sel]; dp2 = xp - xs[sel]; big = mv & (np.linalg.norm(dt2, axis=1) > 2)
        row[f"slope2d{tag}"] = slope(dp2, dt2, big)
        row[f"epe_static{tag}"] = float(np.median(err[~mv])) if (~mv).any() else float("nan")
        row[f"epe_moving{tag}"] = float(np.median(err[mv])) if mv.any() else float("nan")
        pos = {j: i for i, j in enumerate(sel)}
        if all(j in pos for j in idx_sep):
            ii = np.array([pos[j] for j in idx_sep])
            sp, _ = s3.sep_and_noise(x1p[idx_sep], xp[ii], grp[idx_sep])
            row[f"sep_pred{tag}"] = float(sp); row[f"sep_ratio{tag}"] = float(sp / sep_gt) if sep_gt > 1 else float("nan")

    if "pts3d_0" in P and "scene_flow" in P:      # dense (Any4D): exactly the e5_causal.py 'base' computation
        off = np.array(s3.pixel_offset(P)); TV = P["scene_flow"].shape[0] - 1
        xd = s3.pred_correspondences(P, TV, off)
        xp = (xd[v1, u1] + 0.5) / s - 0.5
        score2d("", xp, np.arange(len(x1)))
        score2d("_q", xp[q], q)
        xpq518 = xd[v1[q], u1[q]]
    elif "x2_q" in P:
        xpq518 = P["x2_q"].astype(np.float64)
        score2d("_q", (xpq518 + 0.5) / s - 0.5, q)
        row["track_self_offset"] = float(np.median(np.linalg.norm(P["x2_q0"] - P["q_xy"], axis=1))) if "x2_q0" in P else float("nan")
    else:
        xpq518 = None
    if xpq518 is not None:                          # 3D slope with the predicted (not GT) correspondence
        ut = np.clip(np.round(xpq518[:, 0]).astype(int), 0, M518 - 1); vt = np.clip(np.round(xpq518[:, 1]).astype(int), 0, M518 - 1)
        dpt = (P["ptsL"][vt, ut].astype(np.float64) - P["pts0"][v1[q], u1[q]].astype(np.float64)) @ R.T * sc
        m = big3[q]; row["slope3d_track"] = slope(dpt, dt[q], m)
        row["slope3d_q"] = slope(dpw[q] @ R.T * sc, dt[q], m)   # GT-correspondence slope on the same subset
    return row


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--pred", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = json.load(open(a.out + ".partial")) if os.path.exists(a.out + ".partial") else []
    done = {r["scene"] for r in rows}
    for si, d in enumerate(scene_dirs(a.data, a.limit)):
        n = os.path.basename(d); pz = os.path.join(a.pred, n + ".npz")
        if n in done or not os.path.exists(pz):
            continue
        try:
            r = eval_scene(d, dict(np.load(pz)), si)
        except Exception as e:
            import traceback; r = dict(scene=n, error=repr(e), tb=traceback.format_exc()[-1500:])
        r["model"] = a.model; rows.append(r)
        json.dump(rows, open(a.out + ".partial.tmp", "w")); os.replace(a.out + ".partial.tmp", a.out + ".partial")
        print(n, {k: round(v, 3) for k, v in r.items() if k.startswith(("slope", "sep_ratio")) and isinstance(v, float)}, flush=True)
    json.dump(dict(model=a.model, rows=rows), open(a.out, "w"), indent=1)
    keys = [k for k in rows[0] if isinstance(rows[0].get(k), float)] if rows else []
    with open(a.out[:-5] + "_summary.md" if a.out.endswith(".json") else a.out + "_summary.md", "w") as fh:
        fh.write(f"# {a.model}: output metrics, {len(rows)} scenes\n\n| metric | mean | median | n |\n|---|---|---|---|\n")
        for k in keys:
            v = np.array([r.get(k, np.nan) for r in rows], float); v = v[np.isfinite(v)]
            if len(v):
                fh.write(f"| {k} | {v.mean():.4f} | {np.median(v):.4f} | {len(v)} |\n")
    print("done", len(rows))


if __name__ == "__main__":
    main()
