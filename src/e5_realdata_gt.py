"""Item 8: ground truth for real-data output-level evaluation (KITTI scene flow 2015, MPI Sintel).

Every sample is one reference frame t0 and one target frame t1 (KITTI: frames 10 -> 11; Sintel: t -> t+1).
Stored per sample (native resolution H x W) in <gt_root>/<dataset>/<name>.npz:
  x2_true   (H,W,2) float32  GT position in the target frame of each reference pixel (pixel centres, x right, y down)
  x2_static (H,W,2) float32  where the pixel would land if its 3D point were static (camera / ego motion only)
  group     (H,W)   int16    -1 = no GT or ignored, 0 = static, k > 0 = moving instance k
  label     (H,W)   int16    dense motion label (KITTI: moving obj_map instance id at every pixel, else 0; Sintel:
                             = group); used to exclude a band around motion boundaries in the evaluation
  valid     (H,W)   bool     x2_true and x2_static both defined
  meta      json string: K, frame image paths per pass and frame offset, ego motion, thresholds, per-instance stats
Definitions follow e5_causal.py: d = x - x_static, slope = sum(d_pred . d_true) / sum(|d_true|^2) over moving pixels
with |d_true| > 2 px.

KITTI (Menze and Geiger 2015). 3D at t0 from disp_occ_0 at pixel p; 3D at t1 from disp_occ_1 (stored at the same
reference pixel p) at the flowed location p + flow(p). Ego motion (R, t: camera-t0 coords -> camera-t1 coords of a
static point) is fitted on background pixels (obj_map == 0) by a robust (soft-L1) least-squares fit of the stereo
reprojection residual (du, dv in the left image, d disparity at t1), initialised by a disparity-weighted Kabsch fit
on a RANSAC inlier set. Object motion = residual after ego motion. An obj_map instance is MOVING if the median
|x2_true - x2_static| over its pixels exceeds TAU_KITTI = max(2 px, 3 x the 90th percentile of the same quantity
on background pixels). Rationale: 2 px is the slope eligibility floor of e5_causal; the background percentile
absorbs the scene's own disparity / calibration noise. Parked cars are static instances and join group 0.

Sintel (Butler et al. 2012). x2_true from GT flow, x2_static from GT depth and GT cameras (camdata_left).
Moving pixels: |x2_true - x2_static| > TAU_SINTEL_PX (1 px, the GT is synthetic and exact up to rasterisation;
the smoke test reports the static residual distribution). The moving mask is cleaned by a 3x3 opening and split
into instances by 8-connected components; components < MIN_INST_PX pixels and the invalid mask are ignored (-1).
Characters in Sintel are non-rigid, so an instance is a connected moving region, not a rigid body.
"""
import json, os, sys, glob
import numpy as np
import cv2

TAU_KITTI_MIN = 2.0
TAU_SINTEL_PX = 1.0
MIN_INST_PX = 400
TAG = 202021.25


def _save(path, **arrs):
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def project(K, X):
    x = X @ K.T
    return x[..., :2] / np.clip(x[..., 2:3], 1e-6, None)


def backproject(K, u, v, Z):
    return np.stack([(u - K[0, 2]) / K[0, 0] * Z, (v - K[1, 2]) / K[1, 1] * Z, Z], -1)


def rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def kabsch(A, B, w=None):
    w = np.ones(len(A)) if w is None else w
    w = w / w.sum(); ca = (w[:, None] * A).sum(0); cb = (w[:, None] * B).sum(0)
    H = ((A - ca) * w[:, None]).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


# ------------------------------------------------------------------ KITTI
def kitti_read_calib(path):
    P = {}
    for line in open(path):
        k, _, v = line.partition(":")
        if k.startswith("P_rect_0"):
            P[k] = np.array([float(x) for x in v.split()]).reshape(3, 4)
    P2, P3 = P["P_rect_02"], P["P_rect_03"]
    K = P2[:, :3].copy(); B = (P2[0, 3] - P3[0, 3]) / K[0, 0]
    return K, B


def kitti_flow(path):
    f = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float64)
    return np.stack([(f[..., 2] - 2 ** 15) / 64.0, (f[..., 1] - 2 ** 15) / 64.0], -1), f[..., 0] > 0


def kitti_disp(path):
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float64) / 256.0
    return d, d > 0


def fit_ego(K, B, X0, x1obs, d1obs, rng):
    """Robust ego motion from background correspondences. Returns R, t, residual stats."""
    from scipy.optimize import least_squares
    X1obs = backproject(K, x1obs[:, 0], x1obs[:, 1], K[0, 0] * B / d1obs)
    n = len(X0); best = None
    for _ in range(300):
        i = rng.choice(n, 3, replace=False)
        try:
            R, t = kabsch(X0[i], X1obs[i])
        except np.linalg.LinAlgError:
            continue
        Y = X0 @ R.T + t
        e = np.linalg.norm(project(K, Y) - x1obs, axis=1) + np.abs(K[0, 0] * B / np.clip(Y[:, 2], 1e-3, None) - d1obs)
        inl = e < 2.0
        if best is None or inl.sum() > best.sum():
            best = inl
    w = 1.0 / np.clip(X0[best, 2], 1, None) ** 2
    R, t = kabsch(X0[best], X1obs[best], w)

    def resid(p):
        Rr = rodrigues(p[:3]) @ R; Y = X0 @ Rr.T + (t + p[3:])
        uv = project(K, Y) - x1obs; dd = K[0, 0] * B / np.clip(Y[:, 2], 1e-3, None) - d1obs
        return np.concatenate([uv[:, 0], uv[:, 1], dd])
    sol = least_squares(resid, np.zeros(6), loss="soft_l1", f_scale=1.0)
    R = rodrigues(sol.x[:3]) @ R; t = t + sol.x[3:]
    return R, t, dict(ransac_inlier_frac=float(best.mean()), n_bg=int(n))


def kitti_sample(root, idx, rng):
    tr = os.path.join(root, "training"); s = f"{idx:06d}"
    K, B = kitti_read_calib(os.path.join(tr, "calib_cam_to_cam", f"{s}.txt"))
    d0, v0 = kitti_disp(os.path.join(tr, "disp_occ_0", f"{s}_10.png"))
    d1, v1 = kitti_disp(os.path.join(tr, "disp_occ_1", f"{s}_10.png"))
    fl, vf = kitti_flow(os.path.join(tr, "flow_occ", f"{s}_10.png"))
    obj = cv2.imread(os.path.join(tr, "obj_map", f"{s}_10.png"), cv2.IMREAD_UNCHANGED).astype(int)
    H, W = d0.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    x2_true = np.stack([u, v], -1) + fl
    Z0 = np.where(v0, K[0, 0] * B / np.clip(d0, 1e-6, None), np.nan)
    X0 = backproject(K, u, v, Z0)
    allv = v0 & v1 & vf
    bg = allv & (obj == 0)
    ii = np.flatnonzero(bg.ravel())
    if len(ii) > 30000:
        ii = rng.choice(ii, 30000, replace=False)
    R, t, st = fit_ego(K, B, X0.reshape(-1, 3)[ii], x2_true.reshape(-1, 2)[ii], d1.ravel()[ii], rng)
    x2_static = project(K, X0 @ R.T + t)
    valid = v0 & vf
    d = np.linalg.norm(x2_true - x2_static, axis=-1)
    bgd = d[bg]
    tau = max(TAU_KITTI_MIN, 3.0 * float(np.percentile(bgd, 90)))
    X1 = backproject(K, x2_true[..., 0], x2_true[..., 1], np.where(v1, K[0, 0] * B / np.clip(d1, 1e-6, None), np.nan))
    res3 = np.linalg.norm(X1 - (X0 @ R.T + t), axis=-1)
    group = np.where(valid, 0, -1).astype(np.int16)
    inst = []
    for k in sorted(set(np.unique(obj)) - {0}):
        m = valid & (obj == k)
        if m.sum() < 50:
            group[m] = -1; continue
        med2 = float(np.median(d[m])); m3 = m & v1
        med3 = float(np.nanmedian(res3[m3])) if m3.sum() > 20 else float("nan")
        mov = med2 > tau
        if mov:
            group[m] = k
        inst.append(dict(id=int(k), n_px=int(m.sum()), med_d2_px=med2, med_res3_m=med3, moving=bool(mov)))
    label = np.zeros((H, W), np.int16)
    for it in inst:
        if it["moving"]:
            label[obj == it["id"]] = it["id"]
    frames = {"rgb": {}}
    for off in range(-10, 11):
        f = 10 + off
        p = os.path.join(tr, "image_2", f"{s}_{f:02d}.png")
        if os.path.exists(p):
            frames["rgb"][str(off)] = p
    egoang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    meta = dict(dataset="kitti15", name=s, H=H, W=W, K0=K.tolist(), K1=K.tolist(), baseline=B, frames=frames,
                ego_R=R.tolist(), ego_t=t.tolist(), ego_rot_deg=egoang, ego_trans_m=float(np.linalg.norm(t)), tau_px=tau,
                bg_d2_median=float(np.median(bgd)), bg_d2_p90=float(np.percentile(bgd, 90)), instances=inst, **st)
    return dict(x2_true=x2_true.astype(np.float32), x2_static=x2_static.astype(np.float32), group=group, valid=valid,
                label=label), meta


# ------------------------------------------------------------------ Sintel
def _readbin(path, dtype, shape_from_header=True):
    with open(path, "rb") as fh:
        tag = np.fromfile(fh, np.float32, 1)[0]; assert abs(tag - TAG) < 1e-3, path
        w, h = np.fromfile(fh, np.int32, 2)
        return np.fromfile(fh, dtype, -1), int(w), int(h)


def sintel_flow(p):
    a, w, h = _readbin(p, np.float32); return a.reshape(h, w, 2).astype(np.float64)


def sintel_depth(p):
    a, w, h = _readbin(p, np.float32); return a.reshape(h, w).astype(np.float64)


def sintel_cam(p):
    with open(p, "rb") as fh:
        tag = np.fromfile(fh, np.float32, 1)[0]; assert abs(tag - TAG) < 1e-3, p
        M = np.fromfile(fh, np.float64, 9).reshape(3, 3); N = np.fromfile(fh, np.float64, 12).reshape(3, 4)
    return M, N


def _sintel_static(K0, N0, K1, N1, Zraw, u, v, depth_mode, ext):
    Z = Zraw
    if depth_mode == "ray":
        r = np.linalg.norm(np.stack([(u - K0[0, 2]) / K0[0, 0], (v - K0[1, 2]) / K0[1, 1], np.ones_like(u)], -1), axis=-1)
        Z = Zraw / r
    if ext == "c2w":   # alternative reading of the extrinsic: camera-to-world [R | t]
        N0 = np.c_[N0[:, :3].T, -N0[:, :3].T @ N0[:, 3]]; N1 = np.c_[N1[:, :3].T, -N1[:, :3].T @ N1[:, 3]]
    Xc0 = backproject(K0, u, v, Z)
    Xw = (Xc0 - N0[:, 3]) @ N0[:, :3]          # R^T (X - t), N = world-to-camera [R | t]
    Xc1 = Xw @ N1[:, :3].T + N1[:, 3]
    return project(K1, Xc1), Xc1, N0, N1


def sintel_sample(root, seq, t, depth_mode="auto"):
    """depth_mode 'auto': the depth convention (z vs ray distance) and extrinsic reading (world-to-camera, as in the
    Sintel SDK, vs its inverse) are chosen per sample as the combination with the smallest median
    |x2_true - x2_static| over all valid pixels (the correct one is ~0 on the static majority); the choice is
    recorded in meta and the smoke test checks that it is the documented one (z, w2c)."""
    tr = os.path.join(root, "training"); f = f"frame_{t:04d}"
    fl = sintel_flow(os.path.join(tr, "flow", seq, f + ".flo"))
    Z = sintel_depth(os.path.join(tr, "depth", seq, f + ".dpt"))
    K0, N0 = sintel_cam(os.path.join(tr, "camdata_left", seq, f + ".cam"))
    K1, N1 = sintel_cam(os.path.join(tr, "camdata_left", seq, f"frame_{t+1:04d}.cam"))
    H, W = Z.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    x2_true = np.stack([u, v], -1) + fl
    combos = [(depth_mode, "w2c")] if depth_mode != "auto" else [("z", "w2c"), ("ray", "w2c"), ("z", "c2w"), ("ray", "c2w")]
    scores = {}
    for dm, ex in combos:
        xs_, _, _, _ = _sintel_static(K0, N0, K1, N1, Z, u, v, dm, ex)
        dd = np.linalg.norm(x2_true - xs_, axis=-1)
        scores[f"{dm}_{ex}"] = float(np.nanmedian(dd))
    best = min(scores, key=scores.get); dm, ex = best.split("_")
    x2_static, Xc1, _, _ = _sintel_static(K0, N0, K1, N1, Z, u, v, dm, ex)
    depth_mode = best
    inv_p = os.path.join(tr, "invalid", seq, f + ".png")
    invalid = cv2.imread(inv_p, cv2.IMREAD_GRAYSCALE) > 0 if os.path.exists(inv_p) else np.zeros((H, W), bool)
    valid = ~invalid & np.isfinite(x2_static).all(-1) & (Xc1[..., 2] > 1e-3)
    d = np.linalg.norm(x2_true - x2_static, axis=-1)
    mov = valid & (d > TAU_SINTEL_PX)
    mov = cv2.morphologyEx(mov.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    n, lab = cv2.connectedComponents(mov.astype(np.uint8), connectivity=8)
    group = np.where(valid, 0, -1).astype(np.int16)
    group[valid & (d > TAU_SINTEL_PX) & ~mov] = -1   # isolated moving pixels removed by the opening: ignore
    inst = []; k_out = 0
    for k in range(1, n):
        m = lab == k
        if m.sum() < MIN_INST_PX:
            group[m] = -1; continue
        k_out += 1; group[m] = k_out
        inst.append(dict(id=k_out, n_px=int(m.sum()), med_d2_px=float(np.median(d[m]))))
    st = d[valid & (group == 0)]
    frames = {}
    for ps in ("final", "clean"):
        frames[ps] = {}
        for off in range(-6, 7):
            p = os.path.join(tr, ps, seq, f"frame_{t+off:04d}.png")
            if os.path.exists(p):
                frames[ps][str(off)] = p
    meta = dict(dataset="sintel", name=f"{seq}_{t:04d}", seq=seq, t=t, H=H, W=W, K0=K0.tolist(), K1=K1.tolist(),
                frames=frames, tau_px=TAU_SINTEL_PX, convention=depth_mode, convention_scores=scores, instances=inst,
                static_d2_median=float(np.median(st)) if st.size else float("nan"),
                static_d2_p90=float(np.percentile(st, 90)) if st.size else float("nan"),
                cam_trans=float(np.linalg.norm(N1[:, :3].T @ N1[:, 3] - N0[:, :3].T @ N0[:, 3])))
    return dict(x2_true=x2_true.astype(np.float32), x2_static=x2_static.astype(np.float32), group=group, valid=valid,
                label=group.copy()), meta


def sintel_list(root, stride=4, need_after=3):
    out = []
    for sd in sorted(glob.glob(os.path.join(root, "training", "flow", "*"))):
        seq = os.path.basename(sd); n = len(glob.glob(os.path.join(root, "training", "final", seq, "frame_*.png")))
        for t in range(1, n - need_after + 1, stride):
            out.append((seq, t))
    return out


def _work(job):
    ds, root, out, key = job
    name = f"{key:06d}" if ds == "kitti15" else f"{key[0]}_{key[1]:04d}"
    path = os.path.join(out, name + ".npz")
    if os.path.exists(path):
        return name, "skip"
    try:
        if ds == "kitti15":
            arr, meta = kitti_sample(root, key, np.random.default_rng(key))
        else:
            arr, meta = sintel_sample(root, key[0], key[1])
        _save(path, meta=np.array(json.dumps(meta)), **arr)
        return name, "ok"
    except Exception as e:  # noqa: BLE001
        return name, "error " + repr(e)


if __name__ == "__main__":
    import argparse
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["kitti15", "sintel"]); ap.add_argument("root"); ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    keys = list(range(200)) if a.dataset == "kitti15" else sintel_list(a.root, a.stride)
    if a.limit:
        keys = keys[:: max(1, len(keys) // a.limit)][: a.limit]
    with Pool(a.procs) as pool:
        res = pool.map(_work, [(a.dataset, a.root, a.out, k) for k in keys])
    bad = [r for r in res if r[1].startswith("error")]
    print(a.dataset, "samples", len(res), "new", sum(r[1] == "ok" for r in res), "errors", len(bad))
    for r in bad[:10]:
        print(*r)
