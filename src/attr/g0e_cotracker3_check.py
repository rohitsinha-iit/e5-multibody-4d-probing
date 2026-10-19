
"""Gate G0e (CoTracker3_Kubric contents) on shard 0000: frames readable; 3D from 2D tracks + per-track depth + intrinsics
reprojects into other frames within 0.5 px (static tracks, visible in both frames)."""
import numpy as np, os, json, sys, time
from PIL import Image
d = sys.argv[1]; out = sys.argv[2]
seq = os.path.basename(d.rstrip("/"))
D = np.load(os.path.join(d, f"{seq}.npy"), allow_pickle=True).item()
cam = D["camera"]; print("camera keys", {k: np.shape(v) for k, v in (cam.items() if isinstance(cam, dict) else [])})
print("keys", {k: (np.asarray(v).dtype, np.shape(v)) for k, v in D.items() if k not in ("camera", "metadata")}); print("metadata", str(D["metadata"])[:600])
R = np.load(os.path.join(d, f"{seq}_with_rank.npz")); Kk = R["shared_intrinsics"]; E = R["extrinsics"]
tr = np.asarray(D["coords"], np.float64); vis = np.asarray(D["visibility"]).astype(bool); dep = np.asarray(D["coords_depth"], np.float64)
print("coords", tr.shape, "vis", vis.shape, vis.mean(), "coords_depth", dep.shape, "K", Kk.tolist(), "E0", E[0].tolist())
t2 = np.load(os.path.join(d, f"{seq}_trajs_2d.npy")); print("trajs_2d == coords:", t2.shape, float(np.abs(t2 - tr.reshape(t2.shape) if t2.shape == tr.shape else 0).max()) if t2.shape == tr.shape else "shape differs")
frames = sorted(os.listdir(os.path.join(d, "frames"))); ok = 0; shp = None
for f in frames:
    a = np.array(Image.open(os.path.join(d, "frames", f))); ok += a.ndim == 3 and a.shape[2] >= 3; shp = a.shape
print("frames", len(frames), "readable", ok, shp)
H, W = shp[:2]
N, T = tr.shape[:2] if tr.shape[0] != 120 else tr.shape[1::-1]
if tr.shape[0] == 120: tr = tr.transpose(1, 0, 2); vis = vis.T if vis.shape[0] == 120 else vis; dep = dep.T if dep.shape[0] == 120 else dep
dep = dep.reshape(tr.shape[0], tr.shape[1])
Ki = np.linalg.inv(Kk)
def lift(t, ray):
    p = np.c_[tr[:, t], np.ones(len(tr))] @ Ki.T
    if ray: p = p / np.linalg.norm(p, axis=1, keepdims=True)
    return p * dep[:, t:t + 1]
best = None
for ray in (False, True):
    for inv in (False, True):
        M = [np.r_[E[t], [[0, 0, 0, 1]]] for t in range(T)]
        W2C = [np.linalg.inv(m) if inv else m for m in M]
        C2W = [np.linalg.inv(m) for m in W2C]
        X = [lift(t, ray) @ C2W[t][:3, :3].T + C2W[t][:3, 3] for t in range(T)]
        Xs = np.stack(X, 1)  # N,T,3
        spread = np.nanmax(np.linalg.norm(Xs - Xs[:, :1], axis=2) * np.where(vis, 1, np.nan), axis=1)
        static = spread < 1e-2 * np.nanmedian(np.linalg.norm(Xs[:, 0], axis=1))
        errs = []
        for t in range(1, T, 7):
            m = static & vis[:, 0] & vis[:, t]
            Pc = X[0][m] @ W2C[t][:3, :3].T + W2C[t][:3, 3]; p = Pc @ Kk.T; p = p[:, :2] / p[:, 2:3]
            errs.append(np.linalg.norm(p - tr[m, t], axis=1))
        e = np.concatenate(errs) if errs else np.array([np.inf])
        rt = []
        for t in range(0, T, 7):  # same-frame round trip (convention check only)
            Pc = lift(t, ray); p = Pc @ Kk.T; rt.append(np.abs(p[:, :2] / p[:, 2:3] - tr[:, t]).max())
        r = dict(depth_is_ray=ray, extrinsics_is_c2w=inv, n_static=int(static.sum()), frac_static=float(static.mean()),
                 cross_frame_err_px_median=float(np.median(e)), p90=float(np.percentile(e, 90)), n_pairs=int(len(e)), same_frame_roundtrip_max_px=float(max(rt)))
        print(r)
        if best is None or r["cross_frame_err_px_median"] < best["cross_frame_err_px_median"]:
            best = r
dm = []
for t in range(0, T, 10):  # coords_depth vs depth map at the track pixel
    z = np.load(os.path.join(d, "depths", f"{t:03d}.npy")).squeeze()
    m = vis[:, t]; u = np.clip(np.round(tr[m, t, 0]).astype(int), 0, W - 1); v = np.clip(np.round(tr[m, t, 1]).astype(int), 0, H - 1)
    dm.append(np.abs(z[v, u] - dep[m, t]) / np.maximum(dep[m, t], 1e-6))
dm = np.concatenate(dm)
ent = dict(gate="G0e", time=time.strftime("%Y-%m-%d %H:%M:%S"), host="aicoe-a6000", shard=seq, n_frames=len(frames), frames_readable=int(ok),
           frame_shape=list(shp), n_tracks=int(tr.shape[0]), best_convention=best,
           depthmap_vs_coords_depth_rel_err_median=float(np.median(dm)),
           criteria="RGB frames present and readable; static tracks: frame-0 lift (track + per-track depth + shared intrinsics + extrinsics) reprojects into frames t within 0.5 px (median)",
           status="pass" if ok == len(frames) == T and best["cross_frame_err_px_median"] <= 0.5 else "fail")
print(json.dumps(ent))
for path in (out, "/data2/DATA/rohit_sinha/e5/repo/results/attr_gates.json"):
    g = json.load(open(path)) if os.path.exists(path) else {}
    g["G0e"] = ent; json.dump(g, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)
