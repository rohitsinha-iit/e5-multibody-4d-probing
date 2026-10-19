"""OpenD4RT readout check (time-boxed): official path vs adapter, index-convention sweep, own-camera sanity.
FIT scenes only for tuning (v2 even seeds). Writes one json."""
import os, sys, json, time, argparse
import numpy as np
E5 = "/data2/DATA/rohit_sinha/e5"; REPO = E5 + "/repo"; UP = E5 + "/big/models/src/Open-d4rt"
sys.path += [REPO + "/src", REPO + "/src/attr"]
import e5_models as em, e5_kubric_post as kp, e5_xm_eval as xe
ap = argparse.ArgumentParser(); ap.add_argument("--scenes", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--nq", type=int, default=3000); ap.add_argument("--ckpt_dir", default="")
args = ap.parse_args()
import torch, cv2
from PIL import Image
A = em.OpenD4RTAdapter(ckpt_dir=args.ckpt_dir or None)
sys.path.insert(0, UP)
import infer_track_3d as inf
d4t = A.d4t; h, w = A.h, A.w
rng = np.random.default_rng(0)


def load_video(paths):
    fr = [np.array(Image.open(p).convert("RGB")) for p in paths]
    return inf._resize_video(np.stack(fr), (h, w)), fr[0].shape[:2]


def to_t(vid):
    return torch.from_numpy(vid).to(A.device, torch.float32).permute(0, 3, 1, 2).unsqueeze(0) / 255.0


def run(video, mem, uvn, ts, tt, tc):
    n = len(uvn)
    q = A._q(uvn[:, 0], uvn[:, 1], np.broadcast_to(ts, (n,)), np.broadcast_to(tt, (n,)), np.broadcast_to(tc, (n,)))
    return A._run(video, torch.ones(1, 1, device=A.device), mem, q)


def proj(X, K):
    return np.stack([X[:, 0] / X[:, 2] * K[0, 0] + K[0, 2], X[:, 1] / X[:, 2] * K[1, 1] + K[1, 2]], -1)


def med(e, m):
    return float(np.nanmedian(e[m])) if m.any() else float("nan")


def rot_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


rows = []
for sc in args.scenes.split(","):
    d = os.path.join(E5, "big/data/v2/out", sc)
    meta, gt, _ = kp.load_scene(d)
    F = len(meta["frames"]); assert F == 24, F
    paths = [os.path.join(d, f"rgba_{f:05d}.png") for f in range(24)]
    vid, (H, W) = load_video(paths)
    g = em.kubric_cv_geometry(meta, gt, list(range(24)))
    K = g["K"]; w2c = g["w2c"]; c2w = g["c2w"]
    st0 = xe.static_mask(meta, gt, 0)
    row = dict(scene=sc)
    # ---- correspondences frame 0 -> t for several t; fixed query subsets drawn on the frame-0 -> 23 static set
    T = 23
    c = kp.correspondences(meta, gt, 0, T)
    x1, x2, grp = c["x1"], c["x2"], c["group"]
    s = (grp == 0) & st0[x1[:, 1].astype(int), x1[:, 0].astype(int)]
    mv = grp > 0
    si = rng.choice(np.flatnonzero(s), min(args.nq, s.sum()), replace=False)
    mi = rng.choice(np.flatnonzero(mv), min(args.nq // 3, mv.sum()), replace=False) if mv.any() else np.array([], int)
    idx = np.concatenate([si, mi]); isS = np.r_[np.ones(len(si), bool), np.zeros(len(mi), bool)]
    q1, q2 = x1[idx], x2[idx]
    uvn = (q1 / np.array([W - 1, H - 1])).astype(np.float32)          # repo convention u/(W-1) on native px
    depth = gt["depth"][..., 0]
    ray = np.sqrt(((q1[:, 0] - W / 2) / K[0, 0]) ** 2 + ((q1[:, 1] - H / 2) / K[1, 1]) ** 2 + 1)
    z0 = depth[0][q1[:, 1].astype(int), q1[:, 0].astype(int)] / ray
    Km = K.copy(); Km[0] *= (w - 1) / (W - 1); Km[1] *= (h - 1) / (H - 1)
    nat = lambda p: p * np.array([(W - 1) / (w - 1), (H - 1) / (h - 1)])

    def readouts(video, mem, tmap=lambda t: t, uvq=uvn, tag="", smap=None):
        """r1, r2 (GT K), r4 (GT cams) and r4own (own cameras) for frame 0 -> T."""
        o = {}; smap = smap or tmap
        loc = run(video, mem, uvq, smap(0), tmap(T), tmap(T))
        ref = run(video, mem, uvq, smap(0), tmap(T), tmap(0))
        ref0 = run(video, mem, uvq, smap(0), tmap(0), tmap(0))
        e1 = np.linalg.norm(loc["uv_2d"] * np.array([W - 1, H - 1]) - q2, axis=1)
        e2 = np.linalg.norm(proj(loc["xyz_3d"], K) - q2, axis=1)
        scl = float(np.median(z0[isS] / ref0["xyz_3d"][isS, 2]))
        Xc = (sc_ := scl * ref["xyz_3d"]) @ w2c[T][:3, :3].T + w2c[T][:3, 3]
        e4 = np.linalg.norm(proj(Xc, K) - q2, axis=1)
        # own cameras: D4RT-style pose of frame T from a 32x32 grid of frame-T pixels (t_src = t_tgt = T), t_cam = T vs 0
        gg = np.stack(np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32)), -1).reshape(-1, 2).astype(np.float32)
        pa = run(video, mem, gg, smap(T), tmap(T), tmap(T)); pb = run(video, mem, gg, smap(T), tmap(T), tmap(0))
        Tg, _ = em._robust_rigid(pa["xyz_3d"].astype(np.float64), pb["xyz_3d"].astype(np.float64))   # cam_T -> cam_0
        Xown = (ref["xyz_3d"] - Tg[:3, 3]) @ Tg[:3, :3]                     # world(cam0) -> cam_T, own pose
        e4o = np.linalg.norm(proj(Xown, K) - q2, axis=1)
        # self-consistency on the queried points themselves: rigid fit xyz(0,T,T) -> xyz(0,T,0) on static queries
        Ts, _ = em._robust_rigid(loc["xyz_3d"][isS].astype(np.float64), ref["xyz_3d"][isS].astype(np.float64))
        # GT relative pose cam_T -> cam_0, in model scale (translation scaled by 1/scl)
        Rg = c2w[T][:3, :3]; tg = c2w[T][:3, 3] / scl
        o.update({f"r1{tag}": med(e1, isS), f"r2gtK{tag}": med(e2, isS), f"r4{tag}": med(e4, isS), f"r4own{tag}": med(e4o, isS),
                  f"r1_mov{tag}": med(e1, ~isS), f"r4_mov{tag}": med(e4, ~isS), f"r4own_mov{tag}": med(e4o, ~isS),
                  f"pose_rot_err_deg_grid{tag}": rot_deg(Tg[:3, :3], Rg), f"pose_rot_err_deg_selfq{tag}": rot_deg(Ts[:3, :3], Rg),
                  f"pose_trans_err_rel_grid{tag}": float(np.linalg.norm(Tg[:3, 3] - tg) / max(np.linalg.norm(tg), 1e-9)),
                  f"gt_rot_deg{tag}": rot_deg(np.eye(3), Rg), f"gt_trans_model{tag}": float(np.linalg.norm(tg)),
                  f"r2_vs_r4own_3d_rel{tag}": float(np.median(np.linalg.norm(loc["xyz_3d"][isS] - Xown[isS], axis=1) / np.abs(Xown[isS, 2])))})
        return o

    with torch.no_grad():
        video = to_t(vid); mem = d4t._encode_model_memory(model=A.model, video_b=video, aspect_b=torch.ones(1, 1, device=A.device))
        row.update(readouts(video, mem))
        # (a) official repo path on the same queries (24 targets, t_src 0, aspect from native size)
        t0 = time.time()
        off = inf._infer_tracks(model=A.model, video_model_rgb=vid, native_aspect_ratio=W / H, query_uv_norm=uvn,
                                query_chunk_size=16384)
        uv_off = off["tracks_uv_norm"][:, T] * np.array([W - 1, H - 1])
        e_off = np.linalg.norm(uv_off - q2, axis=1)
        row.update(r1_official=med(e_off, isS), r2gtK_official=med(np.linalg.norm(proj(off["tracks_xyz_local"][:, T], K) - q2, axis=1), isS),
                   official_s=time.time() - t0, official_clip_frames=int(off["clip_frames"]))
        loc = run(video, mem, uvn, 0, T, T)
        row["official_vs_adapter_uv_maxabs_px"] = float(np.nanmax(np.abs(uv_off - loc["uv_2d"] * np.array([W - 1, H - 1]))))
        # adapter dense-grid path, as the gate uses it (targets 0 and T), sampled at the same native queries
        P = A.predict(paths, targets=[0, T], gt=dict(K=Km, w2c=w2c, depth0_z=cv2.resize(g["depth_z"][0], (w, h), interpolation=cv2.INTER_NEAREST)))
        import gate_newmodels as gn
        xp, _ = gn.sample(P["x_r1"][T], P["geom"][0], q1[:, 0], q1[:, 1])
        row["r1_adapter_dense"] = med(np.linalg.norm(nat(xp) - q2, axis=1), isS)
        xp, _ = gn.sample(P["x_r4"][T], P["geom"][0], q1[:, 0], q1[:, 1])
        row["r4_adapter_dense"] = med(np.linalg.norm(nat(xp) - q2, axis=1), isS)
        # (b) index / input convention sweep
        row.update(readouts(video, mem, tmap=lambda t: t + 1, smap=lambda t: t, tag="_onebased"))   # t_src indexes the video tensor, so it stays 0-based
        row.update(readouts(video, mem, uvq=((q1 + 0.5) / np.array([W, H])).astype(np.float32), tag="_uvhalfpix"))
        row.update(readouts(video, mem, tmap=lambda t: 23 - t, smap=lambda t: t, tag="_timereversed"))
        # pad to 32 by repeating the last frame (training convention for 24-frame Kubric scenes)
        vid32 = np.concatenate([vid, np.repeat(vid[-1:], 8, 0)]); v32 = to_t(vid32)
        m32 = d4t._encode_model_memory(model=A.model, video_b=v32, aspect_b=torch.ones(1, 1, device=A.device))
        row.update(readouts(v32, m32, tag="_pad32"))
        # resample to 32 frames (nearest), time indices mapped onto the 32-frame axis
        sel = np.round(np.arange(32) * 23 / 31).astype(int); vr = to_t(vid[sel])
        mr = d4t._encode_model_memory(model=A.model, video_b=vr, aspect_b=torch.ones(1, 1, device=A.device))
        row.update(readouts(vr, mr, tmap=lambda t: int(round(t * 31 / 23)), tag="_resample32"))
        # stride-2 subclip (12 frames 0,2,..,22 + 23 -> 12 even frames): 0..22 even plus 23 replaces 22
        sub = list(range(0, 22, 2)) + [23]; vs = to_t(vid[sub])
        ms = d4t._encode_model_memory(model=A.model, video_b=vs, aspect_b=torch.ones(1, 1, device=A.device))
        row.update(readouts(vs, ms, tmap=lambda t: sub.index(t), tag="_sub12"))
        # does the uv head depend on t_cam? (uv at t_tgt=T with t_cam = 0)
        alt = run(video, mem, uvn, 0, T, 0)
        row["r1_tcam0"] = med(np.linalg.norm(alt["uv_2d"] * np.array([W - 1, H - 1]) - q2, axis=1), isS)
    # ---- error vs time offset (base convention): r1, r2, r4 at t in 4..23
    curve = {}
    with torch.no_grad():
        for t in [2, 4, 8, 12, 16, 20, 23]:
            c = kp.correspondences(meta, gt, 0, t); a1, a2, gr = c["x1"], c["x2"], c["group"]
            ss = (gr == 0) & st0[a1[:, 1].astype(int), a1[:, 0].astype(int)]
            ii = rng.choice(np.flatnonzero(ss), min(1500, ss.sum()), replace=False)
            u_ = (a1[ii] / np.array([W - 1, H - 1])).astype(np.float32)
            lo = run(video, mem, u_, 0, t, t); rf = run(video, mem, u_, 0, t, 0); r0 = run(video, mem, u_, 0, 0, 0)
            ry = np.sqrt(((a1[ii, 0] - W / 2) / K[0, 0]) ** 2 + ((a1[ii, 1] - H / 2) / K[1, 1]) ** 2 + 1)
            zz = depth[0][a1[ii, 1].astype(int), a1[ii, 0].astype(int)] / ry
            scl = float(np.median(zz / r0["xyz_3d"][:, 2]))
            Xc = scl * rf["xyz_3d"] @ w2c[t][:3, :3].T + w2c[t][:3, 3]
            gtflow = float(np.median(np.linalg.norm(a2[ii] - a1[ii], axis=1)))
            curve[t] = dict(r1=float(np.median(np.linalg.norm(lo["uv_2d"] * np.array([W - 1, H - 1]) - a2[ii], axis=1))),
                            r2gtK=float(np.median(np.linalg.norm(proj(lo["xyz_3d"], K) - a2[ii], axis=1))),
                            r4=float(np.median(np.linalg.norm(proj(Xc, K) - a2[ii], axis=1))), gt_flow_px=gtflow,
                            zero_motion=float(np.median(np.linalg.norm(a1[ii] - a2[ii], axis=1))))
    row["curve"] = curve
    rows.append(row); print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "curve"}), flush=True)
    json.dump(dict(rows=rows, nq=args.nq), open(args.out, "w"), indent=1)
print("DONE")
