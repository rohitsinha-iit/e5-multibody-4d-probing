#!/usr/bin/env python3
"""U2 behaviour runs (unified plan U2, pre-registration core): model inference on the factorial Kubric variants, the
E5 real-data samples (KITTI 2015, Sintel) and the pose-set windows. One model per process (loaded once); work units
are (split, variant), (dataset, pass, config) or the pose windows. Every scene / sample / window is written as its own
npz and skipped when it exists; a unit gets a _DONE.json once all its items exist. Scoring is separate (u2_score.py).

Settings fixed by the pre-registration (results/attr_preregistration.json):
  Any4D, VGGT, 4RC, MoRe: views 0,4,8,12,16,20,23 (target 23); OpenD4RT: all 24 frames padded to 32 (pad_to_clip);
  TrackCraft3R: frames 0,2,...,20,23, stretch resize, no t0 re-registration, conditions gt (i), da3 (ii), identity (iii).
Per Kubric scene the npz holds, at the E5 query pixels q_pix of the base scene (identical in A, B, C, D):
  xp (N,2)  predicted native-px position in frame 23 (primary image readout); xp_<r> secondary readouts
  wd (N,3)  predicted world displacement frame 0 -> 23 in the predicted frame-0 camera (model units); wd_<r> secondary
  z0 (N,)   predicted frame-0 z-depth (predicted frame-0 camera); depth0_med median predicted frame-0 depth (dense)
  cams (V,4,4) predicted camera-to-world (OpenCV), views (V,) frame indices; budget arrays for Any4D and 4RC
Model-specific readouts:
  any4d     E5 route (e5_step3_analysis.pred_correspondences with the GT-free pixel_offset convention check), as in
            e5_xm_eval / e5_w2_eval; xp_nooff without the offset
  vggt      VGGT-1B with the facebook/VGGT_tracker_fixed track head (e20) on the 518 query grid (E5 t7); s_w by lifting
            the tracks with VGGT's own depth and cameras (LiftedTrackerAdapter)
  opend4rt  xp = r4 (r3 moved with GT cameras), xp_r1 uv head, xp_r2 xyz(t_cam=t) with predicted K, xp_r2gtK; wd = r3
  4rc       xp = track_t projected with camera t; wd = F_t in the predicted frame-0 camera
  tc3r_<c>  xp = P0(t) projected with the input camera t; wd = P0(t) - P0(0); cams = input cameras
  cotracker3, raft   2D tracks lifted with GT depth and GT cameras; RAFT xp = chained over the 7 views (E5 "chained
            RAFT"), xp_direct = direct flow 0 -> 23 (and wd, wd_direct)
  more      cameras only
"""
import argparse, glob, json, os, sys, time, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.append(HERE); sys.path.append(os.path.dirname(HERE))
import u2_common as u  # noqa: E402
import e5_models as em  # noqa: E402

BIG = u.E5_BIG


def _identity_geom(W, H):
    return dict(W=W, H=H, w=W, h=H, sx=1.0, sy=1.0, left=0.0, top=0.0, ow=W, oh=H)


# =========================================================================================== model wrappers
class Any4DM:
    tag = "any4d"; need_gb = 10

    def __init__(self):
        repo = os.path.join(BIG, "models/src/Any4D"); os.chdir(repo); sys.path.insert(0, os.path.join(repo, "scripts"))
        self.A = em.Any4D(checkpoint_path=os.path.join(BIG, "models/ckpt/any4d/any4d_4v_combined.pth"))
        import e5_step3_analysis as s3; self.s3 = s3

    def _pred(self, paths, mode="square"):
        self.A.resize_mode = mode
        P = self.A.predict(paths); P.pop("_views", None); return P

    def kubric(self, ds, seed, v, q):
        import torch
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        P = self._pred(u.frame_paths(ds, seed, v, u.VIEWS7)); rt = time.time() - t0
        TV = 6; g0 = P["geom"][0]; x, y = q[:, 0], q[:, 1]
        off = np.array(self.s3.pixel_offset(P))
        xd = self.s3.pred_correspondences(P, TV, off); xd0 = self.s3.pred_correspondences(P, TV, np.zeros(2))
        xm, ins = u.sample_nearest(xd, g0, x, y)
        T0 = P["cam_pose"][0]; Pc0 = (P["pts3d_0"] - T0[:3, 3]) @ T0[:3, :3]
        wdm = em._to_cam0(P["scene_flow"][TV], P)
        return dict(xp=u.to_native(P["geom"][TV], xm), xp_nooff=u.to_native(P["geom"][TV], u.sample_nearest(xd0, g0, x, y)[0]),
                    wd=u.sample_nearest(wdm, g0, x, y)[0], z0=u.sample_nearest(Pc0[..., 2], g0, x, y)[0],
                    depth0_med=float(np.median(Pc0[..., 2])), cams=P["cam_pose"], views=np.array(u.VIEWS7), inside=ins,
                    pixel_offset=off, b_P0=u.sample_nearest(P["pts3d_0"], g0, x, y)[0],
                    b_F=u.sample_nearest(P["scene_flow"][TV], g0, x, y)[0],
                    b_K=np.stack([u.K_native(P["K"][i], P["geom"][i]) for i in range(len(P["K"]))]),
                    runtime_s=rt, **u.gpu_stats())

    def real(self, paths, meta, cfg):
        mode = "fixed_mapping" if cfg.get("resize") == "wide" else "square"
        P = self._pred(paths, mode); off = np.array(self.s3.pixel_offset(P))
        return dict(field=self.s3.pred_correspondences(P, 1, off).astype(np.float32), geom=json.dumps(P["geom"][:2]))

    def pose(self, paths):
        return dict(cams=self._pred(paths)["cam_pose"])


class VGGTM:
    tag = "vggt"; need_gb = 16

    def __init__(self):
        import torch
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.load_fn import load_and_preprocess_images
        self.torch, self.pe2, self.lpi = torch, pose_encoding_to_extri_intri, load_and_preprocess_images
        hub = os.path.join(BIG, "models/hf/hub")
        ck = glob.glob(os.path.join(hub, "models--facebook--VGGT-1B/snapshots/*/model.pt"))[0]
        tk = glob.glob(os.path.join(hub, "models--facebook--VGGT_tracker_fixed/snapshots/*/model_tracker_fixed_e20.pt"))[0]
        m = VGGT(); sd = torch.load(ck, map_location="cpu", weights_only=False)
        sd = sd["model"] if isinstance(sd, dict) and isinstance(sd.get("model"), dict) else sd
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        r = m.load_state_dict(sd, strict=False)
        assert not [k for k in r.missing_keys if not k.startswith("track_head")], r.missing_keys[:5]
        tsd = torch.load(tk, map_location="cpu", weights_only=False); tsd = tsd.get("model", tsd)
        tsd = {k[7:] if k.startswith("module.") else k: v for k, v in tsd.items()}
        m.track_head.load_state_dict({k[len("track_head."):]: v for k, v in tsd.items() if k.startswith("track_head.")}, strict=True)
        self.m = m.to("cuda").eval(); self.dt = torch.bfloat16; self.ckpt = dict(vggt=ck, track=tk)

    def _load518(self, paths):
        from PIL import Image
        ims = [np.asarray(Image.open(p).convert("RGB").resize((518, 518), Image.BICUBIC)) for p in paths]
        return self.torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().div(255).to("cuda")

    def _run(self, img, queries=None, chunk=4096):
        torch = self.torch; m = self.m; img = img[None]
        with torch.no_grad():
            with torch.autocast("cuda", dtype=self.dt):
                toks, ps = m.aggregator(img)
            toks = [t.float() if t is not None else None for t in toks]
            with torch.autocast("cuda", enabled=False):
                pe = m.camera_head(toks)[-1]
                depth, _ = m.depth_head(toks, images=img, patch_start_idx=ps)
                tr = vis = None
                if queries is not None:
                    q = torch.as_tensor(np.asarray(queries, np.float32), device="cuda"); outs, viss = [], []
                    for j in range(0, len(q), chunk):
                        tl, vi, _ = m.track_head(toks, images=img, patch_start_idx=ps, query_points=q[None, j:j + chunk])
                        outs.append(tl[-1][0].cpu()); viss.append(vi[0].cpu())
                    tr = torch.cat(outs, 1).numpy(); vis = torch.cat(viss, 1).numpy()
        extr, intr = self.pe2(pe, img.shape[-2:])
        extr = extr[0].cpu().numpy(); intr = intr[0].cpu().numpy(); V = len(extr)
        c2w = np.tile(np.eye(4), (V, 1, 1))
        for i in range(V):
            c2w[i, :3, :3] = extr[i, :, :3].T; c2w[i, :3, 3] = -extr[i, :, :3].T @ extr[i, :, 3]
        return dict(c2w=c2w, K=intr, depth=depth[0, ..., 0].float().cpu().numpy(), tracks=tr, vis=vis)

    def kubric(self, ds, seed, v, q):
        torch = self.torch; torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        s = 518 / 512.0
        q518 = np.clip(np.round((q + 0.5) * s - 0.5), 0, 517).astype(np.float32)   # e5_xm_eval.to518
        R = self._run(self._load518(u.frame_paths(ds, seed, v, u.VIEWS7)), q518); rt = time.time() - t0
        TV = 6; tr = R["tracks"]
        lift = em.LiftedTrackerAdapter("vggt").predict(np.stack([q518, tr[TV]]).astype(np.float64), R["depth"][[0, TV]],
                                                       R["c2w"][[0, TV]], R["K"][[0, TV]])
        qi = q518.astype(int)
        return dict(xp=(tr[TV].astype(np.float64) + 0.5) / s - 0.5, wd=lift["w_disp"][1], z0=R["depth"][0][qi[:, 1], qi[:, 0]],
                    depth0_med=float(np.median(R["depth"][0])), cams=R["c2w"], views=np.array(u.VIEWS7),
                    vis_t=R["vis"][TV], runtime_s=rt, **u.gpu_stats())

    def _geom_crop(self, W, H, w, h):
        s = 518 / W; nh = int(round(H * s / 14) * 14); top = max(0, (nh - 518) // 2)
        return em._geom_affine(W, H, w, h, s, nh / H, 0.5 * s - 0.5, 0.5 * nh / H - 0.5 - top)

    def real(self, paths, meta, cfg):
        img = self.lpi(list(paths)).to("cuda"); h, w = img.shape[-2:]
        vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
        R = self._run(img, np.c_[uu.ravel(), vv.ravel()], chunk=8192)
        g = self._geom_crop(meta["W"], meta["H"], w, h)
        return dict(field=R["tracks"][1].reshape(h, w, 2).astype(np.float32), geom=json.dumps([g, g]))

    def pose(self, paths):
        return dict(cams=self._run(self.lpi(list(paths)).to("cuda"))["c2w"])


class OpenD4RTM:
    tag = "opend4rt"; need_gb = 18

    def __init__(self):
        self.A = em.OpenD4RTAdapter(dense_stride=1)
        assert self.A.pad_to_clip

    def kubric(self, ds, seed, v, q):
        import cv2, e5_kubric_post as kp
        meta, gt, _ = kp.load_scene(u.vdir(ds, seed, v)); fr = list(range(24)); A = self.A
        g = em.kubric_cv_geometry(meta, gt, fr); H, W = g["depth_z"].shape[1:]
        Km = g["K"].copy(); Km[0] *= (A.w - 1) / (W - 1); Km[1] *= (A.h - 1) / (H - 1)
        dz = cv2.resize(g["depth_z"][0], (A.w, A.h), interpolation=cv2.INTER_NEAREST)
        P = A.predict(u.frame_paths(ds, seed, v, fr), targets=[0, 23], gt=dict(K=Km, w2c=g["w2c"], depth0_z=dz))
        gd, gm = P["geom"][0], P["geom_model"][23]; x, y = q[:, 0], q[:, 1]
        smp = lambda a: u.sample_nearest(a, gd, x, y)[0]
        out = dict(xp=u.to_native(gm, smp(P["x_r4"][23])), xp_r1=u.to_native(gm, smp(P["x_r1"][23])),
                   xp_r2=u.to_native(gm, smp(P["x_r2_predK"][23])), xp_r2gtK=u.to_native(gm, smp(P["x_r2_gtK"][23])),
                   wd=smp(P["w_disp"][23]), z0=smp(P["pts3d_0"][..., 2]), depth0_med=float(np.nanmedian(P["pts3d_0"][..., 2])),
                   cams=P["cam_pose"], views=np.array(fr), r4_scale=P.get("r4_scale", np.nan), runtime_s=P["_runtime_s"], **P["_gpu"])
        return out

    def real(self, paths, meta, cfg):
        P = self.A.predict(paths, targets=[0, 1])
        x4 = em._project_cv(em._apply_T(np.linalg.inv(P["cam_pose"][1]), P["xyz_ref0"][1]), P["K"][1])
        return dict(field=P["x_r1"][1].astype(np.float32), field_r2=P["x_r2_predK"][1].astype(np.float32),
                    field_r4own=x4.astype(np.float32), geom=json.dumps([P["geom"][0], P["geom_model"][1]]))

    def pose(self, paths):
        return dict(cams=self.A.predict(paths, targets=[0])["cam_pose"])


class FourRCM:
    tag = "4rc"; need_gb = 15

    def __init__(self):
        self.A = em.FourRCAdapter()

    def kubric(self, ds, seed, v, q):
        P = self.A.predict(u.frame_paths(ds, seed, v, u.VIEWS7)); TV = 6
        g = P["geom"][0]; x, y = q[:, 0], q[:, 1]; smp = lambda a: u.sample_nearest(a, g, x, y)[0]
        Pc0 = em._apply_T(np.linalg.inv(P["cam_pose"][0]), P["pts3d_0"])
        return dict(xp=u.to_native(g, smp(P["x_pred"][TV])), wd=smp(P["w_disp"][TV]), z0=smp(Pc0[..., 2]),
                    depth0_med=float(np.median(Pc0[..., 2])), cams=P["cam_pose"], views=np.array(u.VIEWS7),
                    b_P0=smp(P["pts3d_0"]), b_F=smp(P["scene_flow"][TV]),
                    b_K=np.stack([u.K_native(P["K"][i], g) for i in range(len(P["K"]))]),
                    runtime_s=P["_runtime_s"], **P["_gpu"])

    def real(self, paths, meta, cfg):
        P = self.A.predict(paths)
        return dict(field=P["x_pred"][1].astype(np.float32), geom=json.dumps(P["geom"][:2]))

    def pose(self, paths):
        return dict(cams=self.A.predict(paths)["cam_pose"])


class TC3RM:
    need_gb = 20

    def __init__(self, cond):
        self.cond = cond; self.tag = f"tc3r_{cond}"
        self.A = em.TrackCraft3RAdapter(resize_mode="stretch"); self.A.register_t0 = False

    def kubric(self, ds, seed, v, q):
        import e5_kubric_post as kp
        meta, gt, _ = kp.load_scene(u.vdir(ds, seed, v)); fr = u.TC3R_FRAMES
        if self.cond == "da3":
            dd = os.path.join(u.U2, "da3", ds, str(seed), v)
            assert os.path.exists(os.path.join(dd, "depth.npy")), f"DA3 inputs missing: {dd}"
            d, w2c, intr = em.TrackCraft3RAdapter.make_inputs("da3", da3_dir=dd)
        else:
            d, w2c, intr = em.TrackCraft3RAdapter.make_inputs(self.cond, meta, gt, fr)
        P = self.A.predict(u.frame_paths(ds, seed, v, fr), d, w2c, intr, register_t0=False)
        g = P["geom"][0]; x, y = q[:, 0], q[:, 1]; smp = lambda a: u.sample_nearest(a, g, x, y)[0]; T = len(fr) - 1
        return dict(xp=u.to_native(g, smp(P["x_pred"][T])), wd=smp(P["w_disp"][T]), z0=smp(P["pts3d_0"][..., 2]),
                    depth0_med=float(np.median(P["pts3d_0"][..., 2])), cams=P["cam_pose"], views=np.array(fr),
                    resize_mode=str(P["resize_mode"]), runtime_s=P["_runtime_s"], **P["_gpu"])

    def real(self, paths, meta, cfg):
        dd = cfg["da3_dir"]
        assert os.path.exists(os.path.join(dd, "depth.npy")), f"DA3 inputs missing: {dd}"
        d, w2c, intr = em.TrackCraft3RAdapter.make_inputs("da3", da3_dir=dd)
        P = self.A.predict(paths, d, w2c, intr, register_t0=False)
        return dict(field=P["x_pred"][1].astype(np.float32), geom=json.dumps(P["geom"][:2]))


class MoReM:
    tag = "more"; need_gb = 12

    def __init__(self):
        self.A = em.MoReAdapter()

    def kubric(self, ds, seed, v, q):
        P = self.A.predict(u.frame_paths(ds, seed, v, u.VIEWS7))
        return dict(cams=P["cam_pose"], views=np.array(u.VIEWS7), depth0_med=float(np.median(P["depth"][0])),
                    runtime_s=P["_runtime_s"], **P["_gpu"])

    def pose(self, paths):
        return dict(cams=self.A.predict(paths)["cam_pose"])


class TrackerM:
    """CoTracker3 (offline, scaled_offline.pth) or RAFT (torchvision raft_large C_T_V2, 24 updates), as in e5_w2_predict."""
    need_gb = 8

    def __init__(self, which):
        import torch, torch.nn.functional as F
        self.torch, self.F, self.which, self.tag = torch, F, which, which
        if which == "cotracker3":
            sys.path.insert(0, os.path.join(BIG, "models/src/co-tracker"))
            from cotracker.models.build_cotracker import build_cotracker
            from cotracker.models.core.model_utils import get_points_on_a_grid
            ck = glob.glob(os.path.join(BIG, "models/hf/hub/models--facebook--cotracker3/snapshots/*/scaled_offline.pth"))[0]
            self.CT = build_cotracker(ck, offline=True, window_len=60).to("cuda").eval(); self.grid = get_points_on_a_grid
            self.IS = self.CT.model_resolution
        else:
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            self.RAFT = raft_large(weights=Raft_Large_Weights.C_T_V2, progress=False).to("cuda").eval()

    def _video(self, paths):
        from PIL import Image
        ims = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
        return self.torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().to("cuda")

    def track(self, video, qxy):
        torch, F = self.torch, self.F; IS = self.IS
        T, _, H, W = video.shape
        with torch.no_grad():
            v = F.interpolate(video, tuple(IS), mode="bilinear", align_corners=True)[None]
            q = torch.zeros(1, len(qxy), 3, device="cuda"); q[0, :, 1:] = torch.as_tensor(qxy, device="cuda", dtype=torch.float32)
            q[:, :, 1:] *= q.new_tensor([(IS[1] - 1) / (W - 1), (IS[0] - 1) / (H - 1)])
            g = self.grid(6, IS, device="cuda"); g = torch.cat([torch.zeros_like(g[:, :, :1]), g], 2)
            out = self.CT(video=v, queries=torch.cat([q, g], 1), iters=6)
            n = len(qxy); tr = out[0][0, :, :n] * out[0].new_tensor([(W - 1) / (IS[1] - 1), (H - 1) / (IS[0] - 1)])
        return tr.cpu().numpy().astype(np.float64), out[1][0, :, :n].float().cpu().numpy()

    def flow(self, i0, i1):
        torch, F = self.torch, self.F
        H, W = i0.shape[-2:]; ph, pw = (-H) % 8, (-W) % 8
        x = torch.stack([i0, i1]) / 127.5 - 1.0
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        with torch.no_grad():
            return self.RAFT(x[:1], x[1:], num_flow_updates=24)[-1][0][:, :H, :W]

    def sample(self, fl, pts):
        torch, F = self.torch, self.F
        H, W = fl.shape[-2:]; g = torch.as_tensor(pts, device="cuda", dtype=torch.float32).clone()
        g[:, 0] = 2 * g[:, 0] / (W - 1) - 1; g[:, 1] = 2 * g[:, 1] / (H - 1) - 1
        s = F.grid_sample(fl[None], g[None, None], mode="bilinear", padding_mode="border", align_corners=True)
        return s[0, :, 0].T.cpu().numpy().astype(np.float64)

    def kubric(self, ds, seed, v, q):
        import e5_kubric_post as kp
        torch = self.torch; torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        meta, gt, _ = kp.load_scene(u.vdir(ds, seed, v))
        vid = self._video(u.frame_paths(ds, seed, v, u.VIEWS7)); x1 = q.astype(np.float64)
        g = em.kubric_cv_geometry(meta, gt, [0, 23]); L = em.LiftedTrackerAdapter(self.which)
        lift = lambda x2: L.predict(np.stack([x1, x2]), g["depth_z"], g["c2w"], g["K"])["w_disp"][1]
        qi = q.astype(int); out = dict(z0=g["depth_z"][0][qi[:, 1], qi[:, 0]].astype(np.float64),
                                       depth0_med=float(np.median(g["depth_z"][0])), views=np.array(u.VIEWS7))
        if self.which == "cotracker3":
            tr, vis = self.track(vid, x1); out.update(xp=tr[-1], wd=lift(tr[-1]), vis_t=vis[-1])
        else:
            x2 = x1.copy()
            for k in range(len(vid) - 1):
                x2 = x2 + self.sample(self.flow(vid[k], vid[k + 1]), x2)
            xd = x1 + self.sample(self.flow(vid[0], vid[-1]), x1)
            out.update(xp=x2, wd=lift(x2), xp_direct=xd, wd_direct=lift(xd))
        out.update(runtime_s=time.time() - t0, **u.gpu_stats())
        return out

    def real(self, paths, meta, cfg):
        H, W = meta["H"], meta["W"]; vid = self._video(paths); g = _identity_geom(W, H)
        if self.which == "raft":
            fl = self.flow(vid[0], vid[1]).permute(1, 2, 0).cpu().numpy()
            vv, uu = np.mgrid[0:H, 0:W].astype(np.float32)
            return dict(field=np.stack([uu, vv], -1) + fl, geom=json.dumps([g, g]))
        import e5_realdata_eval as ev
        z = np.load(cfg["gt_file"]); band = ev.boundary_band(z["label"])
        vv, uu = np.nonzero(z["valid"] & (z["group"] >= 0) & ~band)
        field = np.full((H, W, 2), np.nan, np.float32); qxy = np.c_[uu, vv].astype(np.float64); ti = 1
        for j in range(0, len(qxy), 20000):
            tr, _ = self.track(vid, qxy[j:j + 20000]); field[vv[j:j + 20000], uu[j:j + 20000]] = tr[ti]
        return dict(field=field, geom=json.dumps([g, g]))


def build(model):
    if model == "any4d": return Any4DM()
    if model == "vggt": return VGGTM()
    if model == "opend4rt": return OpenD4RTM()
    if model == "4rc": return FourRCM()
    if model.startswith("tc3r_"): return TC3RM(model.split("_", 1)[1])
    if model == "more": return MoReM()
    if model in ("cotracker3", "raft"): return TrackerM(model)
    raise KeyError(model)


# =========================================================================================== drivers
REAL_CFG = {"2v": [0, 1], "mv4": [0, 1, 2, 3], "mv7": [0, 1, 2, 3, 4, 5, 6], "w12": list(range(11)) + [10]}


def run_kubric(M, split, v, scenes_only=None, limit=0):
    ds = u.SPLIT_SET[split]; od = os.path.join(u.U2, "pred", M.tag, split, v)
    sc = u.factorial_scenes(split)
    if scenes_only:
        sc = [s for s in sc if str(s[0]) in scenes_only]
    if limit:
        sc = sc[:limit]
    n_new = n_err = 0
    for seed, name in sc:
        pz = os.path.join(od, f"{seed}.npz")
        if os.path.exists(pz):
            continue
        q = np.load(u.gt_path(ds, seed, "D"))["q_pix"].astype(np.float64)
        try:
            out = M.kubric(ds, seed, v, q)
        except Exception as e:
            n_err += 1; u.save_json(os.path.join(od, "_errors", f"{seed}.json"), dict(error=repr(e), tb=traceback.format_exc()[-3000:]))
            print("ERROR", M.tag, split, v, seed, repr(e), flush=True)
            if "out of memory" in repr(e).lower():
                raise
            continue
        out = {k: (np.asarray(val) if not isinstance(val, str) else np.array(val)) for k, val in out.items()}
        u.save_npz(pz, q_pix=q, scene=np.array(name), seed=np.array(seed), variant=np.array(v), **out)
        n_new += 1
        print(f"[{time.strftime('%H:%M:%S')}] {M.tag} {split} {v} {seed} {name} rt={float(out.get('runtime_s', np.nan)):.1f}s", flush=True)
    have = [s for s in sc if os.path.exists(os.path.join(od, f"{s[0]}.npz"))]
    if not scenes_only and not limit and len(have) == len(sc):
        u.save_json(os.path.join(od, "_DONE.json"), dict(n=len(have), time=time.strftime("%Y-%m-%d %H:%M:%S")))
    print("unit", M.tag, split, v, "new", n_new, "errors", n_err, "have", len(have), "of", len(sc), flush=True)
    return n_err


def real_files(dsname):
    return sorted(f for f in glob.glob(os.path.join(u.REAL_GT, dsname, "*.npz")) if not f.endswith(".tmp.npz"))


def run_real(M, unit, limit=0):
    dsname, pas, cfgname = unit.split(":")[:3]; extra = unit.split(":")[3:]
    resize = extra[0] if extra else ""
    sub = f"{dsname}_{pas}_{cfgname}" + (f"_{resize}" if resize else "")
    od = os.path.join(u.U2, "pred_real", M.tag, sub); files = real_files(dsname)
    if limit:
        files = files[:: max(1, len(files) // limit)][:limit]
    n_new = n_err = n_skip = 0
    for f in files:
        name = os.path.basename(f)[:-4]; pz = os.path.join(od, name + ".npz")
        if os.path.exists(pz):
            continue
        z = np.load(f, allow_pickle=False); meta = json.loads(str(z["meta"])); fr = meta["frames"].get(pas, {})
        offs = REAL_CFG[cfgname]
        if not all(str(o) in fr for o in offs):
            n_skip += 1; continue
        paths = [fr[str(o)] for o in offs]
        cfg = dict(resize=resize, gt_file=f, da3_dir=os.path.join(u.U2, "da3", "real", f"{dsname}_{pas}_{cfgname}", name))
        try:
            t0 = time.time(); out = M.real(paths, meta, cfg); out["runtime_s"] = time.time() - t0
        except Exception as e:
            n_err += 1; u.save_json(os.path.join(od, "_errors", name + ".json"), dict(error=repr(e), tb=traceback.format_exc()[-3000:]))
            print("ERROR", M.tag, sub, name, repr(e), flush=True)
            if "out of memory" in repr(e).lower():
                raise
            continue
        u.save_npz(pz, paths=np.array(json.dumps(paths)), **{k: np.asarray(val) for k, val in out.items()})
        n_new += 1
        if n_new % 25 == 1:
            print(f"[{time.strftime('%H:%M:%S')}] {M.tag} {sub} {name} rt={out['runtime_s']:.1f}s", flush=True)
    have = sum(os.path.exists(os.path.join(od, os.path.basename(f)[:-4] + ".npz")) for f in files)
    if not limit and have + n_skip == len(files) and n_err == 0:
        u.save_json(os.path.join(od, "_DONE.json"), dict(n=have, skipped_missing_frames=n_skip, time=time.strftime("%Y-%m-%d %H:%M:%S")))
    print("unit", M.tag, sub, "new", n_new, "errors", n_err, "skipped(missing frames)", n_skip, "have", have, flush=True)
    return n_err


def run_pose(M, limit=0):
    W = u.make_pose_windows()["windows"]; od = os.path.join(u.U2, "pred_pose", M.tag)
    if limit:
        W = [w for i, w in enumerate(W) if i % max(1, len(W) // limit) == 0][:limit]
    n_new = n_err = 0
    for w in W:
        pz = os.path.join(od, w["id"].replace("/", "__") + ".npz")
        if os.path.exists(pz):
            continue
        try:
            t0 = time.time(); out = M.pose(w["frames"]); rt = time.time() - t0
        except Exception as e:
            n_err += 1; u.save_json(os.path.join(od, "_errors", w["id"].replace("/", "__") + ".json"), dict(error=repr(e), tb=traceback.format_exc()[-3000:]))
            print("ERROR", M.tag, w["id"], repr(e), flush=True)
            if "out of memory" in repr(e).lower():
                raise
            continue
        u.save_npz(pz, cams=np.asarray(out["cams"]), id=np.array(w["id"]), runtime_s=np.array(rt))
        n_new += 1
    have = sum(os.path.exists(os.path.join(od, w["id"].replace("/", "__") + ".npz")) for w in W)
    if not limit and have == len(W):
        u.save_json(os.path.join(od, "_DONE.json"), dict(n=have, time=time.strftime("%Y-%m-%d %H:%M:%S")))
    print("unit", M.tag, "pose", "new", n_new, "errors", n_err, "have", have, "of", len(W), flush=True)
    return n_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kubric", default="", help="comma list split:variant, e.g. test:D,test:C")
    ap.add_argument("--real", default="", help="comma list dataset:pass:config[:resize], e.g. kitti15:rgb:2v:wide")
    ap.add_argument("--pose", action="store_true")
    ap.add_argument("--scenes", default="", help="restrict Kubric units to these seeds (smoke)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time(); M = build(a.model); print("loaded", a.model, f"{time.time() - t0:.0f}s", flush=True)
    err = 0
    for it in [x for x in a.kubric.split(",") if x]:
        sp, v = it.split(":"); err += run_kubric(M, sp, v, set(a.scenes.split(",")) if a.scenes else None, a.limit)
    for it in [x for x in a.real.split(",") if x]:
        err += run_real(M, it, a.limit)
    if a.pose:
        err += run_pose(M, a.limit)
    print("finished", a.model, "errors", err, f"{time.time() - t0:.0f}s", flush=True)
    sys.exit(3 if err else 0)


if __name__ == "__main__":
    main()
