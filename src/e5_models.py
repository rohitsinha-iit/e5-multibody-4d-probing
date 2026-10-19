"""Model adapters for item 8 (real-data output-level evaluation). Model-agnostic interface:

    model = get_model(name, **kw)
    P = model.predict(image_paths)          # list of V paths, index 0 = reference view
P is a dict (numpy, model resolution h x w):
    pts3d_0    (h,w,3)    reference-view pointmap in the model's world frame
    scene_flow (V,h,w,3)  allocentric scene flow from view 0 to each view (index 0 zeros), or None
    cam_pose   (V,4,4)    camera-to-world
    K          (V,3,3)    pinhole intrinsics at model resolution
    conf_0     (h,w)      optional confidence
    x_pred     (V,h,w,2)  optional: direct predicted correspondences of view-0 pixels in each view (model px);
                          if present it is used instead of the pointmap + scene flow projection
    geom       list of V dicts mapping native pixels to model pixels:
               u_model = (u_native + 0.5) * sx - 0.5 - left,  v_model = (v_native + 0.5) * sy - 0.5 - top
               (W, H native size; w, h model size). This is exactly Any4D's load_images resize (isotropic rescale
               so the image covers the target, floor of the scaled size, then a centred crop).
A new model only needs a class with predict() returning this dict.
"""
import math, os, sys
import numpy as np


def resize_geom(W, H, tw, th):
    """Replicates any4d.utils.cropping.crop_resize_if_necessary (no intrinsics given)."""
    scale = max(tw / W, th / H) + 1e-8
    ow, oh = int(math.floor(W * scale)), int(math.floor(H * scale))
    left, top = (ow - tw) // 2, (oh - th) // 2
    return dict(W=W, H=H, w=tw, h=th, sx=ow / W, sy=oh / H, left=left, top=top, ow=ow, oh=oh)


def native_to_model(g, u, v):
    return (u + 0.5) * g["sx"] - 0.5 - g["left"], (v + 0.5) * g["sy"] - 0.5 - g["top"]


def model_to_native(g, um, vm):
    return (um + g["left"] + 0.5) / g["sx"] - 0.5, (vm + g["top"] + 0.5) / g["sy"] - 0.5


class Any4D:
    """Any4D (MapAnything fine-tuned for 4D). Same loading as e5_any4d_infer.py. Run with cwd = Any4D repo root and
    this file importable next to scripts/demo_inference.py."""
    name = "any4d"

    def __init__(self, resize_mode="square", size=518, checkpoint_path="checkpoints/any4d_4v_combined.pth",
                 config_dir="configs"):
        import torch
        from demo_inference import init_inference_model, sample_inference
        from any4d.utils.image import load_images, find_closest_aspect_ratio
        from any4d.utils.geometry import quaternion_to_rotation_matrix, recover_pinhole_intrinsics_from_ray_directions
        from any4d.utils.misc import seed_everything
        seed_everything(0)
        self.torch = torch; self.sample_inference = sample_inference; self.load_images = load_images
        self.q2R = quaternion_to_rotation_matrix; self.rec_K = recover_pinhole_intrinsics_from_ray_directions
        self.fcar = find_closest_aspect_ratio
        self.resize_mode = resize_mode; self.size = size
        cfg = {"path": f"{config_dir}/train.yaml",
               "config_overrides": ["machine=local", "model=any4d", "model.encoder.uses_torch_hub=false", "model/task=images_only"],
               "checkpoint_path": checkpoint_path, "trained_with_amp": True, "data_norm_type": "dinov2"}
        self.cfg = cfg; self.device = torch.device("cuda")
        self.model = init_inference_model(cfg, checkpoint_path, self.device)

    def target_size(self, W, H):
        if self.resize_mode == "square":
            s = (self.size // 14) * 14; return s, s
        if self.resize_mode == "fixed_mapping":
            return self.fcar(W / H, 518)
        raise ValueError(self.resize_mode)

    def predict(self, paths):
        from PIL import Image
        torch = self.torch
        sizes = [Image.open(p).size for p in paths]
        if self.resize_mode == "fixed_mapping":   # load_images uses the mean aspect ratio over the list
            ar = sum(w / h for w, h in sizes) / len(sizes); tw, th = self.fcar(ar, 518)
        else:
            tw, th = self.target_size(*sizes[0])
        views = self.load_images(list(paths), resize_mode=self.resize_mode,
                                 size=self.size if self.resize_mode == "square" else None,
                                 norm_type=self.cfg["data_norm_type"], patch_size=14)
        geom = [resize_geom(w, h, tw, th) for (w, h) in sizes]
        for vw, g in zip(views, geom):
            assert tuple(vw["img"].shape[-2:]) == (g["h"], g["w"]), (vw["img"].shape, g)
        with torch.no_grad():
            res = self.sample_inference(self.model, views, self.device, use_amp=True)
        V = len(paths); pr = [res[f"pred{i+1}"] for i in range(V)]
        out = {"pts3d_0": pr[0]["pts3d"][0].float().cpu().numpy()}
        out["scene_flow"] = np.stack([np.zeros_like(out["pts3d_0"])] +
                                     [pr[i]["scene_flow"][0].float().cpu().numpy() for i in range(1, V)])
        poses, Ks = [], []
        for i in range(V):
            T = np.eye(4); T[:3, :3] = self.q2R(pr[i]["cam_quats"][0].float().cpu()).numpy()
            T[:3, 3] = pr[i]["cam_trans"][0].float().cpu().numpy(); poses.append(T)
            Ks.append(self.rec_K(pr[i]["ray_directions"][0].float().cpu()).numpy())
        out["cam_pose"] = np.stack(poses); out["K"] = np.stack(Ks)
        if "conf" in pr[0]:
            out["conf_0"] = pr[0]["conf"][0].float().cpu().numpy().squeeze()
        out["geom"] = geom
        out["_views"] = views
        return out


def get_model(name, **kw):
    if name == "any4d":
        return Any4D(**kw)
    raise KeyError(f"unknown model {name}; add an adapter class with predict(paths) to e5_models.py")


# =====================================================================================================================
# New adapters for U2 (A6000 new-models track, 2026-10-19). Added as new classes only; get_model() above is unchanged,
# use get_adapter(name, **kw), which also falls back to get_model().
#
# Every adapter returns the shared dict above PLUS
#   w_disp (V,h,w,3)  world displacement of view-0 pixels from frame 0 to each view, expressed in frame-0 camera
#                     coordinates (OpenCV axes: x right, y down, z forward), model units; index 0 is zero.
#                     None if the model has no motion output (MoRe).
# Conventions used by every new adapter:
#   cam_pose  camera-to-world, OpenCV axes, world = the model's own world frame (for the new models this is the
#             frame-0 camera up to the model's own choice; w_disp is always re-expressed in the predicted frame-0 camera)
#   K         pinhole intrinsics at model resolution, pixel-index convention (pixel i has centre at i)
#   geom      same dict layout as resize_geom(); native_to_model / model_to_native work unchanged
# Each model runs in its own venv ($E5_BIG/venvs/<name>); this file only needs numpy at import time.
# =====================================================================================================================

E5_BIG_DEFAULT = os.environ.get("E5_BIG", "/data2/DATA/rohit_sinha/e5/big")


def _geom_affine(W, H, w, h, sx, sy, ox, oy):
    """geom dict for u_model = sx * u_native + ox (pixel-index convention), in the resize_geom layout."""
    return dict(W=W, H=H, w=w, h=h, sx=sx, sy=sy, left=0.5 * sx - 0.5 - ox, top=0.5 * sy - 0.5 - oy,
                ow=int(round(W * sx)), oh=int(round(H * sy)))


def _project_cv(X, K):
    """X (...,3) camera coords (OpenCV) -> (...,2) pixels."""
    z = X[..., 2]
    return np.stack([K[0, 0] * X[..., 0] / z + K[0, 2], K[1, 1] * X[..., 1] / z + K[1, 2]], -1)


def _apply_T(T, X):
    return X @ T[:3, :3].T + T[:3, 3]


def _umeyama(src, dst, with_scale=True):
    mu_s, mu_d = src.mean(0), dst.mean(0); A, B = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(B.T @ A / len(src))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (A ** 2).sum(1).mean()) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def _robust_rigid(src, dst, rounds=2, keep=0.8, with_scale=False):
    ok = np.isfinite(src).all(1) & np.isfinite(dst).all(1); src, dst = src[ok], dst[ok]
    s, R, t = _umeyama(src, dst, with_scale)
    for _ in range(rounds):
        r = np.linalg.norm(src @ R.T * s + t - dst, axis=1); m = r <= np.quantile(r, keep)
        s, R, t = _umeyama(src[m], dst[m], with_scale)
    T = np.eye(4); T[:3, :3] = s * R; T[:3, 3] = t
    return T, s


def _to_cam0(P, out):
    """Re-express w_disp in the predicted frame-0 camera: w_disp_cam0 = R0^T (world displacement)."""
    R0 = out["cam_pose"][0][:3, :3]
    return P @ R0  # (..,3) @ R0  ==  (R0^T x)^T


def _gpu_stats(torch):
    if not torch.cuda.is_available():
        return {}
    return dict(peak_alloc_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                peak_reserved_gb=torch.cuda.max_memory_reserved() / 2 ** 30)


# ----------------------------------------------------------------------------- Kubric GT inputs (OpenCV convention)
def kubric_cv_geometry(meta, gt, frames):
    """GT inputs for the new models from an E5 Kubric scene (kp.load_scene output), for the frame indices `frames`
    (indices into meta['frames']). Returns dict:
      depth_z (T,H,W) z-depth, w2c (T,4,4) OpenCV world-to-camera normalised to frames[0] (w2c[0] = I),
      c2w (T,4,4) its inverse, K (3,3) pixel-index convention, intr4 [fx,fy,cx,cy], c2w_world (T,4,4) un-normalised.
    Kubric depth is distance along the ray; Kubric cameras are Blender (x right, y up, z backward)."""
    W, H = meta["resolution"]; Kb = np.array(meta["K"])
    fx, fy = Kb[0, 0], Kb[1, 1]; cx, cy = W / 2 - 0.5, H / 2 - 0.5
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    ray_norm = np.sqrt(((u - W / 2) / fx) ** 2 + ((v - H / 2) / fy) ** 2 + 1.0)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    depth = gt["depth"][..., 0]
    dz, c2w_w = [], []
    for f in frames:
        dz.append(depth[f] / ray_norm)
        C = np.array(meta["camera_matrix_world"][str(meta["frames"][f])]) @ flip
        c2w_w.append(C)
    c2w_w = np.stack(c2w_w); w2c_w = np.linalg.inv(c2w_w)
    w2c = w2c_w @ c2w_w[0]           # world -> cam_t, with world := cam_0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return dict(depth_z=np.stack(dz).astype(np.float32), w2c=w2c, c2w=np.linalg.inv(w2c), K=K,
                intr4=np.array([fx, fy, cx, cy]), c2w_world=c2w_w)


def kubric_gt_wdisp(meta, gt, a, b):
    """GT world displacement of every frame-a pixel to frame b, in frame-a camera coords (OpenCV), (H,W,3);
    plus the frame-a points X0 in camera-a coords and the moving mask (pixel on an object of group > 0)."""
    import e5_kubric_post as kp
    depth = gt["depth"][..., 0]; seg = gt["segmentation"][..., 0]
    X = kp.backproject(meta, depth[a], a); Xb = X.copy(); moving = np.zeros(X.shape[:2], bool)
    fa, fb = str(meta["frames"][a]), str(meta["frames"][b])
    for sid, (g, k) in kp.seg_to_group(meta).items():
        m = seg[a] == sid
        if k >= 0 and m.any():
            Ma = np.array(meta["objects"][k]["matrix_world"][fa]); Mb = np.array(meta["objects"][k]["matrix_world"][fb])
            T = Mb @ np.linalg.inv(Ma); Xb[m] = X[m] @ T[:3, :3].T + T[:3, 3]
            if g > 0:
                moving |= m
    C = np.array(meta["camera_matrix_world"][fa]) @ np.diag([1.0, -1.0, -1.0, 1.0])
    R = C[:3, :3]
    X0c = (X - C[:3, 3]) @ R
    return (Xb - X) @ R, X0c, moving


# ----------------------------------------------------------------------------- OpenD4RT
class OpenD4RTAdapter:
    """OpenD4RT (unofficial D4RT, ViT-g, 256x256, tubelet 2 so V must be even). Queries (u,v,t_src,t_tgt,t_cam) with
    u,v normalised by (W-1). All view-0 pixels of the model grid (stride dense_stride) are queried at every target.

    Readouts (P3 v2 4.2), all in model pixels / model units:
      x_r1        (V,h,w,2) uv head at t_tgt                          (also returned as x_pred)
      x_r2_predK  (V,h,w,2) xyz(t_cam = t_tgt) projected with the predicted intrinsics of view t
      x_r2_gtK    (V,h,w,2) same, projected with GT intrinsics (needs gt['K'] at model resolution)
      w_disp      (V,h,w,3) r3 = xyz(0,t,0) - xyz(0,0,0), frame-0 camera coordinates
      x_r4        (V,h,w,2) r3 moved into camera t with GT cameras and projected (needs gt['w2c'] frame-0
                            normalised, gt['K'] model res, and gt['depth0_z'] model-res z-depth for the scale)
    Cameras: c2w of view t from a trimmed rigid fit between xyz(t,t,t) and xyz(t,t,0) on a pose_grid^2 grid of
    frame-t pixels; K of view t from the median focal of xyz(t,t,t) vs its uv (repo's estimator, centred principal
    point)."""
    name = "opend4rt"

    def __init__(self, repo=None, ckpt_dir=None, device="cuda", query_chunk=16384, dense_stride=1, pose_grid=32):
        import torch
        repo = repo or os.path.join(E5_BIG_DEFAULT, "models/src/Open-d4rt")
        ckpt_dir = ckpt_dir or os.path.join(E5_BIG_DEFAULT, "models/ckpt/opend4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from src.core import load_checkpoint, load_yaml_config
        from src.model import build_model
        from src.eval import tasks as d4t
        self.torch, self.d4t = torch, d4t
        cfg = load_yaml_config(os.path.join(ckpt_dir, "model.yaml"))
        model = build_model(cfg["model"]).eval()
        payload = load_checkpoint(os.path.join(ckpt_dir, "opend4rt.ckpt"), map_location="cpu")
        sd = payload
        if isinstance(payload, dict):
            for k in ("state_dict", "model", "module", "network", "net"):
                if isinstance(payload.get(k), dict):
                    sd = payload[k]; break
        r = model.load_state_dict(sd, strict=False)
        self.load_info = dict(missing=len(r.missing_keys), unexpected=len(r.unexpected_keys))
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        hw = cfg.get_path("model.input.image_size", [256, 256])
        self.h, self.w = int(hw[0]), int(hw[1])
        self.clip_frames = d4t._model_clip_frames(self.model)
        self.query_chunk, self.stride, self.pose_grid = query_chunk, dense_stride, pose_grid

    def _q(self, u, v, ts, tt, tc):
        torch = self.torch; n = len(u)
        f = lambda a, dt: torch.as_tensor(np.broadcast_to(a, (n,)).copy(), device=self.device, dtype=dt)
        return dict(u=f(u, torch.float32), v=f(v, torch.float32), t_src=f(ts, torch.long), t_tgt=f(tt, torch.long),
                    t_cam=f(tc, torch.long))

    def _run(self, video, aspect, memory, q):
        out = self.d4t._run_model_for_queries(model=self.model, video_b=video, aspect_b=aspect, query=q,
                                              chunk_size=self.query_chunk, memory_b=memory)
        return {k: v.numpy() for k, v in out.items()}

    def predict(self, paths, targets=None, gt=None):
        import cv2, time
        from PIL import Image
        torch = self.torch
        V = len(paths)
        if V % 2:
            raise ValueError("OpenD4RT needs an even number of frames (tubelet 2)")
        if V > self.clip_frames:
            raise ValueError(f"{V} frames > clip length {self.clip_frames}; use the 48-frame checkpoint")
        frames = [np.array(Image.open(p).convert("RGB")) for p in paths]
        H, W = frames[0].shape[:2]; h, w = self.h, self.w
        vid = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA if H >= h else cv2.INTER_LINEAR) for f in frames])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        video = torch.from_numpy(vid).to(self.device, torch.float32).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
        aspect = torch.tensor([[W / H]], device=self.device, dtype=torch.float32)
        targets = list(range(V)) if targets is None else list(targets)
        s = self.stride
        vv, uu = np.meshgrid(np.arange(0, h, s, dtype=np.float32), np.arange(0, w, s, dtype=np.float32), indexing="ij")
        gh, gw = vv.shape
        un, vn = (uu / (w - 1)).ravel(), (vv / (h - 1)).ravel(); n = un.size
        with torch.no_grad():
            memory = self.d4t._encode_model_memory(model=self.model, video_b=video, aspect_b=aspect)
            tt = np.repeat(np.array(targets), n); U = np.tile(un, len(targets)); Vn = np.tile(vn, len(targets))
            loc = self._run(video, aspect, memory, self._q(U, Vn, 0, tt, tt))
            ref = self._run(video, aspect, memory, self._q(U, Vn, 0, tt, 0))
            # cameras and intrinsics for every view
            g = self.pose_grid
            pv, pu = np.meshgrid(np.linspace(0, 1, g), np.linspace(0, 1, g), indexing="ij")
            pu, pv = pu.ravel().astype(np.float32), pv.ravel().astype(np.float32)
            ta = np.repeat(np.arange(V), g * g)
            pa = self._run(video, aspect, memory, self._q(np.tile(pu, V), np.tile(pv, V), ta, ta, ta))
            pb = self._run(video, aspect, memory, self._q(np.tile(pu, V), np.tile(pv, V), ta, ta, 0))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        runtime = time.time() - t0
        T_ = len(targets)
        rs = lambda a, c: a.reshape(T_, gh, gw, c) if c else a.reshape(T_, gh, gw)
        xyz_loc, xyz_ref = rs(loc["xyz_3d"], 3), rs(ref["xyz_3d"], 3)
        uv = rs(loc["uv_2d"], 2) * np.array([w - 1, h - 1], np.float32)
        vis = 1 / (1 + np.exp(-rs(loc["visibility"], 0))); conf = rs(loc["confidence"], 0)
        poses, Ks = [], []
        xa, xb = pa["xyz_3d"].reshape(V, g * g, 3), pb["xyz_3d"].reshape(V, g * g, 3)
        uva = pa["uv_2d"].reshape(V, g * g, 2)
        for t in range(V):
            Tt, _ = _robust_rigid(xa[t].astype(np.float64), xb[t].astype(np.float64)) if t else (np.eye(4), 1.0)
            poses.append(Tt)
            fx, fy, cx, cy = self.d4t._estimate_intrinsics_params_from_predictions(xa[t], uva[t], (h, w))
            Ks.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]))
        poses, Ks = np.stack(poses), np.stack(Ks)
        if 0 not in targets:
            raise ValueError("targets must include view 0")
        i0 = targets.index(0)
        full = lambda a: a  # outputs are on the target list; map to V views below
        Vfull = lambda a: np.stack([a[targets.index(t)] if t in targets else np.full_like(a[0], np.nan) for t in range(V)])
        xyz_loc, xyz_ref, uv, vis, conf = map(Vfull, (xyz_loc, xyz_ref, uv, vis, conf))
        pts0 = xyz_ref[0]
        w_disp = xyz_ref - pts0[None]
        x_r2p = np.stack([_project_cv(xyz_loc[t], Ks[t]) for t in range(V)])
        out = dict(pts3d_0=pts0, scene_flow=w_disp, w_disp=w_disp, cam_pose=poses, K=Ks, conf_0=conf[0],
                   x_pred=uv, x_r1=uv, x_r2_predK=x_r2p, xyz_local=xyz_loc, xyz_ref0=xyz_ref, vis=vis, conf=conf,
                   targets=np.array(targets), dense_stride=s)
        if gt is not None and "K" in gt:
            Kg = np.asarray(gt["K"]); Kg = np.broadcast_to(Kg, (V, 3, 3)) if Kg.ndim == 2 else Kg
            out["x_r2_gtK"] = np.stack([_project_cv(xyz_loc[t], Kg[t]) for t in range(V)])
            if "w2c" in gt and "depth0_z" in gt:
                dz = np.asarray(gt["depth0_z"])[::s, ::s]
                m = np.isfinite(dz) & (dz > 0) & np.isfinite(pts0[..., 2]) & (pts0[..., 2] > 0)
                if "static0" in gt:
                    m &= np.asarray(gt["static0"])[::s, ::s]
                sc = float(np.median(dz[m] / pts0[..., 2][m]))
                out["r4_scale"] = sc
                out["x_r4"] = np.stack([_project_cv(_apply_T(np.asarray(gt["w2c"])[t], sc * xyz_ref[t]), Kg[t])
                                        for t in range(V)])
        sx, sy = (w - 1) / (W - 1) / s, (h - 1) / (H - 1) / s   # native px -> dense-grid index
        out["geom"] = [_geom_affine(W, H, gw, gh, sx, sy, 0.0, 0.0) for _ in range(V)]
        out["geom_model"] = [_geom_affine(W, H, w, h, (w - 1) / (W - 1), (h - 1) / (H - 1), 0.0, 0.0) for _ in range(V)]
        out["_runtime_s"] = runtime; out["_gpu"] = _gpu_stats(torch)
        return out


# ----------------------------------------------------------------------------- TrackCraft3R
class TrackCraft3RAdapter:
    """TrackCraft3R (Wan2.1-1.3B DiT + LoRA, one-step regression). 12 frames per run, 480x832 canvas with
    resize_mode='pad' (a 512x512 input becomes a centred 480x480 region). Inputs per run: z-depth (T,H,W),
    frame-0-normalised W2C (T,4,4, OpenCV) and [fx,fy,cx,cy] at native resolution. Output: frame-0-camera tracks
    P0(t) of every frame-0 pixel, a residual from the input frame-0 pointmap.
    Conditions (unified plan 2.2 / P3 v2): 'gt' = GT depth + GT cameras; 'da3' = DA3 depth + cameras (+ DA3 intrinsics
    unless gt_intrinsics); 'identity' = GT depth + identity cameras. Use make_inputs() to build them.
    The repo predictor is patched (env/patches/trackcraft3r_null_context.patch): with TC3R_NULL_CONTEXT pointing at a
    cached empty-prompt context the umT5 encoder is neither loaded nor moved to the GPU.
    Returned in model pixels of the 480x480 region: w_disp = P0(t) - P0(0), x_pred = projection of P0(t) with the
    input camera t (so image-space beta is input-determined), cam_pose / K = the input cameras."""
    name = "trackcraft3r"
    KUBRIC_FRAMES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23]

    def __init__(self, repo=None, ckpt=None, wan_root=None, null_ctx=None, device="cuda", height=480, width=832,
                 resize_mode="pad"):
        big = E5_BIG_DEFAULT
        repo = repo or os.path.join(big, "models/src/TrackCraft3r")
        ckpt = ckpt or os.path.join(big, "models/ckpt/trackcraft3r/model.safetensors")
        os.environ.setdefault("MODELSCOPE_CACHE", wan_root or os.path.join(big, "models/ckpt/wan_models"))
        os.environ.setdefault("TC3R_NULL_CONTEXT", null_ctx or os.path.join(big, "models/ckpt/trackcraft3r/null_context_bf16.pt"))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import torch
        from evaluation.wan_scene_flow_predictor import WanSceneFlowPredictor
        self.torch = torch
        self.p = WanSceneFlowPredictor(checkpoint_path=ckpt, height=height, width=width, device=device,
                                       resize_mode=resize_mode)

    @staticmethod
    def make_inputs(cond, meta=None, gt=None, frames=None, da3_dir=None, gt_intrinsics=False):
        """Build (depth_z, w2c, intr4) for a condition. Kubric GT needs meta, gt (kp.load_scene) and frames;
        'da3' needs da3_dir with depth.npy, extrinsics.npy (raw W2C), intrinsics.npy from run_da3()."""
        if cond in ("gt", "identity") or gt_intrinsics:
            g = kubric_cv_geometry(meta, gt, frames)
        if cond == "gt":
            return g["depth_z"], g["w2c"], g["intr4"]
        if cond == "identity":
            return g["depth_z"], np.tile(np.eye(4), (len(frames), 1, 1)), g["intr4"]
        if cond == "da3":
            d = np.load(os.path.join(da3_dir, "depth.npy")); E = np.load(os.path.join(da3_dir, "extrinsics.npy"))
            Ki = np.load(os.path.join(da3_dir, "intrinsics.npy"))
            if E.shape[-2] == 3:
                E = np.concatenate([E, np.tile([[[0, 0, 0, 1.0]]], (len(E), 1, 1))], 1)
            E = E @ np.linalg.inv(E[0])
            intr = g["intr4"] if gt_intrinsics else np.array([Ki[0, 0, 0], Ki[0, 1, 1], Ki[0, 0, 2], Ki[0, 1, 2]])
            return d.astype(np.float32), E, intr
        raise ValueError(cond)

    @staticmethod
    def run_da3(frame_paths, out_dir, model_name="depth-anything/DA3NESTED-GIANT-LARGE", venv=None, gpu=None):
        """Runs TrackCraft3R's scripts/preprocess_da3.py in the da3 venv on the given frames (subprocess)."""
        import subprocess, tempfile, shutil
        big = E5_BIG_DEFAULT
        py = os.path.join(venv or os.path.join(big, "venvs/da3"), "bin/python")
        tmp = tempfile.mkdtemp(prefix="da3_")
        for i, p in enumerate(frame_paths):
            os.symlink(os.path.abspath(p), os.path.join(tmp, f"{i:05d}" + os.path.splitext(p)[1]))
        env = dict(os.environ)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [py, os.path.join(big, "models/src/TrackCraft3r/scripts/preprocess_da3.py"), "--frame_dir", tmp,
               "--output_dir", out_dir, "--da3_root", os.path.join(big, "models/src/depth-anything-3"),
               "--model_name", model_name]
        subprocess.run(cmd, check=True, env=env)
        shutil.rmtree(tmp)
        return out_dir

    def predict(self, paths, depth_z, w2c, intr4):
        import time
        from PIL import Image
        torch = self.torch
        imgs = [Image.open(p).convert("RGB") for p in paths]
        w2c = np.asarray(w2c, np.float64); w2c = w2c @ np.linalg.inv(w2c[0])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        _, ar, nw, nh, pl, pt, W, H = self.p._resize(imgs)
        self.p.predict(imgs, np.zeros((1, 2)), None, np.asarray(intr4, np.float64), depth_map=np.asarray(depth_z),
                       extrinsics_w2c=w2c)
        runtime = time.time() - t0
        dense = self.p._last_row_dense[:, pt:pt + nh, pl:pl + nw].astype(np.float32)   # (T,nh,nw,3) cam-0 coords
        pj0 = self.p._last_pj_input[0, pt:pt + nh, pl:pl + nw].astype(np.float32)
        vis = self.p._last_vis_dense
        vis = vis[:, pt:pt + nh, pl:pl + nw] if vis is not None else None
        fx, fy, cx, cy = intr4
        K = np.array([[fx * ar, 0, cx * ar], [0, fy * ar, cy * ar], [0, 0, 1.0]])
        T = len(paths)
        x_pred = np.stack([_project_cv(_apply_T(w2c[t], dense[t]), K) for t in range(T)])
        out = dict(pts3d_0=pj0, tracks_cam0=dense, w_disp=dense - dense[0:1], scene_flow=dense - pj0[None],
                   cam_pose=np.linalg.inv(w2c), K=np.tile(K, (T, 1, 1)), x_pred=x_pred, vis=vis,
                   residual0=dense[0] - pj0)
        out["geom"] = [_geom_affine(W, H, nw, nh, ar, ar, 0.0, 0.0) for _ in range(T)]
        out["_runtime_s"] = runtime; out["_gpu"] = _gpu_stats(torch)
        return out


# ----------------------------------------------------------------------------- 4RC
class FourRCAdapter:
    """4RC (Arc): base geometry (depth + camera -> world points), relative motion of the query frame's pixels
    (motion decoder + track head) and a camera decoder. Query frame 0, all views as targets.
    Per view t from arc: pts (world points of view t), track (= relative motion + world points of query view 0),
    extrinsic (c2w), intrinsic. Returned: pts3d_0 = P0 (view-0 world points), scene_flow = F_t = track_t - P0
    (world frame), w_disp = F_t in predicted frame-0 camera coords, x_pred = projection of track_t with camera t,
    plus budget = dict(P0, F, c2w, K) for budget.py.
    load_images(size=512, patch_size=14); square_ok=True keeps a square 504x504 crop of a 512x512 frame
    (square_ok=False, the repo default, crops square inputs to 504x378)."""
    name = "4rc"

    def __init__(self, repo=None, checkpoint="Luo-Yihang/4RC", device="cuda", size=512, square_ok=True,
                 dtype="bf16-mixed"):
        repo = repo or os.path.join(E5_BIG_DEFAULT, "models/src/4RC")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import torch
        from arc.models.arc.arc import Arc
        from arc.dust3r.inference_multiview import inference
        from arc.dust3r.utils.image import load_images
        self.torch, self.inference, self.load_images = torch, inference, load_images
        self.device = torch.device(device)
        self.model = Arc.from_pretrained(checkpoint).to(self.device).eval()
        self.size, self.square_ok, self.dtype = size, square_ok, dtype

    def _geom(self, W, H):
        S = max(W, H); nw, nh = int(round(W * self.size / S)), int(round(H * self.size / S))
        cx, cy = nw // 2, nh // 2; ps = 14
        halfw, halfh = ((2 * cx) // ps) * ps // 2, ((2 * cy) // ps) * ps // 2
        if not self.square_ok and nw == nh:
            halfh = 3 * halfw / 4
        l, t = cx - halfw, cy - halfh
        s = nw / W
        # PIL resize maps pixel centres: u' = (u + 0.5) * s - 0.5; then crop offset
        return _geom_affine(W, H, int(2 * halfw), int(2 * halfh), s, nh / H, 0.5 * s - 0.5 - l, 0.5 * (nh / H) - 0.5 - t)

    def predict(self, paths, query_idx=0):
        import time
        from PIL import Image
        torch = self.torch
        imgs = self.load_images(list(paths), size=self.size, patch_size=14, square_ok=self.square_ok, verbose=False)
        for im in imgs:
            im["track_query_idx"] = torch.tensor([query_idx])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            res = self.inference(imgs, self.model, self.device, dtype=self.dtype, profiling=True, verbose=False,
                                 use_center_as_anchor=False)
        out_dict = res[0] if isinstance(res, tuple) else res
        runtime = time.time() - t0
        preds = out_dict["preds"]
        f = lambda x: x.detach().float().cpu().numpy()
        sq = lambda a: a[0] if a.ndim == 4 else a
        pts = np.stack([sq(f(p["pts"])) for p in preds]); trk = np.stack([sq(f(p["track"])) for p in preds])
        c2w = np.stack([f(p["extrinsic"]).reshape(-1, 4, 4)[0] if f(p["extrinsic"]).shape[-2] == 4 else
                        np.vstack([f(p["extrinsic"]).reshape(3, 4), [0, 0, 0, 1.0]]) for p in preds])
        K = np.stack([f(p["intrinsic"]).reshape(-1, 3, 3)[0] for p in preds])
        P0 = pts[query_idx]; F = trk - P0[None]
        out = dict(pts3d_0=P0, scene_flow=F, cam_pose=c2w, K=K, track_world=trk, pts_world=pts,
                   conf_0=sq(f(preds[query_idx]["conf"])).squeeze(), conf_track=np.stack([sq(f(p["conf_track"])).squeeze() for p in preds]),
                   output_keys=sorted(preds[0].keys()))
        out["w_disp"] = _to_cam0(F, out)
        out["x_pred"] = np.stack([_project_cv(_apply_T(np.linalg.inv(c2w[t]), trk[t]), K[t]) for t in range(len(preds))])
        out["budget"] = dict(P0=P0, F=F, c2w=c2w, K=K)
        W, H = Image.open(paths[0]).size
        g = self._geom(W, H)
        assert (g["h"], g["w"]) == P0.shape[:2], (g, P0.shape)
        out["geom"] = [g] * len(paths)
        out["_runtime_s"] = runtime; out["_gpu"] = _gpu_stats(torch)
        return out


# ----------------------------------------------------------------------------- MoRe
class MoReAdapter:
    """MoRe (VGGT-based with attention forcing), full-attention model pretrained/more_full.pt, config
    training/config/omniworld_full.yaml. Outputs cameras (VGGT pose encoding -> W2C OpenCV, inverted to c2w), depth
    (unprojected to world points) and, with predict_motion, the motion head map (a per-pixel dynamic score; the
    model gives no displacement, so w_disp and scene_flow are None)."""
    name = "more"

    def __init__(self, repo=None, ckpt=None, config=None, device="cuda", predict_motion=True):
        big = E5_BIG_DEFAULT
        repo = repo or os.path.join(big, "models/src/MoRe")
        ckpt = ckpt or os.path.join(big, "models/ckpt/more/more_full.pt")
        config = config or os.path.join(repo, "training/config/omniworld_full.yaml")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import torch
        from hydra import initialize_config_dir, compose
        from hydra.utils import instantiate
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        self.torch, self.lpi, self.pe2 = torch, load_and_preprocess_images, pose_encoding_to_extri_intri
        with initialize_config_dir(version_base=None, config_dir=os.path.dirname(os.path.abspath(config))):
            cfg = compose(config_name=os.path.basename(config).replace(".yaml", ""))
        model = instantiate(cfg.model, _recursive_=False)
        ck = torch.load(ckpt, map_location="cpu")
        r = model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)
        self.load_info = dict(missing=len(r.missing_keys), unexpected=len(r.unexpected_keys))
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.predict_motion = predict_motion

    def predict(self, paths):
        import time
        from PIL import Image
        torch = self.torch
        images = self.lpi(list(paths)).to(self.device)
        x = images[None]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.dtype):
            tokens, ps_idx, _ = self.model.aggregator(x)
            pose_enc = self.model.camera_head(tokens, first_token=True)[-1]
            extr, intr = self.pe2(pose_enc, x.shape[-2:])
            depth, dconf = self.model.depth_head(tokens, x, ps_idx)
            motion = self.model.motion_head(tokens, x, ps_idx) if (self.predict_motion and getattr(self.model, "motion_head", None) is not None) else None
        runtime = time.time() - t0
        f = lambda a: a.detach().float().cpu().numpy()
        E = f(extr)[0]; K = f(intr)[0]; D = f(depth)[0][..., 0]
        V = len(paths)
        w2c = np.concatenate([E, np.tile([[[0, 0, 0, 1.0]]], (V, 1, 1))], 1); c2w = np.linalg.inv(w2c)
        h, w = D.shape[1:]
        vv, uu = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        def unproj(t):
            Xc = np.stack([(uu - K[t, 0, 2]) / K[t, 0, 0] * D[t], (vv - K[t, 1, 2]) / K[t, 1, 1] * D[t], D[t]], -1)
            return _apply_T(c2w[t], Xc)
        out = dict(pts3d_0=unproj(0), pts_world=np.stack([unproj(t) for t in range(V)]), depth=D,
                   depth_conf=f(dconf)[0], cam_pose=c2w, K=K, scene_flow=None, w_disp=None)
        if motion is not None:
            m = motion[0] if isinstance(motion, (tuple, list)) else motion
            out["motion"] = f(m)[0].squeeze(-1) if f(m)[0].shape[-1] == 1 else f(m)[0]
        W, H = Image.open(paths[0]).size
        # VGGT load_and_preprocess_images (mode 'crop'): width -> 518, height scaled and rounded to a multiple of 14,
        # centre-cropped to 518 if taller
        s = 518 / W; nh = int(round(H * s / 14) * 14); top = max(0, (nh - 518) // 2)
        out["geom"] = [_geom_affine(W, H, w, h, s, nh / H, 0.5 * s - 0.5, 0.5 * nh / H - 0.5 - top) for _ in range(V)]
        out["_runtime_s"] = runtime; out["_gpu"] = _gpu_stats(torch)
        return out


# ----------------------------------------------------------------------------- lifted 2D trackers
class LiftedTrackerAdapter:
    """Lifts 2D tracks of view-0 query points to 3D world displacement using depth and cameras:
       X_0 = unproject(x_0, z_0(x_0), K_0, c2w_0),  X_t = unproject(x_t, z_t(x_t), K_t, c2w_t)  (bilinear z lookup),
       w_disp_t = R_0^T (X_t - X_0)  (frame-0 camera coords).
    For CoTracker3 and RAFT pass GT depth and cameras (kubric_cv_geometry); for the VGGT tracker pass VGGT's own
    depth and cameras. The 2D tracks come from the existing E5 runners (xm venv); this class is numpy only.
    Points whose track is not visible at t (vis False) or lands outside the image get NaN."""
    name = "lifted"

    def __init__(self, tracker="cotracker3"):
        self.tracker = tracker

    @staticmethod
    def _bilinear(img, x, y):
        H, W = img.shape
        x0 = np.clip(np.floor(x).astype(int), 0, W - 2); y0 = np.clip(np.floor(y).astype(int), 0, H - 2)
        ax, ay = np.clip(x - x0, 0, 1), np.clip(y - y0, 0, 1)
        return ((1 - ax) * (1 - ay) * img[y0, x0] + ax * (1 - ay) * img[y0, x0 + 1] +
                (1 - ax) * ay * img[y0 + 1, x0] + ax * ay * img[y0 + 1, x0 + 1])

    def predict(self, tracks, depth_z, c2w, K, vis=None):
        """tracks (V,N,2) pixel-index coords incl. view 0 = query pixels; depth_z (V,H,W); c2w (V,4,4) OpenCV;
        K (3,3) or (V,3,3); vis (V,N) bool or None. Returns dict(w_disp (V,N,3), X_world (V,N,3), x_pred = tracks)."""
        V, N, _ = tracks.shape; H, W = depth_z.shape[1:]
        K = np.broadcast_to(K, (V, 3, 3)) if np.ndim(K) == 2 else np.asarray(K)
        Xw = np.full((V, N, 3), np.nan)
        for t in range(V):
            x, y = tracks[t, :, 0], tracks[t, :, 1]
            ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
            if vis is not None:
                ok &= np.asarray(vis[t], bool)
            z = np.full(N, np.nan); z[ok] = self._bilinear(depth_z[t], x[ok], y[ok])
            Xc = np.stack([(x - K[t, 0, 2]) / K[t, 0, 0] * z, (y - K[t, 1, 2]) / K[t, 1, 1] * z, z], -1)
            Xw[t] = _apply_T(c2w[t], Xc)
        wd = (Xw - Xw[0:1]) @ c2w[0][:3, :3]
        return dict(w_disp=wd, X_world=Xw, x_pred=tracks, cam_pose=np.asarray(c2w), K=K)


_NEW_ADAPTERS = {"opend4rt": OpenD4RTAdapter, "trackcraft3r": TrackCraft3RAdapter, "4rc": FourRCAdapter,
                 "more": MoReAdapter, "lifted": LiftedTrackerAdapter}


def get_adapter(name, **kw):
    """Like get_model(), including the new adapters."""
    if name in _NEW_ADAPTERS:
        return _NEW_ADAPTERS[name](**kw)
    return get_model(name, **kw)
