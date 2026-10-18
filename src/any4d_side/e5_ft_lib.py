"""Item 10: head-targeted and decoder-side fine-tuning of Any4D. Shared library (place next to demo_inference.py).

Variants (all start from checkpoints/any4d_4v_combined.pth; everything else frozen):
  heads    (a) LoRA rank r on the qkv rows and proj input columns of the five supply heads only
               (2:1, 22:9, 17:10, 14:6, 8:11). Every other weight, including the other heads' slices, is frozen.
  sf       (b) full fine-tune of the scene-flow DPT head (feature + regressor) and its adaptor.
  heads_sf (c) (a) + (b).
  ctrl     (d) control: the same LoRA on 5 random heads matched in block (same 5 blocks, a different head in
               each block, so 4 global-attention heads + 1 frame-attention head, at the same depths).
  gain     (e) one learned scalar gain per supply head on its attention output (5 parameters).
Head slices follow uniception's Attention: qkv output index = s * dim + h * head_dim + d (s = q, k, v),
proj input index = h * head_dim + d. LoRA B is zero-initialised, so every variant starts exactly at the base model.
All adapters fold into ordinary weights (merge_adapters), so a fine-tuned checkpoint has the base format and every
existing evaluation script runs on it unchanged via its checkpoint path.

Loss (Any4D paper, Sec. 3.2 and supp.; the official training code is not released, so this is a reimplementation):
  L_sf  = sum_p w_p || flog(F_gt / z_gt) - flog(F_pred / z_pred) || / sum_p w_p,  w = 10 on moving pixels, else 1,
          flog(x) = x / |x| * log(1 + |x|) (MapAnything apply_log_to_norm), z = mean distance of the view-0
          pointmap from the origin (GT: Kubric depth; pred: predicted pts3d_0, detached), averaged over views 1..V-1.
  L_geo = distillation to the frozen model (same weights with adapters disabled): flog distance of world pointmaps
          and |log depth| differences (normalised by the teacher scale), pose translation / teacher scale, quaternion
          distance, and |log metric scale|. Zero by construction for the sf variant (transformer and geometry heads
          untouched), so it is skipped there.
"""
import json, math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FIVE = [(2, 1), (22, 9), (17, 10), (14, 6), (8, 11)]
TEST_FRAMES = [0, 4, 8, 12, 16, 20, 23]
VARIANTS = ["heads", "sf", "heads_sf", "ctrl", "gain"]
BASE_CKPT = "checkpoints/any4d_4v_combined.pth"
CFG = {"path": "configs/train.yaml",
       "config_overrides": ["machine=local", "model=any4d", "model.encoder.uses_torch_hub=false", "model/task=images_only"],
       "trained_with_amp": True, "data_norm_type": "dinov2"}


def control_heads(seed=10, five=FIVE, nh=12):
    """One random different head in each supply head's block (seeded, fixed)."""
    rng = np.random.default_rng(seed)
    return [(b, int(rng.choice([x for x in range(nh) if x != h]))) for b, h in five]


def variant_heads(variant):
    return {"heads": FIVE, "heads_sf": FIVE, "gain": FIVE, "ctrl": control_heads(), "sf": []}[variant]


# ----------------------------------------------------------------------------------------- adapters
class HeadQKV(nn.Module):
    """qkv Linear plus a rank-r update restricted to the q, k and v rows of the given heads."""

    def __init__(self, base, heads, head_dim, r, alpha):
        super().__init__()
        dev = base.weight.device
        self.base = base; self.heads = list(heads); self.enabled = True
        dim = base.in_features
        idx = [s * dim + h * head_dim + d for h in heads for s in range(3) for d in range(head_dim)]
        self.register_buffer("idx", torch.tensor(idx, dtype=torch.long, device=dev), persistent=False)
        self.A = nn.Parameter(torch.empty(r, dim, device=dev)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(len(idx), r, device=dev)); self.scale = alpha / r

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        d = F.linear(F.linear(x, self.A), self.B) * self.scale
        return y.index_add(-1, self.idx, d.to(y.dtype))

    @torch.no_grad()
    def merged(self):
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        W = self.base.weight.detach().clone().float()
        W[self.idx] += (self.B.float() @ self.A.float()) * self.scale
        lin.weight.copy_(W)
        if self.base.bias is not None:
            lin.bias.copy_(self.base.bias.detach())
        return lin.to(self.base.weight.device, self.base.weight.dtype)


class HeadProj(nn.Module):
    """proj Linear plus (optional) rank-r update on the input columns of the given heads and (optional) a scalar
    gain per head on those columns: y = W x + W[:, c] ((g - 1) x_c) + B A x_c."""

    def __init__(self, base, heads, head_dim, r=8, alpha=16, lora=True, gain=False):
        super().__init__()
        dev = base.weight.device
        self.base = base; self.heads = list(heads); self.enabled = True; self.head_dim = head_dim
        cols = [h * head_dim + d for h in heads for d in range(head_dim)]
        self.register_buffer("cols", torch.tensor(cols, dtype=torch.long, device=dev), persistent=False)
        self.lora = lora; self.use_gain = gain
        if lora:
            self.A = nn.Parameter(torch.empty(r, len(cols), device=dev)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev)); self.scale = alpha / r
        if gain:
            self.g = nn.Parameter(torch.ones(len(heads), device=dev))

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        xc = x.index_select(-1, self.cols)
        if self.use_gain:
            gx = (self.g - 1).repeat_interleave(self.head_dim)
            y = y + F.linear(xc * gx.to(xc.dtype), self.base.weight.index_select(1, self.cols)).to(y.dtype)
        if self.lora:
            y = y + (F.linear(F.linear(xc, self.A), self.B) * self.scale).to(y.dtype)
        return y

    @torch.no_grad()
    def merged(self):
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        W = self.base.weight.detach().clone().float()
        if self.use_gain:
            W[:, self.cols] *= self.g.float().repeat_interleave(self.head_dim)[None, :]
        if self.lora:
            W[:, self.cols] += (self.B.float() @ self.A.float()) * self.scale
        lin.weight.copy_(W)
        if self.base.bias is not None:
            lin.bias.copy_(self.base.bias.detach())
        return lin.to(self.base.weight.device, self.base.weight.dtype)


def blocks_of(model):
    return model.info_sharing.self_attention_blocks


def apply_variant(model, variant, r=8, alpha=16, lr_lora=2e-4, lr_sf=2e-5, lr_gain=5e-3):
    """Freeze everything, insert the adapters, return optimizer param groups and a description."""
    assert variant in VARIANTS, variant
    for p in model.parameters():
        p.requires_grad_(False)
    blocks = blocks_of(model)
    hd = blocks[0].attn.head_dim
    heads = variant_heads(variant)
    by_block = {}
    for b, h in heads:
        by_block.setdefault(b, []).append(h)
    groups, lora_p, gain_p = [], [], []
    for b, hs in sorted(by_block.items()):
        attn = blocks[b].attn
        if variant == "gain":
            attn.proj = HeadProj(attn.proj, hs, hd, lora=False, gain=True); gain_p.append(attn.proj.g)
        else:
            attn.qkv = HeadQKV(attn.qkv, hs, hd, r, alpha); attn.proj = HeadProj(attn.proj, hs, hd, r, alpha, lora=True)
            lora_p += [attn.qkv.A, attn.qkv.B, attn.proj.A, attn.proj.B]
    if lora_p:
        groups.append(dict(params=lora_p, lr=lr_lora, name="lora"))
    if gain_p:
        groups.append(dict(params=gain_p, lr=lr_gain, name="gain"))
    if variant in ("sf", "heads_sf"):
        sfp = [p for m in (model.scene_flow_dense_head, model.scene_flow_dense_adaptor) for p in m.parameters()]
        groups.append(dict(params=sfp, lr=lr_sf, name="sf_head"))
    for g in groups:
        for p in g["params"]:
            p.requires_grad_(True)
    n = sum(p.numel() for g in groups for p in g["params"])
    return groups, dict(variant=variant, heads=heads, n_trainable=int(n), rank=r, alpha=alpha)


def adapters(model):
    return [m for m in model.modules() if isinstance(m, (HeadQKV, HeadProj))]


def set_adapters(model, enabled):
    for m in adapters(model):
        m.enabled = enabled


def trainable_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}


def load_trainable_state(model, sd):
    cur = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in sd.items():
            cur[k].copy_(v.to(cur[k].device, cur[k].dtype))


@torch.no_grad()
def merge_adapters(model):
    """Replace adapter wrappers by plain Linear layers with merged weights (state dict keys become the base keys)."""
    for blk in blocks_of(model):
        if isinstance(blk.attn.qkv, HeadQKV):
            blk.attn.qkv = blk.attn.qkv.merged()
        if isinstance(blk.attn.proj, HeadProj):
            blk.attn.proj = blk.attn.proj.merged()
    return model


def enable_checkpointing(model):
    """Activation checkpointing on every transformer block and on the three dense heads (state dict unchanged)."""
    from torch.utils.checkpoint import checkpoint

    def wrap(mod):
        f = mod.forward

        def g(*a, **k):
            if torch.is_grad_enabled():
                return checkpoint(f, *a, use_reentrant=False, **k)
            return f(*a, **k)
        mod.forward = g
    for blk in blocks_of(model):
        wrap(blk)
    for name in ("dense_head", "scene_flow_dense_head", "pose_head"):
        if hasattr(model, name):
            wrap(getattr(model, name))


# ----------------------------------------------------------------------------------------- data
def model_index_map(W=512, M=518):
    s = M / W
    return np.clip(np.round((np.arange(M) + 0.5) / s - 0.5).astype(int), 0, W - 1)


def scene_flow_gt(meta, gtz, frames, M=518):
    """GT allocentric scene flow of view-0 pixels from frame frames[0] to each frame, in the view-0 OpenCV camera
    frame (x right, y down, z forward), sampled at the model's 518 x 518 pixel centres (nearest 512 px pixel).
    Returns F (V,M,M,3), X0 (M,M,3) view-0 camera-frame points, moving (M,M) bool, norm_mask (M,M) bool."""
    import e5_kubric_post as kp
    depth = gtz["depth"][frames[0], ..., 0]; seg = gtz["segmentation"][frames[0], ..., 0]
    W, H = meta["resolution"]
    ix = model_index_map(W, M); iy = model_index_map(H, M)
    depth = depth[np.ix_(iy, ix)]; seg = seg[np.ix_(iy, ix)]
    K = np.array(meta["K"]); f0 = str(meta["frames"][frames[0]])
    C0 = np.array(meta["camera_matrix_world"][f0])
    u, v = np.meshgrid(ix + 0.5, iy + 0.5)
    d = np.stack([(u - W / 2) / K[0, 0], -(v - H / 2) / K[1, 1], -np.ones_like(u)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    Xw = (d * depth[..., None]) @ C0[:3, :3].T + C0[:3, 3]
    s2g = kp.seg_to_group(meta)
    dome = {int(s) for s, k in meta["seg_index_map"].items() if k == "dome"}
    grp = np.zeros(seg.shape, int); known = np.zeros(seg.shape, bool)
    masks = {}
    for sid, (g, k) in s2g.items():
        m = seg == sid
        if m.any():
            masks[sid] = (m, g, k); grp[m] = g; known |= m
    flip = np.array([1.0, -1.0, -1.0])
    Fs = []
    for fi in frames:
        Xt = Xw.copy(); ft = str(meta["frames"][fi])
        for sid, (m, g, k) in masks.items():
            if k >= 0:
                Ma = np.array(meta["objects"][k]["matrix_world"][f0]); Mb = np.array(meta["objects"][k]["matrix_world"][ft])
                T = Mb @ np.linalg.inv(Ma)
                Xt[m] = Xw[m] @ T[:3, :3].T + T[:3, 3]
        Fs.append(((Xt - Xw) @ C0[:3, :3]) * flip)
    X0 = ((Xw - C0[:3, 3]) @ C0[:3, :3]) * flip
    norm_mask = known & ~np.isin(seg, list(dome)) & (depth < 200)
    return np.stack(Fs).astype(np.float32), X0.astype(np.float32), (grp > 0) & known, norm_mask


class KubricFT(torch.utils.data.Dataset):
    """Item k -> one training sample: scene and 6 sorted target frames from 1..23 drawn from rng([seed, k]);
    frame 0 is always the reference. With fixed_frames, item k is scene k with those frames (validation)."""

    def __init__(self, scene_dirs, n_views=7, seed=0, fixed_frames=None, n_frames=24):
        self.scenes = list(scene_dirs); self.n_views = n_views; self.seed = seed
        self.fixed = fixed_frames; self.n_frames = n_frames

    def frames_for(self, k):
        if self.fixed is not None:
            return k % len(self.scenes), list(self.fixed)
        rng = np.random.default_rng([self.seed, k])
        si = int(rng.integers(len(self.scenes)))
        fr = [0] + sorted(int(x) for x in rng.choice(np.arange(1, self.n_frames), self.n_views - 1, replace=False))
        return si, fr

    def __len__(self):
        return 10 ** 9 if self.fixed is None else len(self.scenes)

    def __getitem__(self, k):
        from any4d.utils.image import load_images
        si, fr = self.frames_for(k)
        d = self.scenes[si]
        meta = json.load(open(os.path.join(d, "e5_meta.json")))
        z = np.load(os.path.join(d, "gt.npz"))
        gtz = {"depth": z["depth"], "segmentation": z["segmentation"]}
        Fg, X0, mov, nm = scene_flow_gt(meta, gtz, fr)
        views = load_images([os.path.join(d, f"rgba_{f:05d}.png") for f in fr], resize_mode="square", size=518,
                            norm_type="dinov2", patch_size=14)
        return dict(views=views, F=torch.from_numpy(Fg), X0=torch.from_numpy(X0), moving=torch.from_numpy(mov),
                    norm_mask=torch.from_numpy(nm), scene=os.path.basename(d), frames=fr, k=int(k))


def first(x):
    return x[0]


# ----------------------------------------------------------------------------------------- losses
def flog(x):
    n = x.norm(dim=-1, keepdim=True)
    return x / n.clamp(min=1e-8) * torch.log1p(n)


def forward(model, views, amp=True):
    with torch.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
        return model(views)


def sf_loss(preds, s, dyn_w=10.0):
    dev = preds[0]["pts3d"].device
    Fg = s["F"].to(dev); X0 = s["X0"].to(dev); mov = s["moving"].to(dev); nm = s["norm_mask"].to(dev)
    pts0 = preds[0]["pts3d"][0].float()
    z_pred = pts0[nm].norm(dim=-1).mean().detach().clamp(min=1e-6)
    z_gt = X0[nm].norm(dim=-1).mean().clamp(min=1e-6)
    w = 1.0 + (dyn_w - 1.0) * mov.float()
    tot, dyn, sta = 0.0, 0.0, 0.0
    V = len(preds)
    for i in range(1, V):
        e = (flog(Fg[i] / z_gt) - flog(preds[i]["scene_flow"][0].float() / z_pred)).norm(dim=-1)
        tot = tot + (w * e).sum() / w.sum()
        dyn = dyn + (float(e[mov].mean()) if bool(mov.any()) else 0.0)
        sta = sta + float(e[~mov].mean())
    return tot / (V - 1), dict(sf_dyn=dyn / (V - 1), sf_static=sta / (V - 1))


def geo_loss(preds, teach):
    zt = torch.stack([t["pts3d"][0].float().norm(dim=-1).mean() for t in teach]).mean().clamp(min=1e-6)
    lp = ld = lt = lq = 0.0
    for p, t in zip(preds, teach):
        lp = lp + (flog(p["pts3d"][0].float() / zt) - flog(t["pts3d"][0].float() / zt)).norm(dim=-1).mean()
        ld = ld + (torch.log(p["depth_along_ray"][0].float().clamp(min=1e-4)) -
                   torch.log(t["depth_along_ray"][0].float().clamp(min=1e-4))).abs().mean()
        lt = lt + (p["cam_trans"][0].float() - t["cam_trans"][0].float()).norm() / zt
        q1, q2 = p["cam_quats"][0].float(), t["cam_quats"][0].float()
        lq = lq + torch.minimum((q1 - q2).norm(), (q1 + q2).norm())
    V = len(preds)
    ls = (torch.log(preds[0]["metric_scaling_factor"].float().clamp(min=1e-6)) -
          torch.log(teach[0]["metric_scaling_factor"].float().clamp(min=1e-6))).abs().mean()
    tot = (lp + ld + lt + lq) / V + ls
    return tot, dict(geo_pts=float(lp) / V, geo_depth=float(ld) / V, geo_trans=float(lt) / V, geo_quat=float(lq) / V,
                     geo_scale=float(ls))


def to_device(views, dev):
    out = []
    for v in views:
        w = dict(v)
        for k in list(w.keys()):
            if torch.is_tensor(w[k]) and k not in ("true_shape", "idx", "instance"):
                w[k] = w[k].to(dev, non_blocking=True)
        out.append(w)
    return out


def preds_to_P(preds):
    """Model outputs -> the numpy dict used by e5_step3_analysis (same as e5_any4d_infer.py)."""
    from any4d.utils.geometry import quaternion_to_rotation_matrix, recover_pinhole_intrinsics_from_ray_directions
    V = len(preds)
    out = {"pts3d_0": preds[0]["pts3d"][0].float().cpu().numpy()}
    out["scene_flow"] = np.stack([np.zeros_like(out["pts3d_0"])] +
                                 [preds[i]["scene_flow"][0].float().cpu().numpy() for i in range(1, V)])
    poses, Ks, dr = [], [], []
    for i in range(V):
        R = quaternion_to_rotation_matrix(preds[i]["cam_quats"][0].float().cpu())
        T = np.eye(4); T[:3, :3] = R.numpy(); T[:3, 3] = preds[i]["cam_trans"][0].float().cpu().numpy(); poses.append(T)
        Ks.append(recover_pinhole_intrinsics_from_ray_directions(preds[i]["ray_directions"][0].float().cpu()).numpy())
        dr.append(preds[i]["depth_along_ray"][0].float().cpu().numpy().squeeze(-1))
    out["cam_pose"] = np.stack(poses); out["K"] = np.stack(Ks); out["depth_ray"] = np.stack(dr)
    return out


# ----------------------------------------------------------------------------------------- round-1 metrics
FLOW_THR = 0.1


def kubric_metrics(meta, gt, P, si, frames, ti=None):
    """Round-1 output metrics for target view ti (default: last view), frame 0 -> frames[ti], 512 px GT frame.
    slope, epe_static, epe_moving, sep_ratio exactly as e5_causal.py / e5_e8_knockout.py (300 points per group,
    rng seeded by the scene index si); mask_iou exactly as e5_step3_analysis.py (|scene flow| > 0.1)."""
    import e5_kubric_post as kp
    import e5_step3_analysis as s3
    ti = len(frames) - 1 if ti is None else ti
    W = meta["resolution"][0]; Hm = P["pts3d_0"].shape[0]; s = Hm / W; b = frames[ti]
    c = kp.correspondences(meta, gt, 0, b)
    X = kp.backproject(meta, gt["depth"][0, ..., 0], 0); xs_dense, _ = kp.project(meta, X, b)
    x1 = c["x1"]; grp = c["group"]
    xs = xs_dense[x1[:, 1].astype(int), x1[:, 0].astype(int)]
    u = np.clip(np.round((x1[:, 0] + 0.5) * s - 0.5).astype(int), 0, Hm - 1)
    v = np.clip(np.round((x1[:, 1] + 0.5) * s - 0.5).astype(int), 0, Hm - 1)
    x1p = (np.c_[u, v] + 0.5) / s - 0.5
    rs = np.random.default_rng(si); idx = s3.sample_by_group(grp, 300, rs)
    sep_gt, _ = s3.sep_and_noise(x1[idx], c["x2"][idx], grp[idx])
    off = np.array(s3.pixel_offset(P))
    xp = s3.pred_correspondences(P, ti, off)[v, u]; xp = (xp + 0.5) / s - 0.5
    err = np.linalg.norm(xp - c["x2"], axis=1); mv = grp > 0
    dt = c["x2"] - xs; dp = xp - xs; big = mv & (np.linalg.norm(dt, axis=1) > 2)
    sep_pr, _ = s3.sep_and_noise(x1p[idx], xp[idx], grp[idx])
    row = dict(slope=float((dp[big] * dt[big]).sum() / (dt[big] ** 2).sum()) if big.sum() > 20 else float("nan"),
               epe_static=float(np.median(err[~mv])), epe_moving=float(np.median(err[mv])) if mv.any() else float("nan"),
               sep_gt=float(sep_gt), sep_pred=float(sep_pr), sep_ratio=float(sep_pr / sep_gt) if sep_gt > 1 else float("nan"))
    sf = np.linalg.norm(P["scene_flow"][ti], axis=-1)
    gm = np.zeros((Hm, Hm), bool)
    yy, xx = np.nonzero(c["dense_group"] > 0)
    gm[np.clip(np.round((yy + 0.5) * s - 0.5).astype(int), 0, Hm - 1),
       np.clip(np.round((xx + 0.5) * s - 0.5).astype(int), 0, Hm - 1)] = True
    pm = sf > FLOW_THR
    inter, union = (pm & gm).sum(), (pm | gm).sum()
    row["mask_iou"] = float(inter / union) if union else float("nan")
    # threshold-free version of the same read-out: ROC AUC of |scene flow| for GT moving vs static view-0 pixels
    # (comparable with the probe's motion AUC, which uses the same moving / static split)
    row["mask_auc"] = rank_auc(sf.ravel(), gm.ravel())
    return row


def rank_auc(score, y):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort"); ranks = np.empty(y.size); ranks[order] = np.arange(1, y.size + 1)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def geometry_errors(meta, gt, P, frames):
    """Depth and pose error vs Kubric GT. Depth: ray distance at the model pixel centres (nearest 512 px pixel),
    non-dome pixels; abs_rel in metric units and after a per-scene median scale. Pose: relative pose of each view to
    view 0 (GT Blender cam-to-world converted to OpenCV): rotation error (deg), translation direction error (deg),
    relative translation error |t_pred - t_gt| / |t_gt| (metric) and after the depth median scale."""
    W, H = meta["resolution"]; M = P["depth_ray"].shape[1]
    ix = model_index_map(W, M); iy = model_index_map(H, M)
    dome = {int(s) for s, k in meta["seg_index_map"].items() if k == "dome"}
    ar, arm, scl = [], [], []
    for vi, f in enumerate(frames):
        g = gt["depth"][f, ..., 0][np.ix_(iy, ix)]; sg = gt["segmentation"][f, ..., 0][np.ix_(iy, ix)]
        m = ~np.isin(sg, list(dome)) & (g > 1e-3) & (g < 200)
        p = P["depth_ray"][vi][m]; g = g[m]
        ar.append(float(np.mean(np.abs(p - g) / g)))
        k = float(np.median(g / np.maximum(p, 1e-6))); scl.append(k)
        arm.append(float(np.mean(np.abs(k * p - g) / g)))
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    Cs = [np.array(meta["camera_matrix_world"][str(meta["frames"][f])]) @ flip for f in frames]
    k0 = float(np.median(scl))
    rot, tdir, trel, trel_s = [], [], [], []
    for vi in range(1, len(frames)):
        Tg = np.linalg.inv(Cs[0]) @ Cs[vi]; Tp = np.linalg.inv(P["cam_pose"][0]) @ P["cam_pose"][vi]
        Rd = Tg[:3, :3].T @ Tp[:3, :3]
        rot.append(float(np.degrees(np.arccos(np.clip((np.trace(Rd) - 1) / 2, -1, 1)))))
        tg, tp = Tg[:3, 3], Tp[:3, 3]
        ng, npd = np.linalg.norm(tg), np.linalg.norm(tp)
        if ng > 1e-3 and npd > 1e-6:
            tdir.append(float(np.degrees(np.arccos(np.clip(tg @ tp / (ng * npd), -1, 1)))))
            trel.append(float(np.linalg.norm(tp - tg) / ng)); trel_s.append(float(np.linalg.norm(k0 * tp - tg) / ng))
    nm = lambda a: float(np.mean(a)) if len(a) else float("nan")  # noqa: E731
    return dict(depth_absrel=nm(ar), depth_absrel_med=nm(arm), depth_scale=k0, rot_err_deg=nm(rot),
                tdir_err_deg=nm(tdir), trans_relerr=nm(trel), trans_relerr_med=nm(trel_s))
