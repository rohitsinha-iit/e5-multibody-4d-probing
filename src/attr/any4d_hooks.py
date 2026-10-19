"""Any4D activation cache, interchange patching, head edits and key-region attention masking (unified plan 3.5).

Token layout of Any4D info_sharing (uniception MultiViewAlternatingAttentionTransformerIFR, aat_ifr_24_layers):
  V views x P = 37*37 = 1369 tokens per view (view 0 first), then 1 learned scale token (index V*P).
  Even blocks: global attention over (1, V*P + 1, 768).  Odd blocks: frame attention over (V, P, 768), scale token
  held out of the block (it passes through unchanged).
Canonical positions are FLAT token indices: view v token j -> v*P + j, scale token -> V*P. In frame blocks the scale
index does not exist and is silently ignored (both for caching and for patching).
Components: ('h', block, head)  per-head attention output = input of attn.proj, channels head*64:(head+1)*64
            ('r', block, None)  residual stream after the block (block output)
Usage:
  H = Any4DHooks(model)                       # installs hooks once; inert until configured
  out, cache = H.run(views, capture=True)     # cache = {('h',b): (Ntok,768), ('r',b): (Ntok,768)} on GPU
  H.run(views, patches=[Patch(('h',2,1), pos, cache_src)])
  H.run(views, edits={(b,h): fn})             # fn(x_head (n,64), flat positions) -> new x_head
  H.run(views, masks={b: KeyMask(heads, key_moving, query_views)})
The model stays in its fused SDPA path for every head that is not masked; masked heads are recomputed explicitly
with keys on flagged tokens removed and the softmax renormalised over the remaining keys.
"""
import os, sys, time
from dataclasses import dataclass, field
from types import MethodType
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

E5_ROOT = os.environ.get("E5_ROOT", "/data2/DATA/rohit_sinha/e5")
ANY4D = os.path.realpath(os.environ.get("ANY4D_DIR", os.path.join(E5_ROOT, "Any4D")))
P_TOK = 37 * 37
GRID = 37
FIVE = [(2, 1), (22, 9), (17, 10), (14, 6), (8, 11)]
FRAMES = [0, 4, 8, 12, 16, 20, 23]

CFG = {"path": "configs/train.yaml",
       "config_overrides": ["machine=local", "model=any4d", "model.encoder.uses_torch_hub=false", "model/task=images_only"],
       "checkpoint_path": "checkpoints/any4d_4v_combined.pth", "trained_with_amp": True, "data_norm_type": "dinov2"}


def load_any4d(device="cuda", ckpt=None):
    """Load Any4D exactly as e5_causal.py / e5_xm_any4d.py (cwd must allow relative configs; we chdir to ANY4D)."""
    sys.path.insert(0, os.path.join(ANY4D, "scripts"))
    os.chdir(ANY4D)
    from demo_inference import init_inference_model  # noqa
    from any4d.utils.misc import seed_everything  # noqa
    seed_everything(0)
    cfg = dict(CFG, path=os.path.join(ANY4D, CFG["path"]))
    model = init_inference_model(cfg, ckpt or os.path.join(ANY4D, CFG["checkpoint_path"]), torch.device(device))
    model.eval()
    return model


def load_views(scene_dir, frames=FRAMES):
    from any4d.utils.image import load_images  # noqa
    return load_images([os.path.join(scene_dir, f"rgba_{f:05d}.png") for f in frames], resize_mode="square", size=518,
                       norm_type=CFG["data_norm_type"], patch_size=14)


@dataclass
class Patch:
    comp: Tuple            # ('h', b, h) or ('r', b, None)
    pos: torch.Tensor      # flat destination positions (long, on device)
    src: Dict              # source cache (from run(capture=True))
    src_pos: Optional[torch.Tensor] = None   # flat source positions (default = pos); used by the warp-aligned analysis


@dataclass
class KeyMask:
    heads: Sequence[int]
    key_moving: torch.Tensor          # bool (V*P,) flagged keys (view tokens only)
    query_views: Sequence[int] = ()   # views whose queries are restricted (default: views 1..V-1)


class Any4DHooks:
    def __init__(self, model, cache_dtype=None):
        self.model = model
        self.blocks = model.info_sharing.self_attention_blocks
        self.NB = len(self.blocks); self.NH = self.blocks[0].attn.num_heads; self.HD = self.blocks[0].attn.head_dim
        self.C = self.NH * self.HD
        self.cache_dtype = cache_dtype      # None = keep activation dtype (exact); torch.float16 to halve memory
        self._reset()
        for l, b in enumerate(self.blocks):
            b.attn.proj.register_forward_pre_hook(self._proj_prehook(l))
            b.register_forward_hook(self._block_hook(l))
            b.attn.forward = MethodType(_make_attn_forward(self, l), b.attn)

    # ------------------------------------------------------------------ state
    def _reset(self):
        self.capture = None; self.patch_h = {}; self.patch_r = {}; self.edits = {}; self.masks = {}
        self.capture_only = None

    @staticmethod
    def _flat_index(pos, n):
        return pos[pos < n] if pos is not None else None

    def _apply_patch(self, xf, plist, l, kind):
        for (head, pt) in plist:
            n = xf.shape[0]
            keep = pt.pos < n
            dst = pt.pos[keep]
            srcp = (pt.src_pos if pt.src_pos is not None else pt.pos)[keep]
            sv = pt.src[(kind, l)]
            keep2 = srcp < sv.shape[0]
            dst, srcp = dst[keep2], srcp[keep2]
            if head is None:
                xf[dst] = sv[srcp].to(xf.dtype)
            else:
                sl = slice(head * self.HD, (head + 1) * self.HD)
                xf[dst, sl] = sv[srcp, sl].to(xf.dtype)

    def _proj_prehook(self, l):
        def hook(mod, inp):
            x = inp[0]
            if self.capture is not None and (self.capture_only is None or ("h", l) in self.capture_only):
                self.capture[("h", l)] = x.detach().reshape(-1, x.shape[-1]).to(self.cache_dtype or x.dtype).clone()
            pl = self.patch_h.get(l); ed = [(k, f) for k, f in self.edits.items() if k[0] == l]
            if not pl and not ed:
                return None
            xf = x.reshape(-1, x.shape[-1]).clone()
            if pl:
                self._apply_patch(xf, pl, l, "h")
            for (_, hh), fn in ed:
                sl = slice(hh * self.HD, (hh + 1) * self.HD)
                xf[:, sl] = fn(xf[:, sl].float(), xf.shape[0]).to(xf.dtype)
            return (xf.reshape(x.shape),)
        return hook

    def _block_hook(self, l):
        def hook(mod, inp, out):
            if self.capture is not None and (self.capture_only is None or ("r", l) in self.capture_only):
                self.capture[("r", l)] = out.detach().reshape(-1, out.shape[-1]).to(self.cache_dtype or out.dtype).clone()
            pl = self.patch_r.get(l)
            if not pl:
                return None
            xf = out.reshape(-1, out.shape[-1]).clone()
            self._apply_patch(xf, pl, l, "r")
            return xf.reshape(out.shape)
        return hook

    # ------------------------------------------------------------------ public
    def configure(self, capture=False, capture_only=None, patches=(), edits=None, masks=None):
        self._reset()
        self.capture = {} if capture else None
        self.capture_only = set(capture_only) if capture_only else None
        for pt in patches:
            kind, b, h = pt.comp
            (self.patch_h if kind == "h" else self.patch_r).setdefault(b, []).append((h, pt))
        self.edits = dict(edits or {})
        self.masks = dict(masks or {})

    def run(self, views, capture=False, capture_only=None, patches=(), edits=None, masks=None, use_amp=True,
            seed=0, extract=True):
        """One forward. Returns (outputs dict of torch tensors, cache or None)."""
        from demo_inference import sample_inference  # noqa
        self.configure(capture, capture_only, patches, edits, masks)
        try:
            torch.manual_seed(seed)
            with torch.no_grad():
                res = sample_inference(self.model, views, torch.device("cuda"), use_amp=use_amp)
            cache = self.capture
        finally:
            self._reset()
        out = extract_outputs(res, len(views)) if extract else res
        return out, cache


def _make_attn_forward(H, l):
    """Copy of uniception Attention.forward with optional explicit masked recomputation for selected heads."""
    def forward(self, x, xpos=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        if self.custom_positional_encoding is not None:
            q = self.custom_positional_encoding(q, xpos)
            k = self.custom_positional_encoding(k, xpos)
        if self.use_scalable_softmax:
            q = q * torch.log(torch.tensor(N, device=q.device))
        if self.use_entropy_scaling:
            sf = torch.sqrt((self.entropy_scaling_growth_factor * torch.log(torch.tensor(N, device=q.device)))
                            / torch.log(torch.tensor(self.base_token_count_for_entropy_scaling, device=q.device)))
            q = q * sf
        if self.fused_attn:
            o = F.scaled_dot_product_attention(q, k, v, dropout_p=(self.attn_drop.p if self.training else 0.0), scale=self.scale)
        else:
            a = (q * self.scale) @ k.transpose(-2, -1)
            o = self.attn_drop(a.softmax(dim=-1)) @ v
        km = H.masks.get(l)
        if km is not None and len(km.heads):
            o = _masked_heads(self, q, k, v, o, km, B, N)
        o = o.transpose(1, 2).reshape(B, N, -1)
        o = self.proj(o)
        return self.proj_drop(o)
    return forward


def _masked_heads(attn, q, k, v, o, km, B, N, chunk=4096):
    o = o.clone()
    flags = km.key_moving
    VP = flags.numel(); V = VP // P_TOK
    qv = list(km.query_views) if len(km.query_views) else list(range(1, V))
    if B == 1:   # global block: N = V*P (+1 scale)
        kf = torch.zeros(N, dtype=torch.bool, device=q.device); kf[:VP] = flags
        rows = torch.cat([torch.arange(vv * P_TOK, (vv + 1) * P_TOK, device=q.device) for vv in qv])
        jobs = [(0, rows, kf)]
    else:        # frame block: batch = V, N = P
        jobs = [(vv, torch.arange(P_TOK, device=q.device), flags[vv * P_TOK:(vv + 1) * P_TOK]) for vv in qv]
    for bi, rows, kf in jobs:
        if not kf.any() or kf.all():
            continue
        for h in km.heads:
            kh = k[bi, h].float(); vh = v[bi, h].float()
            for s in range(0, rows.numel(), chunk):
                r = rows[s:s + chunk]
                lg = (q[bi, h, r].float() * attn.scale) @ kh.T
                lg.masked_fill_(kf[None, :], float("-inf"))
                o[bi, h, r] = (lg.softmax(-1) @ vh).to(o.dtype)
    return o


def extract_outputs(res, V):
    """Torch outputs on GPU: pts3d_0 (518,518,3), scene_flow (V,518,518,3) (view 0 zero), cam_pose (V,4,4) c2w,
    quats/trans raw, and K lazily (numpy) via to_numpy()."""
    from any4d.utils.geometry import quaternion_to_rotation_matrix  # noqa
    pr = [res[f"pred{i+1}"] for i in range(V)]
    pts0 = pr[0]["pts3d"][0].float()
    sf = torch.stack([torch.zeros_like(pts0)] + [pr[i]["scene_flow"][0].float() for i in range(1, V)])
    poses = []
    for i in range(V):
        R = quaternion_to_rotation_matrix(pr[i]["cam_quats"][0].float())
        T = torch.eye(4, device=pts0.device); T[:3, :3] = R.to(pts0.device); T[:3, 3] = pr[i]["cam_trans"][0].float()
        poses.append(T)
    out = {"pts3d_0": pts0, "scene_flow": sf, "cam_pose": torch.stack(poses), "_ray": [p["ray_directions"][0] for p in pr]}
    if "conf" in pr[0]:
        out["conf0"] = pr[0]["conf"][0].float()
    return out


def to_numpy(out):
    from any4d.utils.geometry import recover_pinhole_intrinsics_from_ray_directions  # noqa
    o = {k: v.detach().cpu().numpy() for k, v in out.items() if torch.is_tensor(v)}
    o["K"] = np.stack([recover_pinhole_intrinsics_from_ray_directions(r.float().cpu()).numpy() for r in out["_ray"]])
    return o


def all_components(NB=24, NH=12, heads=True, resid=True):
    comps = []
    for b in range(NB):
        if heads:
            comps += [("h", b, h) for h in range(NH)]
        if resid:
            comps.append(("r", b, None))
    return comps


def comp_name(c):
    return f"b{c[1]}h{c[2]}" if c[0] == "h" else f"b{c[1]}r"


def view_positions(views, V, device="cuda", scale=False):
    p = [torch.arange(v * P_TOK, (v + 1) * P_TOK, device=device) for v in views]
    if scale:
        p.append(torch.tensor([V * P_TOK], device=device))
    return torch.cat(p)


def token_mask_from_pixels(mask512, thr=0.5):
    """(T,512,512) bool pixel mask -> (T, 1369) bool token mask: fraction of moving pixels in the token >= thr, after
    the same square 518 resize the views get (nearest)."""
    m = torch.as_tensor(np.asarray(mask512), dtype=torch.float32)
    if m.ndim == 2:
        m = m[None]
    m = F.interpolate(m[:, None], size=(518, 518), mode="nearest")[:, 0]
    f = F.avg_pool2d(m[:, None], 14)[:, 0]   # (T,37,37)
    return (f >= thr).reshape(m.shape[0], -1)


def max_abs_diff(a, b, keys=("pts3d_0", "scene_flow", "cam_pose")):
    return {k: float((a[k] - b[k]).abs().max()) for k in keys}
