#!/usr/bin/env python3
"""VGGT activation cache and interchange patching (unified plan 3.6, U4b), xm venv. Also gate G0h (python vggt_hooks.py).

Token layout (vggt/models/aggregator.py, checked 2026-10-19): per frame s, tokens = [camera (1), register (4), patches
(37*37)], patch_start_idx = 5, PT = 1374 tokens per frame. aa_order = [frame, global], 24 frame + 24 global blocks,
16 heads x 64, dim 1024. Frame blocks see (S, PT, C); global blocks see (1, S*PT, C). Both flatten to the same
canonical index s*PT + j, so one position convention serves both block types. The camera head reads
aggregated_tokens_list[-1][:, :, 0] (camera token of every frame; camera_head.py line 89).
Components: ('fh', i, h) / ('gh', i, h)  per-head output (input of attn.proj) of frame / global block i
            ('fr', i, None) / ('gr', i, None)  residual stream after frame / global block i
G0h: (1) patching the camera token (index 0 of every frame) at all 48 blocks from D into D reproduces D (pose_enc,
depth, world_points) to 1e-5 (fp32); (2) positive check that index 0 is the token the camera head reads: patching the
camera tokens of the LAST frame and global block outputs from C into D reproduces C's pose_enc to 1e-4.
"""
import glob, json, os, sys, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from PIL import Image

E5_BIG = os.environ.get("E5_BIG", "/data2/DATA/rohit_sinha/e5/big")
HUB = os.path.join(E5_BIG, "models", "hf", "hub")
FRAMES = [0, 4, 8, 12, 16, 20, 23]
PT = 5 + 37 * 37


def ckpt_paths():
    v = glob.glob(os.path.join(HUB, "models--facebook--VGGT-1B", "snapshots", "*", "model.pt"))[0]
    t = glob.glob(os.path.join(HUB, "models--facebook--VGGT_tracker_fixed", "snapshots", "*", "model_tracker_fixed_e20.pt"))[0]
    return v, t


def load_vggt(track=True):
    from vggt.models.vggt import VGGT  # noqa
    vp, tp = ckpt_paths()
    m = VGGT(); sd = torch.load(vp, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd); sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    if track:
        tsd = torch.load(tp, map_location="cpu", weights_only=False); tsd = tsd.get("model", tsd)
        tsd = {k[7:] if k.startswith("module.") else k: v for k, v in tsd.items()}
        m.track_head.load_state_dict({k[len("track_head."):]: v for k, v in tsd.items() if k.startswith("track_head.")}, strict=True)
    return m.cuda().eval()


def load_images(scene_dir, frames=FRAMES):
    ims = [np.asarray(Image.open(os.path.join(scene_dir, f"rgba_{f:05d}.png")).convert("RGB").resize((518, 518), Image.BICUBIC)) for f in frames]
    return torch.as_tensor(np.stack(ims)).permute(0, 3, 1, 2).float().div(255).cuda()


@dataclass
class Patch:
    comp: Tuple
    pos: torch.Tensor
    src: Dict
    src_pos: Optional[torch.Tensor] = None


class VGGTHooks:
    def __init__(self, model):
        self.model = model; agg = model.aggregator
        assert list(agg.aa_order) == ["frame", "global"]
        assert agg.patch_start_idx == 5, agg.patch_start_idx
        self.blocks = {"f": agg.frame_blocks, "g": agg.global_blocks}
        self.HD = agg.global_blocks[0].attn.head_dim; self.NH = agg.global_blocks[0].attn.num_heads
        self.capture = None; self.ph = {}; self.pr = {}
        for t, bl in self.blocks.items():
            for i, b in enumerate(bl):
                b.attn.proj.register_forward_pre_hook(self._pre(t, i)); b.register_forward_hook(self._post(t, i))

    def _apply(self, xf, plist, key):
        for h, pt in plist:
            sp = pt.src_pos if pt.src_pos is not None else pt.pos
            sv = pt.src[key]
            if h is None:
                xf[pt.pos] = sv[sp].to(xf.dtype)
            else:
                sl = slice(h * self.HD, (h + 1) * self.HD); xf[pt.pos, sl] = sv[sp, sl].to(xf.dtype)

    def _pre(self, t, i):
        def hook(mod, inp):
            x = inp[0]
            if self.capture is not None:
                self.capture[(t + "h", i)] = x.detach().reshape(-1, x.shape[-1]).clone()
            pl = self.ph.get((t, i))
            if not pl:
                return None
            xf = x.reshape(-1, x.shape[-1]).clone(); self._apply(xf, pl, (t + "h", i)); return (xf.reshape(x.shape),)
        return hook

    def _post(self, t, i):
        def hook(mod, inp, out):
            if self.capture is not None:
                self.capture[(t + "r", i)] = out.detach().reshape(-1, out.shape[-1]).clone()
            pl = self.pr.get((t, i))
            if not pl:
                return None
            xf = out.reshape(-1, out.shape[-1]).clone(); self._apply(xf, pl, (t + "r", i)); return xf.reshape(out.shape)
        return hook

    def run(self, images, capture=False, patches=(), query_points=None, amp=True):
        self.capture = {} if capture else None; self.ph = {}; self.pr = {}
        for pt in patches:
            kind, i, h = pt.comp
            (self.ph if kind[1] == "h" else self.pr).setdefault((kind[0], i), []).append((h, pt))
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = self.model(images, query_points=query_points)
            cache = self.capture
        finally:
            self.capture = None; self.ph = {}; self.pr = {}
        return out, cache


def camera_positions(S, device="cuda"):
    return torch.arange(S, device=device) * PT


def register_positions(S, device="cuda"):
    return (torch.arange(S, device=device)[:, None] * PT + torch.arange(1, 5, device=device)[None]).reshape(-1)


def gate_g0h(scene="/data2/DATA/rohit_sinha/e5/big/attr/factorial/v2/469",
             out=os.path.join(E5_BIG, "results", "attr_mech", "g0h.json"), gates=os.path.join(E5_BIG, "results", "attr_gates.json")):
    import fcntl
    t0 = time.time(); m = load_vggt(track=False); H = VGGTHooks(m)
    imD, imC = load_images(os.path.join(scene, "D")), load_images(os.path.join(scene, "C"))
    S = imD.shape[0]; cam = camera_positions(S)
    keys = ("pose_enc", "depth", "world_points")
    oD, cD = H.run(imD, capture=True, amp=False)
    comps = [(t + "r", i, None) for t in "fg" for i in range(24)] + [(t + "h", i, h) for t in "fg" for i in range(24) for h in range(16)]
    oP, _ = H.run(imD, patches=[Patch(c, cam, cD) for c in comps], amp=False)
    d1 = {k: float((oD[k] - oP[k]).abs().max()) for k in keys}
    oC, cC = H.run(imC, capture=True, amp=False)
    oX, _ = H.run(imD, patches=[Patch(("fr", 23, None), cam, cC), Patch(("gr", 23, None), cam, cC)], amp=False)
    d2 = float((oX["pose_enc"] - oC["pose_enc"]).abs().max()); dDC = float((oD["pose_enc"] - oC["pose_enc"]).abs().max())
    # register-token positions as a second identity check
    oR, _ = H.run(imD, patches=[Patch(c, register_positions(S), cD) for c in comps if c[0][1] == "r"], amp=False)
    d3 = {k: float((oD[k] - oR[k]).abs().max()) for k in keys}
    ok = max(d1.values()) <= 1e-5 and d2 <= 1e-4 and dDC > 1e-3 and max(d3.values()) <= 1e-5
    R = {"gate": "G0h", "status": "pass" if ok else "fail", "patch_start_idx": 5, "tokens_per_frame": PT,
         "camera_token_index": 0, "identity_cam_DtoD_max": d1, "identity_reg_DtoD_max": d3,
         "final_cam_tokens_CtoD_vs_C_pose_enc_max": d2, "pose_enc_D_vs_C_max": dDC,
         "criteria": "fp32; camera-token (index 0 per frame) patch D->D at all 48 blocks (heads + residuals) equals base to 1e-5; "
                     "final frame+global camera tokens C->D reproduce C pose_enc to 1e-4",
         "scene": scene, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": round(time.time() - t0, 1),
         "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(R, open(out, "w"), indent=1)
    with open(gates + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        g = json.load(open(gates)) if os.path.exists(gates) else {}
        g["G0h"] = R; json.dump(g, open(gates + ".tmp", "w"), indent=1); os.replace(gates + ".tmp", gates)
    print(json.dumps(R))


if __name__ == "__main__":
    gate_g0h()
