"""Frozen Wan2.1-T2V-1.3B features for P3 X2 / X3 (P3 plan v2 section 4.1; unified plan U3).

Readout: clean latents, timestep t = 5 (TrackCraft3R convention), empty-prompt umT5 context computed ONCE
(`ctx` subcommand) and cached at $E5_BIG/models/ckpt/wan_empty_ctx.pt; T5 is never loaded afterwards.
Latents: AutoencoderKLWan latent_dist.mode(), then (z - latents_mean) / latents_std from the VAE config.
Frames: 25 frames at 480x480 (E5 clips repeat frame 23); 7 latent frames x 30 x 30 tokens = 6300 tokens, dim 1536.
Hooks (WanFeat.run):
  residual stream after the patch embedding and after each of the 30 blocks (31 states, "res");
  per-head attn1 outputs via a forward pre-hook on attn1.to_out[0], reshaped to (L, 12, 128) ("heads");
  post-RoPE q/k only through QKProbeProcessor, which stores argmax and pmax (full-softmax probability) maps from
  latent-0 queries to latent-6 keys, and the recomputed latent-0 attention output for the identity check.
Subcommands:
  ctx      compute and cache the empty-prompt context (loads umT5 once)
  g0a      gate G0a: VAE round trip PSNR on 10 clips; per-block parity vs official Wan2.1 WanModel on 2 clips
           (fp32 and bf16); q/k processor identity check. Writes $E5_BIG/results/p3_g0a.json
  extract  final-block tokens (fp16, (6300,1536)) for every clip in --clips into --feat (resumable)
"""
import argparse, glob, importlib, json, math, os, sys, time, types
import numpy as np
import torch
import torch.nn.functional as F

E5_BIG = os.environ.get("E5_BIG", "/data2/DATA/rohit_sinha/e5/big")
REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
CTX = os.path.join(E5_BIG, "models", "ckpt", "wan_empty_ctx.pt")
OFFICIAL_SRC = os.path.join(E5_BIG, "models", "src", "Wan2.1")
OFFICIAL_CKPT = os.path.join(E5_BIG, "models", "ckpt", "Wan2.1-T2V-1.3B")
T_READ = 5.0
RES = 480


def repo_path():
    """Local snapshot dir of the Diffusers repo (diffusers 0.39 queries the Hub for sharded checkpoints even with
    HF_HUB_OFFLINE=1, so models are loaded from the resolved local path)."""
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(REPO, local_files_only=True)
    except Exception:
        return snapshot_download(REPO)


def compute_empty_context(dev):
    from transformers import AutoTokenizer, UMT5EncoderModel
    tok = AutoTokenizer.from_pretrained(REPO, subfolder="tokenizer")
    te = UMT5EncoderModel.from_pretrained(REPO, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(dev)
    t = tok([""], padding="max_length", max_length=512, truncation=True, add_special_tokens=True,
            return_attention_mask=True, return_tensors="pt")
    n = int(t.attention_mask.sum())
    with torch.no_grad():
        e = te(t.input_ids.to(dev), t.attention_mask.to(dev)).last_hidden_state.float()
    ctx = torch.zeros(1, 512, e.shape[-1]); ctx[0, :n] = e[0, :n].cpu()
    os.makedirs(os.path.dirname(CTX), exist_ok=True)
    torch.save(dict(ctx=ctx, n_tokens=n, prompt="", dtype_t5="bf16", repo=REPO), CTX)
    return ctx, n


def to_video_tensor(frames_uint8, res=RES):
    """(T,H,W,3) uint8 -> (1,3,T,res,res) float in [-1,1] (bilinear, antialiased resize)."""
    x = torch.from_numpy(np.ascontiguousarray(frames_uint8)).permute(0, 3, 1, 2).float()
    if x.shape[-1] != res or x.shape[-2] != res:
        x = F.interpolate(x, size=(res, res), mode="bilinear", antialias=True, align_corners=False)
    return (x / 127.5 - 1.0).clamp(-1, 1).permute(1, 0, 2, 3)[None]


class QKProbeProcessor:
    """Self-attention processor that reproduces diffusers' WanAttnProcessor and additionally stores, for queries in
    latent frame 0, the argmax key index and its probability (full softmax over all keys) restricted to latent frame 6,
    plus the recomputed attention output for those queries (identity check against the to_out[0] input)."""

    def __init__(self, store, tokens_per_frame=900, q_frame=0, k_frame=6):
        self.store, self.tpf, self.qf, self.kf = store, tokens_per_frame, q_frame, k_frame

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        from diffusers.models.transformers.transformer_wan import _get_qkv_projections, dispatch_attention_fn
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = attn.norm_q(query); key = attn.norm_k(key)
        query = query.unflatten(2, (attn.heads, -1)); key = key.unflatten(2, (attn.heads, -1)); value = value.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            def rot(h, fc, fs):
                x1, x2 = h.unflatten(-1, (-1, 2)).unbind(-1)
                c, s = fc[..., 0::2], fs[..., 1::2]
                o = torch.empty_like(h); o[..., 0::2] = x1 * c - x2 * s; o[..., 1::2] = x1 * s + x2 * c
                return o.type_as(h)
            query = rot(query, *rotary_emb); key = rot(key, *rotary_emb)
        q0 = query[:, self.qf * self.tpf:(self.qf + 1) * self.tpf].float().transpose(1, 2)       # (B,H,900,D)
        kk = key.float().transpose(1, 2); vv = value.float().transpose(1, 2)
        p = torch.softmax(q0 @ kk.transpose(-1, -2) / math.sqrt(q0.shape[-1]), -1)              # (B,H,900,L)
        pk = p[..., self.kf * self.tpf:(self.kf + 1) * self.tpf]
        pmax, amax = pk.max(-1)
        self.store.append(dict(argmax=amax.cpu(), pmax=pmax.cpu(), out0=(p @ vv).transpose(1, 2).flatten(2).cpu()))
        hs = dispatch_attention_fn(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hs = hs.flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](hs))


class WanFeat:
    def __init__(self, dev="cuda", dtype=torch.bfloat16, vae_dtype=torch.float32, load_dit=True):
        from diffusers import AutoencoderKLWan, WanTransformer3DModel
        self.dev, self.dtype = torch.device(dev), dtype
        self.vae = AutoencoderKLWan.from_pretrained(repo_path(), subfolder="vae", torch_dtype=vae_dtype).to(self.dev).eval()
        cfg = self.vae.config
        self.lmean = torch.tensor(cfg.latents_mean, dtype=torch.float32, device=self.dev).view(1, -1, 1, 1, 1)
        self.lstd = torch.tensor(cfg.latents_std, dtype=torch.float32, device=self.dev).view(1, -1, 1, 1, 1)
        self.dit = None
        if load_dit:
            self.dit = WanTransformer3DModel.from_pretrained(repo_path(), subfolder="transformer", torch_dtype=dtype).to(self.dev).eval()
        self.ctx = torch.load(CTX)["ctx"] if os.path.exists(CTX) else None

    @torch.no_grad()
    def encode(self, video):
        z = self.vae.encode(video.to(self.dev, self.vae.dtype)).latent_dist.mode().float()
        return (z - self.lmean) / self.lstd

    @torch.no_grad()
    def decode(self, zn):
        z = zn * self.lstd + self.lmean
        return self.vae.decode(z.to(self.vae.dtype)).sample.float()

    @torch.no_grad()
    def run(self, zn, t=T_READ, want=("final",), qk_blocks=()):
        """zn: normalised latents (1,16,7,60,60). Returns dict with the requested states.
        want: 'final' (block 29 output, (L,1536)), 'res' (31 states), 'heads' (30 x (L,12,128))."""
        dit = self.dit; hooks = []; res = {}; heads = {}; qk = {}
        if "res" in want:
            hooks.append(dit.patch_embedding.register_forward_hook(lambda m, i, o: res.__setitem__(0, o.flatten(2).transpose(1, 2)[0].float().cpu())))
        for bi, blk in enumerate(dit.blocks):
            if "res" in want or ("final" in want and bi == len(dit.blocks) - 1):
                hooks.append(blk.register_forward_hook(lambda m, i, o, bi=bi: res.__setitem__(bi + 1, o if torch.is_tensor(o) else o[0])))
            if "heads" in want:
                hooks.append(blk.attn1.to_out[0].register_forward_pre_hook(
                    lambda m, i, bi=bi: heads.__setitem__(bi, i[0][0].unflatten(-1, (12, -1)).float().cpu())))
        old = {}
        for bi in qk_blocks:
            st = qk.setdefault(bi, []); old[bi] = dit.blocks[bi].attn1.processor
            dit.blocks[bi].attn1.set_processor(QKProbeProcessor(st))
        try:
            ts = torch.tensor([t], device=self.dev, dtype=torch.float32)
            dit(zn.to(self.dev, self.dtype), ts, self.ctx.to(self.dev, self.dtype), return_dict=False)
        finally:
            for h in hooks:
                h.remove()
            for bi, p in old.items():
                dit.blocks[bi].attn1.set_processor(p)
        out = {}
        if "final" in want:
            out["final"] = res[len(dit.blocks)][0]
        if "res" in want:
            out["res"] = [res[k] if k == 0 else res[k][0].float().cpu() for k in sorted(res)]
        if "heads" in want:
            out["heads"] = [heads[k] for k in sorted(heads)]
        if qk_blocks:
            out["qk"] = {k: v[0] for k, v in qk.items()}
        return out


# ------------------------------------------------------------------ official Wan2.1 code for the parity check
def load_official(dtype_attn):
    """Import wan.modules.model from the official repo without its heavy package __init__; replace flash_attention
    by an SDPA implementation (fp32 math for the fp32 check, bf16 for the bf16 check, like flash-attn)."""
    for name, sub in (("wan", ""), ("wan.modules", "modules")):
        m = types.ModuleType(name); m.__path__ = [os.path.join(OFFICIAL_SRC, "wan", sub)]; sys.modules[name] = m
    mm = importlib.import_module("wan.modules.model")

    def sdpa_flash(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None, causal=False,
                   window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, version=None):
        out_dtype = q.dtype
        cd = dtype_attn
        o = F.scaled_dot_product_attention(q.transpose(1, 2).to(cd), k.transpose(1, 2).to(cd), v.transpose(1, 2).to(cd))
        return o.transpose(1, 2).to(out_dtype)
    mm.flash_attention = sdpa_flash
    model = mm.WanModel.from_pretrained(OFFICIAL_CKPT)
    return model.eval().requires_grad_(False)


@torch.no_grad()
def official_states(model, zn, ctx, dev, autocast_bf16):
    st = {}
    hooks = [model.patch_embedding.register_forward_hook(lambda m, i, o: st.__setitem__(0, o.flatten(2).transpose(1, 2)[0].float().cpu()))]
    for bi, blk in enumerate(model.blocks):
        hooks.append(blk.register_forward_hook(lambda m, i, o, bi=bi: st.__setitem__(bi + 1, o[0].float().cpu())))
    n = int((ctx[0].abs().sum(-1) > 0).sum())
    x = zn[0].to(dev, torch.float32); L = x.shape[1] // 1 * (x.shape[2] // 2) * (x.shape[3] // 2)
    t = torch.tensor([T_READ], device=dev)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_bf16):
            model([x], t, [ctx[0, :n].to(dev)], seq_len=L)
    finally:
        for h in hooks:
            h.remove()
    return [st[k] for k in sorted(st)]


def compare(a_list, b_list):
    rows = []
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        a = a.double().flatten(); b = b.double().flatten()
        rows.append(dict(state=i, rel_err=float((a - b).norm() / b.norm()), cos=float(F.cosine_similarity(a, b, dim=0))))
    return rows


def psnr(x, y):
    mse = float(((x.clamp(-1, 1) - y) ** 2).mean()) / 4.0      # [-1,1] -> [0,1] scale
    return 10 * math.log10(1.0 / max(mse, 1e-12))


def clip_frames(clips_root, name):
    return np.load(os.path.join(clips_root, name, "rgb.npy"))


def gate_g0a(a):
    dev = torch.device("cuda")
    names = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(a.clips, "*", "rgb.npy")))
    rng = np.random.default_rng(0); sel = sorted(rng.choice(names, min(10, len(names)), replace=False).tolist())
    rep = dict(gate="G0a", criteria="VAE round-trip PSNR >= 30 dB on 10 clips; per-block rel err < 1e-2 and cos > 0.999 (fp32, bf16)",
               t=T_READ, res=RES, clips=sel)
    wf = WanFeat(dev, dtype=torch.float32)
    ps = []; zs = {}
    for n in sel:
        v = to_video_tensor(clip_frames(a.clips, n)); zn = wf.encode(v); rec = wf.decode(zn).cpu()
        ps.append(psnr(rec, v)); zs[n] = zn.cpu()
        print("vae", n, tuple(zn.shape), round(ps[-1], 2), flush=True)
    rep["vae_psnr"] = dict(zip(sel, ps)); rep["vae_psnr_min"] = float(min(ps)); rep["vae_pass"] = bool(min(ps) >= 30)
    rep["latent_shape"] = list(zs[sel[0]].shape)
    del wf.vae; torch.cuda.empty_cache()
    par = sel[:2]; res_d = {}
    # diffusers fp32 + q/k identity check on block 0 and 15
    for n in par:
        out = wf.run(zs[n], want=("res", "heads"), qk_blocks=(0, 15))
        res_d[("fp32", n)] = out["res"]
        idc = {}
        for bi, q in out["qk"].items():
            h = out["heads"][bi][:900].flatten(1)            # to_out[0] input rows of latent-0 queries
            idc[bi] = float((q["out0"][0] - h).abs().max() / h.abs().max())
        rep.setdefault("qk_identity_maxrel", {})[n] = idc
        rep.setdefault("qk_pmax_mean", {})[n] = {bi: float(q["pmax"].mean()) for bi, q in out["qk"].items()}
    wf.dit = wf.dit.to(torch.bfloat16); wf.dtype = torch.bfloat16
    for n in par:
        res_d[("bf16", n)] = wf.run(zs[n], want=("res",))["res"]
    del wf.dit; torch.cuda.empty_cache()
    ctx = torch.load(CTX)["ctx"]
    rep["parity"] = {}
    for mode, dt_attn, ac in (("fp32", torch.float32, False), ("bf16", torch.bfloat16, True)):
        om = load_official(dt_attn).to(dev)
        for n in par:
            os_ = official_states(om, zs[n], ctx, dev, ac)
            rows = compare(res_d[(mode, n)], os_)
            rep["parity"][f"{mode}/{n}"] = dict(max_rel_err=max(r["rel_err"] for r in rows), min_cos=min(r["cos"] for r in rows),
                                                 final_rel_err=rows[-1]["rel_err"], final_cos=rows[-1]["cos"], per_state=rows)
            if mode == "bf16":   # also diffusers bf16 vs official fp32 reference (informational)
                pass
            print("parity", mode, n, rep["parity"][f"{mode}/{n}"]["max_rel_err"], rep["parity"][f"{mode}/{n}"]["min_cos"], flush=True)
        if mode == "fp32":
            ref32 = {n: official_states(om, zs[n], ctx, dev, False) for n in par}
        del om; torch.cuda.empty_cache()
    rep["bf16_diffusers_vs_official_fp32"] = {n: dict(max_rel_err=max(r["rel_err"] for r in compare(res_d[("bf16", n)], ref32[n])),
                                                      min_cos=min(r["cos"] for r in compare(res_d[("bf16", n)], ref32[n]))) for n in par}
    rep["parity_pass"] = {m: bool(all(v["max_rel_err"] < 1e-2 and v["min_cos"] > 0.999 for k, v in rep["parity"].items() if k.startswith(m)))
                          for m in ("fp32", "bf16")}
    rep["qk_identity_pass"] = bool(all(v < 1e-2 for d in rep["qk_identity_maxrel"].values() for v in d.values()))
    rep["passed"] = bool(rep["vae_pass"] and all(rep["parity_pass"].values()))
    os.makedirs(os.path.join(E5_BIG, "results"), exist_ok=True)
    json.dump(rep, open(os.path.join(E5_BIG, "results", "p3_g0a.json"), "w"), indent=1)
    print("G0a", rep["passed"], rep["vae_psnr_min"], rep["parity_pass"], rep["qk_identity_pass"], flush=True)


def extract(a):
    dev = torch.device("cuda"); os.makedirs(a.feat, exist_ok=True)
    names = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(a.clips, "*", "labels.npz")))
    todo = [n for n in names if not os.path.exists(os.path.join(a.feat, n + ".npy"))]
    if a.shard_n > 1:
        todo = todo[a.shard_i::a.shard_n]
    print("clips", len(names), "todo", len(todo), "dit dtype", a.dit_dtype, flush=True)
    dt = dict(fp32=torch.float32, bf16=torch.bfloat16)[a.dit_dtype]
    json.dump(dict(dit_dtype=a.dit_dtype, vae_dtype="fp32", t=T_READ, res=RES, state="block 29 output", storage="fp16",
                   note="fp32 DiT because G0a bf16 parity failed (fp32 parity passed)"),
              open(os.path.join(a.feat, "_extract_config.json"), "w"), indent=1)
    wf = WanFeat(dev, dtype=dt)
    t0 = time.time()
    for i, n in enumerate(todo):
        if os.path.exists(os.path.join(a.feat, n + ".npy")):
            continue
        v = to_video_tensor(clip_frames(a.clips, n))
        zn = wf.encode(v)
        f = wf.run(zn, want=("final",))["final"].to(torch.float16).cpu().numpy()
        assert f.shape == (6300, 1536), f.shape
        tmp = os.path.join(a.feat, n + ".tmp.npy"); np.save(tmp, f); os.replace(tmp, os.path.join(a.feat, n + ".npy"))
        if i % 10 == 0:
            print(i, n, f.shape, f"{(time.time() - t0) / (i + 1):.2f}s/clip", f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
    print("EXTRACT_DONE", len(todo), f"{time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ctx", "g0a", "extract"])
    ap.add_argument("--clips", default=os.path.join(E5_BIG, "p3", "clips")); ap.add_argument("--feat", default=os.path.join(E5_BIG, "p3", "feat"))
    ap.add_argument("--dit_dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--shard_i", type=int, default=0); ap.add_argument("--shard_n", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "ctx":
        ctx, n = compute_empty_context("cuda"); print("ctx", tuple(ctx.shape), "tokens", n, flush=True)
    elif a.cmd == "g0a":
        gate_g0a(a)
    else:
        extract(a)


if __name__ == "__main__":
    main()
