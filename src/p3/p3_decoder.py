"""P3 X2 decoder on frozen Wan2.1-1.3B final-block tokens (P3 plan v2 section 4.4; unified plan U3).

Tokens:  Linear(1536 -> W) + LayerNorm, plus a linear map of a Fourier embedding of each token's (t, y, x)
         (RoPE acts only inside Wan attention, so the residual stream may lack absolute position).
Query:   Fourier(u, v) + MLP(9x9 RGB patch from the 512 px source frame) + continuous time embeddings of
         t_src, t_tgt and t_cam (sinusoidal on t in [0, 1] followed by an MLP, one MLP per role).
Body:    `depth` cross-attention blocks (query attends to tokens), width W, 8 heads; queries are independent
         (no query self-attention). Default W = 512, depth 4.
Outputs: X in the camera of t_cam at time t_tgt, in clip-normalised units (divided by the clip scale s);
         visibility logit; confidence (1 + softplus).
Parameterisations:
  P-abs: X = head_X(z_q).
  P-res: X = S + head_d(z_q), S = head_X(z_q') with q' = (u, v, t_src, t_tgt := t_src, t_cam): the same decoder
         queried for "where the point would be in camera t_cam had it not moved". No extra supervision.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier(x, n_freq):
    """x (..., d) in roughly [0, 1] -> (..., d * (2 n_freq + 1))."""
    f = (2.0 ** torch.arange(n_freq, device=x.device, dtype=x.dtype)) * math.pi
    xf = x[..., None] * f
    return torch.cat([x, torch.sin(xf).flatten(-2), torch.cos(xf).flatten(-2)], -1)


def sincos_time(t, dim=128, max_period=100.0):
    half = dim // 2
    fr = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = (t.float()[..., None] * 100.0) * fr
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class MLP(nn.Module):
    def __init__(self, i, h, o):
        super().__init__(); self.a = nn.Linear(i, h); self.b = nn.Linear(h, o)

    def forward(self, x):
        return self.b(F.gelu(self.a(x)))


class CrossBlock(nn.Module):
    def __init__(self, w, heads):
        super().__init__()
        self.h = heads; self.ln_q = nn.LayerNorm(w); self.ln_kv = nn.LayerNorm(w)
        self.q = nn.Linear(w, w); self.kv = nn.Linear(w, 2 * w); self.o = nn.Linear(w, w)
        self.ln2 = nn.LayerNorm(w); self.mlp = MLP(w, 4 * w, w)

    def kv_cache(self, tok):
        k, v = self.kv(self.ln_kv(tok)).chunk(2, -1)
        B, N, W = k.shape
        return k.view(B, N, self.h, W // self.h).transpose(1, 2), v.view(B, N, self.h, W // self.h).transpose(1, 2)

    def forward(self, x, kv):
        k, v = kv
        B, Q, W = x.shape
        q = self.q(self.ln_q(x)).view(B, Q, self.h, W // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Q, W)
        x = x + self.o(a)
        return x + self.mlp(self.ln2(x))


class P3Decoder(nn.Module):
    def __init__(self, param="abs", feat_dim=1536, width=512, heads=8, depth=4, patch=9, n_freq_tok=8, n_freq_q=10):
        super().__init__()
        assert param in ("abs", "res")
        self.param = param; self.patch = patch; self.nft = n_freq_tok; self.nfq = n_freq_q
        self.tok_proj = nn.Linear(feat_dim, width); self.tok_ln = nn.LayerNorm(width)
        self.tok_pos = nn.Linear(3 * (2 * n_freq_tok + 1), width)
        self.q_uv = nn.Linear(2 * (2 * n_freq_q + 1), width)
        self.q_patch = MLP(3 * patch * patch, width, width)
        self.t_emb = nn.ModuleList([MLP(128, width, width) for _ in range(3)])
        self.blocks = nn.ModuleList([CrossBlock(width, heads) for _ in range(depth)])
        self.ln_out = nn.LayerNorm(width)
        self.head_X = nn.Linear(width, 3)
        self.head_d = nn.Linear(width, 3) if param == "res" else None
        self.head_vis = nn.Linear(width, 1); self.head_conf = nn.Linear(width, 1)
        if self.head_d is not None:
            nn.init.zeros_(self.head_d.weight); nn.init.zeros_(self.head_d.bias)

    def encode_tokens(self, feat, tok_pos):
        """feat (B,N,1536); tok_pos (N,3) in [0,1] -> per-block (k, v) caches."""
        x = self.tok_ln(self.tok_proj(feat)) + self.tok_pos(fourier(tok_pos, self.nft))[None]
        return [blk.kv_cache(x) for blk in self.blocks]

    def embed_query(self, uv, patch, ts, tt, tc):
        """uv (B,Q,2) in [0,1]; patch (B,Q,3*p*p) in [-1,1]; ts, tt, tc (B,Q) in [0,1]."""
        x = self.q_uv(fourier(uv, self.nfq)) + self.q_patch(patch)
        for emb, t in zip(self.t_emb, (ts, tt, tc)):
            x = x + emb(sincos_time(t))
        return x

    def decode(self, kvs, x):
        for blk, kv in zip(self.blocks, kvs):
            x = blk(x, kv)
        return self.ln_out(x)

    def forward(self, feat, tok_pos, uv, patch, ts, tt, tc, return_static=False):
        kvs = self.encode_tokens(feat, tok_pos)
        z = self.decode(kvs, self.embed_query(uv, patch, ts, tt, tc))
        out = {}
        if self.param == "abs":
            X = self.head_X(z)
        else:
            z0 = self.decode(kvs, self.embed_query(uv, patch, ts, ts, tc))
            S = self.head_X(z0); X = S + self.head_d(z)
            out["S"] = S
        out.update(X=X.float(), vis_logit=self.head_vis(z)[..., 0].float(), conf=1.0 + F.softplus(self.head_conf(z)[..., 0].float()))
        return out


def token_positions(T_lat=7, Hg=30, Wg=30, n_frames=25, device="cpu"):
    """(T_lat*Hg*Wg, 3) normalised (t, y, x) of Wan tokens in diffusers' flatten order (f, h, w).
    Latent 0 holds frame 0; latent k >= 1 holds frames 4k-3 .. 4k (centre 4k - 1.5)."""
    tc = torch.tensor([0.0] + [4 * k - 1.5 for k in range(1, T_lat)], device=device) / (n_frames - 1)
    yc = (torch.arange(Hg, device=device) + 0.5) / Hg; xc = (torch.arange(Wg, device=device) + 0.5) / Wg
    t, y, x = torch.meshgrid(tc, yc, xc, indexing="ij")
    return torch.stack([t, y, x], -1).reshape(-1, 3)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
