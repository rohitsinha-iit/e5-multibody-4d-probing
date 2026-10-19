"""P3 X2 trainer (P3 plan v2 section 4.4; unified plan U3). Resumable; checkpoints every --ckpt_every steps.

Data: clips from kubric_shards.py (<clips>/<name>/{rgb.npy,labels.npz,meta.json}) and Wan final-block features from
wan_feats.py extract (<feat>/<name>.npy, (6300,1536) fp16, tokens in diffusers order over a 7x30x30 grid).
Coordinates: CoTracker3 tracks are continuous pixel coordinates (pixel centre = i + 0.5, cx = 256 reprojects
exactly), so uv = coords / 512; E5 scenes use pixel-index coordinates and are mapped with (x + 0.5) / 512 in x2_eval.
Query sampler: --n_query (2048) per clip; t_src uniform over the 25 frames; the track is drawn among tracks visible
at t_src, from the near-edge set (within 8 px of a segmentation boundary) with probability 0.3; t_tgt uniform;
t_cam = t_tgt with probability 0.4, else uniform.
Targets: X = E_{t_cam} Xw(track, t_tgt) divided by the clip scale s (mean z of the tracks visible in frame 0).
Losses (equal read-out weight 1.0; identical auxiliaries in both loss spaces):
  L3D: sum over xyz of |slog(X_pred) - slog(X_gt)|, slog(x) = sign(x) log(1 + |x|) (D4RT normalised sign-log L1);
  L2D: Huber (delta --huber, default 0.02 in normalised image units, about 9 px at f = 450) on the projection
       (x/z, y/z) of X_pred against the GT projection, i.e. GT intrinsics, normalised image coordinates;
  visibility 0.1 (BCE, only for queries with t_cam = t_tgt, against visibility at t_tgt);
  confidence 0.2: c * l.detach() - log(c), with c = 1 + softplus(.). The error is detached so that the confidence
  term does not reweight the read-out loss (this keeps the read-out weight identical across cells).
Optimiser: AdamW (lr 1e-4, wd 0.05, betas 0.9/0.95), linear warmup 1k, cosine to 1e-6, bf16 autocast, clip 1.0.
Held-out evaluation every --eval_every steps on val windows (sequence id % 10 == 9): frame-0 queries, t_tgt = t_cam =
frame 24, EPE of the 2D projection at 512 px; static = world displacement below 1 cm; per-clip medians, then mean.
"""
import argparse, glob, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_decoder import P3Decoder, token_positions, n_params  # noqa: E402

E5_BIG = os.environ.get("E5_BIG", "/data2/DATA/rohit_sinha/e5/big")
NF = 25
STATIC_M = 0.01


def slog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def clip_list(clips, feat, split):
    out = []
    for d in sorted(glob.glob(os.path.join(clips, "*", "meta.json"))):
        m = json.load(open(d)); name = os.path.basename(os.path.dirname(d))
        if m["split"] == split and os.path.exists(os.path.join(feat, name + ".npy")):
            out.append(name)
    return out


def load_clip(clips, name):
    lab = dict(np.load(os.path.join(clips, name, "labels.npz")))
    Xc0 = np.einsum("ij,nj->ni", lab["E"][0, :, :3], lab["Xw"][:, 0]) + lab["E"][0, :, 3]
    lab["scale"] = float(Xc0[lab["vis"][:, 0], 2].mean())
    return lab


def patches(rgb, t, u, v, p=9):
    r = p // 2; H, W = rgb.shape[1:3]
    ui = np.clip(np.round(u).astype(int)[:, None] + np.arange(-r, r + 1)[None], 0, W - 1)
    vi = np.clip(np.round(v).astype(int)[:, None] + np.arange(-r, r + 1)[None], 0, H - 1)
    pt = rgb[t[:, None, None], vi[:, :, None], ui[:, None, :]]          # (Q,p,p,3)
    return (pt.astype(np.float32) / 127.5 - 1.0).reshape(len(t), -1)


def targets(lab, n, tt, tc):
    E = lab["E"]; Xw = lab["Xw"][n, tt].astype(np.float64)
    X = np.einsum("qij,qj->qi", E[tc, :, :3], Xw) + E[tc, :, 3]
    return (X / lab["scale"]).astype(np.float32)


class ClipSet(Dataset):
    def __init__(self, clips, feat, names, n_query=2048, edge_frac=0.3, p_same=0.4, seed=0, preload=False, tracks_keep=0):
        self.clips, self.feat, self.names = clips, feat, names
        self.nq, self.ef, self.ps, self.seed = n_query, edge_frac, p_same, seed
        self.epoch = 0; self.mem = None
        if preload:   # hold everything in RAM (shared copy-on-write with forked loader workers)
            t0 = time.time(); self.mem = {}
            for k, nm in enumerate(names):
                lab = load_clip(clips, nm)
                if tracks_keep and len(lab["vis"]) > tracks_keep:
                    keep = np.sort(np.random.default_rng(int.from_bytes(nm.encode()[:8].ljust(8, b"0"), "little") % (1 << 31)).choice(len(lab["vis"]), tracks_keep, replace=False))
                    for key in ("coords", "Xw", "vis", "near_edge", "segid"):
                        lab[key] = np.ascontiguousarray(lab[key][keep])
                self.mem[nm] = (np.load(os.path.join(feat, nm + ".npy")), np.load(os.path.join(clips, nm, "rgb.npy")), lab)
                if k % 50 == 0:
                    print(f"preload {k}/{len(names)} {time.time() - t0:.0f}s", flush=True)
            print(f"preload done {len(names)} clips {time.time() - t0:.0f}s", flush=True)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        rng = np.random.default_rng([self.seed, self.epoch, i, int(time.time_ns() % (1 << 31))])
        if self.mem is not None:
            feat, rgb, lab = self.mem[name]
        else:
            lab = load_clip(self.clips, name)
            rgb = np.load(os.path.join(self.clips, name, "rgb.npy"), mmap_mode="r")
            feat = np.load(os.path.join(self.feat, name + ".npy"))
        vis, ne = lab["vis"], lab["near_edge"]
        ok_f = np.where(vis.sum(0) > 0)[0]
        ts = rng.choice(ok_f, self.nq)
        n = np.zeros(self.nq, int)
        for f in np.unique(ts):
            sel = np.where(ts == f)[0]; cv = np.where(vis[:, f])[0]; ce = np.where(ne[:, f])[0]
            use_e = (rng.random(len(sel)) < self.ef) & (len(ce) > 0)
            n[sel] = np.where(use_e, ce[rng.integers(0, max(len(ce), 1), len(sel))] if len(ce) else 0,
                              cv[rng.integers(0, len(cv), len(sel))])
        tt = rng.integers(0, NF, self.nq)
        tc = np.where(rng.random(self.nq) < self.ps, tt, rng.integers(0, NF, self.nq))
        uvp = lab["coords"][n, ts]
        X = targets(lab, n, tt, tc)
        vt = vis[n, tt].astype(np.float32); vmask = (tc == tt).astype(np.float32)
        return dict(feat=torch.from_numpy(feat), uv=torch.from_numpy((uvp / 512.0).astype(np.float32)),   # continuous coords (pixel centre = i + 0.5)
                    patch=torch.from_numpy(patches(rgb, ts, uvp[:, 0] - 0.5, uvp[:, 1] - 0.5)),
                    ts=torch.from_numpy(ts / (NF - 1.0)).float(), tt=torch.from_numpy(tt / (NF - 1.0)).float(),
                    tc=torch.from_numpy(tc / (NF - 1.0)).float(), X=torch.from_numpy(X),
                    vis=torch.from_numpy(vt), vmask=torch.from_numpy(vmask))


def compute_loss(out, b, loss_space, huber):
    Xp, Xg = out["X"], b["X"]
    if loss_space == "L3D":
        l = (slog(Xp) - slog(Xg)).abs().sum(-1); valid = torch.ones_like(l, dtype=torch.bool)
    else:
        zg = Xg[..., 2]; valid = zg > 1e-2
        pg = Xg[..., :2] / zg.clamp(min=1e-2)[..., None]
        zp = Xp[..., 2]; zp = torch.where(zp.abs() < 1e-2, torch.full_like(zp, 1e-2), zp)
        pp = Xp[..., :2] / zp[..., None]
        l = F.huber_loss(pp, pg, delta=huber, reduction="none").sum(-1)
    vf = valid.float(); nv = vf.sum().clamp(min=1)
    main = (l * vf).sum() / nv
    c = out["conf"]; caux = ((c * l.detach() - torch.log(c)) * vf).sum() / nv
    vm = b["vmask"]; vaux = (F.binary_cross_entropy_with_logits(out["vis_logit"], b["vis"], reduction="none") * vm).sum() / vm.sum().clamp(min=1)
    total = main + 0.1 * vaux + 0.2 * caux
    return total, dict(main=main.item(), vis=vaux.item(), conf=caux.item())


_VAL_CACHE = {}


@torch.no_grad()
def evaluate(model, clips, feat, names, tok_pos, dev, max_q=4096, t_tgt=NF - 1):
    model.eval(); rows = []
    for name in names:
        if name not in _VAL_CACHE:
            _VAL_CACHE[name] = (load_clip(clips, name), np.load(os.path.join(clips, name, "rgb.npy")),
                                np.load(os.path.join(feat, name + ".npy")))
        lab, rgb, fnp = _VAL_CACHE[name]
        co = lab["coords"]; vis = lab["vis"]
        inb = (co[:, t_tgt, 0] >= 0) & (co[:, t_tgt, 0] < 512) & (co[:, t_tgt, 1] >= 0) & (co[:, t_tgt, 1] < 512)
        n = np.where(vis[:, 0] & vis[:, t_tgt] & inb)[0]
        if len(n) > max_q:
            n = np.random.default_rng(0).choice(n, max_q, replace=False)
        Q = len(n); ts = np.zeros(Q, int); tt = np.full(Q, t_tgt)
        f = torch.from_numpy(fnp)[None].to(dev)
        uv = torch.from_numpy((co[n, 0] / 512).astype(np.float32))[None].to(dev)
        pt = torch.from_numpy(patches(rgb, ts, co[n, 0, 0] - 0.5, co[n, 0, 1] - 0.5))[None].to(dev)
        z0 = torch.zeros(1, Q, device=dev); z1 = torch.full((1, Q), t_tgt / (NF - 1.0), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(f.to(torch.bfloat16), tok_pos, uv, pt, z0, z1, z1)
        X = out["X"][0].float().cpu().numpy(); K = lab["K"]
        z = np.where(np.abs(X[:, 2]) < 1e-3, 1e-3, X[:, 2])
        xp = np.c_[X[:, 0] / z * K[0, 0] + K[0, 2], X[:, 1] / z * K[1, 1] + K[1, 2]]
        err = np.linalg.norm(xp - co[n, t_tgt], axis=1)
        disp = np.linalg.norm(lab["Xw"][n, t_tgt] - lab["Xw"][n, 0], axis=1); st = disp < STATIC_M
        Xg = targets(lab, n, tt, tt); l3 = float(np.mean(np.abs(np.sign(X) * np.log1p(np.abs(X)) - np.sign(Xg) * np.log1p(np.abs(Xg))).sum(1)))
        rows.append(dict(clip=name, n_static=int(st.sum()), n_moving=int((~st).sum()),
                         epe_static=float(np.median(err[st])) if st.sum() > 10 else float("nan"),
                         epe_moving=float(np.median(err[~st])) if (~st).sum() > 10 else float("nan"), l3d=l3))
    model.train()
    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in ("epe_static", "epe_moving", "l3d")}
    return agg, rows


def lr_at(step, total, base, warm, final=1e-6):
    if step < warm:
        return base * (step + 1) / warm
    p = min(1.0, (step - warm) / max(1, total - warm))
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--param", choices=["abs", "res"], required=True)
    ap.add_argument("--loss", choices=["L3D", "L2D"], required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=512); ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--n_query", type=int, default=2048); ap.add_argument("--huber", type=float, default=0.02)
    ap.add_argument("--ckpt_every", type=int, default=1000); ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--clips", default=os.path.join(E5_BIG, "p3", "clips")); ap.add_argument("--feat", default=os.path.join(E5_BIG, "p3", "feat"))
    ap.add_argument("--runs", default=os.path.join(E5_BIG, "p3", "runs")); ap.add_argument("--max_steps_this_job", type=int, default=0)
    ap.add_argument("--bench_loader", type=int, default=0, help="only time N batches of the data loader and exit")
    ap.add_argument("--preload", type=int, default=0, help="1: hold features, RGB and labels in RAM")
    ap.add_argument("--tracks_keep", type=int, default=0, help="with --preload: fixed random subset of tracks per clip (0 = all)")
    a = ap.parse_args()
    if a.bench_loader:
        tr = clip_list(a.clips, a.feat, "train"); ds = ClipSet(a.clips, a.feat, tr, n_query=a.n_query, preload=bool(a.preload), tracks_keep=a.tracks_keep)
        dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True, prefetch_factor=4)
        t0 = time.time(); nb = 0; mb = 0.0
        for b in dl:
            nb += 1; mb += sum(v.numel() * v.element_size() for v in b.values()) / 2 ** 20
            if nb == 3:
                t0 = time.time(); mb = 0.0
            if nb >= a.bench_loader + 3:
                break
        dt = time.time() - t0; n = nb - 3
        res = dict(batches=n, sec_per_batch=dt / max(n, 1), mb_per_batch=mb / max(n, 1), mb_per_s=mb / max(dt, 1e-9),
                   workers=a.workers, preload=a.preload, n_train_clips=len(tr))
        print("LOADER_BENCH", json.dumps(res), flush=True)
        os.makedirs(a.runs, exist_ok=True); json.dump(res, open(os.path.join(a.runs, f"loader_bench_w{a.workers}_p{a.preload}.json"), "w"))
        return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rd = os.path.join(a.runs, a.name); os.makedirs(rd, exist_ok=True)
    dev = torch.device("cuda")
    tr = clip_list(a.clips, a.feat, "train"); va = clip_list(a.clips, a.feat, "val")
    json.dump(dict(vars(a), n_train_clips=len(tr), n_val_clips=len(va), val_clips=va), open(os.path.join(rd, "config.json"), "w"), indent=1)
    model = P3Decoder(param=a.param, width=a.width, heads=8, depth=a.depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05, betas=(0.9, 0.95))
    step = 0; ck = os.path.join(rd, "ckpt_latest.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step = s["step"]
        print("resumed at", step, flush=True)
    print(f"params {n_params(model)/1e6:.2f}M train clips {len(tr)} val clips {len(va)}", flush=True)
    tok_pos = token_positions(device=dev)
    ds = ClipSet(a.clips, a.feat, tr, n_query=a.n_query, seed=a.seed, preload=bool(a.preload), tracks_keep=a.tracks_keep)
    logf = open(os.path.join(rd, "log.jsonl"), "a"); evf = open(os.path.join(rd, "eval.jsonl"), "a")
    t_stop = step + a.max_steps_this_job if a.max_steps_this_job else a.steps
    torch.cuda.reset_peak_memory_stats()
    while step < min(a.steps, t_stop):
        ds.epoch = step
        dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                        pin_memory=True, prefetch_factor=4, persistent_workers=False)
        t_last = time.time()
        for b in dl:
            t_data = time.time() - t_last
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            for g in opt.param_groups:
                g["lr"] = lr_at(step, a.steps, a.lr, a.warmup)
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(b["feat"].to(torch.bfloat16), tok_pos, b["uv"], b["patch"], b["ts"], b["tt"], b["tc"])
            loss, comp = compute_loss(out, b, a.loss, a.huber)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            torch.cuda.synchronize(); t_comp = time.time() - t0
            step += 1
            if step % 20 == 0 or step <= 5:
                logf.write(json.dumps(dict(step=step, loss=loss.item(), **comp, gn=float(gn), lr=opt.param_groups[0]["lr"],
                                           t_data=t_data, t_compute=t_comp, mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)) + "\n"); logf.flush()
            if step % a.eval_every == 0 and va:
                agg, rows = evaluate(model, a.clips, a.feat, va, tok_pos, dev)
                evf.write(json.dumps(dict(step=step, **agg)) + "\n"); evf.flush()
                print("eval", step, agg, flush=True)
            if step % a.ckpt_every == 0 or step >= min(a.steps, t_stop):
                tmp = ck + ".tmp"; torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, args=vars(a)), tmp)
                os.replace(tmp, ck)
                if step % 5000 == 0:
                    torch.save(dict(model=model.state_dict(), step=step, args=vars(a)), os.path.join(rd, f"model_{step:06d}.pt"))
            t_last = time.time()
            if step >= min(a.steps, t_stop):
                break
    if va:
        agg, rows = evaluate(model, a.clips, a.feat, va, tok_pos, dev)
        json.dump(dict(step=step, **agg, rows=rows), open(os.path.join(rd, "final_eval.json"), "w"), indent=1)
        print("final eval", step, agg, flush=True)
    print("DONE", step, flush=True)


if __name__ == "__main__":
    main()
