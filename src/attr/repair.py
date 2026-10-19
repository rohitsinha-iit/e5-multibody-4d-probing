#!/usr/bin/env python3
"""U5a object-gain edit and U5b contamination masking on Any4D (unified plan 5, U5a and U5b).

U5a directions (mode dirs, fit scenes only), per head in P_obj (top-k* of rank_obj) union the five E5 heads:
  U_O = top-2 right singular vectors of h(D)-h(B) on moving view-0 tokens; U_C = top-2 of h(D)-h(C) on all view-0 tokens;
  P_O projects on U_O orthogonalised against U_C, P_C on U_C orthogonalised against U_O; m = mean head output over
  static tokens (all views) of D. Saved to $E5_BIG/results/u5/u5a_dirs.pt.
U5a edit: h <- h + (g-1) P (h - m) at every token, g in {1.5, 2}. Variants:
  i  P_O    ii P_C    iii 20 random 2D projectors per head (same heads, same rank and norm)
  iv whole-head gain (E5: m = per-run token mean, P = I)    v  heads_sel LoRA checkpoint, no edit (--ckpt)
  base      no edit (reference row; always run)
U5b masking (two-pass: pass 1 = motion probe at L*, pass 2 = masked run). Variants:
  a  probe mask in P_cont (top k* heads of rank_cont)   b  probe mask in all 288 heads
  c  probe mask in k* random heads, block-matched to P_cont, 20 draws   d  GT mask in all heads (Kubric only)
  e  MoRe model itself: run by the New models track (MoReAdapter); not in this file
Evaluation:
  kubric  (split rule as interchange.py) per base scene, variant D: alpha, beta (fit_ab of e5_w2_eval: y = x_pred - x1 =
          alpha A + beta B, moving visible queries, view 23), identifiable flag (cond <= 10, median |A|, |B| >= 2 px),
          s_w and static residual (world_slope.py), moving and static EPE (px, 512), RPE-T, RPE-R, ATE (pose.py), and for
          U5b the recovery (e(D) - e(edit)) / (e(D) - e(B)) with B's base run.
  real    Sintel (MonST3R 14-sequence subset, final pass), TUM-dynamics (rgb_90, groundtruth_90), Bonn (rgb_110,
          groundtruth_110): non-overlapping 8-frame windows (pose.windows), Sim(3) per window; ATE, RPE-T, RPE-R.
Boundary rule (8 px) is not applied here yet (TODO before the Oct 27 freeze; world_slope.boundary_mask).
"""
import argparse, glob, json, os, struct, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import any4d_hooks as ah  # noqa
import interchange as ic  # noqa
import pose as pm  # noqa
import world_slope as ws  # noqa
import e5_step3_analysis as s3  # noqa
from e5_w2_eval import fit_ab  # noqa

E5_BIG = ic.E5_BIG
OUT = os.environ.get("U5_RES", os.path.join(E5_BIG, "results", "u5"))
POSESETS = os.path.join(E5_BIG, "realdata", "posesets")
SINTEL14 = ["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", "market_2", "market_5", "market_6",
            "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"]
LORA = {"heads_sel": os.path.join(E5_BIG, "ft", "runs", "heads_sel", "any4d_ft.pth")}
TV = 6


# ------------------------------------------------------------------ U5a directions
def head_set(kstar_obj, rank_file=None):
    hs = set(ah.FIVE)
    rf = rank_file or os.path.join(ic.RES, "rank_obj.json")
    if os.path.exists(rf) and kstar_obj:
        for n in [n for n in json.load(open(rf))["order"] if not n.endswith("r")][:kstar_obj]:
            c = ic.parse(n); hs.add((c[1], c[2]))
    return sorted(hs)


def top2(X):
    X = X.float()
    _, _, Vt = torch.linalg.svd(X, full_matrices=False)
    return Vt[:2].T            # (64,2)


def orth_proj(U, W):
    Q = U - W @ (W.T @ U)
    Q, _ = torch.linalg.qr(Q)
    return Q @ Q.T


def fit_dirs(H, scenes, heads):
    capt = sorted({("h", b) for b, _ in heads})
    acc = {hh: {"DB": [], "DC": [], "st": []} for hh in heads}
    for d in scenes:
        lab = ah.token_mask_from_pixels(ic.gt_of(d, "D")["moving_mask"][ah.FRAMES], 0.5).reshape(-1).cuda()
        c = {}
        for v in "DBC":
            _, c[v] = H.run(ah.load_views(os.path.join(d, v)), capture=True, capture_only=capt)
        v0 = torch.arange(ah.P_TOK, device="cuda"); mv0 = v0[lab[:ah.P_TOK]]
        VP = lab.numel(); st = torch.nonzero(~lab).squeeze(1)
        for (b, h) in heads:
            sl = slice(h * 64, (h + 1) * 64)
            D, B, C = c["D"][("h", b)][:VP, sl], c["B"][("h", b)][:VP, sl], c["C"][("h", b)][:VP, sl]
            if mv0.numel():
                acc[(b, h)]["DB"].append((D[mv0] - B[mv0]).float().cpu())
            acc[(b, h)]["DC"].append((D[v0] - C[v0]).float().cpu())
            acc[(b, h)]["st"].append(D[st].float().mean(0, keepdim=True).cpu())
        print("[u5a dirs]", ic.sid(d), flush=True)
    dirs = {}
    for hh, a in acc.items():
        UO = top2(torch.cat(a["DB"])); UC = top2(torch.cat(a["DC"]))
        dirs[hh] = {"P_O": orth_proj(UO, UC), "P_C": orth_proj(UC, UO), "m": torch.cat(a["st"]).mean(0),
                    "U_O": UO, "U_C": UC, "cos_OC": float(torch.linalg.svdvals(UO.T @ UC).max())}
    return dirs


def edit_fns(dirs, variant, g, r=0):
    fns = {}
    gen = torch.Generator().manual_seed(1000 + r)
    for (b, h), dd in dirs.items():
        if variant == "i":
            P, m = dd["P_O"].cuda(), dd["m"].cuda()
        elif variant == "ii":
            P, m = dd["P_C"].cuda(), dd["m"].cuda()
        elif variant == "iii":
            Q, _ = torch.linalg.qr(torch.randn(64, 2, generator=gen)); P, m = (Q @ Q.T).cuda(), dd["m"].cuda()
        elif variant == "iv":
            fns[(b, h)] = (lambda x, n, g=g: x.mean(0, keepdim=True) + g * (x - x.mean(0, keepdim=True)))
            continue
        else:
            raise ValueError(variant)
        fns[(b, h)] = (lambda x, n, P=P, m=m, g=g: x + (g - 1) * (x - m) @ P)
    return fns


# ------------------------------------------------------------------ U5b masks
def mask_heads(variant, kstar, rng=None, rank_file=None):
    if variant in ("b", "d"):
        return [(b, h) for b in range(24) for h in range(12)]
    order = [n for n in json.load(open(rank_file or os.path.join(ic.RES, "rank_cont.json")))["order"]][:kstar]
    pc = [(ic.parse(n)[1], ic.parse(n)[2]) for n in order]
    if variant == "a":
        return pc
    out = []
    for b, h in pc:   # c: block-matched random heads, excluding P_cont
        pool = [hh for hh in range(12) if (b, hh) not in pc and (b, hh) not in out]
        out.append((b, int(rng.choice(pool))))
    return out


def masks_for(heads, key):
    m = {}
    for b, h in heads:
        m.setdefault(b, ah.KeyMask([], key)).heads.append(h)
    return m


# ------------------------------------------------------------------ Kubric metrics
def kubric_metrics(o, d, vn="D"):
    g = ic.gt_of(d, vn)
    P = ah.to_numpy(o); off = np.array(s3.pixel_offset(P))
    xp518 = s3.pred_correspondences(P, TV, off)
    uv = ic.to518(g["q_pix"]); xp = xp518[uv[:, 1], uv[:, 0]]; xp = (xp + 0.5) * 512 / 518 - 0.5
    x1 = g["x1"]; A = g["A"][-1]; Bc = g["B"][-1]; xt = g["x_true"][-1]; vis = g["vis"][-1]
    mv = (g["q_group"] > 0) & vis; st = (g["q_group"] == 0) & vis
    y = xp - x1; ab = fit_ab(A[mv], Bc[mv], y[mv])
    ident = bool(ab["cond_AB"] <= 10 and np.median(np.linalg.norm(A[mv], axis=1)) >= 2 and np.median(np.linalg.norm(Bc[mv], axis=1)) >= 2) if mv.sum() > 20 else False
    err = np.linalg.norm(xp - xt, axis=1)
    # s_w: predicted world displacement = scene flow of the query pixel (model world) vs GT world_disp (view 23)
    sf = P["scene_flow"][-1][uv[:, 1], uv[:, 0]]
    T0 = P["cam_pose"][0]; zp = ((P["pts3d_0"][uv[:, 1], uv[:, 0]] - T0[:3, 3]) @ T0[:3, :3])[:, 2]
    w2c0 = g["w2c_cv"][0]
    X = g["pts0"][g["q_pix"][:, 1], g["q_pix"][:, 0]]; zg = (X @ w2c0[:3, :3].T + w2c0[:3, 3])[:, 2]
    sw = ws.world_slope(sf, g["world_disp"][-1], mv, st, z_gt=zg, z_pred=zp, focal=float(g["K"][0, 0]))
    gtc = np.linalg.inv(g["w2c_cv"][ah.FRAMES])
    rt, rr = pm.rpe(P["cam_pose"], gtc)
    return dict(alpha=ab["alpha"], beta=ab["beta"], cond_AB=ab["cond_AB"], identifiable=ident, s_w=sw["s_w"],
                static_residual=sw["static_residual"], epe_moving=float(np.median(err[mv])) if mv.any() else float("nan"),
                epe_static=float(np.median(err[st])), rpe_t=rt, rpe_r=rr, ate=pm.ate(P["cam_pose"], gtc))


def e_norm(r, rD):
    return 0.5 * (r["rpe_r"] / max(rD["rpe_r"], 1e-9) + r["rpe_t"] / max(rD["rpe_t"], 1e-9))


# ------------------------------------------------------------------ real pose sets
def _sintel_cam(f):
    with open(f, "rb") as fh:
        fh.read(4); M = np.frombuffer(fh.read(72), "<f8").reshape(3, 3); N = np.frombuffer(fh.read(96), "<f8").reshape(3, 4)
    w2c = np.eye(4); w2c[:3] = N
    return np.linalg.inv(w2c)


def _quat_pose(t, q):
    x, y, z, w = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def _tum_like(seq_dir, rgb_sub, gt_file):
    imgs = sorted(glob.glob(os.path.join(seq_dir, rgb_sub, "*.png")))
    L = [l.split() for l in open(os.path.join(seq_dir, gt_file)) if l.strip() and not l.startswith("#")]
    ts = np.array([float(l[0]) for l in L]); poses = [_quat_pose(np.array(l[1:4], float), np.array(l[4:8], float)) for l in L]
    if len(L) == len(imgs):
        return imgs, np.stack(poses)
    its = np.array([float(os.path.basename(f)[:-4]) for f in imgs])     # associate by nearest timestamp
    j = np.abs(its[:, None] - ts[None]).argmin(1)
    return imgs, np.stack([poses[k] for k in j])


def pose_sequences(setname):
    if setname == "sintel":
        root = os.path.join(POSESETS, "sintel", "training")
        for s in SINTEL14:
            imgs = sorted(glob.glob(os.path.join(root, "final", s, "*.png")))
            cams = sorted(glob.glob(os.path.join(root, "camdata_left", s, "*.cam")))
            if imgs and len(imgs) == len(cams):
                yield s, imgs, np.stack([_sintel_cam(f) for f in cams])
    elif setname == "tum":
        for d in sorted(glob.glob(os.path.join(POSESETS, "tum", "rgbd_dataset_freiburg3_*"))):
            if os.path.isdir(os.path.join(d, "rgb_90")):
                im, P = _tum_like(d, "rgb_90", "groundtruth_90.txt"); yield os.path.basename(d), im, P
    elif setname == "bonn":
        for d in sorted(glob.glob(os.path.join(POSESETS, "bonn", "rgbd_bonn_dataset", "rgbd_bonn_*"))):
            if os.path.isdir(os.path.join(d, "rgb_110")):
                im, P = _tum_like(d, "rgb_110", "groundtruth_110.txt"); yield os.path.basename(d), im, P


def load_imgs(paths):
    from any4d.utils.image import load_images  # noqa
    return load_images(list(paths), resize_mode="square", size=518, norm_type=ah.CFG["data_norm_type"], patch_size=14)


# ------------------------------------------------------------------ drivers
def run_u5a(H, a, scenes):
    dirs = torch.load(os.path.join(OUT, "u5a_dirs.pt")) if a.variant in ("i", "ii", "iii") else None
    if a.variant == "iv":
        dirs = {hh: None for hh in head_set(a.kstar_obj)}
    draws = range(a.n_random) if a.variant == "iii" else [0]
    od = os.path.join(OUT, f"u5a_{a.variant}_g{a.g:g}{a.out_tag}"); os.makedirs(od, exist_ok=True)
    for d in scenes:
        f = os.path.join(od, ic.sid(d) + ".json")
        if os.path.exists(f):
            continue
        views = ah.load_views(os.path.join(d, "D")); rows = {}
        o, _ = H.run(views); rows["base"] = kubric_metrics(o, d)
        if a.variant != "v":
            for r in draws:
                o, _ = H.run(views, edits=edit_fns(dirs, a.variant, a.g, r)); rows[f"edit_r{r}"] = kubric_metrics(o, d)
        json.dump(dict(scene=ic.sid(d), variant=a.variant, g=a.g, heads=[list(h) for h in (dirs or {})], rows=rows),
                  open(f, "w"), indent=1)
        print("[u5a]", ic.sid(d), a.variant, flush=True)


def run_u5b_kubric(H, a, scenes, probe):
    od = os.path.join(OUT, f"u5b_{a.variant}_kubric{a.out_tag}"); os.makedirs(od, exist_ok=True)
    rng = np.random.default_rng(0)
    for d in scenes:
        f = os.path.join(od, ic.sid(d) + ".json")
        if os.path.exists(f):
            continue
        views = ah.load_views(os.path.join(d, "D"))
        oD, _ = H.run(views); rD = kubric_metrics(oD, d)
        oB, _ = H.run(ah.load_views(os.path.join(d, "B"))); rB = kubric_metrics(oB, d, "B")
        if a.variant == "d":
            key = ah.token_mask_from_pixels(ic.gt_of(d, "D")["moving_mask"][ah.FRAMES], 0.25).reshape(-1).cuda()
        else:
            key = mpa.probe_mask(H, views, probe)
        rows = {"base_D": rD, "base_B": rB}
        for r in range(a.n_random if a.variant == "c" else 1):
            hs = mask_heads(a.variant, a.kstar_cont, rng)
            o, _ = H.run(views, masks=masks_for(hs, key)); m = kubric_metrics(o, d)
            eD, eB, eE = 1.0, e_norm(rB, rD), e_norm(m, rD)
            m["recovery"] = (eD - eE) / (eD - eB) if abs(eD - eB) > 1e-9 else float("nan")
            m["heads"] = [list(h) for h in hs]; rows[f"edit_r{r}"] = m
        rows["mask_frac"] = float(key.float().mean())
        json.dump(dict(scene=ic.sid(d), variant=a.variant, rows=rows), open(f, "w"), indent=1)
        print("[u5b kubric]", ic.sid(d), a.variant, flush=True)


def run_u5b_real(H, a, probe, setname, edit=None):
    od = os.path.join(OUT, f"u5b_{a.variant}_{setname}{a.out_tag}"); os.makedirs(od, exist_ok=True)
    rng = np.random.default_rng(0); n = 0
    for seq, imgs, gt in pose_sequences(setname):
        f = os.path.join(od, seq + ".json")
        if os.path.exists(f):
            continue
        rows = []
        for w in pm.windows(len(imgs), 8, 1):
            views = load_imgs([imgs[i] for i in w]); g = gt[w]
            o, _ = H.run(views); base = pm.pose_metrics(o["cam_pose"].double().cpu().numpy(), g)
            row = {"frames": w, "base": base}
            if a.variant in ("a", "b", "c"):
                key = mpa.probe_mask(H, views, probe); row["mask_frac"] = float(key.float().mean())
                for r in range(a.n_random if a.variant == "c" else 1):
                    hs = mask_heads(a.variant, a.kstar_cont, rng)
                    oe, _ = H.run(views, masks=masks_for(hs, key))
                    row[f"edit_r{r}"] = pm.pose_metrics(oe["cam_pose"].double().cpu().numpy(), g)
            elif a.variant in ("i", "ii", "iii", "iv"):
                pass
            rows.append(row)
        json.dump(dict(set=setname, seq=seq, variant=a.variant, rows=rows), open(f, "w"), indent=1)
        n += 1; print("[u5b real]", setname, seq, len(rows), "windows", flush=True)
        if a.limit and n >= a.limit:
            break


def main():
    global mpa
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["dirs", "u5a", "u5b"])
    p.add_argument("--variant", default="i")
    p.add_argument("--g", type=float, default=2.0)
    p.add_argument("--n_random", type=int, default=20)
    p.add_argument("--kstar_obj", type=int, default=0, help="k* of P_obj (0 = only the five E5 heads)")
    p.add_argument("--kstar_cont", type=int, default=5)
    p.add_argument("--eval", default="kubric", help="kubric | sintel | tum | bonn")
    p.add_argument("--split", default="fit"); p.add_argument("--allow_test", action="store_true")
    p.add_argument("--scenes", default=""); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ckpt", default="", help="heads_sel (variant v) or a path")
    p.add_argument("--out_tag", default="")
    a = p.parse_args()
    if (a.split in ("test", "transfer") or a.eval != "kubric") and not a.allow_test:
        sys.exit("test, transfer and real-data evaluation require --allow_test (after the Oct 27 pre-registration)")
    if a.mode == "dirs" and a.split not in ("fit", "any"):
        sys.exit("directions are fitted on fit scenes only")
    import motion_probe_allviews as _m; mpa = _m
    scenes = ic.split_scenes(a.split)
    if a.scenes:
        w = set(a.scenes.split(",")); scenes = [d for d in scenes if ic.sid(d) in w]
    if a.limit and a.eval == "kubric":
        scenes = scenes[:a.limit]
    ck = LORA.get(a.ckpt, a.ckpt) if (a.ckpt or a.variant == "v") else None
    if a.variant == "v" and not ck:
        ck = LORA["heads_sel"]
    H = ah.Any4DHooks(ah.load_any4d(ckpt=ck))
    os.makedirs(OUT, exist_ok=True)
    print(f"[repair] mode={a.mode} variant={a.variant} g={a.g} eval={a.eval} split={a.split} scenes={len(scenes)} ckpt={ck}", flush=True)
    if a.mode == "dirs":
        heads = head_set(a.kstar_obj); dirs = fit_dirs(H, scenes, heads)
        torch.save(dirs, os.path.join(OUT, "u5a_dirs.pt"))
        json.dump({f"b{b}h{h}": {"cos_OC": v["cos_OC"]} for (b, h), v in dirs.items()},
                  open(os.path.join(OUT, "u5a_dirs_summary.json"), "w"), indent=1)
        print("[u5a dirs] saved", len(dirs), "heads")
    elif a.mode == "u5a":
        run_u5a(H, a, scenes)
    else:
        probe = mpa.load_probe() if a.variant != "d" else None
        if a.eval == "kubric":
            run_u5b_kubric(H, a, scenes, probe)
        else:
            run_u5b_real(H, a, probe, a.eval)


if __name__ == "__main__":
    main()
