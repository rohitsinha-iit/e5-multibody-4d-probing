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
Boundary rule (pre-registration): Kubric metrics exclude query points within 8 px (at 512) of a moving-object boundary at
  frame 0 (primary); every metric is also stored without the rule (suffix _nb).
Frozen inputs: every head list, projector and probe comes from the fit_derived_values block
  ($E5_BIG/results/u4a/fit_derived_values.json, identical to the block appended to attr_preregistration.json).
Outputs: U5a -> $E5_BIG/results/u5a/<eval>/<variant>_g<g>/<scene>.json; U5b -> $E5_BIG/results/u5b/{kubric,real}/.
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
OUT5A = os.environ.get("U5A_RES", os.path.join(E5_BIG, "results", "u5a"))
OUT5B = os.environ.get("U5B_RES", os.path.join(E5_BIG, "results", "u5b"))
FROZEN = os.environ.get("FIT_FROZEN", os.path.join(E5_BIG, "results", "u4a", "fit_derived_values.json"))
POSESETS = os.path.join(E5_BIG, "realdata", "posesets")
REALGT = os.path.join(E5_BIG, "realdata", "gt")
POSE_WINDOWS = os.path.join(E5_BIG, "results", "u2", "pose_windows.json")
SINTEL14 = ["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", "market_2", "market_5", "market_6",
            "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"]
LORA = {"heads_sel": os.path.join(E5_BIG, "ft", "runs", "heads_sel", "any4d_ft.pth")}
TV = 6
GS = [1.5, 2.0]
N_RANDOM = 20


def frozen():
    return json.load(open(FROZEN))


def sha256(f):
    import hashlib
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ U5a directions
def head_set_from_ptobj(p_obj):
    hs = set(ah.FIVE)
    for n in p_obj:
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


def random_projectors(heads, n=N_RANDOM):
    """iii: per draw r a torch.Generator seeded 1000 + r; heads visited in sorted order; one random 2D subspace per
    head (QR of a 64x2 Gaussian), P = Q Q^T (rank 2, Frobenius norm sqrt(2), as P_O and P_C)."""
    out = {hh: [] for hh in heads}
    for r in range(n):
        gen = torch.Generator().manual_seed(1000 + r)
        for hh in sorted(heads):
            Q, _ = torch.linalg.qr(torch.randn(64, 2, generator=gen)); out[hh].append(Q @ Q.T)
    return {hh: torch.stack(v) for hh, v in out.items()}


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


def edit_fns(dirs, rand, variant, g, r=0):
    fns = {}
    for (b, h), dd in dirs.items():
        if variant == "iv":       # E5 whole-head gain: per-run token mean, P = I
            fns[(b, h)] = (lambda x, n, g=g: x.mean(0, keepdim=True) + g * (x - x.mean(0, keepdim=True)))
            continue
        m = dd["m"].cuda()
        P = {"i": lambda: dd["P_O"], "ii": lambda: dd["P_C"], "iii": lambda: rand[(b, h)][r]}[variant]().cuda()
        fns[(b, h)] = (lambda x, n, P=P, m=m, g=g: x + (g - 1) * (x - m) @ P)
    return fns


# ------------------------------------------------------------------ U5b masks
def masks_for(heads, key):
    m = {}
    for b, h in heads:
        m.setdefault(b, ah.KeyMask([], key)).heads.append(h)
    return m


def random_head_sets(p_cont, n=N_RANDOM, seed=0):
    """c: n sets of k* random heads, block-matched to P_cont (one head of the same block per P_cont head), excluding
    P_cont heads; numpy default_rng(seed), drawn once and frozen."""
    rng = np.random.default_rng(seed); pc = [tuple(x) for x in p_cont]; sets = []
    for _ in range(n):
        out = []
        for b, h in pc:
            pool = [hh for hh in range(12) if (b, hh) not in pc and (b, hh) not in out]
            out.append((b, int(rng.choice(pool))))
        sets.append(out)
    return sets


ALL_HEADS = [(b, h) for b in range(24) for h in range(12)]


# ------------------------------------------------------------------ Kubric metrics
def kubric_metrics(o, d, vn="D"):
    g = ic.gt_of(d, vn)
    P = ah.to_numpy(o); off = np.array(s3.pixel_offset(P))
    xp518 = s3.pred_correspondences(P, TV, off)
    uv = ic.to518(g["q_pix"]); xp = xp518[uv[:, 1], uv[:, 0]]; xp = (xp + 0.5) * 512 / 518 - 0.5
    x1 = g["x1"]; A = g["A"][-1]; Bc = g["B"][-1]; xt = g["x_true"][-1]; vis = g["vis"][-1]
    g0 = g["group0"].astype(int)
    keepmap = ws.boundary_mask(g0, moving_ids=[int(i) for i in np.unique(g0[g0 > 0])], radius_px=8, W_ref=512)
    keep = keepmap[g["q_pix"][:, 1].astype(int), g["q_pix"][:, 0].astype(int)]
    sf = P["scene_flow"][-1][uv[:, 1], uv[:, 0]]
    T0 = P["cam_pose"][0]; zp = ((P["pts3d_0"][uv[:, 1], uv[:, 0]] - T0[:3, 3]) @ T0[:3, :3])[:, 2]
    w2c0 = g["w2c_cv"][0]
    X = g["pts0"][g["q_pix"][:, 1], g["q_pix"][:, 0]]; zg = (X @ w2c0[:3, :3].T + w2c0[:3, 3])[:, 2]
    err = np.linalg.norm(xp - xt, axis=1); y = xp - x1
    out = {}
    for suf, kp in (("", keep), ("_nb", np.ones_like(keep))):
        mv = (g["q_group"] > 0) & vis & kp; st = (g["q_group"] == 0) & vis & kp
        ab = fit_ab(A[mv], Bc[mv], y[mv]) if mv.sum() > 20 else {"alpha": float("nan"), "beta": float("nan"), "cond_AB": float("nan")}
        ident = bool(ab["cond_AB"] <= 10 and np.median(np.linalg.norm(A[mv], axis=1)) >= 2
                     and np.median(np.linalg.norm(Bc[mv], axis=1)) >= 2) if mv.sum() > 20 else False
        sw = ws.world_slope(sf, g["world_disp"][-1], (g["q_group"] > 0) & vis, (g["q_group"] == 0) & vis, z_gt=zg,
                            z_pred=zp, focal=float(g["K"][0, 0]), keep=kp)
        out.update({"alpha" + suf: ab["alpha"], "beta" + suf: ab["beta"], "cond_AB" + suf: ab["cond_AB"],
                    "identifiable" + suf: ident, "s_w" + suf: sw["s_w"], "static_residual" + suf: sw["static_residual"],
                    "epe_moving" + suf: float(np.median(err[mv])) if mv.any() else float("nan"),
                    "epe_static" + suf: float(np.median(err[st])) if st.any() else float("nan"),
                    "n_moving" + suf: int(mv.sum())})
    gtc = np.linalg.inv(g["w2c_cv"][ah.FRAMES])
    rt, rr = pm.rpe(P["cam_pose"], gtc)
    out.update(rpe_t=rt, rpe_r=rr, ate=pm.ate(P["cam_pose"], gtc))
    return out


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



# ------------------------------------------------------------------ real flow (E5 KITTI / Sintel protocol)
REAL_FLOW = {"kitti15": ("rgb", "0,1"), "sintel": ("final", "0,1")}   # E5 FT evaluation: 2 views, wide input


def real_flow_samples(ds, limit=0):
    files = sorted(f for f in glob.glob(os.path.join(REALGT, ds, "*.npz")) if not f.endswith(".tmp.npz"))
    pas = REAL_FLOW[ds][0]; out = []
    for f in files:
        z = np.load(f, allow_pickle=False); meta = json.loads(str(z["meta"]))
        fr = meta["frames"].get(pas, {})
        if not all(k in fr for k in ("0", "1")):
            continue
        out.append((os.path.basename(f)[:-4], f, [fr["0"], fr["1"]]))
    return out[:limit] if limit else out


class RealFlow:
    """E5 real-data flow protocol (src/e5_realdata_run.py + e5_realdata_eval.py): Any4D adapter of e5_models with
    resize 'wide' (fixed_mapping), offsets 0,1; edits are installed on the adapter's network through Any4DHooks."""
    def __init__(self, ckpt=None):
        sys.path.insert(0, os.path.join(ah.ANY4D, "scripts")); os.chdir(ah.ANY4D)
        import e5_models, e5_realdata_eval  # noqa
        self.ev = e5_realdata_eval
        kw = {"checkpoint_path": ckpt} if ckpt else {}
        self.m = e5_models.get_model("any4d", resize_mode="fixed_mapping", **kw)
        self.H = ah.Any4DHooks(self.m.model)

    def run(self, f, paths, edits=None, seed=0):
        z = np.load(f, allow_pickle=False); meta = json.loads(str(z["meta"]))
        gt = {k: z[k] for k in ("x2_true", "x2_static", "group", "valid", "label") if k in z.files}
        self.H.configure(edits=edits)
        try:
            R = self.m.predict(paths)
        finally:
            self.H._reset()
        P = dict(pts3d_0=R["pts3d_0"], scene_flow=R["scene_flow"][[0, 1]], cam_pose=R["cam_pose"][[0, 1]],
                 K=R["K"][[0, 1]], geom=R["geom"])
        row = self.ev.evaluate(gt, meta, P, ti=1, seed=seed)
        return {k: row[k] for k in ("slope", "slope_inst_median", "epe_static", "epe_moving", "n_eval", "n_moving_eval")}


# ------------------------------------------------------------------ drivers
def load_dirs():
    fz = frozen()["u5a"]
    dirs = torch.load(fz["projector_file"]); rand = torch.load(fz["random_projector_file"])
    return dirs, rand


def u5a_plan(dirs, rand, variants):
    """(name, edits) for every edit run of variants i..iv at both g values (base run is separate)."""
    plan = []
    for g in GS:
        for v in variants:
            for r in (range(N_RANDOM) if v == "iii" else [0]):
                plan.append((f"{v}_g{g:g}", r, edit_fns(dirs, rand, v, g, r)))
    return plan


def run_u5a(a, scenes):
    """One process: base + variants i, ii, iii (20 draws), iv at g = 1.5 and 2 per scene or sample
    (a.variant = 'all'), or variant v alone (a.variant = 'v', LoRA checkpoint, g-independent)."""
    ev = a.eval
    od = os.path.join(OUT5A, ev, "v" if a.variant == "v" else "edits"); os.makedirs(od, exist_ok=True)
    if a.variant == "v":
        plan = [("v", 0, None)]
    else:
        dirs, rand = load_dirs(); plan = [("base", 0, None)] + u5a_plan(dirs, rand, ["i", "ii", "iii", "iv"])
    ck = LORA["heads_sel"] if a.variant == "v" else None
    if ev == "kubric":
        H = ah.Any4DHooks(ah.load_any4d(ckpt=ck)); items = [(ic.sid(d), d) for d in scenes]
    else:
        RF = RealFlow(ckpt=ck); items = [(n, (f, p)) for n, f, p in real_flow_samples(ev, a.limit)]
    print(f"[u5a] eval={ev} variant={a.variant} items={len(items)} runs/item={len(plan)}", flush=True)
    for name, it in items:
        f = os.path.join(od, name + ".json")
        if os.path.exists(f):
            continue
        t0 = time.time(); rows = {}
        if ev == "kubric":
            views = ah.load_views(os.path.join(it, "D"))
        for (vn, r, fns) in plan:
            if ev == "kubric":
                o, _ = H.run(views, edits=fns); m = kubric_metrics(o, it)
            else:
                m = RF.run(it[0], it[1], edits=fns)
            rows.setdefault(vn, {})[f"r{r}"] = m
        json.dump(dict(item=name, eval=ev, variant=a.variant, rows=rows, wall_s=round(time.time() - t0, 1)),
                  open(f + ".tmp", "w"), indent=1); os.replace(f + ".tmp", f)
        print("[u5a]", ev, name, a.variant, round(time.time() - t0, 1), "s", flush=True)


def u5b_runs(H, views, key_probe, key_gt, fz):
    """Variants a (probe mask in P_cont), b (probe mask, all heads), c (probe mask in 20 frozen random head sets),
    d (GT mask, all heads; only when key_gt is given)."""
    out = {}
    o, _ = H.run(views, masks=masks_for([tuple(x) for x in fz["p_cont"]], key_probe)); out["a"] = {"r0": o}
    o, _ = H.run(views, masks=masks_for(ALL_HEADS, key_probe)); out["b"] = {"r0": o}
    out["c"] = {}
    for r, hs in enumerate(fz["random_head_sets"]):
        o, _ = H.run(views, masks=masks_for([tuple(x) for x in hs], key_probe)); out["c"][f"r{r}"] = o
    if key_gt is not None:
        o, _ = H.run(views, masks=masks_for(ALL_HEADS, key_gt)); out["d"] = {"r0": o}
    return out


def run_u5b_kubric(H, a, scenes, probe):
    import motion_probe_allviews as mpa
    fz = frozen()["u5b"]
    od = os.path.join(OUT5B, "kubric", a.split); os.makedirs(od, exist_ok=True)
    for d in scenes:
        f = os.path.join(od, ic.sid(d) + ".json")
        if os.path.exists(f):
            continue
        t0 = time.time()
        views = ah.load_views(os.path.join(d, "D"))
        oD, _ = H.run(views); rD = kubric_metrics(oD, d)
        oB, _ = H.run(ah.load_views(os.path.join(d, "B"))); rB = kubric_metrics(oB, d, "B")
        key_gt = ah.token_mask_from_pixels(ic.gt_of(d, "D")["moving_mask"][ah.FRAMES], 0.25).reshape(-1).cuda()
        key = mpa.probe_mask(H, views, probe)
        lab = ah.token_mask_from_pixels(ic.gt_of(d, "D")["moving_mask"][ah.FRAMES], 0.5).reshape(-1).cuda()
        rows = {"base_D": rD, "base_B": rB, "mask_frac": float(key.float().mean()),
                "probe_iou_gt": float((key & lab).sum() / max(int((key | lab).sum()), 1)),
                "probe_recall_gt": float((key & lab).sum() / max(int(lab.sum()), 1))}
        eB = e_norm(rB, rD)
        for vn, rr in u5b_runs(H, views, key, key_gt, fz).items():
            for r, o in rr.items():
                m = kubric_metrics(o, d); eE = e_norm(m, rD)
                m["recovery"] = (1.0 - eE) / (1.0 - eB) if abs(1.0 - eB) > 1e-9 else float("nan")
                rows.setdefault(vn, {})[r] = m
        json.dump(dict(scene=ic.sid(d), rows=rows, wall_s=round(time.time() - t0, 1)), open(f + ".tmp", "w"), indent=1)
        os.replace(f + ".tmp", f)
        print("[u5b kubric]", ic.sid(d), round(time.time() - t0, 1), "s", flush=True)


# ------------------------------------------------------------------ pose windows written by the behaviour track
def run_u5b_real(H, a, probe):
    """Exactly the behaviour track's windows (results/u2/pose_windows.json, written once by u2_common.make_pose_windows):
    each window's own image paths and its GT through u2_common.window_gt_c2w (same frames and GT as the MoRe rows)."""
    import motion_probe_allviews as mpa
    import u2_common as u2
    fz = frozen()["u5b"]
    J = json.load(open(a.windows)); wins = J["windows"]
    od = os.path.join(OUT5B, "real"); os.makedirs(od, exist_ok=True)
    json.dump(dict(source=a.windows, sha256=sha256(a.windows), n_windows=len(wins), per_set=J.get("per_set")),
              open(os.path.join(od, "_windows_used.json"), "w"), indent=1)
    print(f"[u5b real] windows {len(wins)} {J.get('per_set')}", flush=True)
    if a.limit:
        wins = wins[:a.limit]
    for w in wins:
        wid = w["id"].replace("/", "__")
        f = os.path.join(od, wid + ".json")
        if os.path.exists(f):
            continue
        t0 = time.time()
        views = load_imgs(w["frames"]); g = u2.window_gt_c2w(w)
        o, _ = H.run(views); base = pm.pose_metrics(o["cam_pose"].double().cpu().numpy(), g)
        key = mpa.probe_mask(H, views, probe)
        row = dict(set=w["set"], seq=w["seq"], frame_idx=w["frame_idx"], id=w["id"], base=base,
                   mask_frac=float(key.float().mean()))
        for vn, rr in u5b_runs(H, views, key, None, fz).items():
            row[vn] = {r: pm.pose_metrics(oo["cam_pose"].double().cpu().numpy(), g) for r, oo in rr.items()}
        json.dump(row, open(f + ".tmp", "w"), indent=1); os.replace(f + ".tmp", f)
        print("[u5b real]", w["id"], round(time.time() - t0, 1), "s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["dirs", "u5a", "u5b"])
    p.add_argument("--variant", default="all", help="u5a: all | v")
    p.add_argument("--eval", default="kubric", help="kubric | kitti15 | sintel (u5a); kubric | real (u5b)")
    p.add_argument("--split", default="fit"); p.add_argument("--allow_test", action="store_true")
    p.add_argument("--scenes", default=""); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--p_obj", default="", help="dirs: comma head list (P_obj = objall top-k*)")
    p.add_argument("--dirs_out", default=os.path.join(OUT5A, "u5a_dirs.pt"))
    p.add_argument("--windows", default=POSE_WINDOWS)
    a = p.parse_args()
    if (a.split in ("test", "transfer") or a.eval != "kubric") and not a.allow_test:
        sys.exit("test, transfer and real-data evaluation require --allow_test (after the fit-derived freeze)")
    if a.mode == "dirs" and a.split not in ("fit", "any"):
        sys.exit("directions are fitted on fit scenes only")
    if a.mode != "dirs" and (a.split in ("test", "transfer") or a.eval != "kubric") and not os.path.exists(FROZEN):
        sys.exit("fit_derived_values not frozen yet: " + FROZEN)
    scenes = ic.split_scenes(a.split)
    if a.scenes:
        w = set(a.scenes.split(",")); scenes = [d for d in scenes if ic.sid(d) in w]
    if a.limit and a.eval == "kubric":
        scenes = scenes[:a.limit]
    os.makedirs(OUT5A, exist_ok=True); os.makedirs(OUT5B, exist_ok=True)
    print(f"[repair] mode={a.mode} variant={a.variant} eval={a.eval} split={a.split} scenes={len(scenes)}", flush=True)
    if a.mode == "dirs":
        heads = head_set_from_ptobj([x for x in a.p_obj.split(",") if x])
        H = ah.Any4DHooks(ah.load_any4d())
        dirs = fit_dirs(H, scenes, heads)
        torch.save(dirs, a.dirs_out)
        rp = random_projectors(heads); torch.save(rp, a.dirs_out.replace(".pt", "_random_projectors.pt"))
        json.dump({f"b{b}h{h}": {"cos_OC": v["cos_OC"]} for (b, h), v in dirs.items()},
                  open(a.dirs_out.replace(".pt", "_summary.json"), "w"), indent=1)
        print("[u5a dirs] saved", len(dirs), "heads", flush=True)
    elif a.mode == "u5a":
        run_u5a(a, scenes)
    else:
        import motion_probe_allviews as mpa
        H = ah.Any4DHooks(ah.load_any4d())
        probe = mpa.load_probe(frozen()["probe"]["file"])
        if a.eval == "kubric":
            run_u5b_kubric(H, a, scenes, probe)
        else:
            run_u5b_real(H, a, probe)


if __name__ == "__main__":
    main()
