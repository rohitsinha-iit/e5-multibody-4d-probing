"""Item 5: is the linearly decodable motion signal used?  LEACE erasure and steering at layers 17-21. Place in Any4D/scripts/.

Concept Z (per view-0 patch, pure patches only): [moving (0/1), dx/50, dy/50] with (dx, dy) = x2_true - x2_static (518 px),
i.e. "is this a moving object" and "how far does it move beyond the camera-induced motion".
Fitting: 2-fold by seed parity; erasers and directions for a scene are fit on the other fold (view-0 tokens from internals/).
LEACE (Belrose et al. 2023), closed form:  x' = x - W+ U U^T W (x - mu),  W = Sigma_xx^-1/2,  U = orthonormal basis of W Sigma_xz.
Random control: same form with Sigma_xz replaced by a Gaussian matrix of the same shape (rank-matched erasure).
Interventions act on every patch token of every view at the residual stream after block L-1 (layer index L, as in internals).
Steering: u = unit(mean_moving - mean_static) at layer L, gap D = (mean_moving - mean_static) . u.
  steer_mov_b{beta}  : tokens on moving objects (all views, from Kubric segmentation) x <- x - beta * D * u  (look static)
  steer_stat_b{beta} : tokens on static objects                                  x <- x + beta * D * u  (look moving)
  steerrand_*        : same with a random unit direction.
Metrics (frame 0 -> 23): slope on moving pixels, epe_moving, epe_static_obj (static-object pixels), epe_bg (background),
  spurious_static_obj = median |pred - static hypothesis| on static-object pixels (px).
Resumable (<out>.partial).
"""
import argparse, glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from demo_inference import init_inference_model, sample_inference  # noqa: E402
from any4d.utils.image import load_images  # noqa: E402
from any4d.utils.geometry import quaternion_to_rotation_matrix, recover_pinhole_intrinsics_from_ray_directions  # noqa: E402
from any4d.utils.misc import seed_everything  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True); p.add_argument("--gt", required=True); p.add_argument("--internals", required=True)
p.add_argument("--e5", required=True); p.add_argument("--out", required=True)
p.add_argument("--layers", default="17,18,19,20,21"); p.add_argument("--steer_layer", type=int, default=19)
p.add_argument("--limit", type=int, default=0)
args = p.parse_args()
sys.path.insert(0, args.e5)
import e5_kubric_post as kp  # noqa: E402
import e5_step3_analysis as s3  # noqa: E402
seed_everything(0)
FRAMES = [0, 4, 8, 12, 16, 20, 23]; V = len(FRAMES); TV = V - 1; G = 37; T = G * G
LAYERS = [int(x) for x in args.layers.split(",")]; SL = args.steer_layer
dev = torch.device("cuda")
cfg = {"path": "configs/train.yaml",
       "config_overrides": ["machine=local", "model=any4d", "model.encoder.uses_torch_hub=false", "model/task=images_only"],
       "checkpoint_path": "checkpoints/any4d_4v_combined.pth", "trained_with_amp": True, "data_norm_type": "dinov2"}
model = init_inference_model(cfg, cfg["checkpoint_path"], dev)
blocks = model.info_sharing.self_attention_blocks

scenes = sorted(d for d in glob.glob(os.path.join(args.data, "*")) if os.path.exists(os.path.join(d, "e5_meta.json"))
                and os.path.exists(os.path.join(args.internals, os.path.basename(d) + "_tok.npy")))
if args.limit:
    scenes = scenes[: args.limit]
names = [os.path.basename(d) for d in scenes]
fold = {n: int(n.split("_s")[-1]) % 2 for n in names}
_sl = os.path.join(args.internals, "stored_layers.json")
STORED = json.load(open(_sl)) if os.path.exists(_sl) else list(range(25))

# ------------------------------------------------ fit erasers and directions per fold
def load_train(f, L):
    Xs, Zs = [], []
    for n in names:
        if fold[n] != f:
            continue
        g = np.load(os.path.join(args.gt, f"{n}_gt.npz"), allow_pickle=True)
        pure = (g["purity"] >= 0.8) & g["vis"]
        tok = np.load(os.path.join(args.internals, f"{n}_tok.npy"), mmap_mode="r")[STORED.index(L), 0]
        X = np.asarray(tok[pure]).astype(np.float64)
        mov = (g["body"][pure] > 0).astype(np.float64)
        d = (g["x2_true"] - g["x2_static"])[pure] / 50.0 * mov[:, None]
        Xs.append(X); Zs.append(np.c_[mov, d])
    return np.concatenate(Xs), np.concatenate(Zs)


def leace(X, Z, rand=None):
    mu = X.mean(0); Xc = X - mu
    Sxx = Xc.T @ Xc / len(X); Sxx += np.eye(len(mu)) * 1e-5 * np.trace(Sxx) / len(mu)
    Sxz = Xc.T @ (Z - Z.mean(0)) / len(X)
    if rand is not None:
        Sxz = rand.normal(size=Sxz.shape) * np.linalg.norm(Sxz) / np.sqrt(Sxz.size)
    ev, evec = np.linalg.eigh(Sxx)
    Wm = evec @ np.diag(ev ** -0.5) @ evec.T; Wp = evec @ np.diag(ev ** 0.5) @ evec.T
    Uu, sv, _ = np.linalg.svd(Wm @ Sxz, full_matrices=False); Uu = Uu[:, sv > sv.max() * 1e-6]
    A = Wp @ Uu @ Uu.T @ Wm
    return torch.tensor(mu, dtype=torch.float32, device=dev), torch.tensor(A, dtype=torch.float32, device=dev)


FIT = {}
rng = np.random.default_rng(0)
for f in (0, 1):
    for L in sorted(set(LAYERS + [SL])):
        X, Z = load_train(f, L)
        mu, A = leace(X, Z)
        mr, Ar = leace(X, Z, rand=np.random.default_rng(100 + L))
        mv = Z[:, 0] > 0.5
        dvec = X[mv].mean(0) - X[~mv].mean(0); D = float(np.linalg.norm(dvec)); uvec = dvec / D
        r = rng.normal(size=uvec.shape); r -= (r @ uvec) * uvec; r /= np.linalg.norm(r)
        # sanity: linear decodability of "moving" before/after erasure on the training fold (least squares)
        Xt = torch.tensor(X, dtype=torch.float32, device=dev)
        Xe = (Xt - (Xt - mu) @ A.T).cpu().numpy()
        def r2(M):
            M1 = np.c_[M, np.ones(len(M))]; w = np.linalg.lstsq(M1, Z[:, 0], rcond=None)[0]
            return float(1 - ((M1 @ w - Z[:, 0]) ** 2).mean() / Z[:, 0].var())
        FIT[(f, L)] = dict(mu=mu, A=A, mur=mr, Ar=Ar, u=torch.tensor(uvec, dtype=torch.float32, device=dev),
                           ur=torch.tensor(r, dtype=torch.float32, device=dev), D=D,
                           r2_before=r2(X), r2_after=r2(Xe), n=len(X))
        print("fit fold", f, "layer", L, "n", len(X), "R2 moving before/after", round(FIT[(f, L)]["r2_before"], 3),
              round(FIT[(f, L)]["r2_after"], 4), "gap", round(D, 2), flush=True)

# ------------------------------------------------ hooks on block L-1 outputs
active = {"erase": {}, "steer": {}}   # erase: {L: (mu, A)}, steer: {L: (vec, mask_tokens (V*T,) float)}


def patch_view(x):
    """Return a (V*T, C) view of the patch tokens and a writer back into x."""
    if x.dim() == 3 and x.shape[0] == V:
        return x[:, :T].reshape(V * T, -1), lambda y: x[:, :T].copy_(y.reshape(V, T, -1))
    return x[0, : V * T], lambda y: x[0, : V * T].copy_(y)


def make_hook(L):
    def h(mod, inp, out):
        if L not in active["erase"] and L not in active["steer"]:
            return None
        x = out.clone(); pt, write = patch_view(x)
        y = pt.float()
        if L in active["erase"]:
            mu, A = active["erase"][L]; y = y - (y - mu) @ A.T
        if L in active["steer"]:
            vec, m = active["steer"][L]; y = y + m[:, None] * vec[None]
        write(y.to(x.dtype)); return x
    return h


for L in sorted(set(LAYERS + [SL])):
    blocks[L - 1].register_forward_hook(make_hook(L))


def run(views):
    with torch.no_grad():
        res = sample_inference(model, views, dev, use_amp=True)
    pr = [res[f"pred{i+1}"] for i in range(V)]
    out = {"pts3d_0": pr[0]["pts3d"][0].float().cpu().numpy()}
    out["scene_flow"] = np.stack([np.zeros_like(out["pts3d_0"])] + [pr[i]["scene_flow"][0].float().cpu().numpy() for i in range(1, V)])
    poses, Ks = [], []
    for i in range(V):
        R = quaternion_to_rotation_matrix(pr[i]["cam_quats"][0].float().cpu())
        Tm = np.eye(4); Tm[:3, :3] = R.numpy(); Tm[:3, 3] = pr[i]["cam_trans"][0].float().cpu().numpy(); poses.append(Tm)
        Ks.append(recover_pinhole_intrinsics_from_ray_directions(pr[i]["ray_directions"][0].float().cpu()).numpy())
    out["cam_pose"] = np.stack(poses); out["K"] = np.stack(Ks)
    return out


def patch_classes(meta, gt):
    """(V*T,) class per patch token: 0 background, 1 static object, 2 moving object, -1 mixed (purity < 0.6)."""
    s2g = kp.seg_to_group(meta); out = []
    for f in FRAMES:
        seg = gt["segmentation"][f, ..., 0]
        cls = np.zeros_like(seg, np.int64)
        for sid, (grp, k) in s2g.items():
            if k is None or k == "dome" or (isinstance(k, (int, np.integer)) and k < 0):
                continue
            cls[seg == sid] = 2 if grp > 0 else 1
        H = seg.shape[0]; ii = np.clip(((np.arange(518) + 0.5) * H / 518).astype(int), 0, H - 1)
        c518 = cls[np.ix_(ii, ii)].reshape(G, 14, G, 14).transpose(0, 2, 1, 3).reshape(T, 196)
        cnt = np.stack([(c518 == k).sum(1) for k in range(3)], 1)
        lab = cnt.argmax(1); lab[cnt.max(1) < 0.6 * 196] = -1
        out.append(lab)
    return np.concatenate(out)


rows = json.load(open(args.out + ".partial"))["rows"] if os.path.exists(args.out + ".partial") else []
done = {r["scene"] for r in rows}
for si, (d, n) in enumerate(zip(scenes, names)):
    if n in done:
        continue
    F_ = 1 - fold[n]
    meta, gt, _ = kp.load_scene(d)
    W = meta["resolution"][0]; s = 518 / W; b = len(meta["frames"]) - 1
    c = kp.correspondences(meta, gt, 0, b)
    X3 = kp.backproject(meta, gt["depth"][0, ..., 0], 0); xs_dense, _ = kp.project(meta, X3, b)
    x1 = c["x1"]; grp = c["group"]
    xs = xs_dense[x1[:, 1].astype(int), x1[:, 0].astype(int)]
    u_ = np.clip(np.round((x1[:, 0] + 0.5) * s - 0.5).astype(int), 0, 517); v_ = np.clip(np.round((x1[:, 1] + 0.5) * s - 0.5).astype(int), 0, 517)
    s2g = kp.seg_to_group(meta); seg0 = gt["segmentation"][0, ..., 0]
    obj_px = np.zeros_like(seg0, bool)
    for sid, (g_, k) in s2g.items():
        if isinstance(k, (int, np.integer)) and k >= 0:
            obj_px |= seg0 == sid
    is_obj = obj_px[x1[:, 1].astype(int), x1[:, 0].astype(int)]
    cls = torch.tensor(patch_classes(meta, gt), device=dev)
    views = load_images([os.path.join(d, f"rgba_{f:05d}.png") for f in FRAMES], resize_mode="square", size=518,
                        norm_type=cfg["data_norm_type"], patch_size=14)

    def measure(cname):
        P = run(views)
        off = np.array(s3.pixel_offset(P))
        xp = s3.pred_correspondences(P, TV, off)[v_, u_]; xp = (xp + 0.5) / s - 0.5
        err = np.linalg.norm(xp - c["x2"], axis=1); mv = grp > 0; so = (~mv) & is_obj; bg = (~mv) & (~is_obj)
        dt = c["x2"] - xs; dp = xp - xs; big = mv & (np.linalg.norm(dt, axis=1) > 2)
        rows.append(dict(scene=n, split=meta["split"], condition=cname,
                         slope=float((dp[big] * dt[big]).sum() / (dt[big] ** 2).sum()) if big.sum() > 20 else float("nan"),
                         epe_moving=float(np.median(err[mv])) if mv.any() else float("nan"),
                         epe_static_obj=float(np.median(err[so])) if so.sum() > 20 else float("nan"),
                         epe_bg=float(np.median(err[bg])),
                         spurious_static_obj=float(np.median(np.linalg.norm(dp[so], axis=1))) if so.sum() > 20 else float("nan")))
        active["erase"] = {}; active["steer"] = {}

    def fit(L):
        return FIT[(F_, L)]

    measure("base")
    for L in LAYERS:
        active["erase"] = {L: (fit(L)["mu"], fit(L)["A"])}; measure(f"leace_L{L}")
    active["erase"] = {L: (fit(L)["mu"], fit(L)["A"]) for L in LAYERS}; measure("leace_all")
    active["erase"] = {SL: (fit(SL)["mur"], fit(SL)["Ar"])}; measure(f"rand_L{SL}")
    active["erase"] = {L: (fit(L)["mur"], fit(L)["Ar"]) for L in LAYERS}; measure("rand_all")
    mov_m = (cls == 2).float(); stat_m = (cls == 1).float()
    for beta in (0.5, 1.0, 2.0):
        fz = fit(SL); active["steer"] = {SL: (-beta * fz["D"] * fz["u"], mov_m)}; measure(f"steer_mov_b{beta:g}")
    for beta in (1.0, 2.0):
        fz = fit(SL); active["steer"] = {SL: (beta * fz["D"] * fz["u"], stat_m)}; measure(f"steer_stat_b{beta:g}")
    fz = fit(SL)
    active["steer"] = {SL: (-2.0 * fz["D"] * fz["ur"], mov_m)}; measure("steerrand_mov_b2")
    active["steer"] = {SL: (2.0 * fz["D"] * fz["ur"], stat_m)}; measure("steerrand_stat_b2")
    active["steer"] = {L: (-1.0 * fit(L)["D"] * fit(L)["u"], mov_m) for L in LAYERS}; measure("steer_mov_all_b1")
    active["steer"] = {L: (1.0 * fit(L)["D"] * fit(L)["u"], stat_m) for L in LAYERS}; measure("steer_stat_all_b1")
    json.dump(dict(rows=rows), open(args.out + ".partial.tmp", "w")); os.replace(args.out + ".partial.tmp", args.out + ".partial")
    print(n, len(rows), flush=True)

fitinfo = {f"fold{f}_L{L}": dict(r2_before=v["r2_before"], r2_after=v["r2_after"], gap=v["D"], n=v["n"]) for (f, L), v in FIT.items()}
json.dump(dict(rows=rows, fit=fitinfo, layers=LAYERS, steer_layer=SL), open(args.out, "w"))
print("done", len(rows))
