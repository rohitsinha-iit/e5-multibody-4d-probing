"""Smoke tests and gates for the new U2 adapters (A6000 new-models track).

Run in the model's own venv, from anywhere:
  python src/attr/gate_newmodels.py --model opend4rt --smoke_dir <frames dir>  --out <json>
  python src/attr/gate_newmodels.py --model opend4rt --data $E5_BIG/data/v2 --n 10 --out <json> [--ref any4d.json]
Models: opend4rt (24 frames), trackcraft3r (12 frames [0,2,...,20,23], --cond gt|identity|da3),
        4rc and more (7 views [0,4,8,12,16,20,23]).
Per scene (frame 0 -> last frame, 512 native px): static EPE and moving EPE (median px) of the adapter's x_pred
(for MoRe, the rigid reprojection of its depth with its cameras), s_w and static residual (world_slope.py),
pose metrics against GT (pose.py), runtime and peak GPU memory. Gate pass rules are in GATES below.
"""
import argparse, glob, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.dirname(HERE)
sys.path += [HERE, SRC]   # appended: src/attr would shadow the attrs package (import attr) if placed first
import e5_models as em  # noqa: E402

VIEWS7 = [0, 4, 8, 12, 16, 20, 23]
FRAMES = dict(opend4rt=list(range(24)), trackcraft3r=em.TrackCraft3RAdapter.KUBRIC_FRAMES, **{"4rc": VIEWS7}, more=VIEWS7,
              any4d=VIEWS7)
GATES = {"opend4rt": "G0d: median static EPE <= 2x Any4D on the same scenes",
         "trackcraft3r": "G0b: median static EPE <= 10 px at 512 (condition gt, 10 v2 scenes)",
         "4rc": "G0g: output keys found (pts, track, extrinsic, intrinsic) and static EPE <= 2x Any4D",
         "more": "G0j: weights load and the full model runs",
         "any4d": "reference for the within-2x gates (existing Any4D class, 7 views, 518 square)"}


def build(model, args):
    if model == "opend4rt":
        return em.OpenD4RTAdapter(dense_stride=args.stride)
    if model == "trackcraft3r":
        return em.TrackCraft3RAdapter()
    if model == "4rc":
        return em.FourRCAdapter(square_ok=not args.no_square)
    if model == "more":
        return em.MoReAdapter()
    if model == "any4d":
        repo = os.path.join(em.E5_BIG_DEFAULT, "models/src/Any4D")
        os.chdir(repo); sys.path.insert(0, os.path.join(repo, "scripts"))
        return em.Any4D(checkpoint_path=os.path.join(em.E5_BIG_DEFAULT, "models/ckpt/any4d/any4d_4v_combined.pth"))
    raise KeyError(model)


def sample(arr, g, x, y):
    """arr (h,w,...) on the grid of geom g, values at native pixels (x,y) by nearest model pixel."""
    um, vm = em.native_to_model(g, x, y)
    ui = np.clip(np.round(um).astype(int), 0, arr.shape[1] - 1); vi = np.clip(np.round(vm).astype(int), 0, arr.shape[0] - 1)
    inside = (um > -0.5) & (um < arr.shape[1] - 0.5) & (vm > -0.5) & (vm < arr.shape[0] - 0.5)
    return arr[vi, ui], inside


def run_model(A, model, paths, meta=None, gt=None, frames=None, cond="gt", da3_dir=None, gpu=None):
    if model == "trackcraft3r":
        if meta is None or cond == "da3":
            import hashlib
            da3_dir = da3_dir or os.path.join(em.E5_BIG_DEFAULT, "results/newmodels/da3", hashlib.md5("|".join(paths).encode()).hexdigest()[:12])
            if not os.path.exists(os.path.join(da3_dir, "depth.npy")):
                em.TrackCraft3RAdapter.run_da3(paths, da3_dir, gpu=gpu)
            d, w2c, intr = em.TrackCraft3RAdapter.make_inputs("da3", da3_dir=da3_dir)
        else:
            d, w2c, intr = em.TrackCraft3RAdapter.make_inputs(cond, meta, gt, frames)
        return A.predict(paths, d, w2c, intr)
    if model == "opend4rt" and meta is not None:
        g = em.kubric_cv_geometry(meta, gt, frames)
        H, W = g["depth_z"].shape[1:]
        sx = (A.w - 1) / (W - 1); sy = (A.h - 1) / (H - 1)
        Km = g["K"].copy(); Km[0] *= sx; Km[1] *= sy
        import cv2
        dz = cv2.resize(g["depth_z"][0], (A.w, A.h), interpolation=cv2.INTER_NEAREST)
        return A.predict(paths, targets=[0, len(paths) - 1], gt=dict(K=Km, w2c=g["w2c"], depth0_z=dz))
    if model == "any4d":
        import torch, time
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        P = A.predict(paths); P.pop("_views", None)
        P["_runtime_s"] = time.time() - t0; P["_gpu"] = em._gpu_stats(torch)
        P["w_disp"] = em._to_cam0(P["scene_flow"], P)
        return P
    return A.predict(paths)


def summarize_keys(P):
    out = {}
    for k, v in P.items():
        if isinstance(v, np.ndarray):
            out[k] = dict(shape=list(v.shape), finite=float(np.isfinite(v).mean()) if v.dtype.kind == "f" else 1.0)
        elif isinstance(v, (int, float, str)):
            out[k] = v
    return out


def score_scene(model, P, meta, gt, frames):
    import e5_kubric_post as kp
    import world_slope as ws, pose as pm
    import e5_xm_eval as xe
    b = frames[-1]; T = len(frames) - 1
    c = kp.correspondences(meta, gt, frames[0], b)
    x1, x2, grp = c["x1"], c["x2"], c["group"]
    st_mask = xe.static_mask(meta, gt, frames[0])[x1[:, 1].astype(int), x1[:, 0].astype(int)]
    g = P["geom"][0]; gm = P.get("geom_model", P["geom"])[T]
    row = {}
    if "x_pred" in P and P["x_pred"] is not None:
        xp, ins = sample(P["x_pred"][T], g, x1[:, 0], x1[:, 1])
        xn = np.stack(em.model_to_native(gm, xp[:, 0], xp[:, 1]), -1)
    else:   # pointmap (+ scene flow if any, Any4D) projected with the model's own camera; MoRe: rigid reprojection
        X0, ins = sample(P["pts3d_0"], g, x1[:, 0], x1[:, 1])
        if P.get("scene_flow") is not None:
            X0 = X0 + sample(P["scene_flow"][T], g, x1[:, 0], x1[:, 1])[0]
        xp = em._project_cv(em._apply_T(np.linalg.inv(P["cam_pose"][T]), X0), P["K"][T])
        xn = np.stack(em.model_to_native(P["geom"][T], xp[:, 0], xp[:, 1]), -1)
    err = np.linalg.norm(xn - x2, axis=1)
    s = (grp == 0) & st_mask & ins; m = (grp > 0) & ins
    row.update(epe_static=float(np.nanmedian(err[s])), epe_moving=float(np.nanmedian(err[m])) if m.any() else float("nan"),
               n_static=int(s.sum()), n_moving=int(m.sum()), frac_inside=float(ins.mean()))
    for k in [k for k in P if k.startswith("x_r") and k != "x_r1"]:
        xp, _ = sample(P[k][T], g, x1[:, 0], x1[:, 1])
        xn2 = np.stack(em.model_to_native(gm, xp[:, 0], xp[:, 1]), -1); e2 = np.linalg.norm(xn2 - x2, axis=1)
        row[f"epe_static_{k}"] = float(np.nanmedian(e2[s])); row[f"epe_moving_{k}"] = float(np.nanmedian(e2[m])) if m.any() else float("nan")
    # world slope
    if P.get("w_disp") is not None:
        dt_full, X0c, moving = em.kubric_gt_wdisp(meta, gt, frames[0], b)
        yi, xi = x1[:, 1].astype(int), x1[:, 0].astype(int)
        dt = dt_full[yi, xi]; zg = X0c[yi, xi, 2]
        dp, _ = sample(P["w_disp"][T], g, x1[:, 0], x1[:, 1])
        P0 = P["pts3d_0"]
        if model in ("4rc", "any4d"):
            P0 = em._apply_T(np.linalg.inv(P["cam_pose"][0]), P0)
        zp, _ = sample(P0[..., 2], g, x1[:, 0], x1[:, 1])
        fx = np.array(meta["K"])[0, 0]
        r = ws.world_slope(dp, dt, (grp > 0) & ins, s, z_gt=zg, z_pred=zp, focal=fx)
        row.update({f"w_{k}": v for k, v in r.items()})
    # pose
    gcv = em.kubric_cv_geometry(meta, gt, frames)
    try:
        row.update({f"pose_{k}": v for k, v in pm.pose_metrics(P["cam_pose"], gcv["c2w"]).items()})
    except Exception as ex:  # noqa
        row["pose_error"] = repr(ex)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data"); ap.add_argument("--smoke_dir")
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--out", required=True)
    ap.add_argument("--cond", default="gt"); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ref", help="json with per-scene Any4D epe_static (dict scene -> value, or rows)")
    ap.add_argument("--no_square", action="store_true"); ap.add_argument("--save_npz", default="")
    args = ap.parse_args()
    t0 = time.time(); A = build(args.model, args); load_s = time.time() - t0
    import torch
    res = dict(model=args.model, load_s=load_s, gate=GATES[args.model], load_info=getattr(A, "load_info", None),
               gpu_after_load_gb=torch.cuda.memory_allocated() / 2 ** 30, rows=[])
    if args.smoke_dir:
        fs = sorted(f for f in glob.glob(os.path.join(args.smoke_dir, "*")) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        n = len(FRAMES[args.model]) if args.model != "opend4rt" else min(24, len(fs) - len(fs) % 2)
        idx = np.linspace(0, len(fs) - 1, n).round().astype(int) if args.model != "opend4rt" else np.arange(n)
        paths = [fs[i] for i in idx]
        P = run_model(A, args.model, paths)
        res["smoke"] = dict(paths=paths, keys=summarize_keys(P), runtime_s=P["_runtime_s"], gpu=P["_gpu"])
        if args.save_npz:
            np.savez_compressed(args.save_npz, **{k: v for k, v in P.items() if isinstance(v, np.ndarray)})
        if args.model == "4rc":
            res["smoke"]["output_keys"] = P["output_keys"]
    if args.data:
        import e5_kubric_post as kp
        scenes = sorted(d for d in glob.glob(os.path.join(args.data, "*")) if os.path.exists(os.path.join(d, "e5_meta.json"))
                        and os.path.exists(os.path.join(d, "gt.npz")))[: args.n]
        res["n_scenes_found"] = len(scenes)
        for d in scenes:
            meta, gt, _ = kp.load_scene(d)
            fr = FRAMES[args.model]
            if len(meta["frames"]) != 24:
                fr = [int(round(f * (len(meta["frames"]) - 1) / 23)) for f in fr]
            paths = [os.path.join(d, f"rgba_{f:05d}.png") for f in fr]
            P = run_model(A, args.model, paths, meta, gt, fr, cond=args.cond)
            row = dict(scene=os.path.basename(d), runtime_s=P["_runtime_s"], **P["_gpu"])
            row.update(score_scene(args.model, P, meta, gt, fr))
            res["rows"].append(row); print(json.dumps(row), flush=True)
        e = np.array([r["epe_static"] for r in res["rows"]], float)
        res["median_epe_static"] = float(np.nanmedian(e)) if len(e) else float("nan")
        res["median_runtime_s"] = float(np.median([r["runtime_s"] for r in res["rows"]])) if res["rows"] else float("nan")
        res["max_peak_reserved_gb"] = float(max([r.get("peak_reserved_gb", 0) for r in res["rows"]] or [0]))
        if args.model == "trackcraft3r":
            res["gate_pass"] = bool(res["median_epe_static"] <= 10.0) if len(e) >= 10 else None
        if args.ref and os.path.exists(args.ref):
            ref = json.load(open(args.ref)); ref = ref.get("rows", ref)
            rm = {r["scene"]: r["epe_static"] for r in ref} if isinstance(ref, list) else ref
            common = [r for r in res["rows"] if r["scene"] in rm]
            if common:
                a = np.nanmedian([rm[r["scene"]] for r in common]); m = np.nanmedian([r["epe_static"] for r in common])
                res["ref_median_epe_static"] = float(a); res["ratio_to_ref"] = float(m / a)
                res["gate_pass"] = bool(m <= 2 * a)
    json.dump(res, open(args.out, "w"), indent=1, default=float)
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
