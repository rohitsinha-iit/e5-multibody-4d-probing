#!/usr/bin/env python3
"""U2 scoring (CPU): per-scene / per-sample / per-window metric rows from the u2_infer.py predictions.
Definitions: unified plan section 6 as frozen in the pre-registration.

Kubric (factorial A, B, C, D; target view 23; E5 query pixels; GT = gt_factorial.npz of the variant):
  points: query pixels visible in view 23 (vis), finite prediction; moving = q_group > 0, static = q_group == 0.
  boundary rule: rows with boundary='excl8' drop points within 8 px (frame 0, 512 px) of a moving-object boundary
                 (Euclidean distance to the frame-0 moving-mask edge, as p3/x2_eval.py); boundary='none' keeps all.
  image (primary readout xp; secondary readouts with a prefix): epe_static, epe_moving (median px);
     slope2d = sum((xp - x_static).A)/sum(|A|^2) over moving points with |A| > 2 px (e5_w2_eval / w2_summary);
     D and B: xp - x1 = alpha A + beta B fit (e5_w2_eval.fit_ab), cond_AB, median |A|, |B|; identifiable =
       cond_AB <= 10 and median |A|, |B| >= 2 px; beta_static on static points with |B| > 2 px.
     C (camera frozen, B = 0): alpha := slope2d, identifiable by construction.  A: no motion, alpha undefined.
     sep_ratio: e5_step3 sep_and_noise on the E5 separation sample (rng = default_rng(scene index)) of the D query set.
  world: s_w, static_residual, scale (src/attr/world_slope.py; GT world displacement in the GT frame-0 OpenCV camera,
     prediction in the predicted frame-0 camera, per-scene median-depth scale on static points, moving points with at
     least 2 px-equivalent motion).
  pose: B, D: ATE (Sim(3) Umeyama on centres), RPE-T, RPE-R (consecutive views, RMSE) on the model's views;
        A, C: spurious ego-motion (mean over views 1.. of the rotation angle in deg and translation / median predicted
        frame-0 depth). OpenD4RT and TrackCraft3R also get *_v7 columns on the views 0,4,...,20,23 subset.
  budget (Any4D, 4RC; C and D): Shapley shares phi_c / sum(phi) of {geometry, flow, camera} (src/attr/budget.py),
        scene median over moving points with |A| >= 2 px.
Real (KITTI 2015, Sintel; E5 protocol, e5_realdata_eval.evaluate + the e5_w2_realdata alpha/beta decomposition).
Pose sets: ATE, RPE-T, RPE-R per 8-frame window (pose.py, Sim(3) per window).
Outputs: results/u2/rows_kubric.csv, rows_real.csv, rows_pose.csv (+ per-item json caches under results/u2/scores/).
"""
import argparse, csv, glob, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.append(HERE); sys.path.append(os.path.dirname(HERE))
import u2_common as u  # noqa: E402

SC = os.path.join(u.U2, "scores")
BUDGET_MODELS = ("any4d", "4rc")


def pslope(dp, dt, m, min_n=20):
    return float((dp[m] * dt[m]).sum() / (dt[m] ** 2).sum()) if m.sum() > min_n else float("nan")


def boundary_far(ds, seed):
    p = os.path.join(u.U2, "cache", "bdist", f"{ds}_{seed}.npy")
    if os.path.exists(p):
        return np.load(p)
    from scipy import ndimage
    g = u.load_gtf(ds, seed, "D"); mov = g["moving_mask"][0]
    b = np.zeros(mov.shape, bool); b[:, 1:] |= mov[:, 1:] != mov[:, :-1]; b[1:, :] |= mov[1:, :] != mov[:-1, :]
    dist = ndimage.distance_transform_edt(~b); q = g["q_pix"].astype(int); d = dist[q[:, 1], q[:, 0]].astype(np.float32)
    os.makedirs(os.path.dirname(p), exist_ok=True); np.save(p, d)
    return d


def sep_positions(ds, seed):
    """Positions (into the query list) of the E5 separation sample of the D query set, with a consistency check."""
    p = os.path.join(u.U2, "cache", "sep", f"{ds}_{seed}.npy")
    if os.path.exists(p):
        return np.load(p)
    import e5_kubric_post as kp, e5_step3_analysis as s3
    d = u.vdir(ds, seed, "D"); info = json.load(open(os.path.join(u.FACT, ds, str(seed), "D_gt", "gt_factorial_info.json")))
    meta = json.load(open(os.path.join(d, "e5_meta.json"))); gt = np.load(os.path.join(d, "gt.npz"))
    c = kp.correspondences(meta, gt, 0, len(meta["frames"]) - 1)
    idx = s3.sample_by_group(c["group"], 300, np.random.default_rng(info["scene_index"]))
    q = np.load(os.path.join(u.FACT, ds, str(seed), "D_gt", "q_idx.npy"))
    qp = u.load_gtf(ds, seed, "D")["q_pix"]
    assert np.array_equal(c["x1"][q].astype(int), qp), "query pixels differ from the E5 correspondence list"
    pos = np.searchsorted(q, idx); assert np.array_equal(q[pos], idx)
    os.makedirs(os.path.dirname(p), exist_ok=True); np.save(p, pos)
    return pos


def fit_ab(A, B, y, min_n=20):
    n = len(A)
    if n <= min_n:
        return dict(alpha=np.nan, beta=np.nan, cos_AB=np.nan, cond_AB=np.nan, r2_ab=np.nan, n_ab=int(n))
    X = np.c_[A.reshape(-1), B.reshape(-1)]; Y = y.reshape(-1)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None); sv = np.linalg.svd(X, compute_uv=False)
    res = Y - X @ coef; r2 = 1 - (res ** 2).sum() / max((Y ** 2).sum(), 1e-12)
    cos = float((A * B).sum() / np.sqrt((A ** 2).sum() * (B ** 2).sum() + 1e-12))
    return dict(alpha=float(coef[0]), beta=float(coef[1]), cos_AB=cos, cond_AB=float(sv[0] / max(sv[-1], 1e-12)), r2_ab=float(r2), n_ab=int(n))


def image_block(xp, G, v, keep, pre, sep_pos):
    import e5_step3_analysis as s3
    x1, xt, xs, grp, vis = G["x1"], G["xt"], G["xs"], G["grp"], G["vis"]
    fin = np.isfinite(xp).all(1); base = vis & fin & keep; mv = grp > 0; st = grp == 0
    A = xt - xs; Bc = xs - x1; y = xp - x1; err = np.linalg.norm(xp - xt, axis=1)
    big = mv & base & (np.linalg.norm(A, axis=1) > 2)
    r = {"epe_static": float(np.median(err[st & base])) if (st & base).any() else np.nan,
         "epe_moving": float(np.median(err[mv & base])) if (mv & base).any() else np.nan,
         "n_static": int((st & base).sum()), "n_moving": int((mv & base).sum()), "n_slope": int(big.sum()),
         "slope2d": pslope(xp - xs, A, big)}
    if v in ("B", "D"):
        ab = fit_ab(A[big], Bc[big], y[big]); r.update(ab)
        r["median_A_px"] = float(np.median(np.linalg.norm(A[big], axis=1))) if big.any() else np.nan
        r["median_B_px"] = float(np.median(np.linalg.norm(Bc[big], axis=1))) if big.any() else np.nan
        r["identifiable"] = bool(np.isfinite(ab["cond_AB"]) and ab["cond_AB"] <= 10 and r["median_A_px"] >= 2 and r["median_B_px"] >= 2)
        sb = st & base & (np.linalg.norm(Bc, axis=1) > 2); r["beta_static"] = pslope(y, Bc, sb)
        if v == "B":
            r["alpha"] = np.nan; r["identifiable"] = False
    elif v == "C":
        r.update(alpha=r["slope2d"], beta=np.nan, identifiable=bool(np.isfinite(r["slope2d"])))
    else:
        r.update(alpha=np.nan, beta=np.nan, identifiable=False)
    # separation ratio on the E5 sample (only points kept by this row's mask)
    sp = sep_pos[base[sep_pos]] if len(sep_pos) else sep_pos
    r["sep_ratio"] = r["sep_gt"] = np.nan
    if len(sp) and len(set(grp[sp].tolist())) >= 2 and (grp[sp] == 0).any():
        try:
            sg, _ = s3.sep_and_noise(x1[sp], xt[sp], grp[sp]); spd, _ = s3.sep_and_noise(x1[sp], xp[sp], grp[sp])
            r["sep_gt"] = float(sg); r["sep_ratio"] = float(spd / sg) if sg > 1 else np.nan
        except Exception:
            pass
    return {pre + k: val for k, val in r.items()}


def world_block(wd, z0, G, keep, pre):
    import world_slope as ws
    r = ws.world_slope(wd, G["dt"], (G["grp"] > 0) & G["vis"] & keep, (G["grp"] == 0) & keep, z_gt=G["zg"], z_pred=z0, focal=G["fx"])
    return {pre + k: val for k, val in r.items()}


def pose_block(cams, views, G, v, depth0_med, pre=""):
    import pose as pm
    gt = np.linalg.inv(G["w2c"][np.asarray(views)])
    if v in ("B", "D"):
        return {pre + "pose_" + k: val for k, val in pm.pose_metrics(cams, gt).items()}
    s = pm.spurious_ego_motion(cams, depth0_med)
    return {pre + "spur_rot_deg": s["rot_deg_mean"], pre + "spur_trans_rel": s["trans_rel_mean"]}


def budget_block(Z, G, keep):
    import budget as bd
    TV = list(Z["views"]).index(23); sel = (G["grp"] > 0) & G["vis"] & keep & np.isfinite(Z["b_P0"]).all(1) & np.isfinite(Z["b_F"]).all(1)
    if sel.sum() < 20:
        return {}
    c2w = Z["cams"]; P0, F, cT = bd.to_frame0(Z["b_P0"][sel], Z["b_F"][sel], c2w[TV], c2w[0])
    pred = dict(P0=P0, F=F, c2w=cT, K=Z["b_K"][TV])
    gt = dict(P0=G["X0c"][sel], F=G["dt"][sel], c2w=np.linalg.inv(G["rel"][23]), K=G["Kcv"])
    st = (G["grp"] == 0) & keep & np.isfinite(Z["z0"]) & (Z["z0"] > 0)
    s = float(np.median(Z["z0"][st] / G["zg"][st]))
    out = bd.shapley_budget(pred, gt, s)
    sh = bd.scene_budget_share(out["phi"], out["valid"])
    # GT-side check: the all-GT projection must equal gt_factorial x_true
    xs_all = bd._proj(G["Kcv"], np.linalg.inv(G["rel"][23]), G["X0c"][sel] + G["dt"][sel])
    r = {f"budget_{k}": val for k, val in sh.items()}
    r["budget_gt_check_px"] = float(np.median(np.linalg.norm(xs_all - G["xt"][sel], axis=1)))
    r["budget_d_pred_med"] = float(np.median(out["d_pred"][out["valid"]])) if out["valid"].any() else np.nan
    return r


def gt_arrays(ds, seed, v):
    g = u.load_gtf(ds, seed, v); assert int(g["target_views"][-1]) == 23
    w2c = g["w2c_cv"]; R0, t0 = w2c[0][:3, :3], w2c[0][:3, 3]
    q = g["q_pix"].astype(int); X0 = g["pts0"][q[:, 1], q[:, 0]].astype(np.float64)
    X0c = X0 @ R0.T + t0; dt = g["world_disp"][-1] @ R0.T
    Kb = g["K"]; W = H = 512
    Kcv = np.array([[Kb[0, 0], 0, W / 2 - 0.5], [0, Kb[1, 1], H / 2 - 0.5], [0, 0, 1.0]])
    return dict(x1=g["x1"], xt=g["x_true"][-1], xs=g["x_static"][-1], vis=g["vis"][-1], grp=g["q_group"].astype(int),
                dt=dt, X0c=X0c, zg=X0c[:, 2], fx=float(Kb[0, 0]), w2c=w2c, rel=g["rel_w2c_cv_to0"], Kcv=Kcv, q=g["q_pix"])


def score_kubric_item(args):
    tag, split, v, pz = args
    ds = u.SPLIT_SET[split]; Z = dict(np.load(pz)); seed = int(Z["seed"])
    G = gt_arrays(ds, seed, v); assert np.array_equal(G["q"], Z["q_pix"].astype(int))
    far = boundary_far(ds, seed) >= u.BOUNDARY_PX; sep = sep_positions(ds, seed)
    rows = []
    for bname, keep in (("excl8", far), ("none", np.ones_like(far))):
        r = dict(model=tag, split=split, set=ds, seed=seed, scene=str(Z["scene"]), variant=v, boundary=bname)
        for k in sorted(k for k in Z if k.startswith("xp")):
            r.update(image_block(Z[k].astype(np.float64), G, v, keep, "" if k == "xp" else k[3:] + "_", sep))
        for k in sorted(k for k in Z if k.startswith("wd")):
            r.update(world_block(Z[k], Z["z0"], G, keep, "" if k == "wd" else k[3:] + "_"))
        if "cams" in Z:
            r.update(pose_block(Z["cams"], Z["views"], G, v, float(Z["depth0_med"])))
            if len(Z["views"]) != 7:
                sel = [list(Z["views"]).index(f) for f in u.VIEWS7]
                r.update(pose_block(Z["cams"][sel], Z["views"][sel], G, v, float(Z["depth0_med"]), pre="v7_"))
        if tag in BUDGET_MODELS and v in ("C", "D"):
            r.update(budget_block(Z, G, keep))
        for k in ("runtime_s", "peak_reserved_gb"):
            if k in Z:
                r[k] = float(Z[k])
        rows.append(r)
    return rows


def decomposition_real(gt, P, ev, ti=1):
    from e5_models import native_to_model, model_to_native
    g0, gt_ = P["geom"][0], P["geom"][ti]
    xm = P["x_pred"][ti].astype(np.float64); h, w = xm.shape[:2]
    band = ev.boundary_band(gt["label"]) if "label" in gt else np.zeros(gt["valid"].shape, bool)
    vv, uu = np.nonzero(gt["valid"] & (gt["group"] >= 0) & ~band)
    um, vm = native_to_model(g0, uu.astype(float), vv.astype(float))
    ins = (um >= -0.5) & (um <= w - 0.5) & (vm >= -0.5) & (vm <= h - 0.5)
    uu, vv, um, vm = uu[ins], vv[ins], um[ins], vm[ins]; grp = gt["group"][vv, uu].astype(int)
    x1 = np.c_[uu, vv].astype(float); x2 = gt["x2_true"][vv, uu].astype(float); xs = gt["x2_static"][vv, uu].astype(float)
    xp_m, _ = ev.sample_pred(xm, um, vm); xp = np.stack(model_to_native(gt_, xp_m[:, 0], xp_m[:, 1]), 1)
    fin = np.isfinite(xp).all(1); mv = grp > 0; A = x2 - xs; B = xs - x1; y = xp - x1
    big = mv & (np.linalg.norm(A, axis=1) > 2) & fin
    r = fit_ab(A[big], B[big], y[big]); st = (~mv) & fin & (np.linalg.norm(B, axis=1) > 2)
    r["beta_static"] = pslope(y, B, st)
    r["median_A_px"] = float(np.median(np.linalg.norm(A[big], axis=1))) if big.any() else np.nan
    r["median_B_px"] = float(np.median(np.linalg.norm(B[big], axis=1))) if big.any() else np.nan
    r["identifiable"] = bool(np.isfinite(r["cond_AB"]) and r["cond_AB"] <= 10 and r["median_A_px"] >= 2 and r["median_B_px"] >= 2)
    return r


def score_real_item(args):
    tag, sub, pz = args
    import e5_realdata_eval as ev
    dsn = sub.split("_")[0]; name = os.path.basename(pz)[:-4]
    files = sorted(f for f in glob.glob(os.path.join(u.REAL_GT, dsn, "*.npz")) if not f.endswith(".tmp.npz"))
    f = os.path.join(u.REAL_GT, dsn, name + ".npz"); fi = files.index(f)
    z = np.load(f, allow_pickle=False); meta = json.loads(str(z["meta"]))
    gt = {k: z[k] for k in ("x2_true", "x2_static", "group", "valid", "label") if k in z.files}
    Z = np.load(pz); geom = json.loads(str(Z["geom"]))
    row = dict(model=tag, unit=sub, dataset=dsn, sample=name)
    for k in sorted(k for k in Z.files if k.startswith("field")):
        P = dict(x_pred=[Z[k], Z[k]], geom=geom); pre = "" if k == "field" else k[6:] + "_"
        r = ev.evaluate(gt, meta, P, ti=1, seed=fi); r.update(decomposition_real(gt, P, ev))
        r.pop("scene", None); r.pop("pixel_offset", None)
        row.update({pre + kk: vv for kk, vv in r.items()})
    row["ego_rot_deg"] = meta.get("ego_rot_deg", np.nan)
    return [row]


def score_pose_item(args):
    tag, pz, w = args
    import pose as pm
    Z = np.load(pz); gt = u.window_gt_c2w(w)
    r = dict(model=tag, window=w["id"], set=w["set"], seq=w["seq"])
    r.update(pm.pose_metrics(Z["cams"].astype(np.float64), gt)); return [r]


def cached(fn, key, args, src):
    p = os.path.join(SC, key + ".json")
    if os.path.exists(p) and os.path.getmtime(p) >= os.path.getmtime(src):
        return json.load(open(p))
    try:
        rows = fn(args)
    except Exception as e:
        import traceback
        rows = [dict(error=repr(e), tb=traceback.format_exc()[-2000:], key=key)]
    u.save_json(p, rows); return rows


def _kub(a): return cached(score_kubric_item, f"kubric/{a[0]}/{a[1]}/{a[2]}/{os.path.basename(a[3])[:-4]}", a, a[3])
def _real(a): return cached(score_real_item, f"real/{a[0]}/{a[1]}/{os.path.basename(a[2])[:-4]}", a, a[2])
def _pose(a): return cached(score_pose_item, f"pose/{a[0]}/{os.path.basename(a[1])[:-4]}", a, a[1])


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        wr = csv.DictWriter(fh, keys); wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{r[k]:.6g}" if isinstance(r.get(k), float) else r.get(k)) for k in keys})
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--only", default="kubric,real,pose"); ap.add_argument("--models", default="")
    a = ap.parse_args(); want = set(a.only.split(",")); mods = set(a.models.split(",")) if a.models else None
    from multiprocessing import Pool
    t0 = time.time(); out = {}
    jobs_k = [(p.split(os.sep)[-4], p.split(os.sep)[-3], p.split(os.sep)[-2], p)
              for p in sorted(glob.glob(os.path.join(u.U2, "pred", "*", "*", "*", "*.npz"))) if not p.endswith(".tmp.npz")]
    jobs_r = [(p.split(os.sep)[-3], p.split(os.sep)[-2], p) for p in sorted(glob.glob(os.path.join(u.U2, "pred_real", "*", "*", "*.npz")))
              if not p.endswith(".tmp.npz")]
    W = {w["id"].replace("/", "__"): w for w in u.make_pose_windows()["windows"]} if "pose" in want else {}
    jobs_p = [(p.split(os.sep)[-2], p, W[os.path.basename(p)[:-4]]) for p in sorted(glob.glob(os.path.join(u.U2, "pred_pose", "*", "*.npz")))
              if not p.endswith(".tmp.npz") and os.path.basename(p)[:-4] in W]
    if mods:
        jobs_k = [j for j in jobs_k if j[0] in mods]; jobs_r = [j for j in jobs_r if j[0] in mods]; jobs_p = [j for j in jobs_p if j[0] in mods]
    # caches that several items share: build once per base scene before the pool
    for j in jobs_k:
        ds = u.SPLIT_SET[j[1]]; seed = int(os.path.basename(j[3])[:-4])
        if not os.path.exists(os.path.join(u.U2, "cache", "sep", f"{ds}_{seed}.npy")):
            sep_positions(ds, seed); boundary_far(ds, seed)
    with Pool(a.procs) as pool:
        for nm, fn, J, fname in (("kubric", _kub, jobs_k, "rows_kubric.csv"), ("real", _real, jobs_r, "rows_real.csv"),
                                 ("pose", _pose, jobs_p, "rows_pose.csv")):
            if nm not in want or not J:
                continue
            rows = [r for rs in pool.imap(fn, J, chunksize=4) for r in rs]
            errs = [r for r in rows if "error" in r]
            write_csv(os.path.join(u.U2, fname), [r for r in rows if "error" not in r])
            if errs:
                u.save_json(os.path.join(u.U2, f"score_errors_{nm}.json"), errs)
            out[nm] = dict(items=len(J), rows=len(rows), errors=len(errs))
            print(nm, out[nm], f"{time.time() - t0:.0f}s", flush=True)
    u.save_json(os.path.join(u.U2, "score_status.json"), dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), **out))


if __name__ == "__main__":
    main()
