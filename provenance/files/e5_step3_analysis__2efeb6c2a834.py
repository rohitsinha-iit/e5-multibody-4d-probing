"""E5 Step 3 (E5-C): multibody analysis of Any4D output correspondences on E5 Kubric scenes.

Predicted correspondence for a view-0 pixel p and target view t:
    X_t = pts3d_0[p] + scene_flow_t[p]                  (allocentric, model world frame)
    x_t = K_t R_t^T (X_t - T_t)                          (predicted camera t, cam-to-world [R_t | T_t])
GT correspondences come from Kubric depth, segmentation and per-frame world matrices (e5_kubric_post).
Everything is compared in the 512 px GT image frame.

Per scene and target view (the last view gives the main numbers):
  epe_static / epe_moving : median endpoint error of predicted vs GT correspondences (px)
  sep_gt / sep_pred       : two-view multibody separation of GT and of predicted correspondences (px)
  noise_pred              : median Sampson residual of predicted correspondences under the F fitted
                            to their own GT body (px) ; snr_pred = sep_pred / (noise_pred * sqrt(2))
  alg / rsc on GT and pred: motion count and segmentation error (Vidal algebraic, n known; seq-RANSAC)
  mask_iou                : IoU of Any4D motion mask (|scene flow| > thr) with GT moving pixels
"""
import glob, json, os, sys
import numpy as np
import e5_multibody as mb
import e5_kubric_post as kp

KAPPA = 2.5e-6
FLOW_THR = 0.1  # metres; the Any4D demo uses |scene flow| > 0.1 as its motion mask


def pred_correspondences(P, ti, offset):
    X = P["pts3d_0"] + P["scene_flow"][ti]
    T = P["cam_pose"][ti]; K = P["K"][ti]
    Pc = (X - T[:3, 3]) @ T[:3, :3]
    x = Pc @ K.T
    return x[..., :2] / x[..., 2:3] - offset


def pixel_offset(P):
    """Convention check: reproject view-0 points into view 0 and compare to the pixel grid."""
    H, W = P["pts3d_0"].shape[:2]
    x = pred_correspondences(dict(P, scene_flow=np.zeros_like(P["scene_flow"])), 0, 0.0)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    return float(np.median(x[..., 0] - u)), float(np.median(x[..., 1] - v))


def sample_by_group(groups, n, rng):
    idx = []
    for g in sorted(set(groups)):
        ii = np.where(groups == g)[0]
        if len(ii) >= 16:
            idx.append(rng.choice(ii, min(n, len(ii)), replace=False))
    return np.concatenate(idx) if idx else np.array([], int)


def multibody_tests(x1, x2, gt_groups, n_true, rng):
    labs = {g: k for k, g in enumerate(sorted(set(gt_groups)))}
    y = np.array([labs[g] for g in gt_groups])
    a, _ = mb.hartley_normalize(x1); b, _ = mb.hartley_normalize(x2)
    st = mb.spectral_stats(a, b, i_max=3)
    d = mb.count_motions(st)["d"]
    alg_n = next((i for i in sorted(d) if d[i] < KAPPA), None)
    lab, _ = mb.segment(a, b, n_true, rng=rng)
    alg_err = mb.seg_error(lab, y, n_true)
    labr, kr = mb.sequential_ransac(x1, x2, thr_px=3.0, rng=rng, iters=500)
    rsc_err = mb.seg_error_partial(labr, y, n_true) if kr <= 5 else np.nan
    return dict(alg_n=alg_n, alg_err=float(alg_err), rsc_n=int(kr), rsc_err=float(rsc_err))


def sep_and_noise(x1, x2, groups):
    P1 = np.c_[x1, np.ones(len(x1))]; P2 = np.c_[x2, np.ones(len(x2))]
    Fs, own = {}, {}
    for g in sorted(set(groups)):
        m = groups == g
        a, Ta = mb.hartley_normalize(x1[m]); b, Tb = mb.hartley_normalize(x2[m])
        Fs[g] = Tb.T @ mb.eight_point(a, b) @ Ta
        own[g] = float(np.median(np.sqrt(mb.sampson(Fs[g], P1[m], P2[m]))))
    seps = [float(np.median(np.sqrt(mb.sampson(Fs[h], P1[groups == g], P2[groups == g]))))
            for g in Fs for h in Fs if g != h]
    return (min(seps) if seps else np.nan), own


def analyse_scene(scene_dir, pred_npz, rng):
    meta, gt, _ = kp.load_scene(scene_dir)
    P = dict(np.load(pred_npz))
    frames = [int(f) for f in P["frames"]]
    Hm = P["pts3d_0"].shape[0]; W = meta["resolution"][0]; s = Hm / W
    off = pixel_offset(P)
    rows = []
    for ti in range(1, len(frames)):
        c = kp.correspondences(meta, gt, 0, frames[ti])
        u = np.clip(np.round((c["x1"][:, 0] + 0.5) * s - 0.5).astype(int), 0, Hm - 1)
        v = np.clip(np.round((c["x1"][:, 1] + 0.5) * s - 0.5).astype(int), 0, Hm - 1)
        xp = pred_correspondences(P, ti, np.array(off))[v, u]
        xp = (xp + 0.5) / s - 0.5  # back to 512 px frame
        # x1 is evaluated at the model pixel centre mapped back to 512 px, to avoid a sub-pixel bias
        x1p = (np.c_[u, v] + 0.5) / s - 0.5
        g = c["group"]
        err = np.linalg.norm(xp - c["x2"], axis=1)
        row = dict(scene=os.path.basename(scene_dir), split=meta["split"], target_frame=frames[ti],
                   n_motions_gt=1 + len(meta["motions"]), n_groups_visible=int(len(set(g))),
                   epe_static=float(np.median(err[g == 0])),
                   epe_moving=float(np.median(err[g > 0])) if (g > 0).any() else np.nan,
                   pixel_offset=off)
        if ti == len(frames) - 1:
            idx = sample_by_group(g, 300, rng)
            gi = g[idx]; n_true = len(set(gi))
            sep_gt, _ = sep_and_noise(c["x1"][idx], c["x2"][idx], gi)
            sep_pr, own_pr = sep_and_noise(x1p[idx], xp[idx], gi)
            noise = float(np.median(list(own_pr.values())))
            row.update(sep_gt=sep_gt, sep_pred=sep_pr, noise_pred=noise, own_residual_pred={str(k): v for k, v in own_pr.items()},
                       snr_pred=sep_pr / (noise * np.sqrt(2)) if noise > 0 else np.nan, n_true=n_true)
            if n_true >= 2:
                for tag, (a1, a2) in {"gt": (c["x1"][idx], c["x2"][idx]), "pred": (x1p[idx], xp[idx])}.items():
                    for k, val in multibody_tests(a1, a2, gi, n_true, rng).items():
                        row[f"{tag}_{k}"] = val
            # Any4D's own motion mask vs GT moving pixels (view-0 pixels, at model resolution)
            sf = np.linalg.norm(P["scene_flow"][ti], axis=-1)
            gm = np.zeros((Hm, Hm), bool)
            mov = c["dense_group"] > 0
            yy, xx = np.nonzero(mov)
            gm[np.clip(np.round((yy + 0.5) * s - 0.5).astype(int), 0, Hm - 1),
               np.clip(np.round((xx + 0.5) * s - 0.5).astype(int), 0, Hm - 1)] = True
            pm = sf > FLOW_THR
            inter, union = (pm & gm).sum(), (pm | gm).sum()
            row["mask_iou"] = float(inter / union) if union else np.nan
            row["mask_frac_pred"] = float(pm.mean()); row["mask_frac_gt"] = float(gm.mean())
        rows.append(row)
    return rows


if __name__ == "__main__":
    data, preds, out = sys.argv[1], sys.argv[2], sys.argv[3]
    rng = np.random.default_rng(0)
    allrows = []
    for pz in sorted(glob.glob(os.path.join(preds, "*.npz"))):
        name = os.path.basename(pz)[:-4]
        try:
            allrows += analyse_scene(os.path.join(data, name), pz, rng)
        except Exception as e:
            import traceback
            allrows.append(dict(scene=name, error=repr(e), tb=traceback.format_exc()[-1500:]))
    json.dump(allrows, open(out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    for r in allrows:
        if "error" in r:
            print(r["scene"], r["error"]); continue
        if "sep_gt" in r:
            print(f"{r['scene']:28s} t={r['target_frame']:2d} EPE stat {r['epe_static']:.1f} mov {r['epe_moving']:.1f} | "
                  f"sep gt {r['sep_gt']:.1f} pred {r['sep_pred']:.1f} noise {r['noise_pred']:.1f} | "
                  f"alg err gt {r.get('gt_alg_err', float('nan')):.2f} pred {r.get('pred_alg_err', float('nan')):.2f} | "
                  f"IoU {r['mask_iou']:.2f} | offset {r['pixel_offset']}")
