"""Item 8: model-agnostic output-level metrics on real data (definitions as in e5_causal.py).

For the reference pixel p (native) with GT target position x2_true and static-world target position x2_static:
  d_true = x2_true - x2_static, d_pred = x_pred - x2_static
  slope      = sum(d_pred . d_true) / sum(|d_true|^2) over moving pixels with |d_true| > 2 px (nan if <= 20 such px)
  epe_static = median |x_pred - x2_true| over static pixels (group 0); epe_moving the same over group > 0
  sep_ratio  = sep(x1p, x_pred) / sep(x1, x2_true) with e5_step3_analysis.sep_and_noise, 300 points per group
               (group 0 = static, one group per moving instance), nan if sep_gt <= 1 px or < 2 groups
Evaluation points are the model's view-0 pixels inside the native image (so only the model's cropped field of view
is evaluated); predictions are mapped back to native pixels with the exact inverse of the input resize
(e5_models.model_to_native) and GT is bilinearly interpolated at the same native location.
"""
import json
import numpy as np
import e5_step3_analysis as s3
from e5_models import native_to_model, model_to_native


def pred_target_px(P, ti):
    """Predicted position (model px of view ti) of every view-0 model pixel."""
    if P.get("x_pred") is not None:
        return P["x_pred"][ti]
    Q = dict(pts3d_0=P["pts3d_0"], scene_flow=P["scene_flow"] if P.get("scene_flow") is not None
             else np.zeros((len(P["cam_pose"]),) + P["pts3d_0"].shape, np.float32),
             cam_pose=P["cam_pose"], K=P["K"])
    off = np.array(s3.pixel_offset(Q))
    return s3.pred_correspondences(Q, ti, off), off


def evaluate(gt, meta, P, ti=1, seed=0):
    """Metrics for one sample. Evaluation points are the model's view-0 pixels: each model pixel centre is mapped to
    its exact native location x1 (inverse of the input resize); GT x2_true and x2_static are bilinearly interpolated
    there, and only points whose 4 native neighbours are valid and share one group are kept. This avoids the
    sub-pixel bias of matching a native pixel to the nearest model pixel (up to 1.1 native px for KITTI at 518x168)."""
    g0, gt_ = P["geom"][0], P["geom"][ti]
    out = pred_target_px(P, ti)
    xm, off = (out if isinstance(out, tuple) else (out, np.zeros(2)))
    h, w = xm.shape[:2]
    valid = gt["valid"] & (gt["group"] >= 0)
    H, W = valid.shape
    # moving pixels in the model's field of view (native count)
    vn_all, un_all = np.nonzero(gt["group"] > 0)
    um_a, vm_a = native_to_model(g0, un_all.astype(float), vn_all.astype(float))
    in_a = (um_a >= -0.5) & (um_a < w - 0.5) & (vm_a >= -0.5) & (vm_a < h - 0.5)
    vm, um = np.mgrid[0:h, 0:w]
    un, vn = model_to_native(g0, um.astype(float).ravel(), vm.astype(float).ravel())
    iu_m, iv_m = um.ravel(), vm.ravel()
    u0 = np.floor(un).astype(int); v0 = np.floor(vn).astype(int)
    ok = (u0 >= 0) & (v0 >= 0) & (u0 + 1 < W) & (v0 + 1 < H)
    un, vn, u0, v0, iu_m, iv_m = un[ok], vn[ok], u0[ok], v0[ok], iu_m[ok], iv_m[ok]
    G = gt["group"].astype(int)
    gs = [G[v0, u0], G[v0, u0 + 1], G[v0 + 1, u0], G[v0 + 1, u0 + 1]]
    vs = [valid[v0, u0], valid[v0, u0 + 1], valid[v0 + 1, u0], valid[v0 + 1, u0 + 1]]
    keep = vs[0] & vs[1] & vs[2] & vs[3] & (gs[0] == gs[1]) & (gs[0] == gs[2]) & (gs[0] == gs[3])
    un, vn, u0, v0, iu_m, iv_m, grp = un[keep], vn[keep], u0[keep], v0[keep], iu_m[keep], iv_m[keep], gs[0][keep]
    a = (un - u0)[:, None]; b = (vn - v0)[:, None]

    def bil(F):
        F = F.astype(np.float64)
        return (F[v0, u0] * (1 - a) * (1 - b) + F[v0, u0 + 1] * a * (1 - b) + F[v0 + 1, u0] * (1 - a) * b
                + F[v0 + 1, u0 + 1] * a * b)
    x1 = np.c_[un, vn]; x2 = bil(gt["x2_true"]); xs = bil(gt["x2_static"])
    xp_m = xm[iv_m, iu_m].astype(np.float64)
    xp = np.stack(model_to_native(gt_, xp_m[:, 0], xp_m[:, 1]), 1)
    fin = np.isfinite(xp).all(1)
    err = np.linalg.norm(xp - x2, axis=1); mv = grp > 0
    dt = x2 - xs; dp = xp - xs; big = mv & (np.linalg.norm(dt, axis=1) > 2) & fin
    row = dict(scene=meta["name"], n_eval=int(len(x1)), n_moving_eval=int(mv.sum()), n_moving_total=int(len(un_all)),
               frac_moving_in_view=float(in_a.mean()) if len(un_all) else float("nan"),
               n_inst=int(len(set(grp[mv].tolist()))), n_slope_px=int(big.sum()),
               slope=float((dp[big] * dt[big]).sum() / (dt[big] ** 2).sum()) if big.sum() > 20 else float("nan"),
               epe_static=float(np.median(err[(~mv) & fin])) if ((~mv) & fin).any() else float("nan"),
               epe_moving=float(np.median(err[mv & fin])) if (mv & fin).any() else float("nan"),
               d_true_moving_med=float(np.median(np.linalg.norm(dt[mv], axis=1))) if mv.any() else float("nan"),
               pixel_offset=[float(off[0]), float(off[1])])
    sl = []
    for k in sorted(set(grp[big].tolist())):
        m = big & (grp == k)
        if m.sum() > 20:
            sl.append(float((dp[m] * dt[m]).sum() / (dt[m] ** 2).sum()))
    row["slope_inst_median"] = float(np.median(sl)) if sl else float("nan")
    rs = np.random.default_rng(seed)
    idx = s3.sample_by_group(np.where(fin, grp, -99), 300, rs)
    idx = idx[grp[idx] >= 0] if len(idx) else idx
    idx = idx[fin[idx]] if len(idx) else idx
    gi = grp[idx] if len(idx) else np.array([], int)
    row["n_groups_sep"] = int(len(set(gi.tolist())))
    if row["n_groups_sep"] >= 2 and (gi == 0).any():
        sep_gt, _ = s3.sep_and_noise(x1[idx], x2[idx], gi)
        sep_pr, _ = s3.sep_and_noise(x1[idx], xp[idx], gi)
        row.update(sep_gt=float(sep_gt), sep_pred=float(sep_pr),
                   sep_ratio=float(sep_pr / sep_gt) if sep_gt > 1 else float("nan"))
    else:
        row.update(sep_gt=float("nan"), sep_pred=float("nan"), sep_ratio=float("nan"))
    return row
