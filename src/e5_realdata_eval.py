"""Item 8: model-agnostic output-level metrics on real data (definitions as in e5_causal.py).

For the reference pixel p (native) with GT target position x2_true and static-world target position x2_static:
  d_true = x2_true - x2_static, d_pred = x_pred - x2_static
  slope      = sum(d_pred . d_true) / sum(|d_true|^2) over moving pixels with |d_true| > 2 px (nan if <= 20 such px)
  epe_static = median |x_pred - x2_true| over static pixels (group 0); epe_moving the same over group > 0
  sep_ratio  = sep(x1p, x_pred) / sep(x1, x2_true) with e5_step3_analysis.sep_and_noise, 300 points per group
               (group 0 = static, one group per moving instance), nan if sep_gt <= 1 px or < 2 groups
Evaluation points are the native GT pixels inside the model's (cropped) field of view; the dense prediction is
interpolated at their exact model coordinates (pixels within BAND_PX of a motion boundary are excluded, since there
the model's coarser grid mixes foreground and background) and mapped back to native pixels with the exact inverse of the input
resize (e5_models.model_to_native). sep_ratio is undefined (nan) when sep_gt <= 1 px, which is the common case on
KITTI because most cars translate along the ego direction (epipole-aligned motion, as in the Kubric epi_aligned split).
"""
import json
import numpy as np
import cv2

BAND_PX = 3   # native px excluded on each side of a motion boundary (>= one model pixel for every input size used:
              # KITTI 518x168 has 2.2 native px per model px, Sintel 518x252 has 2.0)


def boundary_band(label, r=BAND_PX):
    L = label.astype(np.float32); k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    return cv2.dilate(L, k) != cv2.erode(L, k)
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


def sample_pred(xm, um, vm, edge_px=2.0):
    """Bilinear interpolation of the dense predicted field xm (h,w,2) at continuous model coords; where the 4
    neighbours disagree by more than edge_px (object boundary) the nearest neighbour is used instead, so that
    foreground and background predictions are not blended."""
    h, w = xm.shape[:2]
    u0 = np.clip(np.floor(um).astype(int), 0, w - 2); v0 = np.clip(np.floor(vm).astype(int), 0, h - 2)
    a = np.clip(um - u0, 0, 1)[:, None]; b = np.clip(vm - v0, 0, 1)[:, None]
    q = [xm[v0, u0], xm[v0, u0 + 1], xm[v0 + 1, u0], xm[v0 + 1, u0 + 1]]
    bil = q[0] * (1 - a) * (1 - b) + q[1] * a * (1 - b) + q[2] * (1 - a) * b + q[3] * a * b
    spread = np.max([np.linalg.norm(x - bil, axis=1) for x in q], axis=0)
    iu = np.clip(np.round(um).astype(int), 0, w - 1); iv = np.clip(np.round(vm).astype(int), 0, h - 1)
    near = xm[iv, iu]
    edge = ~(spread <= edge_px)
    return np.where(edge[:, None], near, bil), edge


def evaluate(gt, meta, P, ti=1, seed=0):
    """Metrics for one sample. Evaluation points are the native GT pixels (KITTI GT is sparse LiDAR-based, about
    19% of pixels) that fall inside the model's field of view. The dense predicted correspondence field is
    interpolated at each pixel's exact model coordinate (inverse of the input resize) by sample_pred, then mapped
    back to native pixels of the target view."""
    g0, gt_ = P["geom"][0], P["geom"][ti]
    out = pred_target_px(P, ti)
    xm, off = (out if isinstance(out, tuple) else (out, np.zeros(2)))
    xm = xm.astype(np.float64)
    h, w = xm.shape[:2]
    band = boundary_band(gt["label"]) if "label" in gt else np.zeros(gt["valid"].shape, bool)
    vv, uu = np.nonzero(gt["valid"] & (gt["group"] >= 0) & ~band)
    um, vm = native_to_model(g0, uu.astype(float), vv.astype(float))
    inside = (um >= -0.5) & (um <= w - 0.5) & (vm >= -0.5) & (vm <= h - 0.5)
    grp_all = gt["group"][vv, uu].astype(int)
    n_all_mov = int((grp_all > 0).sum())
    uu, vv, um, vm, grp = uu[inside], vv[inside], um[inside], vm[inside], grp_all[inside]
    x1 = np.c_[uu, vv].astype(float)
    x2 = gt["x2_true"][vv, uu].astype(float); xs = gt["x2_static"][vv, uu].astype(float)
    xp_m, edge = sample_pred(xm, um, vm)
    xp = np.stack(model_to_native(gt_, xp_m[:, 0], xp_m[:, 1]), 1)
    fin = np.isfinite(xp).all(1)
    err = np.linalg.norm(xp - x2, axis=1); mv = grp > 0
    dt = x2 - xs; dp = xp - xs; big = mv & (np.linalg.norm(dt, axis=1) > 2) & fin
    row = dict(scene=meta["name"], n_eval=int(len(x1)), n_moving_eval=int(mv.sum()), n_moving_total=n_all_mov,
               frac_moving_in_view=float(mv.sum() / n_all_mov) if n_all_mov else float("nan"),
               frac_edge=float(edge.mean()) if len(edge) else float("nan"),
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
