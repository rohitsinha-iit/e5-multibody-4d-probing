"""World-frame object-motion slope s_w and static residual (unified plan section 6; P3 v2 4.3).

All displacements are world displacements of frame-0 points from frame 0 to the target view, expressed in frame-0
camera coordinates (the w_disp key of src/e5_models.py adapters). Prediction and GT are compared on the same points.

s_w             through-origin slope of predicted on GT world displacement over moving points,
                sum(dp . dt) / sum(|dt|^2), after per-scene median-depth scale alignment of the prediction.
static_residual median |dp| on static points / median |dt| on moving points (after the same scale alignment).
scale           median(z_gt / z_pred) over static frame-0 points (median-depth ratio), applied to dp.
min_disp_px     moving points need |dt| of at least this many pixels-equivalent (|dt| * f / z_gt), default 2.
boundary rule   exclude points within boundary_px (8 at 512) of moving-object boundaries; report with and without.
"""
import numpy as np


def median_depth_scale(z_pred, z_gt, static_mask):
    m = static_mask & np.isfinite(z_pred) & np.isfinite(z_gt) & (z_pred > 0) & (z_gt > 0)
    if m.sum() < 10:
        return float("nan")
    return float(np.median(z_gt[m] / z_pred[m]))


def boundary_mask(seg, moving_ids=None, radius_px=8, W_ref=512):
    """True for pixels at least radius (scaled to this image width from W_ref) away from any boundary of a moving
    object. seg (H,W) int segmentation; moving_ids iterable of segment ids that move (None: every nonzero id)."""
    from scipy.ndimage import binary_dilation, binary_erosion
    H, W = seg.shape
    r = max(1, int(round(radius_px * W / W_ref)))
    ids = np.unique(seg[seg > 0]) if moving_ids is None else list(moving_ids)
    edge = np.zeros_like(seg, bool)
    for i in ids:
        m = seg == i
        if m.any():
            edge |= m ^ binary_erosion(m)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return ~binary_dilation(edge, structure=(xx ** 2 + yy ** 2) <= r * r)


def world_slope(dp, dt, moving, static, z_gt=None, z_pred=None, focal=None, min_disp_px=2.0, keep=None, min_n=20):
    """dp, dt (N,3) or (H,W,3); moving, static bool masks of the same leading shape; z_gt/z_pred frame-0 z-depth of
    the points (needed for the scale alignment and the px-equivalent filter); focal in px. keep: extra bool mask
    (e.g. boundary_mask). Returns dict(s_w, static_residual, scale, n_moving, n_static)."""
    dp = np.asarray(dp, float).reshape(-1, 3); dt = np.asarray(dt, float).reshape(-1, 3)
    mv = np.asarray(moving, bool).ravel(); st = np.asarray(static, bool).ravel()
    ok = np.isfinite(dp).all(1) & np.isfinite(dt).all(1)
    if keep is not None:
        ok &= np.asarray(keep, bool).ravel()
    scale = 1.0
    if z_gt is not None and z_pred is not None:
        scale = median_depth_scale(np.asarray(z_pred, float).ravel(), np.asarray(z_gt, float).ravel(), st & ok)
    dp = dp * scale
    ndt = np.linalg.norm(dt, axis=1)
    big = np.ones_like(mv)
    if focal is not None and z_gt is not None:
        big = ndt * focal / np.maximum(np.asarray(z_gt, float).ravel(), 1e-9) >= min_disp_px
    m = mv & ok & big
    s = ok & st
    sw = float((dp[m] * dt[m]).sum() / (dt[m] ** 2).sum()) if m.sum() > min_n else float("nan")
    sr = float(np.median(np.linalg.norm(dp[s], axis=1)) / np.median(ndt[m])) if (m.sum() > min_n and s.sum() > min_n) else float("nan")
    return dict(s_w=sw, static_residual=sr, scale=scale, n_moving=int(m.sum()), n_static=int(s.sum()))


def bootstrap_ci(values, n_boot=2000, seed=0, stat=np.nanmedian, alpha=0.05):
    """Scene bootstrap CI of a statistic over per-scene values."""
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(est=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    rs = np.random.default_rng(seed)
    b = np.array([stat(v[rs.integers(0, len(v), len(v))]) for _ in range(n_boot)])
    return dict(est=float(stat(v)), lo=float(np.quantile(b, alpha / 2)), hi=float(np.quantile(b, 1 - alpha / 2)), n=len(v))


def one_sided_p_below(values, thr=0.9, n_boot=2000, seed=0, stat=np.nanmean):
    """Bootstrap p-value for H1: stat(values) < thr (T1'). p = fraction of resamples with stat >= thr."""
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    rs = np.random.default_rng(seed)
    b = np.array([stat(v[rs.integers(0, len(v), len(v))]) for _ in range(n_boot)])
    return float((np.sum(b >= thr) + 1) / (n_boot + 1))
