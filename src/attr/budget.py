"""Shapley motion budget over {geometry P0, flow F, camera (R, T, K)} (unified plan 5.3).

For each moving view-0 query point at target view t:
    x(P0, F, cam) = pi(K_t R_t^T (P0 + F_t - T_t))     (cam = c2w (R_t, T_t) and K_t; OpenCV projection)
Prediction components come from a model with composable outputs (Any4D, 4RC); GT components are expressed in the
model's frame and scale: everything is moved into the predicted frame-0 camera (prediction) and the GT frame-0 camera
(GT), and GT lengths are multiplied by the median-depth ratio s = median(z_pred / z_gt) on static view-0 points.
x_true = x(all GT). Deficit along the object-motion direction: d = (x - x_true) . A_hat, A = x_true - x_static,
x_static = x(GT P0, F = 0, GT camera). phi_c = mean over the 6 orders of d(S) - d(S + {c}) (the reduction of the
deficit when c is replaced by GT). sum(phi) = d(none) - d(all) = d(pred).
"""
import itertools
import numpy as np

COMPONENTS = ("geometry", "flow", "camera")


def _proj(K, c2w, X):
    w2c = np.linalg.inv(c2w)
    Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
    return np.stack([K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2], K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]], -1)


def to_frame0(P0, F, c2w_t, c2w_0):
    """Re-express world-frame P0 (N,3), F (N,3) and camera c2w_t in the frame-0 camera of c2w_0."""
    w2c0 = np.linalg.inv(c2w_0)
    return P0 @ w2c0[:3, :3].T + w2c0[:3, 3], F @ w2c0[:3, :3].T, w2c0 @ c2w_t


def shapley_budget(pred, gt, scale_gt_to_pred=1.0, min_A_px=2.0):
    """pred, gt: dicts with P0 (N,3), F (N,3), c2w (4,4), K (3,3) for the same N moving query points at one target
    view t, both already in their own frame-0 camera (use to_frame0). GT lengths are scaled by scale_gt_to_pred.
    Returns per-point phi (N,3) in px, d_pred (N,), and the valid mask (|A| >= min_A_px)."""
    s = scale_gt_to_pred
    G = dict(geometry=gt["P0"] * s, flow=gt["F"] * s)
    cg = np.array(gt["c2w"], float).copy(); cg[:3, 3] *= s
    G["camera"] = (cg, gt["K"]); Pm = dict(geometry=pred["P0"], flow=pred["F"], camera=(pred["c2w"], pred["K"]))

    def x_of(S):
        src = {c: (G[c] if c in S else Pm[c]) for c in COMPONENTS}
        c2w, K = src["camera"]
        return _proj(K, c2w, src["geometry"] + src["flow"])

    x_true = x_of(set(COMPONENTS))
    x_static = _proj(G["camera"][1], G["camera"][0], G["geometry"])
    A = x_true - x_static; nA = np.linalg.norm(A, axis=1); Ah = A / np.maximum(nA, 1e-9)[:, None]
    d = {}
    for r in range(4):
        for S in itertools.combinations(COMPONENTS, r):
            d[frozenset(S)] = ((x_of(set(S)) - x_true) * Ah).sum(1)
    phi = np.zeros((len(nA), 3))
    perms = list(itertools.permutations(COMPONENTS))
    for order in perms:
        S = frozenset()
        for c in order:
            phi[:, COMPONENTS.index(c)] += d[S] - d[S | {c}]
            S = S | {c}
    phi /= len(perms)
    return dict(phi=phi, d_pred=d[frozenset()], valid=nA >= min_A_px, A_norm=nA)


def scene_budget_share(phi, valid):
    """Scene summary: median over valid points of phi_c / sum(phi), and the share of the summed phi."""
    p = phi[valid]
    if len(p) == 0:
        return dict(zip(COMPONENTS, [float("nan")] * 3))
    tot = p.sum(1, keepdims=True)
    ok = np.abs(tot[:, 0]) > 1e-6
    med = np.median(p[ok] / tot[ok], 0) if ok.any() else np.full(3, np.nan)
    agg = p.sum(0) / p.sum()
    return {**{c: float(med[i]) for i, c in enumerate(COMPONENTS)},
            **{c + "_agg": float(agg[i]) for i, c in enumerate(COMPONENTS)}, "n": int(len(p))}
