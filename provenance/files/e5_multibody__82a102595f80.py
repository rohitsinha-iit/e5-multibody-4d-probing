"""E5 multibody toolkit (two-view), following Vidal, Ma, Soatto, Sastry, IJCV 2006.

Pipeline
  1. Hartley normalisation of each view.
  2. Veronese embedding nu_i(x) of degree i (deg-lex monomials, M_i = (i+1)(i+2)/2).
  3. Data matrix V_i with rows kron(nu_i(x2), nu_i(x1))  (M_i^2 columns).
  4. Motion count: exact rank rule (Theorem 3) and noise-tolerant spectral rules.
  5. Multibody fundamental matrix F_mb = null vector of V_n reshaped to M_n x M_n.
  6. Epipolar lines l_j = d/dx2 [nu_n(x2)^T F_mb nu_n(x1)]  (proportional to F_i x1).
  7. Epipoles by hyperplane clustering of the lines (GPCA via polynomial
     differentiation on nu_n(l)), then per-cluster 8-point F_i and Sampson
     reassignment (a few refinement rounds).
"""
import itertools
import numpy as np


# ---------------------------------------------------------------- basics
def hartley_normalize(x):
    """x: (N,2) pixels -> (N,3) homogeneous normalised points, T (3x3)."""
    c = x.mean(0)
    d = np.sqrt(((x - c) ** 2).sum(1)).mean()
    s = np.sqrt(2) / max(d, 1e-12)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    xh = np.c_[x, np.ones(len(x))] @ T.T
    return xh, T


def veronese_exponents(n):
    """Exponents (n1,n2,n3), n1+n2+n3=n, degree-lexicographic order."""
    ex = [e for e in itertools.product(range(n + 1), repeat=3) if sum(e) == n]
    return np.array(sorted(ex, reverse=True))


def M(n):
    return (n + 1) * (n + 2) // 2


def veronese(x, n):
    """x: (N,3) -> (N, M_n)."""
    E = veronese_exponents(n)
    return np.prod(x[:, None, :] ** E[None, :, :], axis=2)


def veronese_jacobian(x, n):
    """d nu_n(x) / dx : (N, M_n, 3)."""
    E = veronese_exponents(n)
    N = len(x)
    J = np.zeros((N, len(E), 3))
    for k in range(3):
        Ek = E.copy()
        coef = Ek[:, k].astype(float)
        Ek[:, k] = np.maximum(Ek[:, k] - 1, 0)
        J[:, :, k] = coef[None, :] * np.prod(x[:, None, :] ** Ek[None, :, :], axis=2)
    return J


def data_matrix(x1h, x2h, i):
    a, b = veronese(x2h, i), veronese(x1h, i)
    return (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)


# ---------------------------------------------------------------- counting
def spectral_stats(x1h, x2h, i_max=4, min_ratio=1.0, row_normalize=True):
    """Per degree i: normalised singular values of V_i (only if N >= min_ratio*M_i^2)."""
    out = {}
    N = len(x1h)
    for i in range(1, i_max + 1):
        if N < min_ratio * M(i) ** 2:
            break
        V = data_matrix(x1h, x2h, i)
        if row_normalize:
            V = V / np.linalg.norm(V, axis=1, keepdims=True)
        s = np.linalg.svd(V, compute_uv=False)
        out[i] = s / s[0]
    return out


def count_motions(stats, tau_exact=1e-8, tau_gap=None, mu=5e-3):
    """Return dict of motion-count estimates under several rules.

    exact : Theorem 3, min i with numerical rank == M_i^2 - 1 (threshold tau_exact).
    gap   : g_i = s_last / s_second_last; exactly one null vector -> g_i small.
            'gap_argmin' = argmin_i g_i ;  'gap_tau' = min i with g_i < tau_gap.
    """
    res = {}
    ex = None
    for i, s in stats.items():
        rank = int((s > tau_exact).sum())
        if rank == M(i) ** 2 - 1:
            ex = i
            break
    res["exact"] = ex
    g = {i: s[-1] / max(s[-2], 1e-300) for i, s in stats.items()}
    res["g"] = g
    res["gap_argmin"] = min(g, key=g.get) if g else None
    # Vidal et al. 2006, Eq. (19): model selection with complexity weight mu
    J = {i: s[-1] ** 2 / (s[:-1] ** 2).sum() + mu * (M(i) ** 2 - 1) for i, s in stats.items()}
    res["J19"] = J
    res["eq19"] = min(J, key=J.get) if J else None  # as printed; cannot select n>=2 for mu=5e-3
    # rank-drop statistic: d_i = s_last^2 / sum_{k<last} s_k^2 ; rule n = min{i : d_i < kappa}
    res["d"] = {i: s[-1] ** 2 / (s[:-1] ** 2).sum() for i, s in stats.items()}
    if tau_gap is not None:
        cand = [i for i in sorted(g) if g[i] < tau_gap]
        res["gap_tau"] = cand[0] if cand else None
    return res


# ---------------------------------------------------------------- estimation
def multibody_F(x1h, x2h, n):
    V = data_matrix(x1h, x2h, n)
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    return Vt[-1].reshape(M(n), M(n))  # rows index nu(x2), cols nu(x1)


def epipolar_lines(Fmb, x1h, x2h, n):
    J2 = veronese_jacobian(x2h, n)              # (N, M, 3)
    v1 = veronese(x1h, n) @ Fmb.T               # (N, M) = F nu(x1)
    l = np.einsum("nmk,nm->nk", J2, v1)
    return l / np.linalg.norm(l, axis=1, keepdims=True)


def _kmeans_lines(normals, n, rng, iters=50):
    """Cluster unit vectors up to sign via projective k-means."""
    N = len(normals)
    centers = normals[rng.choice(N, n, replace=False)]
    for _ in range(iters):
        lab = np.argmax(np.abs(normals @ centers.T), axis=1)
        new = []
        for k in range(n):
            P = normals[lab == k]
            if len(P) == 0:
                new.append(normals[rng.integers(N)])
                continue
            w, U = np.linalg.eigh(P.T @ P)
            new.append(U[:, -1])
        new = np.array(new)
        if np.allclose(np.abs((new * centers).sum(1)), 1, atol=1e-10):
            break
        centers = new
    return lab, centers


def gpca_hyperplanes(l, n, rng):
    """Lines l (N,3) lying on n planes through 0 with normals e_i.
    q(l) = prod_i (e_i^T l) has coefficients in null(nu_n(l)); its gradient at l_j
    is normal to the plane containing l_j. Cluster the normals, refine epipoles."""
    if n == 1:
        w, U = np.linalg.eigh(l.T @ l)
        return np.zeros(len(l), int), U[:, :1].T
    L = veronese(l, n)
    _, _, Vt = np.linalg.svd(L, full_matrices=False)
    c = Vt[-1]
    grad = np.einsum("nmk,m->nk", veronese_jacobian(l, n), c)
    grad /= np.linalg.norm(grad, axis=1, keepdims=True) + 1e-300
    best = None
    for _ in range(5):
        lab, cen = _kmeans_lines(grad, n, rng)
        # epipole per cluster: smallest eigenvector of its lines
        E = []
        for k in range(n):
            P = l[lab == k]
            if len(P) < 2:
                E.append(cen[k]); continue
            w, U = np.linalg.eigh(P.T @ P)
            E.append(U[:, 0])
        E = np.array(E)
        lab2 = np.argmin(np.abs(l @ E.T), axis=1)
        cost = np.abs(l @ E.T).min(1).sum()
        if best is None or cost < best[0]:
            best = (cost, lab2, E)
    return best[1], best[2]


def eight_point(x1h, x2h):
    A = (x2h[:, :, None] * x1h[:, None, :]).reshape(len(x1h), 9)
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    U, s, Vt = np.linalg.svd(F)
    s[2] = 0
    return U @ np.diag(s) @ Vt


def sampson(F, x1h, x2h):
    Fx1 = x1h @ F.T
    Ftx2 = x2h @ F
    num = (x2h * Fx1).sum(1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return num / np.maximum(den, 1e-300)


def segment(x1h, x2h, n, rng=None, refine=5):
    """Full Vidal-style segmentation. Returns labels, list of F_i (normalised coords)."""
    rng = rng or np.random.default_rng(0)
    Fmb = multibody_F(x1h, x2h, n)
    l = epipolar_lines(Fmb, x1h, x2h, n)
    lab, E = gpca_hyperplanes(l, n, rng)
    Fs = None
    for _ in range(refine):
        Fs = []
        for k in range(n):
            idx = lab == k
            Fs.append(eight_point(x1h[idx], x2h[idx]) if idx.sum() >= 8 else np.eye(3))
        D = np.stack([sampson(F, x1h, x2h) for F in Fs], 1)
        new = D.argmin(1)
        if (new == lab).all():
            break
        lab = new
    return lab, Fs


def seg_error(lab, gt, n):
    """Misclassification rate under best label permutation."""
    best = 1.0
    for p in itertools.permutations(range(n)):
        err = np.mean(np.array(p)[lab] != gt)
        best = min(best, err)
    return best


# ---------------------------------------------------------------- synthetic data
def rand_rot(rng, max_deg):
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    th = np.deg2rad(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def synth_scene(n, pts_per_motion=120, rng=None, f=500.0, img=512, depth=(4, 10),
                max_rot_deg=15.0, t_scale=1.0, noise_px=0.0, quantize_px=None):
    """n rigid motions (relative to the camera) between two views.
    Each motion i: X2 = R_i X1 + T_i. Points are kept only if visible in both views.
    noise_px: iid Gaussian pixel noise on both views.
    quantize_px: snap view-1 points to patch-grid centres and view-2 points to the
                 centre of the patch they fall in (models patch-level attention argmax)."""
    rng = rng or np.random.default_rng()
    K = np.array([[f, 0, img / 2], [0, f, img / 2], [0, 0, 1.0]])
    X1s, X2s, gts = [], [], []
    for i in range(n):
        R = rand_rot(rng, max_rot_deg)
        T = rng.normal(size=3); T = t_scale * T / np.linalg.norm(T)
        got = 0
        while got < pts_per_motion:
            m = 4 * pts_per_motion
            z = rng.uniform(*depth, m)
            u = rng.uniform(0, img, m); v = rng.uniform(0, img, m)
            X1 = (np.linalg.inv(K) @ np.c_[u, v, np.ones(m)].T).T * z[:, None]
            X2 = X1 @ R.T + T
            p2 = X2 @ K.T
            ok = (X2[:, 2] > 0.5)
            p2 = p2[:, :2] / p2[:, 2:3]
            ok &= (p2[:, 0] >= 0) & (p2[:, 0] < img) & (p2[:, 1] >= 0) & (p2[:, 1] < img)
            k = min(ok.sum(), pts_per_motion - got)
            X1s.append(X1[ok][:k]); X2s.append(X2[ok][:k]); gts.append(np.full(k, i))
            got += k
    X1 = np.vstack(X1s); X2 = np.vstack(X2s); gt = np.concatenate(gts)
    def proj(X):
        p = X @ K.T
        return p[:, :2] / p[:, 2:3]
    x1, x2 = proj(X1), proj(X2)
    if quantize_px:
        q = quantize_px
        x1 = (np.floor(x1 / q) + 0.5) * q
        x2 = (np.floor(x2 / q) + 0.5) * q
    if noise_px:
        x1 = x1 + rng.normal(scale=noise_px, size=x1.shape)
        x2 = x2 + rng.normal(scale=noise_px, size=x2.shape)
    return x1, x2, gt


# ---------------------------------------------------------------- robust baseline
def sequential_ransac(x1, x2, thr_px=2.0, min_inliers=20, iters=1000, max_models=6, rng=None):
    """Sequential RANSAC on F (normalised 8-point, Sampson distance in pixels).
    Returns labels (-1 = unassigned) and the number of models found."""
    rng = rng or np.random.default_rng(0)
    lab = -np.ones(len(x1), int)
    k = 0
    while k < max_models:
        idx = np.where(lab < 0)[0]
        if len(idx) < max(min_inliers, 8):
            break
        a, b = x1[idx], x2[idx]
        ah, Ta = hartley_normalize(a); bh, Tb = hartley_normalize(b)
        ap = np.c_[a, np.ones(len(a))]; bp = np.c_[b, np.ones(len(b))]
        best = None
        for _ in range(iters):
            s = rng.choice(len(idx), 8, replace=False)
            F = Tb.T @ eight_point(ah[s], bh[s]) @ Ta
            inl = np.sqrt(sampson(F, ap, bp)) < thr_px
            if best is None or inl.sum() > best.sum():
                best = inl
        if best.sum() < min_inliers:
            break
        # refit on inliers, recompute inliers once
        F = Tb.T @ eight_point(ah[best], bh[best]) @ Ta
        best = np.sqrt(sampson(F, ap, bp)) < thr_px
        lab[idx[best]] = k
        k += 1
    return lab, k


def seg_error_partial(lab, gt, n):
    """Misclassification with unassigned (-1) counted as errors; best mapping of found models to GT."""
    found = sorted(set(lab[lab >= 0]))
    best = 1.0
    for p in itertools.permutations(range(max(n, len(found))), len(found)):
        m = dict(zip(found, p))
        pred = np.array([m.get(l, -1) for l in lab])
        best = min(best, np.mean(pred != gt))
    return best
