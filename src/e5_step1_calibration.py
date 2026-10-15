"""Step 1 noise-floor calibration for the E5 multibody toolkit (v2).

Scenes: n rigid motions relative to the camera, two views, image 512 px, f = 500 px,
depth 4-10, rotations up to 15 deg, unit translation, 120 points per motion.
Noise models: iid Gaussian pixel noise sigma, and patch-grid quantisation q (14, 16 px),
which mimics argmax attention matches at patch resolution.

Per trial we record
  sep    : separation = min over motion pairs of the median Sampson residual (px) of one
           motion's clean points under the other motion's GT F.
  s_eff  : effective epipolar noise std (px): sigma*sqrt(2) (Gaussian), q/sqrt(6) (patch).
  snr    : sep / s_eff.
Methods
  algebraic (Vidal et al. 2006): rank-drop statistic d_i = s_last^2 / sum_{k<last} s_k^2 of the
           row-normalised V_i; count n = min{i : d_i < KAPPA}; segmentation with n known.
  seq-RANSAC on F (baseline): threshold 2.5*s_eff, min 20 inliers.
"""
import numpy as np, pandas as pd, time
import e5_multibody as mb

KAPPA = 2.5e-6  # global best from the v1 sweep (Gaussian + patch conditions)
CONDS = [("gauss", s) for s in [0.25, 0.5, 1, 2, 4, 8]] + [("patch", 14), ("patch", 16)]
NS = [2, 3]
TRIALS = 30
rng = np.random.default_rng(2026)


def gt_F(x1, x2):
    a, Ta = mb.hartley_normalize(x1); b, Tb = mb.hartley_normalize(x2)
    return Tb.T @ mb.eight_point(a, b) @ Ta


def separation(x1, x2, gt, n):
    P1 = np.c_[x1, np.ones(len(x1))]; P2 = np.c_[x2, np.ones(len(x2))]
    Fs = [gt_F(x1[gt == k], x2[gt == k]) for k in range(n)]
    s = []
    for a in range(n):
        for b in range(n):
            if a != b:
                s.append(np.median(np.sqrt(mb.sampson(Fs[b], P1[gt == a], P2[gt == a]))))
    return min(s)


def add_noise(x1, x2, kind, lvl, rng):
    if kind == "patch":
        q = lvl
        return (np.floor(x1 / q) + 0.5) * q, (np.floor(x2 / q) + 0.5) * q, q / np.sqrt(6)
    return (x1 + rng.normal(scale=lvl, size=x1.shape), x2 + rng.normal(scale=lvl, size=x2.shape),
            lvl * np.sqrt(2))


def count_alg(d, kappa=KAPPA):
    for i in sorted(d):
        if d[i] < kappa:
            return i
    return None


rows = []
t0 = time.time()
for kind, lvl in CONDS:
    for n in NS:
        for t in range(TRIALS):
            c1, c2, gt = mb.synth_scene(n, rng=rng)
            sep = separation(c1, c2, gt, n)
            x1, x2, s_eff = add_noise(c1, c2, kind, lvl, rng)
            x1h, _ = mb.hartley_normalize(x1); x2h, _ = mb.hartley_normalize(x2)
            c = mb.count_motions(mb.spectral_stats(x1h, x2h))
            lab, _ = mb.segment(x1h, x2h, n, rng=rng)
            labr, kr = mb.sequential_ransac(x1, x2, thr_px=2.5 * s_eff, rng=rng, iters=500)
            rows.append(dict(kind=kind, level=lvl, n=n, trial=t, sep=sep, s_eff=s_eff,
                             snr=sep / s_eff,
                             alg_n=count_alg(c["d"]), alg_err=mb.seg_error(lab, gt, n),
                             rsc_n=kr,
                             rsc_err=mb.seg_error_partial(labr, gt, n) if kr <= 5 else np.nan))
    print(kind, lvl, f"{time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows)
df.to_csv("e5_step1_calibration_raw.csv", index=False)
