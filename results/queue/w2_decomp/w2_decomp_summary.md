# w2_decomp: pull toward camera motion vs shrinkage vs smoothing (review item 1, W2)

Per scene, on moving query points with |x_true - x_static| > 2 px: x_pred - x1 = alpha (x_true - x_static) + beta (x_static - x1) + e (least squares over both coordinates). Pure camera-motion pull: beta near 1, alpha < 1. Shrinkage toward zero image motion: alpha near beta, both < 1. beta_static is the same fit on static points (only beta). alpha and beta are only separable when the object's motion is not parallel to the camera-induced motion and both are visible: alpha, beta and their tests use only scenes where the condition number of [A B] is at most 10 and the median |A| and |B| are at least 2 px (column n: identifiable of all scenes). Epipolar-aligned scenes, most KITTI cars and Sintel shots with a near-static camera fail this guard. slope and beta_static use all scenes. Mean over scenes [bootstrap 95% CI]; sign tests are two-sided.

| model | data | n | slope | alpha | beta | beta_static | alpha - beta [sign p] | beta vs 1 [sign p] | median abs cos(A,B) | median alpha, beta over all scenes (no guard) |
|---|---|---|---|---|---|---|---|---|---|---|
| Any4D | v2 | 84 of 100 | 0.61 [0.56, 0.65] | 0.61 [0.56, 0.67] | 1.03 [0.93, 1.13] | 0.99 [0.99, 1.00] | -0.42 [-0.53, -0.30] [7.9e-11] | 46+/38- [0.445] | 0.64 | 0.67, 1.01 |
| Any4D | v3float | 35 of 69 | 0.56 [0.47, 0.64] | 0.64 [0.56, 0.71] | 1.02 [0.88, 1.16] | 0.99 [0.98, 1.00] | -0.38 [-0.56, -0.22] [5.1e-04] | 17+/18- [1.000] | 0.83 | 0.74, 0.87 |
| VGGT-1B tracker | v2 | 84 of 100 | 0.34 [0.29, 0.41] | 0.22 [0.16, 0.27] | 1.02 [0.92, 1.11] | 1.02 [1.00, 1.04] | -0.80 [-0.90, -0.69] [2.1e-19] | 44+/40- [0.744] | 0.64 | 0.16, 1.05 |
| VGGT-1B tracker | v3float | 35 of 69 | 0.57 [0.47, 0.68] | 0.32 [0.23, 0.42] | 0.95 [0.71, 1.21] | 1.06 [1.02, 1.09] | -0.62 [-0.91, -0.35] [0.006] | 13+/22- [0.175] | 0.83 | 0.39, 0.72 |
| CoTracker3, 24 fr | v2 | 84 of 100 | 0.96 [0.93, 0.97] | 0.95 [0.94, 0.97] | 0.99 [0.97, 1.01] | 0.97 [0.94, 0.99] | -0.04 [-0.07, -0.01] [0.001] | 41+/43- [0.913] | 0.64 | 0.96, 1.00 |
| CoTracker3, 24 fr | v3float | 35 of 69 | 0.97 [0.96, 0.98] | 0.97 [0.96, 0.98] | 1.02 [0.99, 1.05] | 0.97 [0.93, 0.99] | -0.05 [-0.08, -0.02] [5.1e-04] | 15+/20- [0.500] | 0.83 | 0.97, 1.01 |
| CoTracker3, 7 fr | v2 | 84 of 100 | 0.95 [0.92, 0.97] | 0.94 [0.92, 0.97] | 0.98 [0.96, 1.00] | 0.96 [0.93, 0.98] | -0.04 [-0.07, -0.00] [0.038] | 39+/45- [0.586] | 0.64 | 0.97, 1.00 |
| CoTracker3, 7 fr | v3float | 35 of 69 | 0.96 [0.94, 0.97] | 0.96 [0.94, 0.98] | 1.00 [0.98, 1.03] | 0.95 [0.90, 0.98] | -0.05 [-0.07, -0.02] [0.017] | 14+/21- [0.311] | 0.83 | 0.97, 1.00 |
| RAFT, 0 to 23 | v2 | 84 of 100 | 0.59 [0.48, 0.69] | 0.61 [0.49, 0.71] | 0.68 [0.43, 0.90] | 0.60 [0.55, 0.66] | -0.07 [-0.32, 0.18] [0.586] | 27+/57- [0.001] | 0.64 | 0.75, 0.89 |
| RAFT, 0 to 23 | v3float | 35 of 69 | 0.50 [0.33, 0.65] | 0.62 [0.47, 0.75] | 0.78 [0.54, 1.01] | 0.66 [0.58, 0.75] | -0.16 [-0.45, 0.10] [0.736] | 10+/25- [0.017] | 0.83 | 0.62, 0.96 |
| RAFT chain, 7 fr | v2 | 84 of 100 | 0.88 [0.84, 0.90] | 0.85 [0.81, 0.89] | 0.98 [0.93, 1.03] | 0.97 [0.95, 0.99] | -0.13 [-0.20, -0.07] [4.7e-07] | 40+/44- [0.744] | 0.64 | 0.90, 1.00 |
| RAFT chain, 7 fr | v3float | 35 of 69 | 0.89 [0.85, 0.91] | 0.91 [0.87, 0.94] | 1.05 [0.99, 1.11] | 0.96 [0.92, 0.99] | -0.14 [-0.21, -0.08] [0.017] | 14+/21- [0.311] | 0.83 | 0.93, 0.98 |
| RAFT chain, 24 fr | v2 | 84 of 100 | 0.82 [0.79, 0.85] | 0.78 [0.73, 0.82] | 0.99 [0.92, 1.05] | 0.95 [0.92, 0.97] | -0.21 [-0.29, -0.14] [3.5e-08] | 41+/43- [0.913] | 0.64 | 0.82, 1.01 |
| RAFT chain, 24 fr | v3float | 35 of 69 | 0.85 [0.82, 0.88] | 0.85 [0.81, 0.89] | 1.06 [0.97, 1.15] | 0.94 [0.90, 0.98] | -0.21 [-0.30, -0.12] [0.017] | 14+/21- [0.311] | 0.83 | 0.90, 0.95 |
| Any4D | kitti15_rgb | 42 of 200 | 0.58 [0.51, 0.64] | 0.18 [0.01, 0.36] | 0.88 [0.69, 1.07] | 0.84 [0.80, 0.87] | -0.70 [-0.85, -0.54] [4.4e-07] | 20+/22- [0.878] | 0.99 | 0.40, 0.77 |
| Any4D | sintel_clean | 123 of 256 | 0.31 [0.23, 0.39] | 0.38 [0.29, 0.46] | 0.91 [0.73, 1.10] | 0.45 [0.37, 0.53] | -0.53 [-0.75, -0.33] [1.2e-05] | 48+/75- [0.019] | 0.51 | 0.29, 0.73 |
| Any4D | sintel_final | 123 of 256 | 0.33 [0.24, 0.41] | 0.32 [0.24, 0.41] | 1.00 [0.79, 1.25] | 0.48 [0.39, 0.58] | -0.68 [-0.94, -0.45] [2.9e-10] | 44+/79- [0.002] | 0.51 | 0.25, 0.66 |
| RAFT | kitti15_rgb | 41 of 200 | 0.95 [0.94, 0.96] | 0.94 [0.88, 0.98] | 0.97 [0.92, 1.01] | 0.91 [0.88, 0.92] | -0.03 [-0.06, -0.00] [0.028] | 13+/28- [0.028] | 0.99 | 0.97, 0.95 |
| RAFT | sintel_clean | 122 of 256 | 0.91 [0.89, 0.93] | 0.86 [0.82, 0.90] | 0.87 [0.80, 0.93] | 0.92 [0.80, 0.99] | -0.01 [-0.05, 0.05] [0.001] | 36+/86- [6.9e-06] | 0.51 | 0.96, 0.96 |
| RAFT | sintel_final | 122 of 256 | 0.90 [0.88, 0.92] | 0.86 [0.82, 0.90] | 0.89 [0.83, 0.95] | 0.98 [0.95, 1.01] | -0.03 [-0.08, 0.01] [0.085] | 39+/83- [8.4e-05] | 0.51 | 0.95, 0.95 |

## Slope vs distance to the object boundary (Kubric)

Spatial smoothing toward the neighbouring background predicts a lower slope near the boundary and a slope near 1 in the interior. A pull that is the same at all distances is not explained by smoothing.

| model | data | slope, d 0-3 px | slope, d 3-6 px | slope, d 6-12 px | slope, d 12-24 px | slope, d >24 px | edge (d <= 4) | interior (d >= 14) | interior - edge [sign p] |
|---|---|---|---|---|---|---|---|---|---|
| Any4D | v2 | 0.36 [0.31, 0.40] | 0.50 [0.46, 0.55] | 0.68 [0.63, 0.72] | 0.84 [0.80, 0.88] | 0.93 [0.86, 1.00] | 0.40 [0.36, 0.44] | 0.86 [0.82, 0.91] | 0.45 [0.41, 0.49] [1.3e-23] |
| Any4D | v3float | 0.25 [0.16, 0.33] | 0.43 [0.35, 0.50] | 0.62 [0.54, 0.69] | 0.85 [0.78, 0.91] | 0.94 [0.87, 1.02] | 0.30 [0.22, 0.38] | 0.87 [0.80, 0.94] | 0.49 [0.42, 0.56] [4.6e-14] |
| VGGT-1B tracker | v2 | 0.26 [0.20, 0.32] | 0.32 [0.26, 0.39] | 0.35 [0.29, 0.41] | 0.36 [0.30, 0.42] | 0.37 [0.29, 0.46] | 0.28 [0.22, 0.34] | 0.36 [0.30, 0.42] | 0.14 [0.10, 0.19] [2.4e-07] |
| VGGT-1B tracker | v3float | 0.50 [0.39, 0.60] | 0.55 [0.45, 0.66] | 0.57 [0.46, 0.67] | 0.62 [0.51, 0.72] | 0.70 [0.56, 0.84] | 0.52 [0.41, 0.62] | 0.64 [0.52, 0.74] | 0.10 [0.05, 0.16] [0.002] |
| CoTracker3, 24 fr | v2 | 0.90 [0.87, 0.93] | 0.96 [0.93, 0.98] | 0.98 [0.97, 0.99] | 0.99 [0.98, 1.00] | 1.00 [0.98, 1.01] | 0.92 [0.90, 0.94] | 0.99 [0.98, 1.00] | 0.06 [0.04, 0.07] [1.1e-10] |
| CoTracker3, 24 fr | v3float | 0.90 [0.86, 0.93] | 0.98 [0.97, 0.98] | 0.98 [0.98, 0.99] | 0.99 [0.99, 1.00] | 0.99 [0.98, 1.00] | 0.93 [0.91, 0.95] | 0.99 [0.98, 1.00] | 0.07 [0.04, 0.09] [1.2e-07] |
| CoTracker3, 7 fr | v2 | 0.89 [0.86, 0.92] | 0.95 [0.92, 0.97] | 0.97 [0.95, 0.99] | 0.99 [0.97, 1.00] | 0.99 [0.98, 1.01] | 0.91 [0.88, 0.94] | 0.99 [0.98, 1.00] | 0.06 [0.05, 0.08] [5.9e-10] |
| CoTracker3, 7 fr | v3float | 0.87 [0.82, 0.91] | 0.97 [0.95, 0.98] | 0.98 [0.96, 0.99] | 0.99 [0.97, 1.00] | 0.98 [0.96, 1.00] | 0.90 [0.87, 0.93] | 0.98 [0.97, 1.00] | 0.08 [0.06, 0.12] [1.2e-07] |
| RAFT, 0 to 23 | v2 | 0.50 [0.40, 0.60] | 0.58 [0.48, 0.68] | 0.62 [0.51, 0.72] | 0.72 [0.61, 0.81] | 0.85 [0.72, 0.98] | 0.53 [0.42, 0.63] | 0.78 [0.68, 0.87] | 0.17 [0.12, 0.24] [2.9e-05] |
| RAFT, 0 to 23 | v3float | 0.43 [0.27, 0.58] | 0.50 [0.33, 0.65] | 0.54 [0.37, 0.69] | 0.61 [0.43, 0.78] | 0.83 [0.68, 0.96] | 0.45 [0.30, 0.61] | 0.62 [0.44, 0.79] | 0.08 [0.02, 0.14] [0.161] |
| RAFT chain, 7 fr | v2 | 0.70 [0.66, 0.73] | 0.89 [0.85, 0.91] | 0.94 [0.92, 0.96] | 0.97 [0.95, 0.98] | 0.99 [0.97, 1.01] | 0.76 [0.73, 0.79] | 0.97 [0.95, 0.99] | 0.19 [0.17, 0.22] [4.0e-20] |
| RAFT chain, 7 fr | v3float | 0.67 [0.61, 0.73] | 0.92 [0.89, 0.94] | 0.96 [0.94, 0.97] | 0.97 [0.95, 0.99] | 0.97 [0.92, 0.99] | 0.76 [0.71, 0.80] | 0.97 [0.95, 0.99] | 0.19 [0.14, 0.24] [2.3e-09] |
| RAFT chain, 24 fr | v2 | 0.61 [0.57, 0.65] | 0.81 [0.78, 0.85] | 0.90 [0.87, 0.93] | 0.94 [0.92, 0.97] | 0.98 [0.94, 1.01] | 0.67 [0.63, 0.71] | 0.95 [0.93, 0.98] | 0.27 [0.24, 0.31] [1.0e-21] |
| RAFT chain, 24 fr | v3float | 0.59 [0.52, 0.65] | 0.86 [0.83, 0.89] | 0.93 [0.91, 0.95] | 0.95 [0.92, 0.97] | 0.94 [0.90, 0.98] | 0.68 [0.63, 0.73] | 0.95 [0.92, 0.97] | 0.25 [0.19, 0.32] [2.4e-10] |

## Rule-based reading (thresholds: slope CI vs 0.9; beta CI vs 0.95 and 1; sign tests at 0.05)

- Any4D, v2: pull (slope CI below 0.9); camera motion kept (beta near 1), own motion under-recovered (alpha < 1): camera-motion pull; slope lower at object boundaries (spatial smoothing contributes)
- Any4D, v3float: pull (slope CI below 0.9); camera motion kept (beta near 1), own motion under-recovered (alpha < 1): camera-motion pull; slope lower at object boundaries (spatial smoothing contributes)
- VGGT-1B tracker, v2: pull (slope CI below 0.9); camera motion kept (beta near 1), own motion under-recovered (alpha < 1): camera-motion pull; slope lower at object boundaries (spatial smoothing contributes)
- VGGT-1B tracker, v3float: pull (slope CI below 0.9); camera motion kept (beta near 1), own motion under-recovered (alpha < 1): camera-motion pull; slope lower at object boundaries (spatial smoothing contributes)
- CoTracker3, 24 fr, v2: no pull (slope CI above 0.9); camera motion kept (beta near 1); slope lower at object boundaries (spatial smoothing contributes)
- CoTracker3, 24 fr, v3float: no pull (slope CI above 0.9); camera motion kept (beta near 1); slope lower at object boundaries (spatial smoothing contributes)
- CoTracker3, 7 fr, v2: no pull (slope CI above 0.9); camera motion kept (beta near 1); slope lower at object boundaries (spatial smoothing contributes)
- CoTracker3, 7 fr, v3float: no pull (slope CI above 0.9); camera motion kept (beta near 1); slope lower at object boundaries (spatial smoothing contributes)
- RAFT, 0 to 23, v2: pull (slope CI below 0.9); shrinkage toward zero image motion (beta below 1 and alpha not different from beta); slope lower at object boundaries (spatial smoothing contributes)
- RAFT, 0 to 23, v3float: pull (slope CI below 0.9); camera motion kept (beta near 1), own motion under-recovered (alpha < 1): camera-motion pull
- RAFT chain, 7 fr, v2: weak or unclear pull (slope CI includes 0.9); camera motion kept (beta near 1), own motion under-recovered (alpha < 1): camera-motion pull; slope lower at object boundaries (spatial smoothing contributes)
- RAFT chain, 7 fr, v3float: weak or unclear pull (slope CI includes 0.9); camera motion kept (beta near 1), own motion under-recovered (alpha < 1): camera-motion pull; slope lower at object boundaries (spatial smoothing contributes)
- RAFT chain, 24 fr, v2: pull (slope CI below 0.9); camera motion kept (beta near 1), own motion under-recovered (alpha < 1): camera-motion pull; slope lower at object boundaries (spatial smoothing contributes)
- RAFT chain, 24 fr, v3float: pull (slope CI below 0.9); camera motion kept (beta near 1), own motion under-recovered (alpha < 1): camera-motion pull; slope lower at object boundaries (spatial smoothing contributes)
- Any4D, kitti15_rgb: pull (slope CI below 0.9); camera motion kept (beta near 1), own motion under-recovered (alpha < 1): camera-motion pull
- Any4D, sintel_clean: pull (slope CI below 0.9); camera motion kept (beta near 1), own motion under-recovered (alpha < 1): camera-motion pull
- Any4D, sintel_final: pull (slope CI below 0.9); camera motion kept (beta near 1), own motion under-recovered (alpha < 1): camera-motion pull
- raft, kitti15_rgb: no pull (slope CI above 0.9); camera motion kept (beta near 1), own motion under-recovered (alpha < 1): camera-motion pull
- raft, sintel_clean: weak or unclear pull (slope CI includes 0.9); mixed (beta below 1 and alpha different from beta)
- raft, sintel_final: weak or unclear pull (slope CI includes 0.9); shrinkage toward zero image motion (beta below 1 and alpha not different from beta)

![alpha beta](fig_w2_alpha_beta.png)

![boundary](fig_w2_boundary.png)
