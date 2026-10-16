# w2_kubric_baselines: non-reconstruction baselines on Kubric (review item 1, W2)

Same scenes and query sets as e5_xm_eval.py (slope2d_q, epe_*_q, sep_ratio_q). Mean over scenes with bootstrap 95% CI; last column is the paired per-scene difference (model minus Any4D) with a two-sided sign test.

| model | data | n | slope2d_q | epe_static_q | epe_moving_q | sep_ratio_q | paired slope2d_q diff vs Any4D [sign test p] |
|---|---|---|---|---|---|---|---|
| Any4D | v2 | 100 | 0.61 [0.56, 0.65] | 4.10 [3.82, 4.39] | 35.03 [30.93, 39.64] | 0.83 [0.70, 0.96] | reference |
| Any4D | v3float | 69 | 0.56 [0.47, 0.64] | 4.53 [4.11, 5.03] | 47.88 [38.97, 57.61] | 0.73 [0.57, 0.89] | reference |
| VGGT-1B tracker | v2 | 100 | 0.34 [0.29, 0.41] | 9.68 [7.49, 12.32] | 57.55 [51.12, 63.95] | 0.23 [0.16, 0.33] | -0.26 [-0.34, -0.18] [p 3.7e-06, 26+/72-] |
| VGGT-1B tracker | v3float | 69 | 0.57 [0.47, 0.68] | 12.95 [10.55, 15.68] | 50.48 [40.00, 61.35] | 0.27 [0.19, 0.35] | 0.02 [-0.10, 0.14] [p 0.620, 30+/35-] |
| CoTracker3, 24 fr | v2 | 100 | 0.96 [0.93, 0.97] | 3.97 [2.29, 6.05] | 5.56 [3.98, 7.75] | 1.14 [0.88, 1.57] | 0.35 [0.31, 0.39] [p 9.9e-25, 95+/3-] |
| CoTracker3, 24 fr | v3float | 69 | 0.97 [0.96, 0.98] | 4.21 [1.70, 7.76] | 3.43 [2.67, 4.31] | 0.96 [0.86, 1.07] | 0.41 [0.33, 0.50] [p 3.9e-14, 61+/4-] |
| CoTracker3, 7 fr | v2 | 100 | 0.95 [0.92, 0.97] | 4.97 [2.77, 7.68] | 6.48 [4.51, 8.99] | 0.95 [0.84, 1.07] | 0.34 [0.30, 0.38] [p 2.4e-23, 94+/4-] |
| CoTracker3, 7 fr | v3float | 69 | 0.96 [0.94, 0.97] | 5.76 [1.87, 10.96] | 4.31 [2.91, 6.31] | 1.04 [0.87, 1.25] | 0.40 [0.32, 0.49] [p 4.9e-13, 60+/5-] |
| RAFT, 0 to 23 | v2 | 100 | 0.59 [0.48, 0.69] | 67.97 [55.41, 80.73] | 46.28 [35.49, 57.86] | 0.90 [0.66, 1.25] | -0.02 [-0.12, 0.08] [p 0.006, 63+/35-] |
| RAFT, 0 to 23 | v3float | 69 | 0.50 [0.33, 0.65] | 65.34 [48.04, 83.46] | 68.74 [47.83, 91.31] | 0.80 [0.54, 1.17] | -0.06 [-0.20, 0.08] [p 0.136, 39+/26-] |
| RAFT chain, 7 fr | v2 | 100 | 0.88 [0.84, 0.90] | 3.16 [1.41, 5.30] | 10.49 [7.64, 13.85] | 0.84 [0.63, 1.17] | 0.27 [0.23, 0.31] [p 8.1e-16, 87+/11-] |
| RAFT chain, 7 fr | v3float | 69 | 0.89 [0.85, 0.91] | 4.71 [1.48, 9.03] | 8.26 [4.98, 13.36] | 0.66 [0.56, 0.77] | 0.33 [0.26, 0.41] [p 4.3e-11, 58+/7-] |
| RAFT chain, 24 fr | v2 | 100 | 0.82 [0.79, 0.85] | 6.07 [3.87, 8.57] | 12.74 [10.25, 15.59] | 0.77 [0.59, 1.06] | 0.22 [0.17, 0.26] [p 1.4e-12, 83+/15-] |
| RAFT chain, 24 fr | v3float | 69 | 0.85 [0.82, 0.88] | 6.60 [3.43, 10.63] | 10.47 [7.13, 15.76] | 0.65 [0.55, 0.75] | 0.29 [0.22, 0.37] [p 4.3e-11, 58+/7-] |

Scenes with evaluation errors: {'any4d|v2': 0, 'any4d|v3float': 0, 'vggt|v2': 0, 'vggt|v3float': 0, 'cotracker3_24f|v2': 0, 'cotracker3_24f|v3float': 0, 'cotracker3_7f|v2': 0, 'cotracker3_7f|v3float': 0, 'raft_direct|v2': 0, 'raft_direct|v3float': 0, 'raft_chain7|v2': 0, 'raft_chain7|v3float': 0, 'raft_chain24|v2': 0, 'raft_chain24|v3float': 0}

![baselines](fig_w2_baselines_kubric.png)
