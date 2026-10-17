Cells: mean over seeds +/- sd over seeds [95% bootstrap CI over scenes of the seed-averaged per-scene value]. Chance is 0.5 for all AUCs. Folds are grouped by scene (5 folds, as round 1). Sources: trained = Any4D trained; randall = random init (all); randtf = DINOv2 + random transformer; enc_any4d = Any4D encoder tokens (no view PE); enc_dino = public DINOv2 tokens (no view PE).


## Key paired comparisons, layers 0 and 24 (per-scene seed means; mean difference [95% CI], sign-test p, n scenes)

| dataset | comparison | source | probe | layer | metric | difference [95% CI] | sign p | n |
|---|---|---|---|---|---|---|---|---|
| v2 | selectivity real - patch | randall | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.006 [-0.042, +0.056] | 0.75 | 93 |
| v2 | selectivity real - instance | randall | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.022 [-0.050, +0.093] | 0.3 | 93 |
| v2 | selectivity real - patch | randall | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.139 [+0.110, +0.168] | 3.6e-12 | 93 |
| v2 | selectivity real - instance | randall | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.001 [-0.030, +0.029] | 1 | 93 |
| v2 | selectivity real - patch | randall | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | -0.009 [-0.063, +0.047] | 0.41 | 93 |
| v2 | selectivity real - instance | randall | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.014 [-0.061, +0.089] | 0.75 | 93 |
| v2 | L24 - L0 | randall | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | -0.028 [-0.061, +0.007] | 0.39 | 93 |
| v2 | selectivity real - patch | randall | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.148 [+0.114, +0.182] | 1.9e-11 | 93 |
| v2 | selectivity real - instance | randall | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | -0.015 [-0.048, +0.022] | 0.21 | 93 |
| v2 | L24 - L0 | randall | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.000 [-0.014, +0.015] | 0.84 | 93 |
| v2 | selectivity real - patch | enc_dino | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | -0.043 [-0.104, +0.020] | 0.12 | 93 |
| v2 | selectivity real - instance | enc_dino | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | -0.044 [-0.121, +0.033] | 0.84 | 93 |
| v2 | selectivity real - patch | enc_dino | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.242 [+0.214, +0.269] | 2.7e-23 | 93 |
| v2 | selectivity real - instance | enc_dino | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.004 [-0.035, +0.029] | 1 | 93 |
| v2 | selectivity real - patch | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.062 [-0.002, +0.126] | 0.21 | 93 |
| v2 | selectivity real - instance | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.134 [+0.048, +0.214] | 0.0046 | 93 |
| v2 | trained - randall | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.049 [-0.030, +0.127] | 0.12 | 93 |
| v2 | trained - randtf | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.090 [+0.038, +0.141] | 0.12 | 93 |
| v2 | trained L0 - enc_any4d | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.009 [-0.009, +0.026] | 0.59 | 93 |
| v2 | trained L0 - enc_dino | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.085 [+0.036, +0.134] | 0.0067 | 93 |
| v2 | selectivity real - patch | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.242 [+0.213, +0.270] | 6.2e-22 | 93 |
| v2 | selectivity real - instance | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.005 [-0.041, +0.032] | 0.68 | 93 |
| v2 | trained - randall | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.119 [+0.087, +0.152] | 1e-07 | 93 |
| v2 | trained - randtf | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.005 [-0.018, +0.029] | 0.038 | 93 |
| v2 | trained L0 - enc_any4d | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.030 [-0.042, -0.018] | 1.1e-06 | 93 |
| v2 | trained L0 - enc_dino | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.006 [-0.021, +0.031] | 0.061 | 93 |
| v2 | selectivity real - patch | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.492 [+0.469, +0.514] | 2e-28 | 93 |
| v2 | selectivity real - instance | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.595 [+0.537, +0.654] | 8.1e-28 | 93 |
| v2 | trained - randall | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.506 [+0.454, +0.556] | 8.1e-28 | 93 |
| v2 | trained - randtf | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.543 [+0.494, +0.594] | 4e-28 | 93 |
| v2 | L24 - L0 | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.429 [+0.372, +0.488] | 1.6e-27 | 93 |
| v2 | selectivity real - patch | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.392 [+0.361, +0.418] | 2.7e-23 | 93 |
| v2 | selectivity real - instance | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.106 [+0.071, +0.138] | 9.6e-11 | 93 |
| v2 | trained - randall | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.264 [+0.225, +0.300] | 1.7e-19 | 93 |
| v2 | trained - randtf | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.142 [+0.107, +0.175] | 3.6e-12 | 93 |
| v2 | L24 - L0 | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.145 [+0.115, +0.176] | 1.8e-15 | 93 |
| v2 | selectivity real - patch | randtf | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | -0.039 [-0.099, +0.021] | 0.35 | 93 |
| v2 | selectivity real - instance | randtf | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | -0.055 [-0.133, +0.023] | 0.15 | 93 |
| v2 | selectivity real - patch | randtf | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.239 [+0.213, +0.267] | 8.8e-25 | 93 |
| v2 | selectivity real - instance | randtf | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.012 [-0.043, +0.023] | 0.097 | 93 |
| v2 | selectivity real - patch | randtf | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | -0.067 [-0.123, -0.014] | 0.0011 | 93 |
| v2 | selectivity real - instance | randtf | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | -0.065 [-0.140, +0.011] | 0.061 | 93 |
| v2 | L24 - L0 | randtf | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | -0.024 [-0.061, +0.008] | 0.21 | 93 |
| v2 | selectivity real - patch | randtf | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.251 [+0.216, +0.284] | 1.7e-19 | 93 |
| v2 | selectivity real - instance | randtf | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.012 [-0.024, +0.051] | 1 | 93 |
| v2 | L24 - L0 | randtf | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.009 [-0.010, +0.026] | 0.84 | 93 |
| v2 | selectivity real - patch | enc_any4d | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.058 [-0.001, +0.117] | 0.21 | 93 |
| v2 | selectivity real - instance | enc_any4d | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.101 [+0.012, +0.191] | 0.016 | 93 |
| v2 | selectivity real - patch | enc_any4d | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.267 [+0.240, +0.292] | 2.7e-23 | 93 |
| v2 | selectivity real - instance | enc_any4d | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.002 [-0.036, +0.034] | 1 | 93 |
| v3float | selectivity real - patch | randall | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | -0.035 [-0.095, +0.025] | 0.9 | 62 |
| v3float | selectivity real - instance | randall | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | -0.029 [-0.122, +0.065] | 0.25 | 62 |
| v3float | selectivity real - patch | randall | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.143 [+0.099, +0.186] | 1.1e-08 | 62 |
| v3float | selectivity real - instance | randall | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.013 [-0.028, +0.050] | 0.9 | 62 |
| v3float | selectivity real - patch | randall | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.018 [-0.050, +0.087] | 0.37 | 62 |
| v3float | selectivity real - instance | randall | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.012 [-0.075, +0.102] | 1 | 62 |
| v3float | L24 - L0 | randall | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.016 [-0.022, +0.053] | 0.37 | 62 |
| v3float | selectivity real - patch | randall | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.129 [+0.080, +0.180] | 0.0005 | 62 |
| v3float | selectivity real - instance | randall | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.006 [-0.044, +0.051] | 0.25 | 62 |
| v3float | L24 - L0 | randall | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | -0.016 [-0.047, +0.014] | 0.25 | 62 |
| v3float | selectivity real - patch | enc_dino | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.059 [-0.021, +0.139] | 0.098 | 62 |
| v3float | selectivity real - instance | enc_dino | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.052 [-0.049, +0.146] | 0.37 | 62 |
| v3float | selectivity real - patch | enc_dino | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.247 [+0.211, +0.285] | 2.6e-13 | 62 |
| v3float | selectivity real - instance | enc_dino | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.024 [-0.061, +0.013] | 0.9 | 62 |
| v3float | selectivity real - patch | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.073 [-0.028, +0.172] | 0.25 | 62 |
| v3float | selectivity real - instance | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | -0.047 [-0.160, +0.072] | 0.9 | 62 |
| v3float | trained - randall | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.094 [-0.013, +0.200] | 0.098 | 62 |
| v3float | trained - randtf | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.023 [-0.051, +0.097] | 0.53 | 62 |
| v3float | trained L0 - enc_any4d | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.001 [-0.039, +0.035] | 0.51 | 62 |
| v3float | trained L0 - enc_dino | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.011 [-0.065, +0.088] | 1 | 62 |
| v3float | selectivity real - patch | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.270 [+0.234, +0.304] | 8.5e-16 | 62 |
| v3float | selectivity real - instance | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.015 [-0.056, +0.027] | 1 | 62 |
| v3float | trained - randall | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.135 [+0.089, +0.181] | 0.0013 | 62 |
| v3float | trained - randtf | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.012 [-0.018, +0.042] | 0.53 | 62 |
| v3float | trained L0 - enc_any4d | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.027 [-0.049, -0.008] | 0.03 | 62 |
| v3float | trained L0 - enc_dino | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.020 [-0.011, +0.051] | 0.16 | 62 |
| v3float | selectivity real - patch | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.482 [+0.441, +0.519] | 5.4e-17 | 62 |
| v3float | selectivity real - instance | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.407 [+0.322, +0.499] | 9.1e-13 | 62 |
| v3float | trained - randall | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.483 [+0.415, +0.552] | 1.1e-16 | 62 |
| v3float | trained - randtf | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.445 [+0.374, +0.516] | 1.1e-16 | 62 |
| v3float | L24 - L0 | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.405 [+0.314, +0.495] | 5.4e-17 | 62 |
| v3float | selectivity real - patch | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.417 [+0.388, +0.442] | 2.7e-17 | 62 |
| v3float | selectivity real - instance | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.145 [+0.105, +0.186] | 1.1e-08 | 62 |
| v3float | trained - randall | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.308 [+0.256, +0.359] | 3.1e-12 | 62 |
| v3float | trained - randtf | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.178 [+0.140, +0.214] | 1.7e-09 | 62 |
| v3float | L24 - L0 | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.157 [+0.117, +0.196] | 3.1e-12 | 62 |
| v3float | selectivity real - patch | randtf | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.043 [-0.035, +0.122] | 0.098 | 62 |
| v3float | selectivity real - instance | randtf | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.028 [-0.069, +0.130] | 0.37 | 62 |
| v3float | selectivity real - patch | randtf | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.254 [+0.218, +0.290] | 1.7e-14 | 62 |
| v3float | selectivity real - instance | randtf | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.020 [-0.060, +0.022] | 0.9 | 62 |
| v3float | selectivity real - patch | randtf | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.055 [-0.032, +0.143] | 0.37 | 62 |
| v3float | selectivity real - instance | randtf | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | -0.013 [-0.106, +0.082] | 0.7 | 62 |
| v3float | L24 - L0 | randtf | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | -0.018 [-0.056, +0.016] | 0.53 | 62 |
| v3float | selectivity real - patch | randtf | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.247 [+0.214, +0.279] | 8.5e-16 | 62 |
| v3float | selectivity real - instance | randtf | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | -0.037 [-0.083, +0.006] | 0.37 | 62 |
| v3float | L24 - L0 | randtf | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | -0.010 [-0.034, +0.016] | 0.9 | 62 |
| v3float | selectivity real - patch | enc_any4d | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.073 [-0.021, +0.167] | 0.098 | 62 |
| v3float | selectivity real - instance | enc_any4d | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | -0.055 [-0.160, +0.057] | 0.61 | 62 |
| v3float | selectivity real - patch | enc_any4d | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.291 [+0.254, +0.326] | 3.1e-12 | 62 |
| v3float | selectivity real - instance | enc_any4d | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.004 [-0.039, +0.044] | 0.16 | 62 |
| v2 | selectivity real - patch | randall | MLP | 0 | motion_auc_v0_objects | +0.017 [-0.035, +0.068] | 0.84 | 93 |
| v2 | selectivity real - instance | randall | MLP | 0 | motion_auc_v0_objects | -0.012 [-0.067, +0.044] | 0.84 | 93 |
| v2 | MLP - ridge/bilinear | randall | MLP | 0 | motion_auc_v0_objects | -0.007 [-0.039, +0.028] | 0.41 | 93 |
| v2 | selectivity real - patch | randall | MLP | 0 | pair_auc_obj | +0.135 [+0.104, +0.169] | 6.2e-13 | 93 |
| v2 | selectivity real - instance | randall | MLP | 0 | pair_auc_obj | -0.006 [-0.033, +0.021] | 0.53 | 93 |
| v2 | MLP - ridge/bilinear | randall | MLP | 0 | pair_auc_obj | -0.013 [-0.037, +0.011] | 0.41 | 93 |
| v2 | selectivity real - patch | randall | linear (MLP recipe) | 0 | motion_auc_v0_objects | +0.025 [-0.009, +0.057] | 1 | 93 |
| v2 | selectivity real - instance | randall | linear (MLP recipe) | 0 | motion_auc_v0_objects | +0.000 [-0.038, +0.037] | 0.097 | 93 |
| v2 | selectivity real - patch | randall | linear (MLP recipe) | 0 | pair_auc_obj | +0.145 [+0.115, +0.175] | 1.4e-14 | 93 |
| v2 | selectivity real - instance | randall | linear (MLP recipe) | 0 | pair_auc_obj | -0.000 [-0.023, +0.022] | 1 | 93 |
| v2 | selectivity real - patch | randall | MLP | 24 | motion_auc_v0_objects | -0.006 [-0.066, +0.051] | 1 | 93 |
| v2 | selectivity real - instance | randall | MLP | 24 | motion_auc_v0_objects | -0.021 [-0.080, +0.037] | 1 | 93 |
| v2 | MLP - ridge/bilinear | randall | MLP | 24 | motion_auc_v0_objects | +0.005 [-0.047, +0.055] | 0.53 | 93 |
| v2 | L24 - L0 | randall | MLP | 24 | motion_auc_v0_objects | -0.017 [-0.038, +0.004] | 0.75 | 93 |
| v2 | selectivity real - patch | randall | MLP | 24 | pair_auc_obj | +0.110 [+0.077, +0.143] | 3.5e-07 | 93 |
| v2 | selectivity real - instance | randall | MLP | 24 | pair_auc_obj | +0.010 [-0.019, +0.039] | 1 | 93 |
| v2 | MLP - ridge/bilinear | randall | MLP | 24 | pair_auc_obj | -0.039 [-0.063, -0.014] | 0.0067 | 93 |
| v2 | L24 - L0 | randall | MLP | 24 | pair_auc_obj | -0.026 [-0.043, -0.008] | 0.0067 | 93 |
| v2 | selectivity real - patch | randall | linear (MLP recipe) | 24 | motion_auc_v0_objects | +0.051 [+0.018, +0.084] | 0.028 | 93 |
| v2 | selectivity real - instance | randall | linear (MLP recipe) | 24 | motion_auc_v0_objects | +0.044 [+0.004, +0.086] | 0.15 | 93 |
| v2 | L24 - L0 | randall | linear (MLP recipe) | 24 | motion_auc_v0_objects | +0.035 [+0.009, +0.061] | 0.016 | 93 |
| v2 | selectivity real - patch | randall | linear (MLP recipe) | 24 | pair_auc_obj | +0.124 [+0.094, +0.158] | 3.6e-12 | 93 |
| v2 | selectivity real - instance | randall | linear (MLP recipe) | 24 | pair_auc_obj | +0.006 [-0.021, +0.034] | 1 | 93 |
| v2 | L24 - L0 | randall | linear (MLP recipe) | 24 | pair_auc_obj | -0.022 [-0.038, -0.008] | 0.0035 | 93 |
| v2 | selectivity real - patch | enc_dino | MLP | 0 | motion_auc_v0_objects | -0.021 [-0.068, +0.027] | 0.15 | 93 |
| v2 | selectivity real - instance | enc_dino | MLP | 0 | motion_auc_v0_objects | -0.051 [-0.107, +0.010] | 0.15 | 93 |
| v2 | MLP - ridge/bilinear | enc_dino | MLP | 0 | motion_auc_v0_objects | -0.001 [-0.029, +0.027] | 1 | 93 |
| v2 | selectivity real - patch | enc_dino | MLP | 0 | pair_auc_obj | +0.204 [+0.181, +0.227] | 8.8e-25 | 93 |
| v2 | selectivity real - instance | enc_dino | MLP | 0 | pair_auc_obj | -0.001 [-0.025, +0.021] | 0.41 | 93 |
| v2 | MLP - ridge/bilinear | enc_dino | MLP | 0 | pair_auc_obj | -0.044 [-0.064, -0.025] | 6.5e-05 | 93 |
| v2 | selectivity real - patch | enc_dino | linear (MLP recipe) | 0 | motion_auc_v0_objects | -0.029 [-0.065, +0.008] | 0.097 | 93 |
| v2 | selectivity real - instance | enc_dino | linear (MLP recipe) | 0 | motion_auc_v0_objects | -0.048 [-0.087, -0.004] | 0.00037 | 93 |
| v2 | selectivity real - patch | enc_dino | linear (MLP recipe) | 0 | pair_auc_obj | +0.232 [+0.207, +0.259] | 1.9e-26 | 93 |
| v2 | selectivity real - instance | enc_dino | linear (MLP recipe) | 0 | pair_auc_obj | -0.005 [-0.028, +0.020] | 0.3 | 93 |
| v2 | selectivity real - patch | trained | MLP | 0 | motion_auc_v0_objects | +0.130 [+0.077, +0.179] | 0.00081 | 93 |
| v2 | selectivity real - instance | trained | MLP | 0 | motion_auc_v0_objects | +0.119 [+0.060, +0.176] | 0.038 | 93 |
| v2 | trained - randall | trained | MLP | 0 | motion_auc_v0_objects | +0.106 [+0.032, +0.177] | 0.15 | 93 |
| v2 | trained - randtf | trained | MLP | 0 | motion_auc_v0_objects | +0.100 [+0.047, +0.149] | 0.0017 | 93 |
| v2 | trained L0 - enc_any4d | trained | MLP | 0 | motion_auc_v0_objects | +0.001 [-0.015, +0.017] | 1 | 93 |
| v2 | trained L0 - enc_dino | trained | MLP | 0 | motion_auc_v0_objects | +0.136 [+0.082, +0.190] | 0.0023 | 93 |
| v2 | MLP - ridge/bilinear | trained | MLP | 0 | motion_auc_v0_objects | +0.050 [+0.002, +0.096] | 0.3 | 93 |
| v2 | selectivity real - patch | trained | MLP | 0 | pair_auc_obj | +0.229 [+0.205, +0.253] | 8.8e-25 | 93 |
| v2 | selectivity real - instance | trained | MLP | 0 | pair_auc_obj | +0.045 [+0.020, +0.069] | 0.00016 | 93 |
| v2 | trained - randall | trained | MLP | 0 | pair_auc_obj | +0.099 [+0.068, +0.130] | 1e-07 | 93 |
| v2 | trained - randtf | trained | MLP | 0 | pair_auc_obj | -0.027 [-0.048, -0.005] | 6.5e-05 | 93 |
| v2 | trained L0 - enc_any4d | trained | MLP | 0 | pair_auc_obj | -0.013 [-0.023, -0.002] | 0.15 | 93 |
| v2 | trained L0 - enc_dino | trained | MLP | 0 | pair_auc_obj | +0.018 [-0.004, +0.039] | 0.097 | 93 |
| v2 | MLP - ridge/bilinear | trained | MLP | 0 | pair_auc_obj | -0.033 [-0.056, -0.011] | 0.0035 | 93 |
| v2 | selectivity real - patch | trained | linear (MLP recipe) | 0 | motion_auc_v0_objects | +0.022 [-0.017, +0.060] | 0.097 | 93 |
| v2 | selectivity real - instance | trained | linear (MLP recipe) | 0 | motion_auc_v0_objects | +0.027 [-0.021, +0.074] | 0.41 | 93 |
| v2 | trained - randall | trained | linear (MLP recipe) | 0 | motion_auc_v0_objects | +0.016 [-0.026, +0.061] | 0.68 | 93 |
| v2 | trained - randtf | trained | linear (MLP recipe) | 0 | motion_auc_v0_objects | +0.046 [+0.007, +0.087] | 0.038 | 93 |
| v2 | trained L0 - enc_any4d | trained | linear (MLP recipe) | 0 | motion_auc_v0_objects | +0.015 [-0.011, +0.039] | 0.84 | 93 |
| v2 | trained L0 - enc_dino | trained | linear (MLP recipe) | 0 | motion_auc_v0_objects | +0.059 [+0.011, +0.104] | 0.038 | 93 |
| v2 | selectivity real - patch | trained | linear (MLP recipe) | 0 | pair_auc_obj | +0.213 [+0.188, +0.237] | 1.1e-20 | 93 |
| v2 | selectivity real - instance | trained | linear (MLP recipe) | 0 | pair_auc_obj | +0.014 [-0.010, +0.040] | 0.15 | 93 |
| v2 | trained - randall | trained | linear (MLP recipe) | 0 | pair_auc_obj | +0.074 [+0.044, +0.104] | 9.4e-06 | 93 |
| v2 | trained - randtf | trained | linear (MLP recipe) | 0 | pair_auc_obj | -0.028 [-0.049, -0.007] | 0.012 | 93 |
| v2 | trained L0 - enc_any4d | trained | linear (MLP recipe) | 0 | pair_auc_obj | +0.021 [+0.007, +0.033] | 0.0035 | 93 |
| v2 | trained L0 - enc_dino | trained | linear (MLP recipe) | 0 | pair_auc_obj | -0.028 [-0.050, -0.006] | 0.097 | 93 |
| v2 | selectivity real - patch | trained | MLP | 24 | motion_auc_v0_objects | +0.490 [+0.480, +0.501] | 2e-28 | 93 |
| v2 | selectivity real - instance | trained | MLP | 24 | motion_auc_v0_objects | +0.499 [+0.471, +0.526] | 2e-28 | 93 |
| v2 | trained - randall | trained | MLP | 24 | motion_auc_v0_objects | +0.502 [+0.447, +0.556] | 4e-28 | 93 |
| v2 | trained - randtf | trained | MLP | 24 | motion_auc_v0_objects | +0.512 [+0.462, +0.561] | 2e-28 | 93 |
| v2 | MLP - ridge/bilinear | trained | MLP | 24 | motion_auc_v0_objects | +0.000 [-0.001, +0.002] | 0.79 | 93 |
| v2 | L24 - L0 | trained | MLP | 24 | motion_auc_v0_objects | +0.379 [+0.326, +0.431] | 4e-28 | 93 |
| v2 | selectivity real - patch | trained | MLP | 24 | pair_auc_obj | +0.456 [+0.442, +0.468] | 2e-28 | 93 |
| v2 | selectivity real - instance | trained | MLP | 24 | pair_auc_obj | +0.209 [+0.186, +0.232] | 6.2e-22 | 93 |
| v2 | trained - randall | trained | MLP | 24 | pair_auc_obj | +0.357 [+0.323, +0.390] | 1.9e-26 | 93 |
| v2 | trained - randtf | trained | MLP | 24 | pair_auc_obj | +0.291 [+0.262, +0.319] | 8.8e-25 | 93 |
| v2 | MLP - ridge/bilinear | trained | MLP | 24 | pair_auc_obj | +0.054 [+0.032, +0.079] | 1.7e-06 | 93 |
| v2 | L24 - L0 | trained | MLP | 24 | pair_auc_obj | +0.232 [+0.203, +0.260] | 1.7e-19 | 93 |
| v2 | selectivity real - patch | trained | linear (MLP recipe) | 24 | motion_auc_v0_objects | +0.411 [+0.383, +0.434] | 8.8e-25 | 93 |
| v2 | selectivity real - instance | trained | linear (MLP recipe) | 24 | motion_auc_v0_objects | +0.426 [+0.392, +0.460] | 1.9e-26 | 93 |
| v2 | trained - randall | trained | linear (MLP recipe) | 24 | motion_auc_v0_objects | +0.356 [+0.320, +0.391] | 2.7e-23 | 93 |
| v2 | trained - randtf | trained | linear (MLP recipe) | 24 | motion_auc_v0_objects | +0.408 [+0.375, +0.440] | 1.9e-26 | 93 |
| v2 | L24 - L0 | trained | linear (MLP recipe) | 24 | motion_auc_v0_objects | +0.376 [+0.342, +0.412] | 4e-28 | 93 |
| v2 | selectivity real - patch | trained | linear (MLP recipe) | 24 | pair_auc_obj | +0.302 [+0.262, +0.338] | 6.2e-22 | 93 |
| v2 | selectivity real - instance | trained | linear (MLP recipe) | 24 | pair_auc_obj | +0.085 [+0.047, +0.119] | 1.1e-06 | 93 |
| v2 | trained - randall | trained | linear (MLP recipe) | 24 | pair_auc_obj | +0.195 [+0.151, +0.240] | 1.9e-11 | 93 |
| v2 | trained - randtf | trained | linear (MLP recipe) | 24 | pair_auc_obj | +0.080 [+0.039, +0.117] | 2.5e-05 | 93 |
| v2 | L24 - L0 | trained | linear (MLP recipe) | 24 | pair_auc_obj | +0.099 [+0.061, +0.132] | 7.7e-09 | 93 |
| v2 | selectivity real - patch | randtf | MLP | 0 | motion_auc_v0_objects | +0.021 [-0.027, +0.069] | 0.41 | 93 |
| v2 | selectivity real - instance | randtf | MLP | 0 | motion_auc_v0_objects | -0.000 [-0.056, +0.058] | 1 | 93 |
| v2 | MLP - ridge/bilinear | randtf | MLP | 0 | motion_auc_v0_objects | +0.040 [+0.005, +0.072] | 0.038 | 93 |
| v2 | selectivity real - patch | randtf | MLP | 0 | pair_auc_obj | +0.253 [+0.229, +0.277] | 1.9e-26 | 93 |
| v2 | selectivity real - instance | randtf | MLP | 0 | pair_auc_obj | +0.021 [-0.005, +0.047] | 0.0067 | 93 |
| v2 | MLP - ridge/bilinear | randtf | MLP | 0 | pair_auc_obj | -0.000 [-0.026, +0.024] | 0.41 | 93 |
| v2 | selectivity real - patch | randtf | linear (MLP recipe) | 0 | motion_auc_v0_objects | -0.018 [-0.054, +0.018] | 0.53 | 93 |
| v2 | selectivity real - instance | randtf | linear (MLP recipe) | 0 | motion_auc_v0_objects | -0.016 [-0.062, +0.029] | 0.84 | 93 |
| v2 | selectivity real - patch | randtf | linear (MLP recipe) | 0 | pair_auc_obj | +0.236 [+0.211, +0.261] | 2e-28 | 93 |
| v2 | selectivity real - instance | randtf | linear (MLP recipe) | 0 | pair_auc_obj | -0.005 [-0.029, +0.020] | 0.061 | 93 |
| v2 | selectivity real - patch | randtf | MLP | 24 | motion_auc_v0_objects | -0.022 [-0.071, +0.030] | 0.53 | 93 |
| v2 | selectivity real - instance | randtf | MLP | 24 | motion_auc_v0_objects | -0.042 [-0.098, +0.015] | 0.15 | 93 |
| v2 | MLP - ridge/bilinear | randtf | MLP | 24 | motion_auc_v0_objects | +0.032 [-0.012, +0.073] | 0.097 | 93 |
| v2 | L24 - L0 | randtf | MLP | 24 | motion_auc_v0_objects | -0.033 [-0.072, +0.007] | 0.0067 | 93 |
| v2 | selectivity real - patch | randtf | MLP | 24 | pair_auc_obj | +0.166 [+0.139, +0.192] | 1.8e-15 | 93 |
| v2 | selectivity real - instance | randtf | MLP | 24 | pair_auc_obj | -0.009 [-0.035, +0.017] | 0.84 | 93 |
| v2 | MLP - ridge/bilinear | randtf | MLP | 24 | pair_auc_obj | -0.095 [-0.122, -0.070] | 1e-07 | 93 |
| v2 | L24 - L0 | randtf | MLP | 24 | pair_auc_obj | -0.087 [-0.110, -0.063] | 1.1e-06 | 93 |
| v2 | selectivity real - patch | randtf | linear (MLP recipe) | 24 | motion_auc_v0_objects | -0.009 [-0.042, +0.023] | 1 | 93 |
| v2 | selectivity real - instance | randtf | linear (MLP recipe) | 24 | motion_auc_v0_objects | -0.015 [-0.049, +0.019] | 1 | 93 |
| v2 | L24 - L0 | randtf | linear (MLP recipe) | 24 | motion_auc_v0_objects | +0.015 [-0.016, +0.043] | 0.21 | 93 |
| v2 | selectivity real - patch | randtf | linear (MLP recipe) | 24 | pair_auc_obj | +0.227 [+0.195, +0.256] | 2.1e-18 | 93 |
| v2 | selectivity real - instance | randtf | linear (MLP recipe) | 24 | pair_auc_obj | -0.002 [-0.029, +0.025] | 1 | 93 |
| v2 | L24 - L0 | randtf | linear (MLP recipe) | 24 | pair_auc_obj | -0.009 [-0.024, +0.006] | 1 | 93 |
| v2 | selectivity real - patch | enc_any4d | MLP | 0 | motion_auc_v0_objects | +0.113 [+0.057, +0.168] | 0.00016 | 93 |
| v2 | selectivity real - instance | enc_any4d | MLP | 0 | motion_auc_v0_objects | +0.100 [+0.036, +0.163] | 0.061 | 93 |
| v2 | MLP - ridge/bilinear | enc_any4d | MLP | 0 | motion_auc_v0_objects | +0.058 [+0.018, +0.095] | 0.022 | 93 |
| v2 | selectivity real - patch | enc_any4d | MLP | 0 | pair_auc_obj | +0.240 [+0.219, +0.261] | 2e-28 | 93 |
| v2 | selectivity real - instance | enc_any4d | MLP | 0 | pair_auc_obj | +0.051 [+0.029, +0.073] | 6.5e-05 | 93 |
| v2 | MLP - ridge/bilinear | enc_any4d | MLP | 0 | pair_auc_obj | -0.050 [-0.071, -0.028] | 0.00081 | 93 |
| v2 | selectivity real - patch | enc_any4d | linear (MLP recipe) | 0 | motion_auc_v0_objects | +0.020 [-0.013, +0.053] | 0.21 | 93 |
| v2 | selectivity real - instance | enc_any4d | linear (MLP recipe) | 0 | motion_auc_v0_objects | +0.009 [-0.028, +0.046] | 0.84 | 93 |
| v2 | selectivity real - patch | enc_any4d | linear (MLP recipe) | 0 | pair_auc_obj | +0.190 [+0.164, +0.216] | 2.1e-18 | 93 |
| v2 | selectivity real - instance | enc_any4d | linear (MLP recipe) | 0 | pair_auc_obj | +0.009 [-0.019, +0.038] | 0.15 | 93 |
| v3float | selectivity real - patch | randall | MLP | 0 | motion_auc_v0_objects | -0.008 [-0.076, +0.058] | 0.7 | 62 |
| v3float | selectivity real - instance | randall | MLP | 0 | motion_auc_v0_objects | -0.011 [-0.084, +0.060] | 0.7 | 62 |
| v3float | MLP - ridge/bilinear | randall | MLP | 0 | motion_auc_v0_objects | +0.014 [-0.051, +0.074] | 0.7 | 62 |
| v3float | selectivity real - patch | randall | MLP | 0 | pair_auc_obj | +0.132 [+0.092, +0.174] | 1.7e-05 | 62 |
| v3float | selectivity real - instance | randall | MLP | 0 | pair_auc_obj | +0.004 [-0.031, +0.039] | 0.9 | 62 |
| v3float | MLP - ridge/bilinear | randall | MLP | 0 | pair_auc_obj | -0.011 [-0.048, +0.025] | 0.7 | 62 |
| v3float | selectivity real - patch | randall | linear (MLP recipe) | 0 | motion_auc_v0_objects | -0.001 [-0.043, +0.045] | 0.53 | 62 |
| v3float | selectivity real - instance | randall | linear (MLP recipe) | 0 | motion_auc_v0_objects | -0.014 [-0.060, +0.032] | 0.9 | 62 |
| v3float | selectivity real - patch | randall | linear (MLP recipe) | 0 | pair_auc_obj | +0.139 [+0.098, +0.183] | 4.8e-06 | 62 |
| v3float | selectivity real - instance | randall | linear (MLP recipe) | 0 | pair_auc_obj | +0.012 [-0.020, +0.045] | 0.16 | 62 |
| v3float | selectivity real - patch | randall | MLP | 24 | motion_auc_v0_objects | -0.036 [-0.102, +0.031] | 0.53 | 62 |
| v3float | selectivity real - instance | randall | MLP | 24 | motion_auc_v0_objects | -0.033 [-0.107, +0.042] | 0.7 | 62 |
| v3float | MLP - ridge/bilinear | randall | MLP | 24 | motion_auc_v0_objects | -0.019 [-0.086, +0.048] | 0.9 | 62 |
| v3float | L24 - L0 | randall | MLP | 24 | motion_auc_v0_objects | -0.017 [-0.055, +0.018] | 1 | 62 |
| v3float | selectivity real - patch | randall | MLP | 24 | pair_auc_obj | +0.093 [+0.055, +0.132] | 0.0005 | 62 |
| v3float | selectivity real - instance | randall | MLP | 24 | pair_auc_obj | -0.010 [-0.042, +0.020] | 0.37 | 62 |
| v3float | MLP - ridge/bilinear | randall | MLP | 24 | pair_auc_obj | -0.041 [-0.076, -0.006] | 0.015 | 62 |
| v3float | L24 - L0 | randall | MLP | 24 | pair_auc_obj | -0.046 [-0.070, -0.023] | 0.00018 | 62 |
| v3float | selectivity real - patch | randall | linear (MLP recipe) | 24 | motion_auc_v0_objects | +0.003 [-0.040, +0.046] | 1 | 62 |
| v3float | selectivity real - instance | randall | linear (MLP recipe) | 24 | motion_auc_v0_objects | +0.014 [-0.037, +0.068] | 0.9 | 62 |
| v3float | L24 - L0 | randall | linear (MLP recipe) | 24 | motion_auc_v0_objects | -0.003 [-0.033, +0.026] | 0.53 | 62 |
| v3float | selectivity real - patch | randall | linear (MLP recipe) | 24 | pair_auc_obj | +0.108 [+0.073, +0.147] | 4.8e-06 | 62 |
| v3float | selectivity real - instance | randall | linear (MLP recipe) | 24 | pair_auc_obj | -0.001 [-0.033, +0.029] | 1 | 62 |
| v3float | L24 - L0 | randall | linear (MLP recipe) | 24 | pair_auc_obj | -0.038 [-0.061, -0.014] | 0.098 | 62 |
| v3float | selectivity real - patch | enc_dino | MLP | 0 | motion_auc_v0_objects | +0.058 [-0.010, +0.123] | 0.16 | 62 |
| v3float | selectivity real - instance | enc_dino | MLP | 0 | motion_auc_v0_objects | +0.078 [+0.013, +0.143] | 0.03 | 62 |
| v3float | MLP - ridge/bilinear | enc_dino | MLP | 0 | motion_auc_v0_objects | -0.005 [-0.044, +0.033] | 1 | 62 |
| v3float | selectivity real - patch | enc_dino | MLP | 0 | pair_auc_obj | +0.189 [+0.158, +0.218] | 1.7e-14 | 62 |
| v3float | selectivity real - instance | enc_dino | MLP | 0 | pair_auc_obj | -0.017 [-0.047, +0.014] | 0.056 | 62 |
| v3float | MLP - ridge/bilinear | enc_dino | MLP | 0 | pair_auc_obj | -0.061 [-0.086, -0.034] | 0.0013 | 62 |
| v3float | selectivity real - patch | enc_dino | linear (MLP recipe) | 0 | motion_auc_v0_objects | +0.028 [-0.019, +0.071] | 0.53 | 62 |
| v3float | selectivity real - instance | enc_dino | linear (MLP recipe) | 0 | motion_auc_v0_objects | +0.013 [-0.044, +0.066] | 0.25 | 62 |
| v3float | selectivity real - patch | enc_dino | linear (MLP recipe) | 0 | pair_auc_obj | +0.238 [+0.206, +0.271] | 1.7e-14 | 62 |
| v3float | selectivity real - instance | enc_dino | linear (MLP recipe) | 0 | pair_auc_obj | -0.010 [-0.036, +0.017] | 0.25 | 62 |
| v3float | selectivity real - patch | trained | MLP | 0 | motion_auc_v0_objects | +0.038 [-0.033, +0.104] | 0.9 | 62 |
| v3float | selectivity real - instance | trained | MLP | 0 | motion_auc_v0_objects | +0.035 [-0.036, +0.110] | 0.53 | 62 |
| v3float | trained - randall | trained | MLP | 0 | motion_auc_v0_objects | +0.045 [-0.056, +0.140] | 0.7 | 62 |
| v3float | trained - randtf | trained | MLP | 0 | motion_auc_v0_objects | -0.026 [-0.081, +0.029] | 0.7 | 62 |
| v3float | trained L0 - enc_any4d | trained | MLP | 0 | motion_auc_v0_objects | -0.028 [-0.058, +0.002] | 0.056 | 62 |
| v3float | trained L0 - enc_dino | trained | MLP | 0 | motion_auc_v0_objects | -0.019 [-0.087, +0.046] | 0.7 | 62 |
| v3float | MLP - ridge/bilinear | trained | MLP | 0 | motion_auc_v0_objects | -0.035 [-0.097, +0.023] | 0.098 | 62 |
| v3float | selectivity real - patch | trained | MLP | 0 | pair_auc_obj | +0.245 [+0.220, +0.270] | 4.3e-19 | 62 |
| v3float | selectivity real - instance | trained | MLP | 0 | pair_auc_obj | +0.037 [+0.009, +0.068] | 0.16 | 62 |
| v3float | trained - randall | trained | MLP | 0 | pair_auc_obj | +0.125 [+0.082, +0.167] | 1.7e-05 | 62 |
| v3float | trained - randtf | trained | MLP | 0 | pair_auc_obj | +0.029 [-0.000, +0.062] | 0.25 | 62 |
| v3float | trained L0 - enc_any4d | trained | MLP | 0 | pair_auc_obj | -0.005 [-0.017, +0.008] | 0.37 | 62 |
| v3float | trained L0 - enc_dino | trained | MLP | 0 | pair_auc_obj | +0.060 [+0.026, +0.093] | 0.03 | 62 |
| v3float | MLP - ridge/bilinear | trained | MLP | 0 | pair_auc_obj | -0.021 [-0.049, +0.006] | 0.7 | 62 |
| v3float | selectivity real - patch | trained | linear (MLP recipe) | 0 | motion_auc_v0_objects | +0.030 [-0.028, +0.086] | 0.37 | 62 |
| v3float | selectivity real - instance | trained | linear (MLP recipe) | 0 | motion_auc_v0_objects | +0.065 [+0.002, +0.126] | 0.098 | 62 |
| v3float | trained - randall | trained | linear (MLP recipe) | 0 | motion_auc_v0_objects | +0.032 [-0.034, +0.093] | 0.7 | 62 |
| v3float | trained - randtf | trained | linear (MLP recipe) | 0 | motion_auc_v0_objects | +0.047 [-0.017, +0.105] | 0.098 | 62 |
| v3float | trained L0 - enc_any4d | trained | linear (MLP recipe) | 0 | motion_auc_v0_objects | -0.003 [-0.031, +0.025] | 0.9 | 62 |
| v3float | trained L0 - enc_dino | trained | linear (MLP recipe) | 0 | motion_auc_v0_objects | +0.014 [-0.043, +0.073] | 0.37 | 62 |
| v3float | selectivity real - patch | trained | linear (MLP recipe) | 0 | pair_auc_obj | +0.223 [+0.189, +0.259] | 8.5e-16 | 62 |
| v3float | selectivity real - instance | trained | linear (MLP recipe) | 0 | pair_auc_obj | +0.001 [-0.029, +0.030] | 0.7 | 62 |
| v3float | trained - randall | trained | linear (MLP recipe) | 0 | pair_auc_obj | +0.083 [+0.040, +0.126] | 0.0071 | 62 |
| v3float | trained - randtf | trained | linear (MLP recipe) | 0 | pair_auc_obj | -0.011 [-0.043, +0.022] | 0.9 | 62 |
| v3float | trained L0 - enc_any4d | trained | linear (MLP recipe) | 0 | pair_auc_obj | -0.009 [-0.026, +0.007] | 0.37 | 62 |
| v3float | trained L0 - enc_dino | trained | linear (MLP recipe) | 0 | pair_auc_obj | -0.017 [-0.050, +0.017] | 0.37 | 62 |
| v3float | selectivity real - patch | trained | MLP | 24 | motion_auc_v0_objects | +0.460 [+0.425, +0.490] | 2.7e-17 | 62 |
| v3float | selectivity real - instance | trained | MLP | 24 | motion_auc_v0_objects | +0.478 [+0.424, +0.526] | 8.5e-16 | 62 |
| v3float | trained - randall | trained | MLP | 24 | motion_auc_v0_objects | +0.483 [+0.406, +0.553] | 8.5e-16 | 62 |
| v3float | trained - randtf | trained | MLP | 24 | motion_auc_v0_objects | +0.387 [+0.310, +0.460] | 8.5e-16 | 62 |
| v3float | MLP - ridge/bilinear | trained | MLP | 24 | motion_auc_v0_objects | -0.018 [-0.037, -0.001] | 0.024 | 62 |
| v3float | L24 - L0 | trained | MLP | 24 | motion_auc_v0_objects | +0.422 [+0.350, +0.492] | 2.7e-17 | 62 |
| v3float | selectivity real - patch | trained | MLP | 24 | pair_auc_obj | +0.428 [+0.405, +0.449] | 4.3e-19 | 62 |
| v3float | selectivity real - instance | trained | MLP | 24 | pair_auc_obj | +0.153 [+0.118, +0.188] | 1.7e-09 | 62 |
| v3float | trained - randall | trained | MLP | 24 | pair_auc_obj | +0.350 [+0.309, +0.389] | 2.7e-17 | 62 |
| v3float | trained - randtf | trained | MLP | 24 | pair_auc_obj | +0.275 [+0.235, +0.311] | 8.5e-16 | 62 |
| v3float | MLP - ridge/bilinear | trained | MLP | 24 | pair_auc_obj | +0.001 [-0.024, +0.025] | 1 | 62 |
| v3float | L24 - L0 | trained | MLP | 24 | pair_auc_obj | +0.179 [+0.143, +0.213] | 3e-11 | 62 |
| v3float | selectivity real - patch | trained | linear (MLP recipe) | 24 | motion_auc_v0_objects | +0.261 [+0.203, +0.314] | 1.7e-09 | 62 |
| v3float | selectivity real - instance | trained | linear (MLP recipe) | 24 | motion_auc_v0_objects | +0.310 [+0.248, +0.374] | 1.7e-09 | 62 |
| v3float | trained - randall | trained | linear (MLP recipe) | 24 | motion_auc_v0_objects | +0.267 [+0.206, +0.326] | 1.1e-08 | 62 |
| v3float | trained - randtf | trained | linear (MLP recipe) | 24 | motion_auc_v0_objects | +0.266 [+0.206, +0.325] | 1.7e-09 | 62 |
| v3float | L24 - L0 | trained | linear (MLP recipe) | 24 | motion_auc_v0_objects | +0.231 [+0.192, +0.270] | 3.1e-12 | 62 |
| v3float | selectivity real - patch | trained | linear (MLP recipe) | 24 | pair_auc_obj | +0.391 [+0.359, +0.422] | 2.7e-17 | 62 |
| v3float | selectivity real - instance | trained | linear (MLP recipe) | 24 | pair_auc_obj | +0.130 [+0.096, +0.162] | 1.2e-06 | 62 |
| v3float | trained - randall | trained | linear (MLP recipe) | 24 | pair_auc_obj | +0.291 [+0.244, +0.333] | 3.1e-12 | 62 |
| v3float | trained - randtf | trained | linear (MLP recipe) | 24 | pair_auc_obj | +0.176 [+0.134, +0.218] | 5.7e-08 | 62 |
| v3float | L24 - L0 | trained | linear (MLP recipe) | 24 | pair_auc_obj | +0.170 [+0.129, +0.211] | 3e-11 | 62 |
| v3float | selectivity real - patch | randtf | MLP | 0 | motion_auc_v0_objects | +0.060 [-0.012, +0.125] | 0.37 | 62 |
| v3float | selectivity real - instance | randtf | MLP | 0 | motion_auc_v0_objects | +0.088 [+0.018, +0.154] | 0.098 | 62 |
| v3float | MLP - ridge/bilinear | randtf | MLP | 0 | motion_auc_v0_objects | +0.014 [-0.030, +0.053] | 0.53 | 62 |
| v3float | selectivity real - patch | randtf | MLP | 0 | pair_auc_obj | +0.220 [+0.191, +0.250] | 2.7e-17 | 62 |
| v3float | selectivity real - instance | randtf | MLP | 0 | pair_auc_obj | -0.007 [-0.036, +0.023] | 0.7 | 62 |
| v3float | MLP - ridge/bilinear | randtf | MLP | 0 | pair_auc_obj | -0.038 [-0.065, -0.009] | 0.16 | 62 |
| v3float | selectivity real - patch | randtf | linear (MLP recipe) | 0 | motion_auc_v0_objects | -0.026 [-0.066, +0.012] | 0.31 | 62 |
| v3float | selectivity real - instance | randtf | linear (MLP recipe) | 0 | motion_auc_v0_objects | -0.010 [-0.052, +0.033] | 0.9 | 62 |
| v3float | selectivity real - patch | randtf | linear (MLP recipe) | 0 | pair_auc_obj | +0.232 [+0.202, +0.262] | 1.7e-14 | 62 |
| v3float | selectivity real - instance | randtf | linear (MLP recipe) | 0 | pair_auc_obj | -0.011 [-0.037, +0.016] | 0.25 | 62 |
| v3float | selectivity real - patch | randtf | MLP | 24 | motion_auc_v0_objects | +0.083 [+0.015, +0.151] | 0.53 | 62 |
| v3float | selectivity real - instance | randtf | MLP | 24 | motion_auc_v0_objects | +0.073 [+0.001, +0.145] | 0.098 | 62 |
| v3float | MLP - ridge/bilinear | randtf | MLP | 24 | motion_auc_v0_objects | +0.041 [-0.006, +0.088] | 0.16 | 62 |
| v3float | L24 - L0 | randtf | MLP | 24 | motion_auc_v0_objects | +0.009 [-0.036, +0.055] | 0.9 | 62 |
| v3float | selectivity real - patch | randtf | MLP | 24 | pair_auc_obj | +0.161 [+0.133, +0.190] | 1.7e-14 | 62 |
| v3float | selectivity real - instance | randtf | MLP | 24 | pair_auc_obj | -0.014 [-0.046, +0.017] | 0.37 | 62 |
| v3float | MLP - ridge/bilinear | randtf | MLP | 24 | pair_auc_obj | -0.096 [-0.126, -0.065] | 5.8e-05 | 62 |
| v3float | L24 - L0 | randtf | MLP | 24 | pair_auc_obj | -0.067 [-0.096, -0.040] | 0.0005 | 62 |
| v3float | selectivity real - patch | randtf | linear (MLP recipe) | 24 | motion_auc_v0_objects | +0.000 [-0.037, +0.035] | 1 | 62 |
| v3float | selectivity real - instance | randtf | linear (MLP recipe) | 24 | motion_auc_v0_objects | -0.009 [-0.051, +0.031] | 1 | 62 |
| v3float | L24 - L0 | randtf | linear (MLP recipe) | 24 | motion_auc_v0_objects | +0.012 [-0.025, +0.048] | 0.7 | 62 |
| v3float | selectivity real - patch | randtf | linear (MLP recipe) | 24 | pair_auc_obj | +0.218 [+0.185, +0.249] | 8.5e-16 | 62 |
| v3float | selectivity real - instance | randtf | linear (MLP recipe) | 24 | pair_auc_obj | -0.007 [-0.043, +0.030] | 0.53 | 62 |
| v3float | L24 - L0 | randtf | linear (MLP recipe) | 24 | pair_auc_obj | -0.017 [-0.042, +0.006] | 0.9 | 62 |
| v3float | selectivity real - patch | enc_any4d | MLP | 0 | motion_auc_v0_objects | +0.064 [-0.014, +0.148] | 0.7 | 62 |
| v3float | selectivity real - instance | enc_any4d | MLP | 0 | motion_auc_v0_objects | +0.046 [-0.042, +0.123] | 0.7 | 62 |
| v3float | MLP - ridge/bilinear | enc_any4d | MLP | 0 | motion_auc_v0_objects | -0.006 [-0.066, +0.055] | 0.7 | 62 |
| v3float | selectivity real - patch | enc_any4d | MLP | 0 | pair_auc_obj | +0.253 [+0.227, +0.278] | 4.3e-19 | 62 |
| v3float | selectivity real - instance | enc_any4d | MLP | 0 | pair_auc_obj | +0.031 [+0.000, +0.061] | 0.25 | 62 |
| v3float | MLP - ridge/bilinear | enc_any4d | MLP | 0 | pair_auc_obj | -0.043 [-0.069, -0.018] | 0.015 | 62 |
| v3float | selectivity real - patch | enc_any4d | linear (MLP recipe) | 0 | motion_auc_v0_objects | +0.042 [-0.016, +0.095] | 0.25 | 62 |
| v3float | selectivity real - instance | enc_any4d | linear (MLP recipe) | 0 | motion_auc_v0_objects | +0.044 [-0.017, +0.105] | 0.16 | 62 |
| v3float | selectivity real - patch | enc_any4d | linear (MLP recipe) | 0 | pair_auc_obj | +0.225 [+0.190, +0.258] | 1.7e-14 | 62 |
| v3float | selectivity real - instance | enc_any4d | linear (MLP recipe) | 0 | pair_auc_obj | +0.001 [-0.025, +0.025] | 0.9 | 62 |

## v2: motion_auc_v0_objects

| source | probe | labels | L0 | L4 | L8 | L12 | L16 | L20 | L24 |
|---|---|---|---|---|---|---|---|---|---|
| trained | linear (round-1 ridge / bilinear) | real | 0.569 +/- 0.000 [0.508, 0.628] | 0.613 +/- 0.000 [0.550, 0.674] | 0.652 +/- 0.000 [0.588, 0.712] | 0.720 +/- 0.000 [0.655, 0.785] | 0.760 +/- 0.000 [0.699, 0.816] | 0.970 +/- 0.000 [0.949, 0.986] | 0.998 +/- 0.000 [0.997, 0.999] |
| trained | linear (round-1 ridge / bilinear) | patch | 0.507 +/- 0.000 [0.483, 0.534] | 0.514 +/- 0.000 [0.488, 0.539] | 0.489 +/- 0.000 [0.466, 0.512] | 0.485 +/- 0.000 [0.462, 0.507] | 0.498 +/- 0.000 [0.474, 0.523] | 0.488 +/- 0.000 [0.466, 0.508] | 0.506 +/- 0.000 [0.485, 0.527] |
| trained | linear (round-1 ridge / bilinear) | instance | 0.435 +/- 0.000 [0.378, 0.490] | 0.449 +/- 0.000 [0.392, 0.504] | 0.454 +/- 0.000 [0.399, 0.509] | 0.432 +/- 0.000 [0.376, 0.489] | 0.406 +/- 0.000 [0.350, 0.461] | 0.387 +/- 0.000 [0.331, 0.446] | 0.403 +/- 0.000 [0.347, 0.458] |
| trained | linear (MLP recipe) | real | 0.531 +/- 0.004 [0.496, 0.570] | 0.544 +/- 0.027 [0.509, 0.578] | 0.550 +/- 0.006 [0.512, 0.588] | 0.570 +/- 0.010 [0.530, 0.608] | 0.557 +/- 0.019 [0.516, 0.593] | 0.743 +/- 0.025 [0.704, 0.784] | 0.907 +/- 0.018 [0.885, 0.926] |
| trained | linear (MLP recipe) | patch | 0.509 +/- 0.016 [0.496, 0.523] | 0.505 +/- 0.017 [0.494, 0.516] | 0.506 +/- 0.013 [0.497, 0.516] | 0.502 +/- 0.018 [0.491, 0.513] | 0.498 +/- 0.015 [0.488, 0.509] | 0.509 +/- 0.011 [0.498, 0.520] | 0.497 +/- 0.020 [0.487, 0.506] |
| trained | linear (MLP recipe) | instance | 0.504 +/- 0.046 [0.480, 0.530] | 0.492 +/- 0.028 [0.470, 0.513] | 0.482 +/- 0.021 [0.459, 0.504] | 0.478 +/- 0.012 [0.455, 0.502] | 0.483 +/- 0.031 [0.460, 0.506] | 0.468 +/- 0.020 [0.441, 0.495] | 0.481 +/- 0.025 [0.452, 0.511] |
| trained | MLP | real | 0.619 +/- 0.014 [0.566, 0.673] | 0.640 +/- 0.021 [0.583, 0.693] | 0.684 +/- 0.014 [0.627, 0.741] | 0.780 +/- 0.020 [0.729, 0.830] | 0.793 +/- 0.031 [0.739, 0.843] | 0.994 +/- 0.001 [0.989, 0.997] | 0.998 +/- 0.000 [0.996, 1.000] |
| trained | MLP | patch | 0.489 +/- 0.014 [0.479, 0.498] | 0.498 +/- 0.018 [0.488, 0.508] | 0.506 +/- 0.023 [0.495, 0.517] | 0.501 +/- 0.016 [0.488, 0.513] | 0.509 +/- 0.016 [0.499, 0.518] | 0.510 +/- 0.015 [0.501, 0.519] | 0.508 +/- 0.014 [0.497, 0.518] |
| trained | MLP | instance | 0.501 +/- 0.070 [0.477, 0.524] | 0.503 +/- 0.055 [0.482, 0.527] | 0.485 +/- 0.076 [0.456, 0.513] | 0.496 +/- 0.058 [0.468, 0.523] | 0.492 +/- 0.052 [0.465, 0.520] | 0.495 +/- 0.055 [0.470, 0.519] | 0.499 +/- 0.053 [0.472, 0.526] |
| randall | linear (round-1 ridge / bilinear) | real | 0.520 +/- 0.000 [0.473, 0.569] | 0.484 +/- 0.000 [0.432, 0.535] | 0.484 +/- 0.000 [0.435, 0.533] | 0.485 +/- 0.000 [0.432, 0.538] | 0.482 +/- 0.000 [0.426, 0.536] | 0.492 +/- 0.000 [0.438, 0.543] | 0.492 +/- 0.000 [0.437, 0.544] |
| randall | linear (round-1 ridge / bilinear) | patch | 0.514 +/- 0.000 [0.493, 0.536] | 0.505 +/- 0.000 [0.484, 0.526] | 0.509 +/- 0.000 [0.487, 0.531] | 0.504 +/- 0.000 [0.483, 0.523] | 0.503 +/- 0.000 [0.482, 0.524] | 0.505 +/- 0.000 [0.482, 0.526] | 0.501 +/- 0.000 [0.480, 0.523] |
| randall | linear (round-1 ridge / bilinear) | instance | 0.498 +/- 0.000 [0.447, 0.547] | 0.500 +/- 0.000 [0.448, 0.549] | 0.497 +/- 0.000 [0.445, 0.551] | 0.509 +/- 0.000 [0.460, 0.560] | 0.499 +/- 0.000 [0.448, 0.550] | 0.472 +/- 0.000 [0.422, 0.523] | 0.478 +/- 0.000 [0.427, 0.530] |
| randall | linear (MLP recipe) | real | 0.516 +/- 0.029 [0.487, 0.545] | 0.517 +/- 0.016 [0.488, 0.544] | 0.545 +/- 0.019 [0.516, 0.574] | 0.513 +/- 0.023 [0.484, 0.543] | 0.513 +/- 0.015 [0.478, 0.548] | 0.533 +/- 0.018 [0.504, 0.565] | 0.551 +/- 0.025 [0.521, 0.582] |
| randall | linear (MLP recipe) | patch | 0.491 +/- 0.014 [0.476, 0.504] | n/a | n/a | n/a | n/a | n/a | 0.500 +/- 0.019 [0.488, 0.511] |
| randall | linear (MLP recipe) | instance | 0.515 +/- 0.037 [0.495, 0.536] | n/a | n/a | n/a | n/a | n/a | 0.507 +/- 0.036 [0.485, 0.528] |
| randall | MLP | real | 0.514 +/- 0.005 [0.464, 0.562] | 0.506 +/- 0.010 [0.454, 0.554] | 0.505 +/- 0.008 [0.452, 0.555] | 0.498 +/- 0.012 [0.445, 0.548] | 0.497 +/- 0.009 [0.441, 0.549] | 0.499 +/- 0.008 [0.445, 0.554] | 0.497 +/- 0.003 [0.440, 0.550] |
| randall | MLP | patch | 0.496 +/- 0.011 [0.485, 0.507] | n/a | n/a | n/a | n/a | n/a | 0.503 +/- 0.007 [0.490, 0.516] |
| randall | MLP | instance | 0.526 +/- 0.013 [0.500, 0.553] | n/a | n/a | n/a | n/a | n/a | 0.518 +/- 0.018 [0.491, 0.545] |
| randtf | linear (round-1 ridge / bilinear) | real | 0.479 +/- 0.000 [0.421, 0.533] | 0.468 +/- 0.000 [0.414, 0.521] | 0.476 +/- 0.000 [0.423, 0.529] | 0.463 +/- 0.000 [0.409, 0.515] | 0.460 +/- 0.000 [0.406, 0.513] | 0.464 +/- 0.000 [0.413, 0.520] | 0.455 +/- 0.000 [0.404, 0.505] |
| randtf | linear (round-1 ridge / bilinear) | patch | 0.518 +/- 0.000 [0.498, 0.540] | 0.525 +/- 0.000 [0.501, 0.550] | 0.527 +/- 0.000 [0.502, 0.553] | 0.524 +/- 0.000 [0.498, 0.548] | 0.525 +/- 0.000 [0.502, 0.548] | 0.515 +/- 0.000 [0.488, 0.541] | 0.522 +/- 0.000 [0.496, 0.547] |
| randtf | linear (round-1 ridge / bilinear) | instance | 0.534 +/- 0.000 [0.475, 0.589] | 0.505 +/- 0.000 [0.456, 0.557] | 0.500 +/- 0.000 [0.446, 0.552] | 0.488 +/- 0.000 [0.434, 0.543] | 0.504 +/- 0.000 [0.453, 0.554] | 0.510 +/- 0.000 [0.459, 0.562] | 0.520 +/- 0.000 [0.466, 0.572] |
| randtf | linear (MLP recipe) | real | 0.485 +/- 0.014 [0.449, 0.520] | 0.502 +/- 0.019 [0.470, 0.536] | 0.489 +/- 0.026 [0.456, 0.520] | 0.480 +/- 0.024 [0.452, 0.508] | 0.476 +/- 0.024 [0.446, 0.505] | 0.470 +/- 0.028 [0.440, 0.501] | 0.500 +/- 0.013 [0.472, 0.530] |
| randtf | linear (MLP recipe) | patch | 0.503 +/- 0.014 [0.493, 0.513] | n/a | n/a | n/a | n/a | n/a | 0.509 +/- 0.017 [0.499, 0.518] |
| randtf | linear (MLP recipe) | instance | 0.501 +/- 0.029 [0.475, 0.527] | n/a | n/a | n/a | n/a | n/a | 0.515 +/- 0.040 [0.493, 0.536] |
| randtf | MLP | real | 0.519 +/- 0.014 [0.469, 0.569] | 0.482 +/- 0.014 [0.433, 0.533] | 0.492 +/- 0.017 [0.442, 0.541] | 0.480 +/- 0.020 [0.430, 0.527] | 0.484 +/- 0.024 [0.434, 0.534] | 0.477 +/- 0.020 [0.432, 0.527] | 0.486 +/- 0.025 [0.439, 0.534] |
| randtf | MLP | patch | 0.498 +/- 0.023 [0.486, 0.510] | n/a | n/a | n/a | n/a | n/a | 0.508 +/- 0.013 [0.499, 0.518] |
| randtf | MLP | instance | 0.520 +/- 0.030 [0.493, 0.548] | n/a | n/a | n/a | n/a | n/a | 0.528 +/- 0.046 [0.502, 0.554] |
| enc_any4d | linear (round-1 ridge / bilinear) | real | 0.560 +/- 0.000 [0.498, 0.624] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (round-1 ridge / bilinear) | patch | 0.502 +/- 0.000 [0.478, 0.525] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (round-1 ridge / bilinear) | instance | 0.459 +/- 0.000 [0.400, 0.518] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (MLP recipe) | real | 0.516 +/- 0.025 [0.485, 0.547] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (MLP recipe) | patch | 0.496 +/- 0.018 [0.486, 0.505] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (MLP recipe) | instance | 0.508 +/- 0.046 [0.483, 0.532] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | MLP | real | 0.618 +/- 0.019 [0.564, 0.670] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | MLP | patch | 0.505 +/- 0.010 [0.495, 0.516] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | MLP | instance | 0.518 +/- 0.063 [0.495, 0.543] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | real | 0.484 +/- 0.000 [0.431, 0.537] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | patch | 0.527 +/- 0.000 [0.504, 0.550] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | instance | 0.528 +/- 0.000 [0.475, 0.582] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (MLP recipe) | real | 0.472 +/- 0.029 [0.438, 0.505] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (MLP recipe) | patch | 0.501 +/- 0.008 [0.489, 0.514] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (MLP recipe) | instance | 0.520 +/- 0.007 [0.493, 0.545] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | MLP | real | 0.483 +/- 0.012 [0.437, 0.531] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | MLP | patch | 0.503 +/- 0.017 [0.492, 0.515] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | MLP | instance | 0.533 +/- 0.030 [0.506, 0.560] | n/a | n/a | n/a | n/a | n/a | n/a |

## v2: pair_auc_obj

| source | probe | labels | L0 | L4 | L8 | L12 | L16 | L20 | L24 |
|---|---|---|---|---|---|---|---|---|---|
| trained | linear (round-1 ridge / bilinear) | real | 0.770 +/- 0.001 [0.744, 0.796] | 0.795 +/- 0.001 [0.767, 0.823] | 0.810 +/- 0.000 [0.783, 0.838] | 0.842 +/- 0.000 [0.816, 0.867] | 0.862 +/- 0.000 [0.837, 0.887] | 0.890 +/- 0.000 [0.860, 0.916] | 0.915 +/- 0.001 [0.890, 0.937] |
| trained | linear (round-1 ridge / bilinear) | patch | 0.528 +/- 0.001 [0.518, 0.541] | 0.526 +/- 0.000 [0.515, 0.539] | 0.526 +/- 0.000 [0.516, 0.536] | 0.530 +/- 0.001 [0.519, 0.541] | 0.526 +/- 0.001 [0.518, 0.536] | 0.525 +/- 0.001 [0.515, 0.535] | 0.524 +/- 0.001 [0.515, 0.533] |
| trained | linear (round-1 ridge / bilinear) | instance | 0.775 +/- 0.001 [0.749, 0.801] | 0.791 +/- 0.000 [0.766, 0.816] | 0.801 +/- 0.000 [0.777, 0.826] | 0.802 +/- 0.001 [0.777, 0.826] | 0.815 +/- 0.001 [0.791, 0.841] | 0.812 +/- 0.000 [0.786, 0.836] | 0.810 +/- 0.002 [0.784, 0.835] |
| trained | linear (MLP recipe) | real | 0.728 +/- 0.025 [0.703, 0.753] | 0.712 +/- 0.022 [0.684, 0.740] | 0.733 +/- 0.015 [0.706, 0.760] | 0.754 +/- 0.021 [0.726, 0.783] | 0.780 +/- 0.022 [0.749, 0.809] | 0.836 +/- 0.018 [0.804, 0.864] | 0.827 +/- 0.011 [0.787, 0.863] |
| trained | linear (MLP recipe) | patch | 0.515 +/- 0.008 [0.510, 0.521] | 0.517 +/- 0.007 [0.512, 0.522] | 0.517 +/- 0.006 [0.513, 0.522] | 0.521 +/- 0.005 [0.515, 0.526] | 0.521 +/- 0.006 [0.516, 0.525] | 0.522 +/- 0.002 [0.518, 0.528] | 0.525 +/- 0.002 [0.520, 0.531] |
| trained | linear (MLP recipe) | instance | 0.714 +/- 0.027 [0.701, 0.728] | 0.703 +/- 0.021 [0.688, 0.719] | 0.706 +/- 0.019 [0.690, 0.722] | 0.708 +/- 0.023 [0.690, 0.727] | 0.700 +/- 0.032 [0.683, 0.715] | 0.721 +/- 0.020 [0.704, 0.739] | 0.742 +/- 0.018 [0.724, 0.760] |
| trained | MLP | real | 0.738 +/- 0.017 [0.714, 0.761] | 0.766 +/- 0.012 [0.741, 0.791] | 0.810 +/- 0.008 [0.784, 0.834] | 0.848 +/- 0.005 [0.823, 0.872] | 0.862 +/- 0.008 [0.839, 0.884] | 0.944 +/- 0.005 [0.924, 0.961] | 0.970 +/- 0.003 [0.956, 0.980] |
| trained | MLP | patch | 0.508 +/- 0.006 [0.502, 0.515] | 0.515 +/- 0.008 [0.509, 0.522] | 0.515 +/- 0.010 [0.508, 0.520] | 0.516 +/- 0.010 [0.510, 0.522] | 0.510 +/- 0.008 [0.505, 0.515] | 0.516 +/- 0.010 [0.510, 0.522] | 0.514 +/- 0.009 [0.509, 0.518] |
| trained | MLP | instance | 0.693 +/- 0.013 [0.679, 0.706] | 0.703 +/- 0.017 [0.686, 0.719] | 0.741 +/- 0.013 [0.724, 0.757] | 0.757 +/- 0.020 [0.739, 0.776] | 0.769 +/- 0.016 [0.752, 0.786] | 0.769 +/- 0.013 [0.751, 0.788] | 0.761 +/- 0.013 [0.742, 0.779] |
| randall | linear (round-1 ridge / bilinear) | real | 0.651 +/- 0.001 [0.621, 0.682] | 0.656 +/- 0.001 [0.626, 0.686] | 0.650 +/- 0.002 [0.615, 0.682] | 0.644 +/- 0.002 [0.612, 0.676] | 0.642 +/- 0.001 [0.608, 0.675] | 0.652 +/- 0.001 [0.620, 0.683] | 0.652 +/- 0.001 [0.617, 0.685] |
| randall | linear (round-1 ridge / bilinear) | patch | 0.512 +/- 0.001 [0.500, 0.523] | 0.515 +/- 0.001 [0.506, 0.524] | 0.509 +/- 0.001 [0.498, 0.519] | 0.506 +/- 0.002 [0.494, 0.519] | 0.503 +/- 0.001 [0.490, 0.514] | 0.499 +/- 0.003 [0.486, 0.510] | 0.504 +/- 0.000 [0.492, 0.516] |
| randall | linear (round-1 ridge / bilinear) | instance | 0.653 +/- 0.002 [0.624, 0.683] | 0.664 +/- 0.001 [0.635, 0.698] | 0.661 +/- 0.002 [0.629, 0.692] | 0.662 +/- 0.001 [0.629, 0.699] | 0.666 +/- 0.001 [0.630, 0.701] | 0.657 +/- 0.001 [0.623, 0.689] | 0.666 +/- 0.002 [0.633, 0.699] |
| randall | linear (MLP recipe) | real | 0.654 +/- 0.005 [0.626, 0.684] | 0.642 +/- 0.010 [0.611, 0.673] | 0.641 +/- 0.007 [0.611, 0.674] | 0.639 +/- 0.007 [0.608, 0.669] | 0.637 +/- 0.003 [0.604, 0.671] | 0.631 +/- 0.006 [0.600, 0.662] | 0.632 +/- 0.008 [0.601, 0.666] |
| randall | linear (MLP recipe) | patch | 0.509 +/- 0.007 [0.503, 0.515] | n/a | n/a | n/a | n/a | n/a | 0.508 +/- 0.004 [0.503, 0.513] |
| randall | linear (MLP recipe) | instance | 0.655 +/- 0.018 [0.632, 0.677] | n/a | n/a | n/a | n/a | n/a | 0.626 +/- 0.017 [0.603, 0.650] |
| randall | MLP | real | 0.639 +/- 0.005 [0.609, 0.668] | 0.631 +/- 0.011 [0.600, 0.663] | 0.638 +/- 0.008 [0.606, 0.668] | 0.626 +/- 0.005 [0.594, 0.659] | 0.622 +/- 0.012 [0.588, 0.655] | 0.613 +/- 0.005 [0.581, 0.646] | 0.613 +/- 0.006 [0.581, 0.646] |
| randall | MLP | patch | 0.503 +/- 0.006 [0.497, 0.509] | n/a | n/a | n/a | n/a | n/a | 0.503 +/- 0.005 [0.496, 0.509] |
| randall | MLP | instance | 0.645 +/- 0.023 [0.624, 0.667] | n/a | n/a | n/a | n/a | n/a | 0.603 +/- 0.021 [0.579, 0.625] |
| randtf | linear (round-1 ridge / bilinear) | real | 0.765 +/- 0.003 [0.741, 0.791] | 0.777 +/- 0.002 [0.749, 0.804] | 0.771 +/- 0.001 [0.741, 0.800] | 0.778 +/- 0.002 [0.751, 0.808] | 0.780 +/- 0.001 [0.750, 0.808] | 0.772 +/- 0.002 [0.742, 0.801] | 0.774 +/- 0.001 [0.743, 0.804] |
| randtf | linear (round-1 ridge / bilinear) | patch | 0.526 +/- 0.002 [0.518, 0.535] | 0.523 +/- 0.001 [0.516, 0.531] | 0.515 +/- 0.002 [0.507, 0.523] | 0.531 +/- 0.002 [0.521, 0.541] | 0.517 +/- 0.003 [0.508, 0.527] | 0.521 +/- 0.002 [0.513, 0.528] | 0.523 +/- 0.003 [0.512, 0.534] |
| randtf | linear (round-1 ridge / bilinear) | instance | 0.777 +/- 0.002 [0.751, 0.802] | 0.763 +/- 0.002 [0.736, 0.795] | 0.768 +/- 0.001 [0.741, 0.794] | 0.760 +/- 0.002 [0.730, 0.790] | 0.758 +/- 0.002 [0.729, 0.788] | 0.756 +/- 0.002 [0.726, 0.787] | 0.762 +/- 0.002 [0.734, 0.789] |
| randtf | linear (MLP recipe) | real | 0.757 +/- 0.008 [0.730, 0.782] | 0.760 +/- 0.003 [0.733, 0.788] | 0.757 +/- 0.008 [0.728, 0.787] | 0.761 +/- 0.002 [0.733, 0.789] | 0.754 +/- 0.010 [0.726, 0.782] | 0.748 +/- 0.007 [0.718, 0.777] | 0.747 +/- 0.005 [0.718, 0.776] |
| randtf | linear (MLP recipe) | patch | 0.520 +/- 0.003 [0.516, 0.525] | n/a | n/a | n/a | n/a | n/a | 0.521 +/- 0.003 [0.515, 0.526] |
| randtf | linear (MLP recipe) | instance | 0.762 +/- 0.017 [0.745, 0.778] | n/a | n/a | n/a | n/a | n/a | 0.749 +/- 0.011 [0.728, 0.770] |
| randtf | MLP | real | 0.765 +/- 0.005 [0.739, 0.790] | 0.728 +/- 0.005 [0.701, 0.753] | 0.708 +/- 0.007 [0.681, 0.734] | 0.693 +/- 0.010 [0.666, 0.719] | 0.695 +/- 0.007 [0.668, 0.721] | 0.685 +/- 0.012 [0.658, 0.710] | 0.679 +/- 0.009 [0.651, 0.705] |
| randtf | MLP | patch | 0.512 +/- 0.002 [0.506, 0.517] | n/a | n/a | n/a | n/a | n/a | 0.512 +/- 0.007 [0.506, 0.519] |
| randtf | MLP | instance | 0.744 +/- 0.010 [0.728, 0.760] | n/a | n/a | n/a | n/a | n/a | 0.688 +/- 0.016 [0.670, 0.706] |
| enc_any4d | linear (round-1 ridge / bilinear) | real | 0.800 +/- 0.001 [0.774, 0.825] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (round-1 ridge / bilinear) | patch | 0.533 +/- 0.001 [0.523, 0.544] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (round-1 ridge / bilinear) | instance | 0.802 +/- 0.001 [0.778, 0.826] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (MLP recipe) | real | 0.708 +/- 0.021 [0.680, 0.736] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (MLP recipe) | patch | 0.517 +/- 0.009 [0.511, 0.523] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (MLP recipe) | instance | 0.699 +/- 0.024 [0.683, 0.715] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | MLP | real | 0.751 +/- 0.015 [0.728, 0.772] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | MLP | patch | 0.510 +/- 0.003 [0.504, 0.517] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | MLP | instance | 0.699 +/- 0.012 [0.686, 0.713] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | real | 0.764 +/- 0.002 [0.738, 0.791] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | patch | 0.523 +/- 0.002 [0.515, 0.531] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | instance | 0.768 +/- 0.001 [0.743, 0.793] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (MLP recipe) | real | 0.757 +/- 0.003 [0.731, 0.782] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (MLP recipe) | patch | 0.524 +/- 0.001 [0.519, 0.529] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (MLP recipe) | instance | 0.761 +/- 0.014 [0.744, 0.779] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | MLP | real | 0.720 +/- 0.007 [0.697, 0.742] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | MLP | patch | 0.516 +/- 0.002 [0.510, 0.521] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | MLP | instance | 0.721 +/- 0.010 [0.706, 0.735] | n/a | n/a | n/a | n/a | n/a | n/a |

## v3float: motion_auc_v0_objects

| source | probe | labels | L0 | L4 | L8 | L12 | L16 | L20 | L24 |
|---|---|---|---|---|---|---|---|---|---|
| trained | linear (round-1 ridge / bilinear) | real | 0.569 +/- 0.000 [0.479, 0.658] | 0.569 +/- 0.000 [0.487, 0.650] | 0.591 +/- 0.000 [0.504, 0.679] | 0.699 +/- 0.000 [0.622, 0.779] | 0.784 +/- 0.000 [0.708, 0.855] | 0.938 +/- 0.000 [0.902, 0.968] | 0.973 +/- 0.000 [0.950, 0.992] |
| trained | linear (round-1 ridge / bilinear) | patch | 0.496 +/- 0.000 [0.462, 0.530] | 0.485 +/- 0.000 [0.449, 0.519] | 0.508 +/- 0.000 [0.477, 0.543] | 0.494 +/- 0.000 [0.461, 0.526] | 0.497 +/- 0.000 [0.466, 0.533] | 0.498 +/- 0.000 [0.465, 0.535] | 0.491 +/- 0.000 [0.462, 0.522] |
| trained | linear (round-1 ridge / bilinear) | instance | 0.616 +/- 0.000 [0.530, 0.696] | 0.559 +/- 0.000 [0.470, 0.641] | 0.549 +/- 0.000 [0.458, 0.638] | 0.570 +/- 0.000 [0.481, 0.662] | 0.568 +/- 0.000 [0.480, 0.655] | 0.540 +/- 0.000 [0.451, 0.628] | 0.566 +/- 0.000 [0.480, 0.655] |
| trained | linear (MLP recipe) | real | 0.531 +/- 0.032 [0.476, 0.585] | 0.539 +/- 0.019 [0.474, 0.609] | 0.548 +/- 0.019 [0.479, 0.617] | 0.557 +/- 0.028 [0.492, 0.624] | 0.585 +/- 0.015 [0.521, 0.648] | 0.664 +/- 0.031 [0.595, 0.729] | 0.763 +/- 0.015 [0.712, 0.810] |
| trained | linear (MLP recipe) | patch | 0.502 +/- 0.020 [0.490, 0.514] | 0.502 +/- 0.008 [0.485, 0.519] | 0.500 +/- 0.008 [0.484, 0.514] | 0.502 +/- 0.011 [0.486, 0.518] | 0.506 +/- 0.006 [0.492, 0.520] | 0.512 +/- 0.013 [0.501, 0.524] | 0.502 +/- 0.015 [0.487, 0.518] |
| trained | linear (MLP recipe) | instance | 0.467 +/- 0.029 [0.438, 0.499] | 0.478 +/- 0.037 [0.455, 0.500] | 0.473 +/- 0.018 [0.442, 0.505] | 0.478 +/- 0.025 [0.439, 0.510] | 0.487 +/- 0.033 [0.454, 0.519] | 0.485 +/- 0.026 [0.447, 0.522] | 0.452 +/- 0.018 [0.418, 0.486] |
| trained | MLP | real | 0.534 +/- 0.021 [0.469, 0.599] | 0.601 +/- 0.022 [0.538, 0.662] | 0.624 +/- 0.030 [0.556, 0.690] | 0.753 +/- 0.034 [0.694, 0.811] | 0.800 +/- 0.023 [0.742, 0.854] | 0.946 +/- 0.004 [0.916, 0.974] | 0.955 +/- 0.013 [0.925, 0.979] |
| trained | MLP | patch | 0.496 +/- 0.019 [0.483, 0.508] | 0.491 +/- 0.016 [0.476, 0.505] | 0.502 +/- 0.012 [0.488, 0.516] | 0.505 +/- 0.008 [0.491, 0.521] | 0.504 +/- 0.012 [0.490, 0.518] | 0.487 +/- 0.018 [0.472, 0.502] | 0.496 +/- 0.014 [0.482, 0.510] |
| trained | MLP | instance | 0.499 +/- 0.029 [0.463, 0.534] | 0.500 +/- 0.026 [0.467, 0.530] | 0.494 +/- 0.037 [0.464, 0.526] | 0.490 +/- 0.030 [0.455, 0.523] | 0.506 +/- 0.043 [0.472, 0.539] | 0.489 +/- 0.034 [0.457, 0.520] | 0.478 +/- 0.020 [0.438, 0.518] |
| randall | linear (round-1 ridge / bilinear) | real | 0.475 +/- 0.000 [0.414, 0.535] | 0.501 +/- 0.000 [0.434, 0.564] | 0.503 +/- 0.000 [0.439, 0.567] | 0.517 +/- 0.000 [0.457, 0.580] | 0.484 +/- 0.000 [0.423, 0.547] | 0.488 +/- 0.000 [0.421, 0.553] | 0.491 +/- 0.000 [0.426, 0.551] |
| randall | linear (round-1 ridge / bilinear) | patch | 0.510 +/- 0.000 [0.477, 0.543] | 0.498 +/- 0.000 [0.469, 0.526] | 0.487 +/- 0.000 [0.454, 0.520] | 0.479 +/- 0.000 [0.445, 0.516] | 0.474 +/- 0.000 [0.436, 0.511] | 0.484 +/- 0.000 [0.450, 0.518] | 0.473 +/- 0.000 [0.440, 0.506] |
| randall | linear (round-1 ridge / bilinear) | instance | 0.504 +/- 0.000 [0.448, 0.560] | 0.496 +/- 0.000 [0.435, 0.558] | 0.479 +/- 0.000 [0.416, 0.540] | 0.488 +/- 0.000 [0.430, 0.547] | 0.506 +/- 0.000 [0.449, 0.559] | 0.484 +/- 0.000 [0.423, 0.540] | 0.479 +/- 0.000 [0.419, 0.540] |
| randall | linear (MLP recipe) | real | 0.500 +/- 0.041 [0.459, 0.539] | 0.483 +/- 0.025 [0.446, 0.525] | 0.477 +/- 0.017 [0.435, 0.520] | 0.488 +/- 0.027 [0.445, 0.530] | 0.500 +/- 0.019 [0.456, 0.542] | 0.509 +/- 0.020 [0.468, 0.550] | 0.496 +/- 0.034 [0.454, 0.539] |
| randall | linear (MLP recipe) | patch | 0.501 +/- 0.013 [0.485, 0.517] | n/a | n/a | n/a | n/a | n/a | 0.493 +/- 0.018 [0.480, 0.507] |
| randall | linear (MLP recipe) | instance | 0.514 +/- 0.038 [0.487, 0.541] | n/a | n/a | n/a | n/a | n/a | 0.482 +/- 0.032 [0.452, 0.512] |
| randall | MLP | real | 0.489 +/- 0.017 [0.428, 0.548] | 0.469 +/- 0.015 [0.408, 0.532] | 0.470 +/- 0.019 [0.407, 0.535] | 0.474 +/- 0.018 [0.413, 0.539] | 0.472 +/- 0.015 [0.406, 0.537] | 0.475 +/- 0.020 [0.409, 0.540] | 0.472 +/- 0.022 [0.408, 0.533] |
| randall | MLP | patch | 0.497 +/- 0.017 [0.481, 0.513] | n/a | n/a | n/a | n/a | n/a | 0.508 +/- 0.022 [0.495, 0.521] |
| randall | MLP | instance | 0.500 +/- 0.058 [0.462, 0.536] | n/a | n/a | n/a | n/a | n/a | 0.505 +/- 0.070 [0.472, 0.540] |
| randtf | linear (round-1 ridge / bilinear) | real | 0.546 +/- 0.000 [0.475, 0.618] | 0.547 +/- 0.000 [0.475, 0.621] | 0.548 +/- 0.000 [0.474, 0.618] | 0.545 +/- 0.000 [0.468, 0.614] | 0.533 +/- 0.000 [0.461, 0.604] | 0.526 +/- 0.000 [0.451, 0.603] | 0.528 +/- 0.000 [0.453, 0.605] |
| randtf | linear (round-1 ridge / bilinear) | patch | 0.504 +/- 0.000 [0.477, 0.530] | 0.516 +/- 0.000 [0.485, 0.548] | 0.479 +/- 0.000 [0.448, 0.509] | 0.500 +/- 0.000 [0.468, 0.531] | 0.494 +/- 0.000 [0.461, 0.524] | 0.483 +/- 0.000 [0.450, 0.514] | 0.474 +/- 0.000 [0.439, 0.507] |
| randtf | linear (round-1 ridge / bilinear) | instance | 0.519 +/- 0.000 [0.444, 0.590] | 0.521 +/- 0.000 [0.450, 0.590] | 0.511 +/- 0.000 [0.442, 0.577] | 0.510 +/- 0.000 [0.445, 0.574] | 0.515 +/- 0.000 [0.449, 0.578] | 0.533 +/- 0.000 [0.468, 0.594] | 0.541 +/- 0.000 [0.476, 0.603] |
| randtf | linear (MLP recipe) | real | 0.485 +/- 0.031 [0.448, 0.520] | 0.509 +/- 0.028 [0.474, 0.541] | 0.505 +/- 0.040 [0.471, 0.541] | 0.509 +/- 0.051 [0.472, 0.547] | 0.506 +/- 0.034 [0.473, 0.538] | 0.487 +/- 0.038 [0.452, 0.521] | 0.496 +/- 0.038 [0.463, 0.528] |
| randtf | linear (MLP recipe) | patch | 0.511 +/- 0.012 [0.496, 0.524] | n/a | n/a | n/a | n/a | n/a | 0.496 +/- 0.022 [0.482, 0.509] |
| randtf | linear (MLP recipe) | instance | 0.495 +/- 0.038 [0.463, 0.524] | n/a | n/a | n/a | n/a | n/a | 0.505 +/- 0.023 [0.481, 0.531] |
| randtf | MLP | real | 0.560 +/- 0.012 [0.493, 0.626] | 0.574 +/- 0.013 [0.504, 0.637] | 0.562 +/- 0.024 [0.496, 0.624] | 0.564 +/- 0.023 [0.494, 0.629] | 0.572 +/- 0.014 [0.507, 0.642] | 0.571 +/- 0.021 [0.503, 0.635] | 0.569 +/- 0.011 [0.499, 0.640] |
| randtf | MLP | patch | 0.500 +/- 0.021 [0.489, 0.510] | n/a | n/a | n/a | n/a | n/a | 0.486 +/- 0.026 [0.473, 0.498] |
| randtf | MLP | instance | 0.472 +/- 0.016 [0.436, 0.509] | n/a | n/a | n/a | n/a | n/a | 0.496 +/- 0.057 [0.461, 0.530] |
| enc_any4d | linear (round-1 ridge / bilinear) | real | 0.568 +/- 0.000 [0.490, 0.646] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (round-1 ridge / bilinear) | patch | 0.495 +/- 0.000 [0.465, 0.528] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (round-1 ridge / bilinear) | instance | 0.624 +/- 0.000 [0.540, 0.710] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (MLP recipe) | real | 0.534 +/- 0.019 [0.480, 0.589] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (MLP recipe) | patch | 0.492 +/- 0.017 [0.481, 0.504] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (MLP recipe) | instance | 0.490 +/- 0.041 [0.466, 0.516] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | MLP | real | 0.562 +/- 0.017 [0.480, 0.635] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | MLP | patch | 0.498 +/- 0.018 [0.485, 0.510] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | MLP | instance | 0.516 +/- 0.059 [0.481, 0.550] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | real | 0.558 +/- 0.000 [0.482, 0.632] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | patch | 0.499 +/- 0.000 [0.470, 0.530] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | instance | 0.506 +/- 0.000 [0.428, 0.580] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (MLP recipe) | real | 0.518 +/- 0.021 [0.477, 0.557] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (MLP recipe) | patch | 0.490 +/- 0.019 [0.473, 0.507] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (MLP recipe) | instance | 0.505 +/- 0.030 [0.472, 0.536] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | MLP | real | 0.553 +/- 0.011 [0.491, 0.616] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | MLP | patch | 0.495 +/- 0.010 [0.484, 0.507] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | MLP | instance | 0.475 +/- 0.032 [0.445, 0.504] | n/a | n/a | n/a | n/a | n/a | n/a |

## v3float: pair_auc_obj

| source | probe | labels | L0 | L4 | L8 | L12 | L16 | L20 | L24 |
|---|---|---|---|---|---|---|---|---|---|
| trained | linear (round-1 ridge / bilinear) | real | 0.780 +/- 0.001 [0.747, 0.813] | 0.811 +/- 0.000 [0.777, 0.844] | 0.819 +/- 0.001 [0.783, 0.853] | 0.877 +/- 0.000 [0.848, 0.906] | 0.909 +/- 0.001 [0.884, 0.932] | 0.941 +/- 0.001 [0.917, 0.961] | 0.937 +/- 0.001 [0.912, 0.959] |
| trained | linear (round-1 ridge / bilinear) | patch | 0.511 +/- 0.001 [0.499, 0.522] | 0.520 +/- 0.001 [0.508, 0.534] | 0.521 +/- 0.001 [0.505, 0.541] | 0.509 +/- 0.001 [0.492, 0.527] | 0.524 +/- 0.001 [0.509, 0.542] | 0.517 +/- 0.002 [0.507, 0.527] | 0.521 +/- 0.002 [0.509, 0.534] |
| trained | linear (round-1 ridge / bilinear) | instance | 0.795 +/- 0.001 [0.762, 0.831] | 0.801 +/- 0.000 [0.761, 0.838] | 0.811 +/- 0.002 [0.778, 0.844] | 0.816 +/- 0.000 [0.784, 0.849] | 0.824 +/- 0.000 [0.789, 0.857] | 0.808 +/- 0.001 [0.774, 0.842] | 0.792 +/- 0.003 [0.754, 0.826] |
| trained | linear (MLP recipe) | real | 0.732 +/- 0.012 [0.697, 0.768] | 0.749 +/- 0.008 [0.713, 0.785] | 0.765 +/- 0.016 [0.726, 0.804] | 0.822 +/- 0.020 [0.785, 0.861] | 0.851 +/- 0.024 [0.815, 0.888] | 0.901 +/- 0.006 [0.870, 0.930] | 0.902 +/- 0.009 [0.871, 0.932] |
| trained | linear (MLP recipe) | patch | 0.509 +/- 0.012 [0.504, 0.515] | 0.513 +/- 0.005 [0.506, 0.519] | 0.517 +/- 0.005 [0.512, 0.523] | 0.513 +/- 0.005 [0.505, 0.520] | 0.511 +/- 0.007 [0.505, 0.517] | 0.518 +/- 0.007 [0.509, 0.527] | 0.511 +/- 0.005 [0.503, 0.518] |
| trained | linear (MLP recipe) | instance | 0.732 +/- 0.019 [0.709, 0.755] | 0.743 +/- 0.022 [0.718, 0.767] | 0.728 +/- 0.028 [0.705, 0.752] | 0.744 +/- 0.022 [0.721, 0.769] | 0.745 +/- 0.026 [0.719, 0.773] | 0.761 +/- 0.029 [0.736, 0.786] | 0.773 +/- 0.021 [0.744, 0.799] |
| trained | MLP | real | 0.760 +/- 0.010 [0.733, 0.787] | 0.758 +/- 0.011 [0.734, 0.782] | 0.826 +/- 0.007 [0.800, 0.851] | 0.864 +/- 0.011 [0.831, 0.893] | 0.877 +/- 0.007 [0.846, 0.907] | 0.924 +/- 0.009 [0.901, 0.945] | 0.938 +/- 0.004 [0.915, 0.959] |
| trained | MLP | patch | 0.514 +/- 0.008 [0.507, 0.522] | 0.516 +/- 0.007 [0.509, 0.524] | 0.518 +/- 0.003 [0.510, 0.525] | 0.511 +/- 0.006 [0.505, 0.518] | 0.513 +/- 0.008 [0.505, 0.521] | 0.508 +/- 0.006 [0.499, 0.516] | 0.511 +/- 0.002 [0.503, 0.519] |
| trained | MLP | instance | 0.723 +/- 0.028 [0.701, 0.744] | 0.727 +/- 0.022 [0.708, 0.748] | 0.761 +/- 0.032 [0.740, 0.784] | 0.776 +/- 0.026 [0.753, 0.799] | 0.791 +/- 0.032 [0.766, 0.817] | 0.787 +/- 0.021 [0.763, 0.814] | 0.786 +/- 0.022 [0.759, 0.809] |
| randall | linear (round-1 ridge / bilinear) | real | 0.646 +/- 0.001 [0.605, 0.688] | 0.641 +/- 0.002 [0.601, 0.682] | 0.629 +/- 0.002 [0.588, 0.673] | 0.634 +/- 0.001 [0.587, 0.677] | 0.633 +/- 0.001 [0.587, 0.675] | 0.627 +/- 0.004 [0.585, 0.672] | 0.629 +/- 0.002 [0.582, 0.673] |
| randall | linear (round-1 ridge / bilinear) | patch | 0.502 +/- 0.002 [0.486, 0.519] | 0.499 +/- 0.001 [0.484, 0.514] | 0.496 +/- 0.004 [0.482, 0.512] | 0.500 +/- 0.004 [0.486, 0.514] | 0.492 +/- 0.001 [0.473, 0.512] | 0.502 +/- 0.002 [0.481, 0.525] | 0.500 +/- 0.004 [0.483, 0.516] |
| randall | linear (round-1 ridge / bilinear) | instance | 0.633 +/- 0.001 [0.600, 0.668] | 0.626 +/- 0.002 [0.587, 0.668] | 0.630 +/- 0.003 [0.593, 0.667] | 0.636 +/- 0.002 [0.597, 0.676] | 0.622 +/- 0.002 [0.581, 0.662] | 0.624 +/- 0.002 [0.586, 0.663] | 0.623 +/- 0.004 [0.590, 0.661] |
| randall | linear (MLP recipe) | real | 0.649 +/- 0.008 [0.609, 0.691] | 0.625 +/- 0.007 [0.587, 0.665] | 0.620 +/- 0.008 [0.580, 0.659] | 0.619 +/- 0.003 [0.580, 0.659] | 0.615 +/- 0.006 [0.577, 0.656] | 0.613 +/- 0.009 [0.574, 0.653] | 0.612 +/- 0.008 [0.574, 0.650] |
| randall | linear (MLP recipe) | patch | 0.510 +/- 0.005 [0.503, 0.518] | n/a | n/a | n/a | n/a | n/a | 0.504 +/- 0.007 [0.497, 0.510] |
| randall | linear (MLP recipe) | instance | 0.637 +/- 0.008 [0.607, 0.669] | n/a | n/a | n/a | n/a | n/a | 0.613 +/- 0.008 [0.584, 0.643] |
| randall | MLP | real | 0.635 +/- 0.006 [0.595, 0.676] | 0.611 +/- 0.007 [0.574, 0.653] | 0.611 +/- 0.006 [0.575, 0.650] | 0.605 +/- 0.007 [0.570, 0.645] | 0.601 +/- 0.005 [0.564, 0.639] | 0.595 +/- 0.011 [0.561, 0.631] | 0.589 +/- 0.005 [0.553, 0.624] |
| randall | MLP | patch | 0.503 +/- 0.006 [0.494, 0.511] | n/a | n/a | n/a | n/a | n/a | 0.495 +/- 0.008 [0.487, 0.503] |
| randall | MLP | instance | 0.631 +/- 0.007 [0.602, 0.662] | n/a | n/a | n/a | n/a | n/a | 0.598 +/- 0.017 [0.568, 0.631] |
| randtf | linear (round-1 ridge / bilinear) | real | 0.769 +/- 0.002 [0.737, 0.800] | 0.761 +/- 0.003 [0.730, 0.792] | 0.768 +/- 0.002 [0.738, 0.798] | 0.773 +/- 0.002 [0.741, 0.803] | 0.768 +/- 0.002 [0.737, 0.801] | 0.755 +/- 0.003 [0.725, 0.785] | 0.759 +/- 0.001 [0.728, 0.791] |
| randtf | linear (round-1 ridge / bilinear) | patch | 0.515 +/- 0.002 [0.502, 0.526] | 0.518 +/- 0.003 [0.507, 0.529] | 0.509 +/- 0.005 [0.499, 0.518] | 0.510 +/- 0.001 [0.500, 0.520] | 0.515 +/- 0.003 [0.503, 0.528] | 0.505 +/- 0.002 [0.492, 0.516] | 0.513 +/- 0.002 [0.501, 0.524] |
| randtf | linear (round-1 ridge / bilinear) | instance | 0.788 +/- 0.002 [0.751, 0.825] | 0.795 +/- 0.001 [0.758, 0.832] | 0.793 +/- 0.004 [0.754, 0.830] | 0.801 +/- 0.003 [0.764, 0.837] | 0.796 +/- 0.002 [0.758, 0.835] | 0.795 +/- 0.001 [0.757, 0.834] | 0.796 +/- 0.003 [0.757, 0.833] |
| randtf | linear (MLP recipe) | real | 0.744 +/- 0.006 [0.711, 0.775] | 0.760 +/- 0.008 [0.731, 0.790] | 0.739 +/- 0.009 [0.708, 0.771] | 0.736 +/- 0.006 [0.704, 0.770] | 0.729 +/- 0.010 [0.699, 0.759] | 0.725 +/- 0.009 [0.695, 0.757] | 0.726 +/- 0.004 [0.695, 0.758] |
| randtf | linear (MLP recipe) | patch | 0.511 +/- 0.007 [0.505, 0.517] | n/a | n/a | n/a | n/a | n/a | 0.509 +/- 0.007 [0.501, 0.516] |
| randtf | linear (MLP recipe) | instance | 0.754 +/- 0.021 [0.729, 0.780] | n/a | n/a | n/a | n/a | n/a | 0.733 +/- 0.023 [0.708, 0.759] |
| randtf | MLP | real | 0.731 +/- 0.010 [0.700, 0.761] | 0.705 +/- 0.009 [0.675, 0.733] | 0.689 +/- 0.008 [0.658, 0.722] | 0.685 +/- 0.015 [0.656, 0.714] | 0.674 +/- 0.008 [0.644, 0.703] | 0.660 +/- 0.005 [0.631, 0.692] | 0.664 +/- 0.012 [0.635, 0.693] |
| randtf | MLP | patch | 0.511 +/- 0.009 [0.504, 0.519] | n/a | n/a | n/a | n/a | n/a | 0.503 +/- 0.011 [0.495, 0.510] |
| randtf | MLP | instance | 0.738 +/- 0.027 [0.714, 0.762] | n/a | n/a | n/a | n/a | n/a | 0.678 +/- 0.014 [0.654, 0.702] |
| enc_any4d | linear (round-1 ridge / bilinear) | real | 0.808 +/- 0.002 [0.776, 0.841] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (round-1 ridge / bilinear) | patch | 0.517 +/- 0.002 [0.507, 0.527] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (round-1 ridge / bilinear) | instance | 0.804 +/- 0.001 [0.767, 0.837] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (MLP recipe) | real | 0.742 +/- 0.014 [0.707, 0.778] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (MLP recipe) | patch | 0.517 +/- 0.010 [0.511, 0.524] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (MLP recipe) | instance | 0.741 +/- 0.025 [0.718, 0.764] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | MLP | real | 0.764 +/- 0.005 [0.735, 0.791] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | MLP | patch | 0.511 +/- 0.008 [0.503, 0.518] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | MLP | instance | 0.734 +/- 0.025 [0.714, 0.755] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | real | 0.760 +/- 0.003 [0.728, 0.791] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | patch | 0.513 +/- 0.002 [0.501, 0.526] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | instance | 0.785 +/- 0.002 [0.745, 0.822] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (MLP recipe) | real | 0.749 +/- 0.004 [0.718, 0.782] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (MLP recipe) | patch | 0.511 +/- 0.004 [0.504, 0.517] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (MLP recipe) | instance | 0.759 +/- 0.026 [0.733, 0.784] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | MLP | real | 0.700 +/- 0.008 [0.669, 0.729] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | MLP | patch | 0.511 +/- 0.012 [0.504, 0.518] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | MLP | instance | 0.716 +/- 0.021 [0.692, 0.743] | n/a | n/a | n/a | n/a | n/a | n/a |

All metrics (including all-patch AUCs) per layer: probe_controls_tables.md, probe_controls_per_layer.csv, probe_controls_comparisons.csv.

