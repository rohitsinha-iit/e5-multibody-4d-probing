Cells: mean over seeds +/- sd over seeds [95% bootstrap CI over scenes of the seed-averaged per-scene value]. Chance is 0.5 for all AUCs. Folds are grouped by scene (5 folds, as round 1). Sources: trained = Any4D trained; randall = random init (all); randtf = DINOv2 + random transformer; enc_any4d = Any4D encoder tokens (no view PE); enc_dino = public DINOv2 tokens (no view PE).


## Key paired comparisons, layers 0 and 24 (per-scene seed means; mean difference [95% CI], sign-test p, n scenes)

| dataset | comparison | source | probe | layer | metric | difference [95% CI] | sign p | n |
|---|---|---|---|---|---|---|---|---|
| v2 | selectivity real - patch | randall | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.006 [-0.045, +0.055] | 0.75 | 93 |
| v2 | selectivity real - instance | randall | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.022 [-0.045, +0.092] | 0.3 | 93 |
| v2 | selectivity real - patch | randall | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.139 [+0.110, +0.169] | 3.6e-12 | 93 |
| v2 | selectivity real - instance | randall | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.001 [-0.031, +0.028] | 1 | 93 |
| v2 | selectivity real - patch | randall | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | -0.009 [-0.063, +0.046] | 0.41 | 93 |
| v2 | selectivity real - instance | randall | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.014 [-0.063, +0.090] | 0.75 | 93 |
| v2 | L24 - L0 | randall | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | -0.028 [-0.062, +0.004] | 0.39 | 93 |
| v2 | selectivity real - patch | randall | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.148 [+0.113, +0.183] | 1.9e-11 | 93 |
| v2 | selectivity real - instance | randall | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | -0.015 [-0.052, +0.021] | 0.21 | 93 |
| v2 | L24 - L0 | randall | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.000 [-0.014, +0.015] | 0.84 | 93 |
| v2 | selectivity real - patch | enc_dino | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | -0.043 [-0.104, +0.019] | 0.12 | 93 |
| v2 | selectivity real - instance | enc_dino | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | -0.044 [-0.122, +0.030] | 0.84 | 93 |
| v2 | selectivity real - patch | enc_dino | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.242 [+0.214, +0.269] | 2.7e-23 | 93 |
| v2 | selectivity real - instance | enc_dino | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.004 [-0.036, +0.029] | 1 | 93 |
| v2 | selectivity real - patch | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.062 [-0.001, +0.128] | 0.21 | 93 |
| v2 | selectivity real - instance | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.134 [+0.052, +0.213] | 0.0046 | 93 |
| v2 | trained - randall | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.049 [-0.025, +0.120] | 0.12 | 93 |
| v2 | trained - randtf | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.090 [+0.036, +0.144] | 0.12 | 93 |
| v2 | trained L0 - enc_any4d | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.009 [-0.007, +0.026] | 0.59 | 93 |
| v2 | trained L0 - enc_dino | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.085 [+0.037, +0.137] | 0.0067 | 93 |
| v2 | selectivity real - patch | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.242 [+0.212, +0.272] | 6.2e-22 | 93 |
| v2 | selectivity real - instance | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.005 [-0.040, +0.031] | 0.68 | 93 |
| v2 | trained - randall | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.119 [+0.086, +0.151] | 1e-07 | 93 |
| v2 | trained - randtf | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.005 [-0.019, +0.030] | 0.038 | 93 |
| v2 | trained L0 - enc_any4d | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.030 [-0.042, -0.018] | 1.1e-06 | 93 |
| v2 | trained L0 - enc_dino | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.006 [-0.019, +0.031] | 0.061 | 93 |
| v2 | selectivity real - patch | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.492 [+0.470, +0.514] | 2e-28 | 93 |
| v2 | selectivity real - instance | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.595 [+0.537, +0.650] | 8.1e-28 | 93 |
| v2 | trained - randall | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.506 [+0.454, +0.556] | 8.1e-28 | 93 |
| v2 | trained - randtf | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.543 [+0.493, +0.591] | 4e-28 | 93 |
| v2 | L24 - L0 | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.429 [+0.365, +0.492] | 1.6e-27 | 93 |
| v2 | selectivity real - patch | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.392 [+0.361, +0.418] | 2.7e-23 | 93 |
| v2 | selectivity real - instance | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.106 [+0.070, +0.139] | 9.6e-11 | 93 |
| v2 | trained - randall | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.264 [+0.227, +0.301] | 1.7e-19 | 93 |
| v2 | trained - randtf | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.142 [+0.110, +0.176] | 3.6e-12 | 93 |
| v2 | L24 - L0 | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.145 [+0.115, +0.176] | 1.8e-15 | 93 |
| v2 | selectivity real - patch | randtf | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | -0.039 [-0.098, +0.023] | 0.35 | 93 |
| v2 | selectivity real - instance | randtf | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | -0.055 [-0.132, +0.028] | 0.15 | 93 |
| v2 | selectivity real - patch | randtf | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.239 [+0.213, +0.268] | 8.8e-25 | 93 |
| v2 | selectivity real - instance | randtf | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.012 [-0.044, +0.020] | 0.097 | 93 |
| v2 | selectivity real - patch | randtf | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | -0.067 [-0.123, -0.016] | 0.0011 | 93 |
| v2 | selectivity real - instance | randtf | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | -0.065 [-0.139, +0.014] | 0.061 | 93 |
| v2 | L24 - L0 | randtf | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | -0.024 [-0.058, +0.009] | 0.21 | 93 |
| v2 | selectivity real - patch | randtf | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.251 [+0.217, +0.285] | 1.7e-19 | 93 |
| v2 | selectivity real - instance | randtf | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.012 [-0.028, +0.051] | 1 | 93 |
| v2 | L24 - L0 | randtf | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.009 [-0.009, +0.027] | 0.84 | 93 |
| v2 | selectivity real - patch | enc_any4d | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.058 [-0.007, +0.116] | 0.21 | 93 |
| v2 | selectivity real - instance | enc_any4d | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.101 [+0.015, +0.184] | 0.016 | 93 |
| v2 | selectivity real - patch | enc_any4d | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.267 [+0.242, +0.293] | 2.7e-23 | 93 |
| v2 | selectivity real - instance | enc_any4d | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.002 [-0.035, +0.033] | 1 | 93 |
| v3float | selectivity real - patch | randall | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | -0.035 [-0.100, +0.030] | 0.9 | 62 |
| v3float | selectivity real - instance | randall | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | -0.029 [-0.127, +0.069] | 0.25 | 62 |
| v3float | selectivity real - patch | randall | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.143 [+0.100, +0.189] | 1.1e-08 | 62 |
| v3float | selectivity real - instance | randall | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.013 [-0.030, +0.051] | 0.9 | 62 |
| v3float | selectivity real - patch | randall | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.018 [-0.055, +0.088] | 0.37 | 62 |
| v3float | selectivity real - instance | randall | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.012 [-0.078, +0.098] | 1 | 62 |
| v3float | L24 - L0 | randall | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.016 [-0.022, +0.053] | 0.37 | 62 |
| v3float | selectivity real - patch | randall | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.129 [+0.079, +0.176] | 0.0005 | 62 |
| v3float | selectivity real - instance | randall | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.006 [-0.040, +0.051] | 0.25 | 62 |
| v3float | L24 - L0 | randall | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | -0.016 [-0.046, +0.014] | 0.25 | 62 |
| v3float | selectivity real - patch | enc_dino | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.059 [-0.029, +0.140] | 0.098 | 62 |
| v3float | selectivity real - instance | enc_dino | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.052 [-0.050, +0.156] | 0.37 | 62 |
| v3float | selectivity real - patch | enc_dino | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.247 [+0.211, +0.283] | 2.6e-13 | 62 |
| v3float | selectivity real - instance | enc_dino | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.024 [-0.062, +0.014] | 0.9 | 62 |
| v3float | selectivity real - patch | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.073 [-0.035, +0.180] | 0.25 | 62 |
| v3float | selectivity real - instance | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | -0.047 [-0.169, +0.071] | 0.9 | 62 |
| v3float | trained - randall | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.094 [-0.012, +0.198] | 0.098 | 62 |
| v3float | trained - randtf | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.023 [-0.051, +0.097] | 0.53 | 62 |
| v3float | trained L0 - enc_any4d | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.001 [-0.037, +0.034] | 0.51 | 62 |
| v3float | trained L0 - enc_dino | trained | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.011 [-0.069, +0.085] | 1 | 62 |
| v3float | selectivity real - patch | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.270 [+0.233, +0.305] | 8.5e-16 | 62 |
| v3float | selectivity real - instance | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.015 [-0.057, +0.028] | 1 | 62 |
| v3float | trained - randall | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.135 [+0.088, +0.181] | 0.0013 | 62 |
| v3float | trained - randtf | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.012 [-0.020, +0.042] | 0.53 | 62 |
| v3float | trained L0 - enc_any4d | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.027 [-0.048, -0.008] | 0.03 | 62 |
| v3float | trained L0 - enc_dino | trained | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.020 [-0.012, +0.050] | 0.16 | 62 |
| v3float | selectivity real - patch | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.482 [+0.444, +0.517] | 5.4e-17 | 62 |
| v3float | selectivity real - instance | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.407 [+0.320, +0.495] | 9.1e-13 | 62 |
| v3float | trained - randall | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.483 [+0.414, +0.547] | 1.1e-16 | 62 |
| v3float | trained - randtf | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.445 [+0.375, +0.513] | 1.1e-16 | 62 |
| v3float | L24 - L0 | trained | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.405 [+0.316, +0.490] | 5.4e-17 | 62 |
| v3float | selectivity real - patch | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.417 [+0.389, +0.441] | 2.7e-17 | 62 |
| v3float | selectivity real - instance | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.145 [+0.103, +0.187] | 1.1e-08 | 62 |
| v3float | trained - randall | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.308 [+0.258, +0.359] | 3.1e-12 | 62 |
| v3float | trained - randtf | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.178 [+0.142, +0.214] | 1.7e-09 | 62 |
| v3float | L24 - L0 | trained | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.157 [+0.118, +0.197] | 3.1e-12 | 62 |
| v3float | selectivity real - patch | randtf | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.043 [-0.033, +0.118] | 0.098 | 62 |
| v3float | selectivity real - instance | randtf | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.028 [-0.065, +0.124] | 0.37 | 62 |
| v3float | selectivity real - patch | randtf | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.254 [+0.219, +0.289] | 1.7e-14 | 62 |
| v3float | selectivity real - instance | randtf | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | -0.020 [-0.061, +0.021] | 0.9 | 62 |
| v3float | selectivity real - patch | randtf | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | +0.055 [-0.029, +0.144] | 0.37 | 62 |
| v3float | selectivity real - instance | randtf | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | -0.013 [-0.106, +0.078] | 0.7 | 62 |
| v3float | L24 - L0 | randtf | linear (round-1 ridge / bilinear) | 24 | motion_auc_v0_objects | -0.018 [-0.056, +0.018] | 0.53 | 62 |
| v3float | selectivity real - patch | randtf | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | +0.247 [+0.211, +0.279] | 8.5e-16 | 62 |
| v3float | selectivity real - instance | randtf | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | -0.037 [-0.082, +0.007] | 0.37 | 62 |
| v3float | L24 - L0 | randtf | linear (round-1 ridge / bilinear) | 24 | pair_auc_obj | -0.010 [-0.035, +0.016] | 0.9 | 62 |
| v3float | selectivity real - patch | enc_any4d | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | +0.073 [-0.022, +0.168] | 0.098 | 62 |
| v3float | selectivity real - instance | enc_any4d | linear (round-1 ridge / bilinear) | 0 | motion_auc_v0_objects | -0.055 [-0.167, +0.055] | 0.61 | 62 |
| v3float | selectivity real - patch | enc_any4d | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.291 [+0.258, +0.326] | 3.1e-12 | 62 |
| v3float | selectivity real - instance | enc_any4d | linear (round-1 ridge / bilinear) | 0 | pair_auc_obj | +0.004 [-0.037, +0.043] | 0.16 | 62 |

## v2: motion_auc_v0_objects

| source | probe | labels | L0 | L4 | L8 | L12 | L16 | L20 | L24 |
|---|---|---|---|---|---|---|---|---|---|
| trained | linear (round-1 ridge / bilinear) | real | 0.569 +/- 0.000 [0.508, 0.628] | 0.613 +/- 0.000 [0.550, 0.674] | 0.652 +/- 0.000 [0.588, 0.712] | 0.720 +/- 0.000 [0.655, 0.785] | 0.760 +/- 0.000 [0.699, 0.816] | 0.970 +/- 0.000 [0.949, 0.986] | 0.998 +/- 0.000 [0.997, 0.999] |
| trained | linear (round-1 ridge / bilinear) | patch | 0.507 +/- 0.000 [0.483, 0.534] | 0.514 +/- 0.000 [0.488, 0.539] | 0.489 +/- 0.000 [0.466, 0.512] | 0.485 +/- 0.000 [0.462, 0.507] | 0.498 +/- 0.000 [0.474, 0.523] | 0.488 +/- 0.000 [0.466, 0.508] | 0.506 +/- 0.000 [0.485, 0.527] |
| trained | linear (round-1 ridge / bilinear) | instance | 0.435 +/- 0.000 [0.378, 0.490] | 0.449 +/- 0.000 [0.392, 0.504] | 0.454 +/- 0.000 [0.399, 0.509] | 0.432 +/- 0.000 [0.376, 0.489] | 0.406 +/- 0.000 [0.350, 0.461] | 0.387 +/- 0.000 [0.331, 0.446] | 0.403 +/- 0.000 [0.347, 0.458] |
| randall | linear (round-1 ridge / bilinear) | real | 0.520 +/- 0.000 [0.473, 0.569] | 0.484 +/- 0.000 [0.432, 0.535] | 0.484 +/- 0.000 [0.435, 0.533] | 0.485 +/- 0.000 [0.432, 0.538] | 0.482 +/- 0.000 [0.426, 0.536] | 0.492 +/- 0.000 [0.438, 0.543] | 0.492 +/- 0.000 [0.437, 0.544] |
| randall | linear (round-1 ridge / bilinear) | patch | 0.514 +/- 0.000 [0.493, 0.536] | 0.505 +/- 0.000 [0.484, 0.526] | 0.509 +/- 0.000 [0.487, 0.531] | 0.504 +/- 0.000 [0.483, 0.523] | 0.503 +/- 0.000 [0.482, 0.524] | 0.505 +/- 0.000 [0.482, 0.526] | 0.501 +/- 0.000 [0.480, 0.523] |
| randall | linear (round-1 ridge / bilinear) | instance | 0.498 +/- 0.000 [0.447, 0.547] | 0.500 +/- 0.000 [0.448, 0.549] | 0.497 +/- 0.000 [0.445, 0.551] | 0.509 +/- 0.000 [0.460, 0.560] | 0.499 +/- 0.000 [0.448, 0.550] | 0.472 +/- 0.000 [0.422, 0.523] | 0.478 +/- 0.000 [0.427, 0.530] |
| randtf | linear (round-1 ridge / bilinear) | real | 0.479 +/- 0.000 [0.421, 0.533] | 0.468 +/- 0.000 [0.414, 0.521] | 0.476 +/- 0.000 [0.423, 0.529] | 0.463 +/- 0.000 [0.409, 0.515] | 0.460 +/- 0.000 [0.406, 0.513] | 0.464 +/- 0.000 [0.413, 0.520] | 0.455 +/- 0.000 [0.404, 0.505] |
| randtf | linear (round-1 ridge / bilinear) | patch | 0.518 +/- 0.000 [0.498, 0.540] | 0.525 +/- 0.000 [0.501, 0.550] | 0.527 +/- 0.000 [0.502, 0.553] | 0.524 +/- 0.000 [0.498, 0.548] | 0.525 +/- 0.000 [0.502, 0.548] | 0.515 +/- 0.000 [0.488, 0.541] | 0.522 +/- 0.000 [0.496, 0.547] |
| randtf | linear (round-1 ridge / bilinear) | instance | 0.534 +/- 0.000 [0.475, 0.589] | 0.505 +/- 0.000 [0.456, 0.557] | 0.500 +/- 0.000 [0.446, 0.552] | 0.488 +/- 0.000 [0.434, 0.543] | 0.504 +/- 0.000 [0.453, 0.554] | 0.510 +/- 0.000 [0.459, 0.562] | 0.520 +/- 0.000 [0.466, 0.572] |
| enc_any4d | linear (round-1 ridge / bilinear) | real | 0.560 +/- 0.000 [0.498, 0.624] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (round-1 ridge / bilinear) | patch | 0.502 +/- 0.000 [0.478, 0.525] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (round-1 ridge / bilinear) | instance | 0.459 +/- 0.000 [0.400, 0.518] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | real | 0.484 +/- 0.000 [0.431, 0.537] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | patch | 0.527 +/- 0.000 [0.504, 0.550] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | instance | 0.528 +/- 0.000 [0.475, 0.582] | n/a | n/a | n/a | n/a | n/a | n/a |

## v2: pair_auc_obj

| source | probe | labels | L0 | L4 | L8 | L12 | L16 | L20 | L24 |
|---|---|---|---|---|---|---|---|---|---|
| trained | linear (round-1 ridge / bilinear) | real | 0.770 +/- 0.001 [0.744, 0.796] | 0.795 +/- 0.001 [0.767, 0.823] | 0.810 +/- 0.000 [0.783, 0.838] | 0.842 +/- 0.000 [0.816, 0.867] | 0.862 +/- 0.000 [0.837, 0.887] | 0.890 +/- 0.000 [0.860, 0.916] | 0.915 +/- 0.001 [0.890, 0.937] |
| trained | linear (round-1 ridge / bilinear) | patch | 0.528 +/- 0.001 [0.518, 0.541] | 0.526 +/- 0.000 [0.515, 0.539] | 0.526 +/- 0.000 [0.516, 0.536] | 0.530 +/- 0.001 [0.519, 0.541] | 0.526 +/- 0.001 [0.518, 0.536] | 0.525 +/- 0.001 [0.515, 0.535] | 0.524 +/- 0.001 [0.515, 0.533] |
| trained | linear (round-1 ridge / bilinear) | instance | 0.775 +/- 0.001 [0.749, 0.801] | 0.791 +/- 0.000 [0.766, 0.816] | 0.801 +/- 0.000 [0.777, 0.826] | 0.802 +/- 0.001 [0.777, 0.826] | 0.815 +/- 0.001 [0.791, 0.841] | 0.812 +/- 0.000 [0.786, 0.836] | 0.810 +/- 0.002 [0.784, 0.835] |
| randall | linear (round-1 ridge / bilinear) | real | 0.651 +/- 0.001 [0.621, 0.682] | 0.656 +/- 0.001 [0.626, 0.686] | 0.650 +/- 0.002 [0.615, 0.682] | 0.644 +/- 0.002 [0.612, 0.676] | 0.642 +/- 0.001 [0.608, 0.675] | 0.652 +/- 0.001 [0.620, 0.683] | 0.652 +/- 0.001 [0.617, 0.685] |
| randall | linear (round-1 ridge / bilinear) | patch | 0.512 +/- 0.001 [0.500, 0.523] | 0.515 +/- 0.001 [0.506, 0.524] | 0.509 +/- 0.001 [0.498, 0.519] | 0.506 +/- 0.002 [0.494, 0.519] | 0.503 +/- 0.001 [0.490, 0.514] | 0.499 +/- 0.003 [0.486, 0.510] | 0.504 +/- 0.000 [0.492, 0.516] |
| randall | linear (round-1 ridge / bilinear) | instance | 0.653 +/- 0.002 [0.624, 0.683] | 0.664 +/- 0.001 [0.635, 0.698] | 0.661 +/- 0.002 [0.629, 0.692] | 0.662 +/- 0.001 [0.629, 0.699] | 0.666 +/- 0.001 [0.630, 0.701] | 0.657 +/- 0.001 [0.623, 0.689] | 0.666 +/- 0.002 [0.633, 0.699] |
| randtf | linear (round-1 ridge / bilinear) | real | 0.765 +/- 0.003 [0.741, 0.791] | 0.777 +/- 0.002 [0.749, 0.804] | 0.771 +/- 0.001 [0.741, 0.800] | 0.778 +/- 0.002 [0.751, 0.808] | 0.780 +/- 0.001 [0.750, 0.808] | 0.772 +/- 0.002 [0.742, 0.801] | 0.774 +/- 0.001 [0.743, 0.804] |
| randtf | linear (round-1 ridge / bilinear) | patch | 0.526 +/- 0.002 [0.518, 0.535] | 0.523 +/- 0.001 [0.516, 0.531] | 0.515 +/- 0.002 [0.507, 0.523] | 0.531 +/- 0.002 [0.521, 0.541] | 0.517 +/- 0.003 [0.508, 0.527] | 0.521 +/- 0.002 [0.513, 0.528] | 0.523 +/- 0.003 [0.512, 0.534] |
| randtf | linear (round-1 ridge / bilinear) | instance | 0.777 +/- 0.002 [0.751, 0.802] | 0.763 +/- 0.002 [0.736, 0.795] | 0.768 +/- 0.001 [0.741, 0.794] | 0.760 +/- 0.002 [0.730, 0.790] | 0.758 +/- 0.002 [0.729, 0.788] | 0.756 +/- 0.002 [0.726, 0.787] | 0.762 +/- 0.002 [0.734, 0.789] |
| enc_any4d | linear (round-1 ridge / bilinear) | real | 0.800 +/- 0.001 [0.774, 0.825] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (round-1 ridge / bilinear) | patch | 0.533 +/- 0.001 [0.523, 0.544] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (round-1 ridge / bilinear) | instance | 0.802 +/- 0.001 [0.778, 0.826] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | real | 0.764 +/- 0.002 [0.738, 0.791] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | patch | 0.523 +/- 0.002 [0.515, 0.531] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | instance | 0.768 +/- 0.001 [0.743, 0.793] | n/a | n/a | n/a | n/a | n/a | n/a |

## v3float: motion_auc_v0_objects

| source | probe | labels | L0 | L4 | L8 | L12 | L16 | L20 | L24 |
|---|---|---|---|---|---|---|---|---|---|
| trained | linear (round-1 ridge / bilinear) | real | 0.569 +/- 0.000 [0.479, 0.658] | 0.569 +/- 0.000 [0.487, 0.650] | 0.591 +/- 0.000 [0.504, 0.679] | 0.699 +/- 0.000 [0.622, 0.779] | 0.784 +/- 0.000 [0.708, 0.855] | 0.938 +/- 0.000 [0.902, 0.968] | 0.973 +/- 0.000 [0.950, 0.992] |
| trained | linear (round-1 ridge / bilinear) | patch | 0.496 +/- 0.000 [0.462, 0.530] | 0.485 +/- 0.000 [0.449, 0.519] | 0.508 +/- 0.000 [0.477, 0.543] | 0.494 +/- 0.000 [0.461, 0.526] | 0.497 +/- 0.000 [0.466, 0.533] | 0.498 +/- 0.000 [0.465, 0.535] | 0.491 +/- 0.000 [0.462, 0.522] |
| trained | linear (round-1 ridge / bilinear) | instance | 0.616 +/- 0.000 [0.530, 0.696] | 0.559 +/- 0.000 [0.470, 0.641] | 0.549 +/- 0.000 [0.458, 0.638] | 0.570 +/- 0.000 [0.481, 0.662] | 0.568 +/- 0.000 [0.480, 0.655] | 0.540 +/- 0.000 [0.451, 0.628] | 0.566 +/- 0.000 [0.480, 0.655] |
| randall | linear (round-1 ridge / bilinear) | real | 0.475 +/- 0.000 [0.414, 0.535] | 0.501 +/- 0.000 [0.434, 0.564] | 0.503 +/- 0.000 [0.439, 0.567] | 0.517 +/- 0.000 [0.457, 0.580] | 0.484 +/- 0.000 [0.423, 0.547] | 0.488 +/- 0.000 [0.421, 0.553] | 0.491 +/- 0.000 [0.426, 0.551] |
| randall | linear (round-1 ridge / bilinear) | patch | 0.510 +/- 0.000 [0.477, 0.543] | 0.498 +/- 0.000 [0.469, 0.526] | 0.487 +/- 0.000 [0.454, 0.520] | 0.479 +/- 0.000 [0.445, 0.516] | 0.474 +/- 0.000 [0.436, 0.511] | 0.484 +/- 0.000 [0.450, 0.518] | 0.473 +/- 0.000 [0.440, 0.506] |
| randall | linear (round-1 ridge / bilinear) | instance | 0.504 +/- 0.000 [0.448, 0.560] | 0.496 +/- 0.000 [0.435, 0.558] | 0.479 +/- 0.000 [0.416, 0.540] | 0.488 +/- 0.000 [0.430, 0.547] | 0.506 +/- 0.000 [0.449, 0.559] | 0.484 +/- 0.000 [0.423, 0.540] | 0.479 +/- 0.000 [0.419, 0.540] |
| randtf | linear (round-1 ridge / bilinear) | real | 0.546 +/- 0.000 [0.475, 0.618] | 0.547 +/- 0.000 [0.475, 0.621] | 0.548 +/- 0.000 [0.474, 0.618] | 0.545 +/- 0.000 [0.468, 0.614] | 0.533 +/- 0.000 [0.461, 0.604] | 0.526 +/- 0.000 [0.451, 0.603] | 0.528 +/- 0.000 [0.453, 0.605] |
| randtf | linear (round-1 ridge / bilinear) | patch | 0.504 +/- 0.000 [0.477, 0.530] | 0.516 +/- 0.000 [0.485, 0.548] | 0.479 +/- 0.000 [0.448, 0.509] | 0.500 +/- 0.000 [0.468, 0.531] | 0.494 +/- 0.000 [0.461, 0.524] | 0.483 +/- 0.000 [0.450, 0.514] | 0.474 +/- 0.000 [0.439, 0.507] |
| randtf | linear (round-1 ridge / bilinear) | instance | 0.519 +/- 0.000 [0.444, 0.590] | 0.521 +/- 0.000 [0.450, 0.590] | 0.511 +/- 0.000 [0.442, 0.577] | 0.510 +/- 0.000 [0.445, 0.574] | 0.515 +/- 0.000 [0.449, 0.578] | 0.533 +/- 0.000 [0.468, 0.594] | 0.541 +/- 0.000 [0.476, 0.603] |
| enc_any4d | linear (round-1 ridge / bilinear) | real | 0.568 +/- 0.000 [0.490, 0.646] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (round-1 ridge / bilinear) | patch | 0.495 +/- 0.000 [0.465, 0.528] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (round-1 ridge / bilinear) | instance | 0.624 +/- 0.000 [0.540, 0.710] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | real | 0.558 +/- 0.000 [0.482, 0.632] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | patch | 0.499 +/- 0.000 [0.470, 0.530] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | instance | 0.506 +/- 0.000 [0.428, 0.580] | n/a | n/a | n/a | n/a | n/a | n/a |

## v3float: pair_auc_obj

| source | probe | labels | L0 | L4 | L8 | L12 | L16 | L20 | L24 |
|---|---|---|---|---|---|---|---|---|---|
| trained | linear (round-1 ridge / bilinear) | real | 0.780 +/- 0.001 [0.747, 0.813] | 0.811 +/- 0.000 [0.777, 0.844] | 0.819 +/- 0.001 [0.783, 0.853] | 0.877 +/- 0.000 [0.848, 0.906] | 0.909 +/- 0.001 [0.884, 0.932] | 0.941 +/- 0.001 [0.917, 0.961] | 0.937 +/- 0.001 [0.912, 0.959] |
| trained | linear (round-1 ridge / bilinear) | patch | 0.511 +/- 0.001 [0.499, 0.522] | 0.520 +/- 0.001 [0.508, 0.534] | 0.521 +/- 0.001 [0.505, 0.541] | 0.509 +/- 0.001 [0.492, 0.527] | 0.524 +/- 0.001 [0.509, 0.542] | 0.517 +/- 0.002 [0.507, 0.527] | 0.521 +/- 0.002 [0.509, 0.534] |
| trained | linear (round-1 ridge / bilinear) | instance | 0.795 +/- 0.001 [0.762, 0.831] | 0.801 +/- 0.000 [0.761, 0.838] | 0.811 +/- 0.002 [0.778, 0.844] | 0.816 +/- 0.000 [0.784, 0.849] | 0.824 +/- 0.000 [0.789, 0.857] | 0.808 +/- 0.001 [0.774, 0.842] | 0.792 +/- 0.003 [0.754, 0.826] |
| randall | linear (round-1 ridge / bilinear) | real | 0.646 +/- 0.001 [0.605, 0.688] | 0.641 +/- 0.002 [0.601, 0.682] | 0.629 +/- 0.002 [0.588, 0.673] | 0.634 +/- 0.001 [0.587, 0.677] | 0.633 +/- 0.001 [0.587, 0.675] | 0.627 +/- 0.004 [0.585, 0.672] | 0.629 +/- 0.002 [0.582, 0.673] |
| randall | linear (round-1 ridge / bilinear) | patch | 0.502 +/- 0.002 [0.486, 0.519] | 0.499 +/- 0.001 [0.484, 0.514] | 0.496 +/- 0.004 [0.482, 0.512] | 0.500 +/- 0.004 [0.486, 0.514] | 0.492 +/- 0.001 [0.473, 0.512] | 0.502 +/- 0.002 [0.481, 0.525] | 0.500 +/- 0.004 [0.483, 0.516] |
| randall | linear (round-1 ridge / bilinear) | instance | 0.633 +/- 0.001 [0.600, 0.668] | 0.626 +/- 0.002 [0.587, 0.668] | 0.630 +/- 0.003 [0.593, 0.667] | 0.636 +/- 0.002 [0.597, 0.676] | 0.622 +/- 0.002 [0.581, 0.662] | 0.624 +/- 0.002 [0.586, 0.663] | 0.623 +/- 0.004 [0.590, 0.661] |
| randtf | linear (round-1 ridge / bilinear) | real | 0.769 +/- 0.002 [0.737, 0.800] | 0.761 +/- 0.003 [0.730, 0.792] | 0.768 +/- 0.002 [0.738, 0.798] | 0.773 +/- 0.002 [0.741, 0.803] | 0.768 +/- 0.002 [0.737, 0.801] | 0.755 +/- 0.003 [0.725, 0.785] | 0.759 +/- 0.001 [0.728, 0.791] |
| randtf | linear (round-1 ridge / bilinear) | patch | 0.515 +/- 0.002 [0.502, 0.526] | 0.518 +/- 0.003 [0.507, 0.529] | 0.509 +/- 0.005 [0.499, 0.518] | 0.510 +/- 0.001 [0.500, 0.520] | 0.515 +/- 0.003 [0.503, 0.528] | 0.505 +/- 0.002 [0.492, 0.516] | 0.513 +/- 0.002 [0.501, 0.524] |
| randtf | linear (round-1 ridge / bilinear) | instance | 0.788 +/- 0.002 [0.751, 0.825] | 0.795 +/- 0.001 [0.758, 0.832] | 0.793 +/- 0.004 [0.754, 0.830] | 0.801 +/- 0.003 [0.764, 0.837] | 0.796 +/- 0.002 [0.758, 0.835] | 0.795 +/- 0.001 [0.757, 0.834] | 0.796 +/- 0.003 [0.757, 0.833] |
| enc_any4d | linear (round-1 ridge / bilinear) | real | 0.808 +/- 0.002 [0.776, 0.841] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (round-1 ridge / bilinear) | patch | 0.517 +/- 0.002 [0.507, 0.527] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_any4d | linear (round-1 ridge / bilinear) | instance | 0.804 +/- 0.001 [0.767, 0.837] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | real | 0.760 +/- 0.003 [0.728, 0.791] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | patch | 0.513 +/- 0.002 [0.501, 0.526] | n/a | n/a | n/a | n/a | n/a | n/a |
| enc_dino | linear (round-1 ridge / bilinear) | instance | 0.785 +/- 0.002 [0.745, 0.822] | n/a | n/a | n/a | n/a | n/a | n/a |

All metrics (including all-patch AUCs) per layer: probe_controls_tables.md, probe_controls_per_layer.csv, probe_controls_comparisons.csv.

