# Causal specificity of the five supply heads (review W5), v2 out100

Scenes: 100. Heads (block:head): 2:1, 22:9, 17:10, 14:6, 8:11. Base: slope 0.607, static EPE 4.10 px, moving EPE 34.96 px, separation ratio 0.825.

Each null set has 4 global-attention heads and 1 frame-attention head, drawn without the five (seeded; the first three are the round-1 sets). Statistic: mean over scenes of (condition minus base). z uses the null mean and sd; robust z uses the null median and MAD; p is the empirical one-sided p, (1 + number of null sets at least as extreme) / (N + 1).


## Mean ablation (abl5 vs 40 random sets)

| metric | five heads, mean [95% CI] | null mean (sd) | null median | z | robust z | p (one-sided) | p (two-sided) | percentile | five vs per-scene null median [95% CI] (sign p) | null sets with static EPE > +10 px |
|---|---|---|---|---|---|---|---|---|---|---|
| slope | -0.599 [-0.641, -0.552] | -0.067 (0.156) | -0.011 | -3.42 | -22.50 | 0.073 | 0.073 | 5 | -0.586 [-0.629, -0.538] (9.9e-25) | 3 |
| epe_static | +3.028 [+2.628, +3.428] | +5.061 (18.690) | -0.011 | -0.11 | 41.49 | 0.122 | 0.122 | 90 | +3.010 [+2.615, +3.412] (6.4e-24) | 3 |
| epe_moving | +56.046 [+49.307, +63.367] | +6.299 (16.474) | +0.855 | 3.02 | 37.51 | 0.073 | 0.073 | 95 | +55.193 [+48.611, +62.105] (2.4e-23) | 3 |
| sep_ratio | -0.796 [-0.924, -0.678] | -0.083 (0.169) | -0.021 | -4.21 | -23.67 | 0.024 | 0.024 | 0 | -0.763 [-0.873, -0.655] (3.2e-27) | 3 |

## Resample ablation (res5 vs 10 random sets)

| metric | five heads, mean [95% CI] | null mean (sd) | null median | z | robust z | p (one-sided) | p (two-sided) | percentile | five vs per-scene null median [95% CI] (sign p) | null sets with static EPE > +10 px |
|---|---|---|---|---|---|---|---|---|---|---|
| slope | -0.525 [-0.585, -0.468] | -0.056 (0.171) | -0.005 | -2.75 | -36.91 | 0.182 | 0.182 | 10 | -0.519 [-0.580, -0.462] (2.4e-23) | 1 |
| epe_static | +4.457 [+3.196, +5.959] | +11.596 (36.297) | +0.131 | -0.20 | 16.98 | 0.182 | 0.182 | 90 | +4.353 [+3.076, +5.936] (2.6e-12) | 1 |
| epe_moving | +52.731 [+45.061, +60.591] | +6.531 (20.287) | +0.277 | 2.28 | 70.51 | 0.182 | 0.182 | 90 | +52.438 [+45.077, +60.605] (9.9e-25) | 1 |
| sep_ratio | -0.616 [-0.739, -0.497] | -0.080 (0.168) | -0.037 | -3.19 | -15.48 | 0.091 | 0.091 | 0 | -0.584 [-0.708, -0.468] (1.9e-14) | 1 |

## Amplification x1.5 (amp5_a1.5 vs 20 random sets)

| metric | five heads, mean [95% CI] | null mean (sd) | null median | z | robust z | p (one-sided) | p (two-sided) | percentile | five vs per-scene null median [95% CI] (sign p) | null sets with static EPE > +10 px |
|---|---|---|---|---|---|---|---|---|---|---|
| slope | +0.346 [+0.320, +0.371] | -0.005 (0.023) | +0.000 | 15.12 | 49.77 | 0.048 | 0.048 | 100 | +0.346 [+0.319, +0.371] (9.9e-25) | 0 |
| epe_static | +0.609 [+0.393, +0.824] | +0.332 (0.809) | +0.047 | 0.34 | 12.92 | 0.143 | 0.143 | 90 | +0.548 [+0.344, +0.747] (0.00041) | 0 |
| epe_moving | +6.702 [+2.168, +10.611] | +0.539 (1.739) | -0.023 | 3.54 | 18.70 | 0.048 | 0.048 | 100 | +6.638 [+2.442, +10.719] (0.00077) | 0 |
| sep_ratio | +0.102 [-0.007, +0.212] | -0.016 (0.049) | -0.005 | 2.42 | 6.56 | 0.048 | 0.143 | 100 | +0.107 [-0.005, +0.215] (0.033) | 0 |

## Single conditions (change vs base, mean [95% CI], sign-test p)

| condition | slope | static EPE (px) | moving EPE (px) | sep ratio |
|---|---|---|---|---|
| res1_b2h1 | -0.160 [-0.185, -0.134] (8.1e-16) | +0.734 [+0.487, +1.003] (0.00087) | +12.735 [+9.011, +17.170] (2.9e-09) | -0.175 [-0.269, -0.080] (0.00027) |
| res1_b22h9 | -0.265 [-0.298, -0.233] (7.1e-21) | +0.345 [+0.216, +0.476] (4.7e-06) | +22.285 [+18.302, +26.367] (1.1e-18) | -0.349 [-0.463, -0.241] (1.8e-07) |
| res1_b17h10 | -0.320 [-0.355, -0.281] (2.4e-23) | +0.148 [+0.072, +0.226] (0.00041) | +29.100 [+24.176, +34.114] (4.5e-22) | -0.431 [-0.553, -0.329] (9e-13) |
| res1_b14h6 | -0.117 [-0.142, -0.091] (2.5e-13) | +0.158 [+0.090, +0.220] (7.9e-05) | +8.430 [+5.990, +11.029] (7.1e-10) | -0.223 [-0.325, -0.140] (1.8e-07) |
| res1_b8h11 | -0.062 [-0.082, -0.044] (4.2e-07) | +0.190 [+0.041, +0.357] (0.62) | +4.875 [+2.965, +7.166] (3.7e-06) | -0.104 [-0.174, -0.036] (0.089) |
| res5 | -0.525 [-0.588, -0.467] (2.4e-23) | +4.457 [+3.150, +6.025] (2.6e-12) | +52.731 [+45.305, +60.262] (9.9e-25) | -0.616 [-0.754, -0.494] (1.9e-14) |
| abl5 | -0.599 [-0.644, -0.553] (9.9e-25) | +3.028 [+2.668, +3.421] (6.4e-24) | +56.046 [+49.383, +62.746] (2.4e-23) | -0.796 [-0.920, -0.683] (3.2e-27) |
| amp5_a1.5 | +0.346 [+0.319, +0.369] (9.9e-25) | +0.609 [+0.405, +0.828] (0.00041) | +6.702 [+2.392, +10.799] (0.00077) | +0.102 [-0.012, +0.215] (0.033) |

## Plain statements (generated from the numbers above)

- Mean ablation: the five heads' change in static EPE (+3.03 px) is within the null of random sets (two-sided empirical p = 0.122, percentile 90).
- Mean ablation: change in slope -0.599 vs null median -0.011 (z = -3.4, empirical p = 0.073).
- Resample ablation: the five heads' change in static EPE (+4.46 px) is within the null of random sets (two-sided empirical p = 0.182, percentile 90).
- Resample ablation: change in slope -0.525 vs null median -0.005 (z = -2.7, empirical p = 0.182).
- Amplification x1.5: the five heads' change in static EPE (+0.61 px) is within the null of random sets (two-sided empirical p = 0.143, percentile 90).
- Amplification x1.5: change in slope +0.346 vs null median +0.000 (z = 15.1, empirical p = 0.048).
- Amplification fixes magnitude, not accuracy: the slope rises by +0.346 while the moving-object EPE also rises by +6.70 px.
