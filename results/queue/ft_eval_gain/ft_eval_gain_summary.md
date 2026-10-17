# ft_eval_gain

Evaluation of **gain** with the round-1 metric code (output metrics: e5_causal / e5_step3 definitions).

| dataset | n | slope (mean) | EPE static (median of scenes) | EPE moving | sep ratio (median) | mask IoU | mask IoU, epi_aligned |
|---|---|---|---|---|---|---|---|
| v2 | 100 | 0.607 | 3.82 | 30.7 | 0.735 | 0.644 | 0.532 |
| v3float | 69 | 0.559 | 4.02 | 33.2 | 0.657 | 0.496 | 0.278 |
| v3far | 19 | 0.648 | 4.38 | 31.6 | 0.566 | 0.663 | nan |
| kitti15_rgb_2v_wide | 200 | 0.575 | 3.67 | 11.6 | 0.740 | - | - |
| kitti15_rgb_mv4_wide | 200 | 0.707 | 4.26 | 12.7 | 1.086 | - | - |
| sintel_clean_2v_wide | 256 | 0.314 | 3.98 | 11.9 | 0.539 | - | - |
| sintel_final_2v_wide | 256 | 0.327 | 4.30 | 12.2 | 0.510 | - | - |
| sintel_final_mv4_wide | 256 | 0.334 | 4.13 | 11.3 | 0.458 | - | - |

Internal motion decodability (view-0 linear probe, 5-fold by scene, floating set):

| layer | motion AUC, all splits | motion AUC, epi_aligned |
|---|---|---|
| 0 | 0.569 | 0.515 |
| 17 | 0.843 | 0.905 |
| 24 | 0.973 | 0.999 |
