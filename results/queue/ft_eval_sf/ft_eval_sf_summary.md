# ft_eval_sf

Evaluation of **sf** with the round-1 metric code (output metrics: e5_causal / e5_step3 definitions).

| dataset | n | slope (mean) | EPE static (median of scenes) | EPE moving | sep ratio (median) | mask IoU | mask IoU, epi_aligned |
|---|---|---|---|---|---|---|---|
| v2 | 100 | 0.767 | 3.82 | 26.7 | 0.737 | 0.561 | 0.464 |
| v3float | 69 | 0.724 | 4.08 | 31.5 | 0.814 | 0.537 | 0.323 |
| v3far | 19 | 0.797 | 4.49 | 27.4 | 0.513 | 0.589 | nan |
| kitti15_rgb_2v_wide | 200 | 0.542 | 3.54 | 14.5 | 0.670 | - | - |
| kitti15_rgb_mv4_wide | 200 | 0.672 | 4.03 | 17.0 | 0.610 | - | - |
| sintel_clean_2v_wide | 256 | 0.211 | 3.76 | 13.4 | 0.430 | - | - |
| sintel_final_2v_wide | 256 | 0.171 | 4.20 | 13.3 | 0.539 | - | - |
| sintel_final_mv4_wide | 256 | 0.180 | 4.08 | 12.3 | 0.456 | - | - |

Internal motion decodability (view-0 linear probe, 5-fold by scene, floating set):

| layer | motion AUC, all splits | motion AUC, epi_aligned |
|---|---|---|
| 0 | 0.569 | 0.515 |
| 17 | 0.843 | 0.905 |
| 24 | 0.973 | 0.999 |
