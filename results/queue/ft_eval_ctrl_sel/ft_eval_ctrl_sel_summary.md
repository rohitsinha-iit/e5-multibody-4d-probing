# ft_eval_ctrl_sel

Evaluation of **ctrl_sel** with the round-1 metric code (output metrics: e5_causal / e5_step3 definitions).

| dataset | n | slope (mean) | EPE static (median of scenes) | EPE moving | sep ratio (median) | mask IoU | mask IoU, epi_aligned |
|---|---|---|---|---|---|---|---|
| v2 | 100 | 0.615 | 3.83 | 30.5 | 0.714 | 0.643 | 0.527 |
| v3float | 69 | 0.567 | 4.02 | 32.5 | 0.673 | 0.496 | 0.290 |
| v3far | 19 | 0.655 | 4.39 | 31.6 | 0.582 | 0.677 | nan |
| kitti15_rgb_2v_wide | 200 | 0.578 | 3.66 | 11.6 | 0.726 | - | - |
| kitti15_rgb_mv4_wide | 200 | 0.710 | 4.26 | 12.7 | 1.152 | - | - |
| sintel_clean_2v_wide | 256 | 0.314 | 3.99 | 11.9 | 0.541 | - | - |
| sintel_final_2v_wide | 256 | 0.327 | 4.31 | 12.1 | 0.525 | - | - |
| sintel_final_mv4_wide | 256 | 0.335 | 4.13 | 11.3 | 0.461 | - | - |

Internal motion decodability (view-0 linear probe, 5-fold by scene, floating set):

| layer | motion AUC, all splits | motion AUC, epi_aligned |
|---|---|---|
| 0 | 0.569 | 0.515 |
| 17 | 0.844 | 0.906 |
| 24 | 0.974 | 0.999 |
