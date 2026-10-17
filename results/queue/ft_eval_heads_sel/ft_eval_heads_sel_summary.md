# ft_eval_heads_sel

Evaluation of **heads_sel** with the round-1 metric code (output metrics: e5_causal / e5_step3 definitions).

| dataset | n | slope (mean) | EPE static (median of scenes) | EPE moving | sep ratio (median) | mask IoU | mask IoU, epi_aligned |
|---|---|---|---|---|---|---|---|
| v2 | 100 | 0.716 | 3.85 | 23.6 | 0.729 | 0.608 | 0.439 |
| v3float | 69 | 0.667 | 4.10 | 28.7 | 0.714 | 0.469 | 0.292 |
| v3far | 19 | 0.733 | 4.49 | 29.9 | 0.621 | 0.622 | nan |
| kitti15_rgb_2v_wide | 200 | 0.598 | 3.63 | 11.6 | 0.969 | - | - |
| kitti15_rgb_mv4_wide | 200 | 0.732 | 4.24 | 12.4 | 1.035 | - | - |
| sintel_clean_2v_wide | 256 | 0.322 | 4.05 | 11.8 | 0.564 | - | - |
| sintel_final_2v_wide | 256 | 0.333 | 4.31 | 11.8 | 0.460 | - | - |
| sintel_final_mv4_wide | 256 | 0.345 | 4.16 | 11.0 | 0.482 | - | - |

Internal motion decodability (view-0 linear probe, 5-fold by scene, floating set):

| layer | motion AUC, all splits | motion AUC, epi_aligned |
|---|---|---|
| 0 | 0.569 | 0.515 |
| 17 | 0.844 | 0.905 |
| 24 | 0.974 | 0.999 |
