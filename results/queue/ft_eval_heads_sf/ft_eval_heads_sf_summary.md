# ft_eval_heads_sf

Evaluation of **heads_sf** with the round-1 metric code (output metrics: e5_causal / e5_step3 definitions).

| dataset | n | slope (mean) | EPE static (median of scenes) | EPE moving | sep ratio (median) | mask IoU | mask IoU, epi_aligned |
|---|---|---|---|---|---|---|---|
| v2 | 100 | 0.776 | 3.82 | 26.4 | 0.759 | 0.560 | 0.439 |
| v3float | 69 | 0.735 | 4.06 | 31.9 | 0.806 | 0.534 | 0.364 |
| v3far | 19 | 0.790 | 4.51 | 27.2 | 0.360 | 0.595 | nan |
| kitti15_rgb_2v_wide | 200 | 0.512 | 3.53 | 13.9 | 0.638 | - | - |
| kitti15_rgb_mv4_wide | 200 | 0.642 | 3.98 | 16.6 | 0.738 | - | - |
| sintel_clean_2v_wide | 256 | 0.217 | 3.76 | 13.4 | 0.435 | - | - |
| sintel_final_2v_wide | 256 | 0.173 | 4.21 | 13.4 | 0.528 | - | - |
| sintel_final_mv4_wide | 256 | 0.186 | 4.07 | 12.1 | 0.468 | - | - |

Internal motion decodability (view-0 linear probe, 5-fold by scene, floating set):

| layer | motion AUC, all splits | motion AUC, epi_aligned |
|---|---|---|
| 0 | 0.569 | 0.515 |
| 17 | 0.846 | 0.907 |
| 24 | 0.973 | 0.999 |
