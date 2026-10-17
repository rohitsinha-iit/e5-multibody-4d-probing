# ft_eval_gain_sel

Evaluation of **gain_sel** with the round-1 metric code (output metrics: e5_causal / e5_step3 definitions).

| dataset | n | slope (mean) | EPE static (median of scenes) | EPE moving | sep ratio (median) | mask IoU | mask IoU, epi_aligned |
|---|---|---|---|---|---|---|---|
| v2 | 100 | 0.620 | 3.84 | 30.3 | 0.724 | 0.642 | 0.527 |
| v3float | 69 | 0.572 | 4.01 | 30.1 | 0.676 | 0.491 | 0.271 |
| v3far | 19 | 0.661 | 4.44 | 31.5 | 0.583 | 0.664 | nan |
| kitti15_rgb_2v_wide | 200 | 0.590 | 3.69 | 11.6 | 0.815 | - | - |
| kitti15_rgb_mv4_wide | 200 | 0.722 | 4.25 | 12.6 | 1.097 | - | - |
| sintel_clean_2v_wide | 256 | 0.319 | 3.97 | 11.8 | 0.539 | - | - |
| sintel_final_2v_wide | 256 | 0.332 | 4.30 | 12.0 | 0.532 | - | - |
| sintel_final_mv4_wide | 256 | 0.340 | 4.15 | 11.3 | 0.479 | - | - |

Internal motion decodability (view-0 linear probe, 5-fold by scene, floating set):

| layer | motion AUC, all splits | motion AUC, epi_aligned |
|---|---|---|
| 0 | 0.569 | 0.515 |
| 17 | 0.844 | 0.906 |
| 24 | 0.973 | 0.999 |
