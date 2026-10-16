# ft_render_v2

Training scenes for item 10 (fine-tuning), rendered with the approved generator settings into `/media/workbench/zone_x/e5_multibody/data/ft_train_v2`.

- Rendered: 197 of 200 planned scenes; render failures: 3 (confound_glued_s2178, confound_glued_s2184, confound_glued_s2196).
- Seeds 2001 to 2200 (all at or above 2001, disjoint from every evaluation set, whose seeds are at most 720).
- GT flow self-check (geometric GT vs Kubric optical flow, frame 0 to 1): median EPE 0.000 px, max 0.000 px.

| split | scenes | median sep (px) | median SNR at 1 px |
|---|---|---|---|
| camrot | 25 | 17.7 | 12.5 |
| confound_glued | 22 | 29.2 | 20.6 |
| confound_same | 25 | 32.4 | 22.9 |
| epi_aligned | 40 | 0.0 | 0.0 |
| n2_generic | 30 | 30.5 | 21.6 |
| n3_generic | 30 | 29.5 | 20.9 |
| planar | 25 | 33.6 | 23.8 |
