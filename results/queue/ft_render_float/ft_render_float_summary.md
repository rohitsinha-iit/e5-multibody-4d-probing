# ft_render_float

Training scenes for item 10 (fine-tuning), rendered with the approved generator settings into `/media/workbench/zone_x/e5_multibody/data/ft_train_float`.

- Rendered: 194 of 200 planned scenes; render failures: 6 (confound_glued_s2397, confound_glued_s2398, confound_glued_s2400, n3_generic_s2272, n3_generic_s2285, n3_generic_s2296).
- Seeds 2201 to 2400 (all at or above 2001, disjoint from every evaluation set, whose seeds are at most 720).
- GT flow self-check (geometric GT vs Kubric optical flow, frame 0 to 1): median EPE 0.000 px, max 0.000 px.

| split | scenes | median sep (px) | median SNR at 1 px |
|---|---|---|---|
| camrot | 25 | 20.2 | 14.3 |
| confound_glued | 22 | 23.3 | 16.5 |
| confound_same | 25 | 32.9 | 23.2 |
| epi_aligned | 40 | 0.0 | 0.0 |
| n2_generic | 30 | 30.9 | 21.9 |
| n3_generic | 27 | 25.1 | 17.7 |
| planar | 25 | 34.5 | 24.4 |
