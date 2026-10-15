# E5: Multibody structure probing of 4D feed-forward reconstruction models (Any4D)

Code and small results for experiment E5 of "Probing 4D Feed-Forward Reconstruction Models: An Epipolar-Geometry Analysis".
Mathematical reference: Vidal, Ma, Soatto, Sastry, "Two-View Multibody Structure from Motion", IJCV 68(1), 2006.
The full plan, methods, results and caveats are in docs/E5_multibody_plan.md (sections 7 to 10 and later).

## Layout

- src/e5_multibody.py: two-view multibody toolkit (Veronese embedding, rank-drop motion count, multibody F, GPCA epipoles, segmentation, sequential RANSAC baseline).
- src/e5_step1_calibration.py: Step 1 noise-floor calibration (synthetic).
- src/kubric_e5_scene.py: Kubric generator for the controlled multibody scenes (v1 floating objects; v2 with --grounded 1 --size_lo 1.4 --size_hi 2.2).
- src/e5_kubric_post.py, src/e5_post_driver.py: ground-truth correspondences from depth, segmentation and world matrices; flow check; multibody SNR; contact sheets.
- src/e5_any4d_infer.py, src/e5_step3_analysis.py: Step 3 (E5-C), Any4D output correspondences vs ground truth.
- src/e5_any4d_extract.py, src/e5_gt_patches.py, src/e5_internal_analysis.py, src/e5_adjacency_control.py: Steps 4 to 6 (E5-A attention, E5-F clustering, E5-P probes) and the adjacency control.
- src/e5_e8_knockout.py: E8 interventions (block skip, head knockouts, single-unit mean-ablation screen).
- scripts/render, scripts/jobs: render and job scripts as run on the GPU server.
- results/: small result tables and JSON (v1 unless marked v2); figures/: summary figures.

## Running

- Kubric: docker image kubricdockerhub/kubruntu (kubric 2022.4.1, Blender 2.93); renders run on CPU.
- Any4D: github.com/Any-4D/Any4D (commit aa9f1b0), checkpoint any4d_4v_combined.pth; scripts in src/e5_any4d_*.py and src/e5_e8_knockout.py are placed in Any4D/scripts/.
- Rendered data and extracted internals are not stored in this repository.
