# E5 queue status

updated 2026-10-17 02:42:54

| # | task | status | attempts | started | finished | pushed |
|---|---|---|---|---|---|---|
| 1 | v3_pilot: Items 1+2 pilot renders (far-glued C3, floating shadowless C4) for visual approval | done | 1 | 2026-10-15 19:49:59 | 2026-10-15 20:00:30 | True |
| 2 | t4_causal_v2: Item 4: five heads together, amplification, path patching into scene-flow decoder (v2) | done | 1 | 2026-10-15 20:00:44 | 2026-10-15 20:18:45 | True |
| 3 | t5_subspace_v2: Item 5: LEACE erasure and steering along the motion subspace, layers 17-21 (v2) | done | 1 | 2026-10-15 20:19:23 | 2026-10-15 20:36:24 | True |
| 4 | t3_viewsweep_v2: Item 3: 2, 3, 4, 7 input views, outputs + probes + attention (v2) | done | 1 | 2026-10-15 20:36:30 | 2026-10-15 21:01:31 | True |
| 5 | v3_scale: (split into v3_scale_float and v3_scale_far) | waiting_gate | 0 |  |  |  |
| 6 | v3_pipeline: (split into v3_pipeline_float and v3_pipeline_far) | pending | 0 |  |  |  |
| 7 | v3_viewsweep: Items 2+3: view-count sweep on the floating set | done | 1 | 2026-10-16 01:22:26 | 2026-10-16 01:40:04 | True |
| 8 | v3_pilot_far2: Item 1 pilot re-render: far-glued pairs kept far apart in the image | done | 1 | 2026-10-15 21:35:58 | 2026-10-15 21:40:05 | True |
| 9 | t8_realdata_kitti: Item 8a: KITTI scene flow 2015 (200 scenes), Any4D slope / EPE / sep_ratio, 2, 4, 7 views, square and wide input | done | 1 | 2026-10-15 21:40:18 | 2026-10-15 22:01:41 | True |
| 10 | t8_realdata_sintel: Item 8b: MPI Sintel training (final, clean; every 4th frame), Any4D slope / EPE / sep_ratio, 2 and 4 views, square and wide input | done | 1 | 2026-10-15 22:01:54 | 2026-10-15 22:32:35 | True |
| 11 | v3_scale_float: Item 2: render 70 floating shadowless scenes + 20 extra grounded epi (v2) | done | 1 | 2026-10-15 22:32:43 | 2026-10-16 00:03:34 | True |
| 12 | v3_pipeline_float: Item 2: Steps 3-6 + adjacency control on the floating set and v2 with 30 epi | done | 1 | 2026-10-16 01:05:40 | 2026-10-16 01:22:18 | True |
| 13 | v3_scale_far: Item 1: render 20 far-glued scenes (image-gap constraint) | done | 1 | 2026-10-16 00:48:25 | 2026-10-16 01:05:33 | True |
| 14 | v3_pipeline_far: Item 1: Steps 3-6 + adjacency control on the far-glued set | done | 1 | 2026-10-16 01:40:13 | 2026-10-16 01:43:20 | True |
| 15 | t7_any4d_v2: Item 7 cross-model: Any4D reference (3D/2D slope, sep_ratio, probes P + attention A at layers 0,4..24) | done | 1 | 2026-10-16 00:11:50 | 2026-10-16 00:17:28 | True |
| 16 | t7_mapanything_v2: Item 7 cross-model: MapAnything V1 (static, Any4D's base) outputs, probes, attention | done | 1 | 2026-10-16 01:43:28 | 2026-10-16 01:46:05 | True |
| 17 | t7_vggt_v2: Item 7 cross-model: VGGT-1B (static) outputs + tracks (fixed tracker head), probes, attention | done | 1 | 2026-10-16 01:46:13 | 2026-10-16 01:49:51 | True |
| 18 | t7_page4d_v2: Item 7 cross-model: PAGE-4D (dynamic VGGT fine-tune) outputs, probes, attention | done | 1 | 2026-10-16 01:49:59 | 2026-10-16 01:53:36 | True |
| 19 | t7_xm_summary: Item 7: cross-model comparison table (Any4D, MapAnything, VGGT, PAGE-4D) | done | 1 | 2026-10-16 01:55:59 | 2026-10-16 01:56:07 | True |
| 20 | t7_any4d_v2_100: Item 7: Any4D reference rerun on the fixed 100 v2 scenes (writes into t7_any4d_v2 results) | done | 1 | 2026-10-16 01:53:44 | 2026-10-16 01:55:51 | True |
| 21 | w2_kubric_baselines: Round 2 item 1 (W2): CoTracker3 + RAFT on Kubric v2 (100) and floating (70), same query sets as Any4D/VGGT; Any4D + VGGT on floating | done | 1 | 2026-10-16 15:11:22 | 2026-10-16 15:46:33 | True |
| 22 | w2_real_baselines: Round 2 item 1 (W2): RAFT on KITTI 2015 and Sintel (2 views), metrics + alpha/beta vs Any4D cached | done | 1 | 2026-10-16 15:46:42 | 2026-10-16 15:54:49 | True |
| 23 | w2_decomp: Round 2 item 1 (W2): alpha/beta decomposition and slope vs boundary distance, all models | done | 1 | 2026-10-16 15:54:57 | 2026-10-16 15:55:05 | True |
| 24 | w2_dose: Round 2 item 1 (W2): dose-response of the slope (speed, area, texture, occlusion, camera rotation, motions, confidence) | done | 1 | 2026-10-16 15:55:12 | 2026-10-16 15:55:49 | True |
| 25 | w2_summary: Round 2 item 1 (W2): summary table, figure and verdict (pull: 4D-model property or generic shrinkage?) | done | 1 | 2026-10-16 15:55:58 | 2026-10-16 15:56:05 | True |
| 26 | ft_render_v2: Item 10: render 200 v2-grounded training scenes (seeds 2001-2200, approved settings) | done | 1 | 2026-10-16 15:56:13 | 2026-10-16 19:20:03 | True |
| 27 | ft_render_float: Item 10: render 200 floating training scenes (seeds 2201-2400, approved v3float settings) | done | 1 | 2026-10-16 19:20:12 | 2026-10-16 22:48:04 | True |
| 28 | ft_eval_base: Item 10: evaluate base Any4D with the fine-tune evaluation (Kubric v2/v3float/v3far, internals, KITTI, Sintel) | pending | 2 | 2026-10-17 02:41:37 | 2026-10-17 02:42:15 | False |
| 29 | ft_pilot: Item 10: pilot fine-tune (heads_sf, 1000 steps) to choose the number of steps | done | 3 | 2026-10-17 01:01:23 | 2026-10-17 02:31:45 | True |
| 30 | ft_train_heads: Item 10: fine-tune (a) LoRA r8 on the five supply heads' slices | pending | 2 | 2026-10-17 02:42:15 | 2026-10-17 02:42:54 | False |
| 31 | ft_eval_heads: Item 10: evaluate fine-tune (a) LoRA r8 on the five supply heads' slices | pending | 0 |  |  |  |
| 32 | ft_train_ctrl: Item 10: fine-tune (d) control: LoRA r8 on 5 matched random heads | running | 2 | 2026-10-17 02:42:54 |  | False |
| 33 | ft_eval_ctrl: Item 10: evaluate fine-tune (d) control: LoRA r8 on 5 matched random heads | pending | 0 |  |  |  |
| 34 | ft_train_sf: Item 10: fine-tune (b) scene-flow DPT head + adaptor | pending | 1 | 2026-10-17 02:40:19 | 2026-10-17 02:40:58 | False |
| 35 | ft_eval_sf: Item 10: evaluate fine-tune (b) scene-flow DPT head + adaptor | pending | 0 |  |  |  |
| 36 | ft_train_heads_sf: Item 10: fine-tune (c) five-head LoRA + scene-flow head | pending | 1 | 2026-10-17 02:40:58 | 2026-10-17 02:41:37 | False |
| 37 | ft_eval_heads_sf: Item 10: evaluate fine-tune (c) five-head LoRA + scene-flow head | pending | 0 |  |  |  |
| 38 | ft_train_gain: Item 10: fine-tune (e) 5 learned per-head gains | pending | 0 |  |  |  |
| 39 | ft_eval_gain: Item 10: evaluate fine-tune (e) 5 learned per-head gains | pending | 0 |  |  |  |
| 40 | ft_summary: Item 10: comparison table and figure (base vs variants; a vs d head specificity; transfer; decodability gap) | pending | 0 |  |  |  |
| 41 | v4_pilot: Review item W3 pilot: 2 scenes each of epi_accel, epi_const (paired), epi_sizerand, epi_sizematch, with GT checks | done | 1 | 2026-10-16 22:48:29 | 2026-10-16 22:53:06 | True |
| 42 | v4_scale_epi: Review item W3: render 30 scenes per new epipolar split (120 scenes) + GT checks | done | 1 | 2026-10-16 22:53:45 | 2026-10-17 01:01:15 | True |
| 43 | v4_viewsweep_epi: Review item W3: 2/3/4/7-view sweep on v4 epipolar set + v3float: outputs, probes (pooled and transfer), attention, stats | pending | 0 |  |  |  |
| 44 | r2_probe_extract: Round 2 W4: control-model internals (random-init Any4D, DINOv2 + random transformer, raw encoder tokens) on v2 out100 and v3float | pending | 0 |  |  |  |
| 45 | r2_heads_null: Round 2 W5: five supply heads vs 40 matched random head sets (mean ablation), resample ablation, amplification x1.5 (v2 out100) | pending | 0 |  |  |  |
| 46 | r2_probe_linear: Round 2 W4: seeded linear probes (5 seeds) with random-init, DINOv2-only and label-shuffle controls (v2 out100, v3float) | pending | 0 |  |  |  |
| 47 | r2_probe_mlp: Round 2 W4: MLP probes vs linear, 5 seeds, label-shuffle selectivity, combined figure (v2 out100, v3float) | pending | 0 |  |  |  |
