# E5 queue status

updated 2026-10-16 00:03:42

| # | task | status | attempts | started | finished | pushed |
|---|---|---|---|---|---|---|
| 1 | v3_pilot: Items 1+2 pilot renders (far-glued C3, floating shadowless C4) for visual approval | done | 1 | 2026-10-15 19:49:59 | 2026-10-15 20:00:30 | True |
| 2 | t4_causal_v2: Item 4: five heads together, amplification, path patching into scene-flow decoder (v2) | done | 1 | 2026-10-15 20:00:44 | 2026-10-15 20:18:45 | True |
| 3 | t5_subspace_v2: Item 5: LEACE erasure and steering along the motion subspace, layers 17-21 (v2) | done | 1 | 2026-10-15 20:19:23 | 2026-10-15 20:36:24 | True |
| 4 | t3_viewsweep_v2: Item 3: 2, 3, 4, 7 input views, outputs + probes + attention (v2) | done | 1 | 2026-10-15 20:36:30 | 2026-10-15 21:01:31 | True |
| 5 | v3_scale: (split into v3_scale_float and v3_scale_far) | waiting_gate | 0 |  |  |  |
| 6 | v3_pipeline: (split into v3_pipeline_float and v3_pipeline_far) | pending | 0 |  |  |  |
| 7 | v3_viewsweep: Items 2+3: view-count sweep on the floating set | pending | 0 |  |  |  |
| 8 | v3_pilot_far2: Item 1 pilot re-render: far-glued pairs kept far apart in the image | done | 1 | 2026-10-15 21:35:58 | 2026-10-15 21:40:05 | True |
| 9 | t8_realdata_kitti: Item 8a: KITTI scene flow 2015 (200 scenes), Any4D slope / EPE / sep_ratio, 2, 4, 7 views, square and wide input | done | 1 | 2026-10-15 21:40:18 | 2026-10-15 22:01:41 | True |
| 10 | t8_realdata_sintel: Item 8b: MPI Sintel training (final, clean; every 4th frame), Any4D slope / EPE / sep_ratio, 2 and 4 views, square and wide input | done | 1 | 2026-10-15 22:01:54 | 2026-10-15 22:32:35 | True |
| 11 | v3_scale_float: Item 2: render 70 floating shadowless scenes + 20 extra grounded epi (v2) | done | 1 | 2026-10-15 22:32:43 | 2026-10-16 00:03:34 | True |
| 12 | v3_pipeline_float: Item 2: Steps 3-6 + adjacency control on the floating set and v2 with 30 epi | running | 1 | 2026-10-16 00:03:42 |  | False |
| 13 | v3_scale_far: Item 1: render 20 far-glued scenes (image-gap constraint) | pending | 0 |  |  |  |
| 14 | v3_pipeline_far: Item 1: Steps 3-6 + adjacency control on the far-glued set | pending | 0 |  |  |  |
| 15 | t7_any4d_v2: Item 7 cross-model: Any4D reference (3D/2D slope, sep_ratio, probes P + attention A at layers 0,4..24) | pending | 0 |  |  |  |
| 16 | t7_mapanything_v2: Item 7 cross-model: MapAnything V1 (static, Any4D's base) outputs, probes, attention | pending | 0 |  |  |  |
| 17 | t7_vggt_v2: Item 7 cross-model: VGGT-1B (static) outputs + tracks (fixed tracker head), probes, attention | pending | 0 |  |  |  |
| 18 | t7_page4d_v2: Item 7 cross-model: PAGE-4D (dynamic VGGT fine-tune) outputs, probes, attention | pending | 0 |  |  |  |
| 19 | t7_xm_summary: Item 7: cross-model comparison table (Any4D, MapAnything, VGGT, PAGE-4D) | pending | 0 |  |  |  |
