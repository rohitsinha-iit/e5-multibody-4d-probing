# OpenD4RT camera-frame readout check (G0d), A6000, 2026-10-20

Time box: one working day, approved by the user, before the U2 decision on OpenD4RT readouts. Finished well within the box.
Commit (local on the A6000, not pushed): `b5cfd6f1a9632606d4384b20fd032883f23c7862`.

## Summary

1. **There is no index or coordinate bug in how the adapter conditions on (t_src, t_tgt, t_cam).** The adapter builds the same queries as the official repo code. On 5 FIT scenes the official `_infer_tracks` path and the adapter differ by at most 0.00 px in the uv output.
2. **There was an input convention mismatch: clip length.** Open-d4rt trains on 32-frame clips. It pads shorter scenes, including the 24-frame Kubric scenes, to 32 frames by repeating the last frame (`src/data/raw_augment.py`, `sample_frame_indices_with_stride` / `_pad_to_clip`). The adapter fed the 24-frame clip without padding, and so do the repo's own inference helpers (`infer_track_3d._infer_tracks` pads nothing when num_frames <= clip_frames). So the adapter followed the official inference code, but that code is out of distribution for clips shorter than 32 frames.
3. **Fix:** `OpenD4RTAdapter(pad_to_clip=True)` is now the default. It pads the video to `clip_frames` by repeating the last frame, and queries still index only the 24 real frames. The flag `--no_pad_clip` in `gate_newmodels.py` restores the old input.
4. **G0d re-run on the same 10 scenes:** r1 static EPE fell from 16.31 to 7.99 px (Any4D 4.30 px). The ratio fell from 3.79 to 1.86, so the gate now **passes, but narrowly** (the threshold is 8.60 px). r2 fell from 17.38 to 9.18 px. r4 (world readout moved with GT cameras) is unchanged at 3.99 to 4.01 px. Pose ATE improved from 0.094 to 0.053.
5. **Recommendation for U2:** always use the padded input. Keep the world readouts (r3 for s_w and r4 for image space) as primary. Use r1 and r2 only as secondary, flagged readouts. Reasons: r1 is still twice the world readout error (8.0 vs 4.0 px), it passes G0d by only 0.6 px, and it stays poor on rotation-heavy scenes (camrot_s305 20.5 px, epi_aligned_s435 25.6 px).

## 1. Code comparison against the official repo

The repo was checked at `big/models/src/Open-d4rt`, commit 403290a. The table compares the adapter (`src/e5_models.py`, class `OpenD4RTAdapter`) with the official files `infer_track_3d.py`, `eval_track3d_in_worldtrack.py`, `src/eval/tasks.py`, `src/model/query_embedding.py`, `src/model/d4rt.py`, `src/model/encoder.py`, `src/losses/d4rt_loss.py` and `src/data/raw_augment.py`.

| Item | Official code | Adapter | Match |
|---|---|---|---|
| Time index convention | 0-based and clip-relative. `nn.Embedding(clip_frames)` for each of t_src, t_tgt, t_cam, clamped to [0, max_frames-1]. t_src also indexes the video tensor directly (local patch via `video[b, t_src]`). | 0-based, clip-relative | yes |
| Argument order | Passed as a dict keyed by `t_src`, `t_tgt`, `t_cam`. The time token is the sum of three embeddings. | Dict with the same keys, built by `_q(u, v, ts, tt, tc)` | yes; a positional swap is not possible |
| Readout queries | local: (t_src=0, t_tgt=t, t_cam=t); ref: (0, t, 0) (`_run_full_clip_queries`) | identical | yes |
| (u, v) normalisation | `u_px / (W-1)` on native pixels (eval script). Local patches use `grid_sample(align_corners=True)` and the Fourier embedding takes [0, 1]. | model pixel index / (w-1), the same mapping because native-to-model is the (W-1)/(w-1) affine | yes |
| uv head target frame | uv of the point at t_tgt in image t_tgt (the loss compares `uv_2d` with the target, and the reprojection term maps xyz from the t_cam camera into t_tgt) | uv at t_tgt, scaled by (w-1, h-1) | yes |
| xyz output frame | the camera frame of t_cam, OpenCV axes (loss: `T_wc[t_cam] @ xyz` is world) | same | yes |
| Output scale or shift | none at inference. The training loss applies mean-depth normalisation and sign-log1p to both prediction and target, so the raw output is linear in camera units. | none; r4 takes a median z-depth scale from GT depth | yes |
| Aspect token | native W/H | W/H = 1.0 | yes |
| Image normalisation | RGB / 255 and resize with INTER_AREA (`_resize_video`) | identical | yes |
| Encoder positional code | 1D sinusoidal over flattened (T', H', W') tokens, so a shorter clip keeps per-frame indices | n/a | consistent |
| **Clip length** | **training:** always clip_frames = 32; a 24-frame Kubric scene becomes frames 0..23 followed by 8 repeats of frame 23. **Inference helpers:** a clip shorter than 32 is fed unpadded. | **24 frames unpadded (before); padded to 32 (after)** | **mismatch with training; fixed** |
| Frame subsampling | training uses a temporal stride of 1 or 2 when the scene is longer than the clip, which a 24-frame scene never is | all 24 frames at stride 1 | yes |

## 2. Controlled tests on FIT scenes

Scenes: n2_generic_s302, n3_generic_s304, camrot_s306, confound_glued_s470, planar_s444 (v2 even seeds, `$E5_BIG/data/v2/out`). No odd seeds and no v3float were used for any choice. Each scene used 3000 random static and up to 1000 moving GT correspondences, from frame 0 to frame 23, queried at their native positions. Values are the median static EPE in native px (512), with the median over the 5 scenes. Script: `src/attr/opend4rt_readout_check.py`. Output: `$E5_BIG/results/opend4rt_check/fit5.json`.

### (a) Official code path vs adapter

| Readout | Adapter, direct queries | Official `_infer_tracks` | Adapter dense grid (gate code path) |
|---|---|---|---|
| r1 uv head | 17.49 | 17.49 | 17.43 |
| r2 xyz(t_cam=23), GT K | 18.01 | 18.01 | n/a |
| r4 world, GT cameras | 3.10 | n/a (the official eval scores the t_cam=0 route only) | 3.13 |

The maximum absolute difference in uv between the official path and the adapter was 0.00 px on all 5 scenes.

### (b) Convention sweep (median over 5 FIT scenes)

| Variant | r1 | r2 (GT K) | r4 (GT cams) | r4own (own cams) | own-pose rotation error at t=23 (deg) |
|---|---|---|---|---|---|
| baseline (24 frames, 0-based, u/(W-1)) | 17.49 | 18.01 | 3.10 | 18.64 | 5.99 |
| 1-based t_tgt and t_cam (t_src stays 0-based because it indexes the video) | 17.06 | 16.72 | 4.73 | 16.86 | 6.80 |
| uv half-pixel convention, (x+0.5)/W | 17.61 | 17.92 | 3.02 | 18.64 | 5.99 |
| time-reversed embeddings (23 - t) | 120.68 | 121.12 | 93.43 | 118.58 | 59.69 |
| subclip of 12 frames (0, 2, ..., 20, 23) | 33.65 | 35.38 | 6.67 | 36.11 | 11.23 |
| resampled to 32 frames (nearest), indices mapped onto the 32-frame axis | 10.76 | 8.54 | 3.15 | 10.84 | 3.45 |
| **padded to 32 frames by repeating frame 23 (training convention)** | **7.54** | **8.36** | 3.62 | 11.30 | **3.16** |
| uv head queried with t_cam = 0 instead of t | 19.22 | n/a | n/a | n/a | n/a |

Notes:
- 1-based indexing cannot be the intended convention. A 1-based t_src runs out of bounds of the video tensor at t_src = 24 (a CUDA index assert on the first attempt), and shifting only t_tgt and t_cam changes nothing that matters.
- Time reversal and half-pixel uv rule out the other index and normalisation hypotheses.
- Only the clip-length variants help, and padding (the exact training convention) helps most. It roughly halves r1 and r2 and the own-pose rotation error.
- GT camera rotation over 0 to 23 is 15 to 39 deg (median 32.6 deg).
- For camrot_s306 the GT translation is zero, so the relative translation error is undefined there.

### (c) Sanity readout: r4 with the model's own cameras

Own cameras come from a D4RT-style Umeyama fit (rigid, trimmed). It fits xyz(t, t, t) to xyz(t, t, 0) on a 32x32 grid of frame-t pixels. Applying the inverse of this camera to the world readout xyz(0, 23, 0) gives "r4own".

- r4own agrees with the direct camera-frame readout r2 to a median of 1 to 2% of depth in 3D (`r2_vs_r4own_3d_rel` is 0.01 to 0.02 on every scene and variant). Its EPE (18.64 px) matches r1 and r2 (17.5 to 18.0 px).
- So the model is self-consistent. Its camera-frame outputs equal its world outputs moved by its own implied camera. The camera-frame error comes from the implied camera, not from the wrapper.
- That camera had a 6.0 deg median rotation error at t = 23 without padding, and 3.2 deg with padding.
- A rigid fit on the queried static points themselves gives the same picture (5.3 deg unpadded, 2.6 deg padded).

### Error growth over the clip (baseline, unpadded, median of 5 FIT scenes)

| t | 2 | 4 | 8 | 12 | 16 | 20 | 23 |
|---|---|---|---|---|---|---|---|
| r1 (px) | 3.24 | 5.09 | 8.07 | 8.81 | 8.71 | 13.70 | 18.01 |
| r2 GT K (px) | 4.49 | 6.07 | 8.55 | 9.86 | 8.76 | 13.69 | 18.55 |
| r4 GT cams (px) | 3.03 | 3.15 | 2.95 | 3.01 | 3.32 | 3.30 | 3.17 |
| GT static flow (px) | 12.0 | 24.0 | 47.8 | 71.2 | 90.2 | 110.2 | 126.2 |

The jump at the last frames (t = 20, 23) fits a clip-end effect. In training the clip never ends at frame 23 of a 24-frame scene; padding puts 8 more frames after it.

## 3. G0d re-run (same 10 scenes as the New models track)

The scenes are camrot_s305, confound_glued_s469, confound_same_s451, epi_aligned_s435, n2_generic_s301, n2_generic_s302, n2_generic_s401, n3_generic_s303, n3_generic_s409 and planar_s443. This is the same list as the original gate, read from `results/attr_gates.json` (G0d). It includes 8 odd seeds. They were used only for this re-evaluation, as instructed; the padding choice comes from the training code and the FIT scenes. The Any4D reference is the same per-scene values as before (`gate_any4d*.json`, 10 common scenes).

| Median static EPE (px, 512) | Before (unpadded) | After (pad to 32) |
|---|---|---|
| r1 uv head (primary) | 16.31 | **7.99** |
| r2 xyz(t_cam=t), predicted K | 17.46 | 8.91 |
| r2 xyz(t_cam=t), GT K | 17.38 | 9.18 |
| r4 world readout, GT cameras | 3.99 | 4.01 |
| Any4D reference | 4.30 | 4.30 |
| Ratio r1 / Any4D | 3.79 | **1.86** |
| G0d (ratio <= 2) | fail | **pass (margin 0.6 px)** |
| Pose ATE (median) | 0.094 | 0.053 |

Per scene, r1 before / r1 after / r2 GT K after / r4 after / Any4D:

| Scene | r1 before | r1 after | r2 after | r4 after | Any4D |
|---|---|---|---|---|---|
| camrot_s305 | 31.02 | 20.48 | 21.20 | 4.17 | 2.29 |
| confound_glued_s469 | 16.23 | 8.11 | 8.70 | 3.26 | 3.94 |
| confound_same_s451 | 17.56 | 7.17 | 10.32 | 9.95 | 5.85 |
| epi_aligned_s435 | 40.04 | 25.56 | 27.09 | 6.88 | 2.99 |
| n2_generic_s301 | 8.49 | 7.87 | 9.58 | 3.54 | 5.42 |
| n2_generic_s302 | 15.59 | 6.79 | 6.37 | 3.85 | 5.54 |
| n2_generic_s401 | 16.39 | 6.67 | 6.32 | 5.06 | 4.66 |
| n3_generic_s303 | 14.47 | 6.37 | 8.14 | 3.05 | 2.63 |
| n3_generic_s409 | 18.99 | 10.17 | 9.10 | 3.17 | 2.95 |
| planar_s443 | 16.21 | 9.54 | 9.25 | 6.43 | 4.92 |

The 48-frame checkpoint, padded to 48 frames in the same way, also improves a lot: r1 went from 34.37 to 8.85 px (ratio 7.99 to 2.06) and r4 from 5.41 to 5.45 px. It still misses the gate narrowly and stays behind the 32CLIP checkpoint.

`results/attr_gates.json` G0d now holds the after values as the gate result, plus `before_unpadded`, `after_pad32`, `fallback_48CLIP` (before and after) and a `fix` note. It was merged with a file lock and atomic replace. A backup of the previous file is at `results/attr_gates.json.bak_20261020_g0d`. Gate jsons: `results/newmodels/gate_opend4rt.json` (before, untouched), `gate_opend4rt_pad32.json`, `gate_opend4rt48.json` and `gate_opend4rt48_pad48.json`.

## 4. Recommendation

1. **Use the padded input for every OpenD4RT run** (now the adapter default). Without padding, the camera-frame readouts and the model's own poses are out of distribution.
2. **For U2, keep the world readouts r3 (w_disp, for s_w) and r4 as primary.** Report r1 and r2 as secondary readouts, now valid by G0d, with three caveats:
   - they pass by only 0.6 px;
   - they carry about twice the world-readout error;
   - they stay poor on rotation-heavy scenes (camrot, epi_aligned at 20 to 26 px), where the model's own camera is the bottleneck.
   This is a change from "world readouts only": r1 and r2 no longer need to be excluded, but they should not carry headline claims.
3. s_w values from OpenD4RT computed before this fix (r3 median 0.79 in the New models report) used the unpadded input. Recompute them with the padded input before any use. On the 10 gate scenes the padded w_s_w per scene is in `gate_opend4rt_pad32.json`.

## 5. Small related task (TrackCraft3R and decisions file)

- `TrackCraft3RAdapter(resize_mode="stretch")` is now the default, with no t0 re-registration (`register_t0=False` was already the default). The docstring records the 2026-10-20 decision.
- `gate_newmodels.py --resize_mode` now defaults to `stretch`, with choices stretch and pad, so pad mode stays available as a flag.
- `src/attr/gates_newmodels_chain.sh` relies on the default and therefore now runs stretch. No other U2 task script sets the TrackCraft3R resize mode.
- Created `env/a6000_decisions.md` with the line exactly as approved.

## 6. Deviations and limits

1. The convention sweep and sanity tests used 5 FIT scenes, with 3000 static and up to 1000 moving queries each. They did not use dense grids. These numbers are provisional and only steer the diagnosis. The G0d numbers come from the full dense gate code on the 10 gate scenes.
2. The official code path (a) was tested through the repo's `_infer_tracks` helper. That is the function the WorldTrack evaluation calls. `infer_track_3d.py` has no standalone entry point. The repo has no separate depth, pose or pointmap inference script. Its intrinsics and Umeyama helpers live in `src/eval/tasks.py`, and the adapter already uses them.
3. Padding to 32 is the only fix adopted. It reproduces the training convention exactly, rather than being picked as the best of the sweep, although it was also the best variant on FIT. Other padding schemes (for example mirroring) were not tried.
4. n2_generic_s302 is both a FIT scene and one of the 10 gate scenes.
5. The 48CLIP padded run is for information only.

## Files changed (commit b5cfd6f)

- `src/e5_models.py`: OpenD4RTAdapter `pad_to_clip=True`; TrackCraft3RAdapter default `resize_mode="stretch"`.
- `src/attr/gate_newmodels.py`: `--no_pad_clip`; `--resize_mode` default stretch.
- `src/attr/opend4rt_readout_check.py`: diagnostic for (a), (b) and (c).
- `src/attr/update_g0d_padfix.py`: G0d merge with before and after values.
- `env/a6000_decisions.md`: new decisions log.
