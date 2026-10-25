# U2 behaviour track: launch report (A6000, GPU 0)

Written 2026-10-24 03:05 IST. Server aicoe-a6000 (ssh:192.168.50.207), user rohit_sinha.
Paths: E5_BIG = /data2/DATA/rohit_sinha/e5/big, REPO = /data2/DATA/rohit_sinha/e5/repo.
Local commit (not pushed): `4b2d6f9` (U2 code only). The mechanism track later committed `1533cc7` and `92d23e1` on top.

## 1. What is running

| Item | Value |
|---|---|
| Pipeline | `scripts/attr/u2_pipeline.sh`, started 2026-10-23 23:16:46 IST, PID 4174143 (setsid, detached; launcher job 87980491) |
| GPU / CPU | GPU 0 only (`CUDA_VISIBLE_DEVICES=0`), 12 threads pinned to cores 0 to 11 |
| Logs | `$E5_BIG/logs/u2/pipeline.log`, `pipeline_status.txt`, one log per step (`any4d.log`, `vggt.log`, ...), `score.log`, `summary.log` |
| Outputs | `$E5_BIG/results/u2/` (pred*, rows_*.csv, summary_u2.md/json, confirmatory_u2.json, fig_F1_factorial.png, fig_F2_crossfamily.png) |
| Done at 03:00 IST | Any4D, VGGT, 4RC: all units complete (52 units with _DONE.json), 0 item errors, 0 scoring errors |
| Remaining order | MoRe, CoTracker3 (Kubric + KITTI), RAFT, OpenD4RT, DA3 batch, TrackCraft3R (i), (ii), (iii), CoTracker3 on Sintel |
| Expected finish | about 2026-10-25 03:00 to 06:00 IST if GPU 0 stays free (about 24 GPU hours remain; CoTracker3 on Sintel alone is about 12 h because every evaluation pixel, about 0.4 M per sample, is queried) |

Design: one process per model (model loaded once) over its units (split:variant, dataset:pass:config, pose windows). Every scene, sample and window is its own npz and is skipped when it exists; each finished unit gets `_DONE.json`. `u2_step.sh` waits until GPU 0 has the step's need plus 4 GB free and retries a failed step (up to 30 times, 5 min apart), so an OOM caused by another job costs only the interrupted item. After every model step the incremental CPU scoring (`u2_score.py`) and the summary (`u2_summary.py`) are refreshed, so tables and `confirmatory_u2.json` exist from the first step on and are final after the last step.

GPU 0 contention: a 4 GPU Flash-WAM distillation job (`/data1/rohit_sinha/flashwampp/...`, same account) occupied 42 to 47 GB on every GPU twice on 2026-10-23 (about 22:29 and 22:43 IST) and then stopped. It was not touched. If it comes back, U2 waits for it.

## 2. Coverage (all settings from the frozen pre-registration)

Splits: fit 50 (v2 even seeds), test 48 (v2 odd seeds), transfer 69 (v3float), excluded confound_glued s471, s477, s593 (read from the pre-registration). Views: Any4D, VGGT, 4RC, MoRe 0,4,8,12,16,20,23; OpenD4RT all 24 frames padded to 32 (pad_to_clip=True); TrackCraft3R 0,2,...,20,23, stretch, no t0 registration.

| Model (tag) | Factorial | Real data | Pose sets |
|---|---|---|---|
| Any4D (any4d) | A, B, C, D on fit, test, transfer; budget on C, D | KITTI 2v and Sintel final/clean 2v, wide and square | yes |
| VGGT + tracker_fixed e20 (vggt), s_w by lifting with its own depth and cameras | A to D | KITTI 2v, Sintel final/clean 2v | yes |
| OpenD4RT 32CLIP (opend4rt): xp = r4, r1, r2 (pred K), r2 (GT K); wd = r3 | A to D | same (r1 primary image readout, r2 and r4own secondary) | yes |
| 4RC (4rc); budget on C, D | A to D | same | yes |
| TrackCraft3R (tc3r_gt, tc3r_da3, tc3r_identity) | (i), (ii) on C, D all splits; (iii) on test D (exploratory) | (ii) on KITTI | no |
| CoTracker3 lifted with GT depth and cameras (cotracker3) | C, D | KITTI 2v and mv7; Sintel final/clean 2v and mv4 | no |
| RAFT lifted (raft): chained over the 7 views (primary) and direct 0 to 23 | C, D | KITTI 2v, Sintel final/clean 2v (direct, as E5) | no |
| MoRe (more), cameras | B, D | no | yes |

Pose-set windows (written once, shared with U5b): `$E5_BIG/results/u2/pose_windows.json`, sha256 `c34a0d0edce7300d7370c76a9fcccc7afa0e9f6342c80d8b0e29a95b465b7df6`. 230 non-overlapping windows of 8 consecutive evaluation frames (remainder dropped): Sintel 77 (MonST3R 14 sequences, final pass), TUM-dynamics 88 (8 sequences, rgb_90 = first 90 frames at stride 3, 0.1 s spacing), Bonn 65 (MonST3R protocol: balloon2, crowd2, crowd3, person_tracking2, synchronous; rgb_110). The mechanism track's repair.py now reads this file (commit 1533cc7). Note for that track: its earlier loader iterated over all 26 Bonn sequences; the file restricts Bonn to the 5 protocol sequences.

Metrics per scene (rows_kubric.csv, one row per scene x variant x boundary in {excl8, none}; excl8 is primary): alpha, beta (D: fit_ab with the identifiability flag cond <= 10, median |A|, |B| >= 2 px; C: alpha = image slope), slope2d, beta_static, s_w, static residual, scale, static and moving EPE, sep_ratio, ATE / RPE-T / RPE-R (B, D), spurious ego-motion (A, C), budget shares (Any4D, 4RC on C, D), plus the secondary readouts with prefixes (r1_, r2_, r2gtK_, nooff_, direct_) and v7_ pose columns for 24 or 12 view models. Paired contrasts (dRPE D-B, dATE, spurious C-A, I = alpha_D - alpha_C, I_sw) are computed in the summary. Real data: rows_real.csv (e5_realdata_eval.evaluate plus the w2 alpha/beta decomposition). Pose sets: rows_pose.csv.

## 3. Smoke checks (2 test scenes, s301 and s303, every model; results/u2_smoke)

All units finished with 0 errors; all stored arrays have the expected shapes and are finite (lifted trackers 98.6 to 99.9% finite world displacement, the rest is tracks leaving the image). Static EPE (median px, boundary none, D unless stated) against the gate values:

| Model | Static EPE smoke | Gate reference | Other checks |
|---|---|---|---|
| Any4D | 5.31, 2.32 | 4.30 (G0d reference) | alpha 1.06 / 0.90, slope2d 0.68 / 0.78 |
| VGGT | 2.82, 2.53 | n/a | slope2d 0.38 / 0.10 (E5 4090: 0.34 / 0.13) |
| 4RC | 4.93, 3.86 | 3.52 (G0g) | s_w 0.89 / 1.04 |
| OpenD4RT r4 / r1 | 4.39, 3.88 / 10.54, 8.43 | r4 3.99, r1 7.99 (G0d after padding) | s_w (r3) 0.91 / 0.86; RPE-R 0.41 / 0.37 deg |
| TrackCraft3R (i) | 9.27, 6.49 | 7.22 (G0b stretch) | (ii) 12.4, 15.7 px with DA3 cameras; (iii) 93 to 115 px as expected for identity cameras |
| CoTracker3 | 3.76, 1.24 | n/a | slope2d 0.96 / 0.91 (E5 4090 7f: 0.97 / 0.92) |
| RAFT chained | 1.79, 0.59 | n/a | slope2d 0.94 / 0.71 (E5 4090 chain7: 0.95 / 0.70) |
| MoRe | cameras only | n/a | RPE-R 0.57 to 0.90 deg (gate: 0.2 to 1.4) |

Scorer validation: re-scoring the smoke predictions of Any4D and VGGT with the E5 evaluator (e5_w2_eval.eval_scene) gives identical slope2d, alpha, beta, static and moving EPE to the U2 scorer (boundary none). The Any4D static EPE on s301 (5.31 px) is lower than the 4090 E5 value (10.06 px) while s303 matches (2.32 vs 2.40); since the scorer is identical, this is a difference of the model run on the re-render, not of the metric. Budget GT check: the all-GT projection reproduces gt_factorial x_true to about 2e-13 px.

First full outputs (Any4D, VGGT, 4RC complete) look consistent: Any4D test D static EPE 4.11 px [3.74, 4.47] over 48 scenes, A 1.33 px, spurious rotation in A 0.08 deg; VGGT test D slope2d 0.44, alpha 0.31 (E5: VGGT slope 0.34, alpha 0.22).

Because Any4D is complete, two U2 confirmatory statistics are already final (deterministic, raw p, Holm by the lead): T7 I = -0.098 [-0.173, -0.023] over 39 test scenes identifiable in D, two-sided p 0.008, decision by the frozen rule "inconclusive" (|I| between 0.05 and 0.10); T8 dRPE-R = +0.008 deg [0.002, 0.015] over 48 test scenes, one-sided p 0.0035. T1a, T1b and the T1 validity check fill in once OpenD4RT, TrackCraft3R (ii), CoTracker3 and RAFT finish. Read the final values from confirmatory_u2.json.

## 4. Deviations from the pre-registration

None. No frozen field was changed. The pre-registration file now has sha256 69bdf016... because the mechanism track appended the planned fit_derived_values block and corrected the 'written' date (commit 92d23e1); I checked that every other field is identical to the core of 678689f (sha 088e5bd2). confirmatory_u2.json records the sha it was computed against.

Implementation choices where the pre-registration or plan is silent (fixed before any full test result; the lead may want to confirm them):
1. T1a/T1b scenes: the frozen text says "one-sided scene bootstrap on v2 test + v3float transfer", so the confirmatory p pools test and transfer D scenes; test-only and transfer-only p are reported alongside. (My task text said test split with transfer alongside; I followed the frozen text and report both.)
2. Statistic for every test: scene mean (the existing world_slope.one_sided_p_below convention); bootstrap p = (count on the null side + 1) / 2001; two-sided p = min(1, 2 min). T1 validity reads the mean lifted s_w on the same pooled scenes.
3. Lifted RAFT for the T1 validity check: chained over the 7 views (the E5 "chained RAFT" reference) is primary; direct 0 to 23 is reported alongside. CoTracker3 on Kubric uses the same 7 views as E5.
4. T1 and T7 use variant D for s_w; C is reported descriptively.
5. Pose metrics use each model's own pre-registered view set; OpenD4RT (24) and TrackCraft3R (12) also get the 7-view subset (v7_ columns). TrackCraft3R cameras are its input cameras (GT in (i), DA3 in (ii)), so its pose rows are not model estimates.
6. Real data: the Kubric 8 px boundary rule is not applied (the pre-registration states it for Kubric metrics); real data use the E5 3 px band. OpenD4RT on real data has no GT cameras, so r1 (uv head) is the primary image readout and r2 (predicted K) and r4own (r3 moved with its own cameras) are secondary. TrackCraft3R (ii) on KITTI uses offsets 0 to 10 with offset 10 repeated (12 frames; the clip has only 11 frames from the reference on). CoTracker3 "full multiview window" = the E5 windows (KITTI mv7, Sintel mv4).
7. Any4D image readout follows the E5 route (pred_correspondences with the GT-free pixel_offset check); the plain projection is kept as nooff_. Dense outputs are read at the nearest model pixel (the E5 to518 rule). sep_ratio uses the exact query pixels on the prediction side (E5 used the 518-rounded pixel); the E5 separation sample itself is reproduced exactly.
8. 4RC on pose sets uses load_images size 512 as on Kubric (4RC's own relpose evaluator uses 518). Pose-set windows are not stratified by dynamic fraction (no masks in the tarballs; U5b owns that analysis).
9. Fit split is run for every model but is descriptive only.

Other notes: `env/a6000_decisions.md` is empty (0 bytes) in the working tree and in git since b5cfd6f; the decisions were taken from the pre-registration's decisions_approved_by_user list. The smoke step computed T7/T8 on its 2 test scenes; that file was deleted and those values were not used.

## 5. Commands

```bash
. /data2/DATA/rohit_sinha/e5/big/queue/site_env.sh
# progress (pipeline status, units with item counts and DONE markers, errors, GPU 0)
bash $E5_ROOT/repo/scripts/attr/u2_status.sh
tail -f $E5_BIG/logs/u2/pipeline_status.txt
# re-run scoring and the summary at any time (incremental, CPU, 12 cores)
taskset -c 0-11 $E5_BIG/venvs/post/bin/python $E5_ROOT/repo/src/attr/u2_score.py --procs 10
taskset -c 0-11 $E5_BIG/venvs/post/bin/python $E5_ROOT/repo/src/attr/u2_summary.py
# restart or resume after an interruption (skips everything already written)
cd $E5_BIG/logs/u2 && setsid nohup bash $E5_ROOT/repo/scripts/attr/u2_pipeline.sh > pipeline.log 2>&1 < /dev/null &
# one model by hand, e.g. OpenD4RT on test D
bash $E5_ROOT/repo/scripts/attr/u2_step.sh 18 opend4rt u2_infer.py --model opend4rt --kubric test:D
```
