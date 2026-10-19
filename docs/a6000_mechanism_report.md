# A6000 mechanism and repair track report (U4a, U4b, U5a, U5b)

Date: 2026-10-19 (IST). Server: aicoe-a6000 (ssh:192.168.50.207). Repo: /data2/DATA/rohit_sinha/e5/repo, local commits `094f408` and `ce26d37` (not pushed).
E5_BIG = /data2/DATA/rohit_sinha/e5/big. All numbers below are from files on the server (results/attr_mech/g0i.json, g0h.json, smoke logs).

## 1. Summary
| Item | Status |
|---|---|
| any4d_hooks.py (cache, patch, edits, key-region masking, two-pass helper) | Done, gate G0i passed |
| interchange.py (obj, cam, camwarp, cont screens; added objall; rank; joint top-k with matched nulls; k*; necessity) | Done, smoke-tested |
| motion_probe_allviews.py (extract, CV over L*, 90% recall threshold, eval AUC and IoU) | Done, smoke-tested |
| repair.py (U5a directions and variants i to v; U5b variants a to d; Kubric and real pose-set evaluation) | Done, smoke-tested |
| vggt_hooks.py (cache and patch over frame and global blocks) | Done, gate G0h passed |
| VGGT screens (u4b_screen_*, u4b_joint) and VGGT probe | Not written (should tier, see deviations) |
| Queue tasks (plan section 9 names), launcher, fit-split watcher | Done. Queue tasks not enqueued. Watcher running |
| Fit-split screens | Not started: no fit-split (v2 even seed) factorial scene exists yet. The watcher starts them when renders land |

## 2. Gates
| Gate | Status | Value |
|---|---|---|
| G0i (Any4D interchange identity) | pass | fp32, all 288 heads + 24 residual states patched at all positions: max abs diff 0.0 for D into D and 0.0 for B into B (pts3d_0, scene_flow, cam_pose); 24 residuals + 24 random heads patched one at a time: max 0.0. Scene v2/469 |
| G0h (VGGT token layout) | pass | Per frame: camera token at index 0, then 4 register tokens, patch_start_idx 5, 1374 tokens per frame. Camera-token and register-token identity patches D into D at all 48 blocks: 0.0. Patching only the final frame and global camera tokens from C into D reproduces C's pose_enc exactly (max diff 0.0; D vs C differ by 0.206), so index 0 is the token the camera head reads |

Both gates are merged into $E5_BIG/results/attr_gates.json (keys G0i, G0h; written under a file lock).

G0i diagnostics (AMP, bf16): repeat forward, identity patch of everything, and an empty key mask are all bit-exact (0.0). Masking keys on GT moving tokens (566 of 9583 tokens, 5.9%) in one head (b10h3) changes scene flow by up to 0.054; in all 12 heads of block 10, by up to 0.54. Positive control: patching every residual state from B into D gives scene flow at mean distance 4.9e-6 from B's and 0.0152 from D's (the D to B distance is 0.0152), so the patches take effect.

## 3. Timings and memory (shared A6000; other users' jobs on the same GPU)
| Operation (7 views, AMP) | Time | Peak GPU |
|---|---|---|
| Model load | 20 s to 3 min (depends on /data2 I/O load) | |
| Plain forward | 0.40 s | |
| Forward with full capture (288 head outputs + 24 residuals, bf16, 1.06 GB) | 0.52 s | 8.9 GB |
| Forward with all components patched | 0.49 s | |
| Key-region masking: 1 head / 12 heads of one block / all 288 heads | 0.50 s / 0.44 s / 1.22 s | 9.1 GB |
| Screen setup per scene (3 base runs with caches) plus 5 to 7 patched forwards | 4.8 to 15.5 s | 10.9 to 11.0 GB |
| Joint top-k (5 k values) with 2 to 3 nulls per k | 9.0 to 10.9 s | 11.0 GB |
| Necessity (donor cache + 5 k values) | 4.6 s | 12.0 GB |
| VGGT G0h (fp32, incl. load) | 92 s | 15.3 GB |

Extrapolated full-screen cost (provisional; from per-forward times on a shared GPU): obj, cam and objall about 314 forwards each, cont 289, camwarp 48 (24 residuals with their round-trip floors). That is about 10 GPU-minutes per fit scene, about 8 to 9 GPU-hours for 50 fit scenes. Test-split joint with 40 nulls: 205 forwards per scene per screen, about 1.7 min.
GPU_NEED_GB is set to 14 for all Any4D tasks (peak 12 GB) and 12 for G0h (the queue adds 4 GB headroom).

## 4. Definitions as implemented (to be frozen at the Oct 27 pre-registration)
- Views 0, 4, 8, 12, 16, 20, 23 at 518 square; target view 23. Positions are flat token indices v*1369 + j; the scale token is index V*1369 (present only in global blocks).
- Components: head output = input of attn.proj, channels h*64 to h*64+63, every token; residual = block output.
- F(y): scene flow of view-0 moving query points (gt_factorial q_pix, q_group > 0) to view 23, model scale. pose(y): views 1 to 6 relative to view 0, first two rotation columns and translation divided by the median predicted view-0 depth.
- tau_O, lambda_O, tau_C, lambda_C as in the plan. kappa uses e = 0.5 * (RPE-R / RPE-R(D) + RPE-T / RPE-T(D)) (pose.py, Sim(3) on centres); kappa_R and kappa_T are stored as well. Because e(D) minus e(B) can be tiny (0.012 on v2/469), the cont ranking pools across scenes: sum(1 - e_y) / sum(1 - e_B).
- Contamination key mask: GT moving tokens with at least 25% moving pixels; queries of views 1 to 6 only; explicit masked softmax for the selected heads; fused SDPA for all others. Probe labels: at least 50% moving pixels; B tokens all labelled 0.
- Nulls: for each component of a top-k set, a different head of the same block among the 6 nearest in mean activation norm (residual states: same block parity). Completeness: effect of the top-k head set divided by the effect of all 288 heads patched at the same positions. k* = smallest k in {1, 3, 5, 10, 20} reaching 80% scene-median completeness on fit scenes (None = distributed, risk 4).
- U5a: U_O, U_C = top-2 right singular vectors of h(D)-h(B) on moving view-0 tokens and h(D)-h(C) on all view-0 tokens; mutual orthogonalisation; m = mean over static tokens of D; edit at every token. Variant iv uses the E5 per-run token mean. Variant v loads $E5_BIG/ft/runs/heads_sel/any4d_ft.pth.
- U5b recovery = (e(D) - e(edit)) / (e(D) - e(B)).

## 5. Important observation for the lead (affects H3 and the obj screen)
At view-0 positions, patches in blocks 22 and 23 have exactly zero effect on F. This is not a bug. F (view 23 scene flow) is decoded from view-6 tokens. The last block that moves information across views is global block 22, so a view-0 change after block 21 cannot reach view 6. Smoke values on v2/469: b22h9 and b23r give tau_O = 0.0 exactly, while b10r gives 0.166. Consequence: under the plan's view-0 object screen, E5 head b22h9 ranks last by construction, so H3 (all five heads in the top 20) cannot hold as worded. I added a secondary screen `objall` (base D, source B at all view tokens; B and D share the camera, so background tokens are aligned and object tokens carry the counterfactual). The H3 rank for blocks 22 and 23 should be read from objall. Please decide at pre-registration which screen H3 uses.
I also restricted P_obj, P_cam and P_cont (joint, necessity, k*) to heads. A residual-state patch carries everything upstream, so mixing residuals and heads in one top-k list makes completeness meaningless. In the first smoke run, completeness came out at 48 because of this. Residual states remain in the screens as a layer profile.

## 6. Smoke outputs (code validation only)
All smoke runs used pilot scenes from the test or transfer split (no fit scene exists), with at most 5 components per screen. Outputs are in $E5_BIG/results/attr_mech/smoke and must not be used for analysis. Logs: $E5_BIG/logs/mech/smoke.log, smoke2.log, smoke3.log.
| Step | Result |
|---|---|
| obj, cam, cont, camwarp, objall screens on v2_469 | ran; v2_469: B vs D |F| difference squared 776, pose difference squared (C vs D) 0.93 |
| rank, joint (nulls), kstar, necessity | ran after two fixes (null pool from the full component universe; head-only top-k) |
| probe (6 v2 odd-seed scenes, smoke) | CV AUC by L: 13 0.983, 15 0.982, 17 0.994, 19 0.994, 21 0.996; L* = 21; threshold 0.567 at OOF recall 0.90, FPR 0.010 |
| probe eval on v3float_531 | AUC 0.998, mask IoU 0.27, B false-positive rate 0.031 (low IoU reflects the 90% recall threshold) |
| U5a dirs (five heads, 3 scenes) | max principal cosine between U_O and U_C: b2h1 0.53, b8h11 0.89, b14h6 0.83, b17h10 0.92, b22h9 0.93 |
| U5a i, iii (2 draws), iv, v on v2_469 | ran; all metrics finite |
| U5b a, c (2 draws), d on v2_469; a on Sintel alley_2 (6 windows); b on TUM sitting_halfsphere (11 windows) | ran; probe mask flags 5.4% of tokens on v2_469 and 0.8 to 2.7% on the real windows |

## 7. What is running
| What | How to check | Expected finish |
|---|---|---|
| Fit-split watcher, PID 2563469 (setsid nohup, `scripts/mech/fit_watch.sh`). Every 20 min it lists complete fit scenes and runs obj, cam, cont, objall and camwarp screens on the new ones (one GPU via run_mech_gpu.sh), then re-ranks | `tail $E5_BIG/logs/mech/fit_watch.out`; per-screen logs `$E5_BIG/logs/mech/fit_screen_<screen>.log`; results `$E5_BIG/results/u4a/screen_<screen>/`, ranks `$E5_BIG/results/u4a/rank_<screen>.json` | Idle until fit renders land (the Data track is rendering variant D first). About 10 GPU-minutes per fit scene after that. Stops at 50 fully screened fit scenes, at 2026-10-27 23:59 IST, or when `$E5_BIG/logs/mech/STOP_WATCH` exists |

Nothing else is running. No queue task is enqueued and the supervisor is not started.

## 8. Ready-to-launch commands
```
. /data2/DATA/rohit_sinha/e5/big/queue/site_env.sh
# fit-split pipeline through the queue (rank, joint and k*, necessity, probe, U5a dirs, G0h). Use either the queue or
# the watcher for the screens, not both: touch $E5_BIG/logs/mech/STOP_WATCH first if switching to the queue.
bash $E5_ROOT/repo/scripts/mech/launch_mech.sh fit
# after the Oct 27 pre-registration: enqueue test, transfer and real-data tasks, then open their gate
bash $E5_ROOT/repo/scripts/mech/launch_mech.sh test
touch $E5_QUEUE/approvals/attr_prereg
# one-off direct runs (one GPU)
R=$E5_ROOT/repo; A=$R/src/attr; L=$E5_BIG/logs/mech
$R/scripts/mech/run_mech_gpu.sh 14 any4d $L/rank.log $A/interchange.py --mode rank --screen obj --split fit
$R/scripts/mech/run_mech_gpu.sh 14 any4d $L/joint_fit.log $A/interchange.py --mode joint --screen obj --split fit --n_null 0
$R/scripts/mech/run_mech_gpu.sh 14 any4d $L/kstar.log $A/interchange.py --mode kstar --screen obj --split fit
$R/scripts/mech/run_mech_gpu.sh 14 any4d $L/probe.log $A/motion_probe_allviews.py --mode all --split fit
$R/scripts/mech/run_mech_gpu.sh 14 any4d $L/dirs.log $A/repair.py --mode dirs --split fit --kstar_obj <k*>
```
Task files: scripts/queue/task_defs/attr_mech_fit.json (23 tasks) and attr_mech_test.json (29 tasks, gate attr_prereg; test, transfer and real data). Every script refuses test, transfer or real-data evaluation without --allow_test, and fitting on anything other than fit scenes.

## 9. Deviations from the plan and the task
1. No fit-split factorial scenes exist yet, so no fit-split screen has run. All smoke tests used test or transfer pilot scenes (mainly v2_469; probe fit on 6 v2 odd-seed scenes; directions on 3), with at most 5 components, for code validation only. The smoke numbers above are provisional and not findings.
2. P_obj, P_cam and P_cont are head sets; residual states are a layer profile only (section 5).
3. Added the secondary objall screen because view-0 patches after block 21 cannot reach F (section 5).
4. kappa normalisation and pooled ranking as in section 4 (the plan leaves "normalised" open).
5. Completeness denominator = all 288 heads patched at the same positions.
6. camwarp is restricted to the 24 residual states (with round-trip floors) to fit the budget.
7. The 8 px boundary rule is not yet applied in the repair metrics (TODO before the freeze).
8. U5a is evaluated on Kubric only; the KITTI and Sintel flow evaluation for U5a is not implemented. U5b real pose-set evaluation is implemented.
9. U4b: only vggt_hooks.py and G0h are done. VGGT screens, the VGGT joint step and u5_probe_vggt are not written.
10. U5b variant (e) MoRe is not in repair.py; it belongs to the New models track's MoReAdapter.
11. The real pose-set loader is my own minimal implementation (none existed): Sintel final pass with the MonST3R 14-sequence list (written from memory of the protocol; to be checked against 4RC's evaluator at G0k), TUM rgb_90 with groundtruth_90, Bonn rgb_110 with groundtruth_110, as preprocessed in the tarballs; square 518 resize; non-overlapping 8-frame windows. I did not check whether the tarballs include dynamic masks.
12. pose.py from the New models track is used unchanged.
13. Smoke jobs picked a GPU through run_mech_gpu.sh and then ran outside its lock; the watcher and the queue tasks use the lock or pick_gpu properly.
14. The local sandbox failed, so all files were written directly on the server.
