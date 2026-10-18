# Implementation and experiment plan: E5 + P3 + Idea 1 (motion attribution)

Version 1, 2026-10-18. Companion to Idea1_research_plan.md (the why) and to P3_plan.md version 2 (whose X2 design is reused unchanged). For execution, this file replaces the P3 v2 schedule and task list.

Hard dates:
- engineering freeze Oct 27 (all code written and smoke-tested);
- results freeze Nov 6;
- CVPR 2027 paper registration Nov 10 AoE and submission Nov 16 AoE.

Compute: the shared 4090 + A5000 server (10.0.117.206), through the E5 queue, about 1.5 effective GPUs. Nothing runs before the user's go-ahead.

## 0. Scope in tiers

| Tier | Experiments | Carries |
|---|---|---|
| Must | U0 setup and gates; U1 factorial renders; U2 behaviour (factorial, cross-family, real data, motion budget); U3 cause (P3 X2); U4a interchange maps on Any4D; U5a object-gain edit; U5b contamination masking on Any4D | Contributions 1 to 5 in one architecture |
| Should | U4b interchange on VGGT; MoRe as a repair baseline; TrackCraft3R on KITTI | Cross-architecture mechanism; architectural baseline |
| Could | 4RC interchange; 3 x 3 dose subset; TrackCraft3R identity-camera condition; DiTracker | Extra rows |
| After submission | U6 = P2 (factorial oracle pairs, ReCamMaster pairs, pseudo-label baseline) | Next paper or rebuttal |

Dropped from P3 v2:
- X3 descriptive probes: only the all-view internal motion probe is kept, because the repair needs it;
- the X5 regression patterns: replaced by counterfactual activation differences in U5a.

Cut order if late:
1. DiTracker and the TrackCraft3R identity condition;
2. 4RC interchange and the dose subset;
3. U4b on VGGT;
4. MoRe;
5. the X2 capacity cell;
6. U5a on real data.

Never cut: U1, U2 on Any4D, VGGT and OpenD4RT, the factorial interaction test, or U4a.

## 1. Models, environments and assets (checked 2026-10-18)

| Model | Use | Source | Env | Notes |
|---|---|---|---|---|
| Any4D | main model for U2, U4a, U5 | on server (checkpoint airlabshare/any4d-checkpoint any4d_4v_combined.pth) | existing E5 venv | 24-block alternating attention, 12 heads of 64, 37x37 tokens per view at 518; PoseHead reads each view's final dense tokens (checked in any4d/models/any4d/model.py); one learned scale token |
| VGGT-1B + tracker_fixed | U2, U4b | on server | xm venv | per-frame camera token plus register tokens; camera head reads camera tokens, so camera interchange is position-free |
| OpenD4RT | U2 | HF Lijiaxin0111/OpenD4RT, 32CLIP checkpoint | new opend4rt | 256 px, even frame count; readouts r1 to r4 (P3) |
| TrackCraft3R | U2 object side | HF trackcraft3r/checkpoint | new trackcraft3r | 12-frame runs at 480x832 (pad); needs depth + W2C + intrinsics; patch out the T5 GPU transfer; GPU_NEED_GB 20 |
| 4RC | U2; U4 could | GitHub Luo-Yihang/4RC, HF Luo-Yihang/4RC (model.safetensors) | new 4rc (python 3.11, torch built for the server CUDA) | Arc.from_pretrained; load_images(size=512, patch_size=14); base geometry plus relative motion plus camera decoder (arc/models/arc/heads/cam_dec.py); the output keys are checked at gate G0g |
| MoRe | U5b baseline (should) | GitHub HellexF/MoRe; weights on Google Drive (more_full.pt) | new more | VGGT-based with attention forcing; MagiAttention needed only for the stream model, so use the full-attention model; needs gdown and Drive access from the server |
| CoTracker3, RAFT | lifted controls | on server | existing | |
| Wan2.1-T2V-1.3B | U3 features | HF Wan-AI/Wan2.1-T2V-1.3B(-Diffusers) | new p3-wan | P3 v2 section 4.1 |

Real data:
- the existing KITTI 2015 and Sintel (E5 protocols);
- new: HF datasets yslan/pointmap_regression_evalsets sintel.tar.gz, tum.tar.gz, bonn.tar.gz (MonST3R-format pose sets, not gated), plus davis.tar.gz for qualitative figures.

X2 training data: facebook/CoTracker3_Kubric, about 400 shards (P3 v2).

Disk on $BIG = /media/workbench/zone_x/e5_multibody:

| Item | Size |
|---|---|
| Factorial renders (510 scenes, 24 frames, 512 px) | about 25 GB |
| Pose sets | about 10 GB |
| Shards | about 120 GB |
| Checkpoints | about 50 GB |
| X2 features | about 40 GB |
| Interchange caches (temporary) | about 20 GB |

Check free space in U0.

## 2. Data

### 2.1 Factorial renders (U1)
- Generator: kubric_e5_scene.py, with two new flags:
  - --freeze_camera: camera held at its frame-0 pose for all frames;
  - --freeze_objects: every moving object held at its frame-0 pose, shadows included.
- Motion is scripted (no physics), so a freeze is a constant pose and nothing else changes.
- Per base scene, render A (both frozen), B (objects frozen), C (camera frozen). D is the existing render; if the existing D was rendered with older code, re-render it with the same seed.
- Frame-0 identity: after rendering, copy D's frame 0 RGB over A, B and C's frame 0. This is valid because the frame-0 scene state is identical. Log the PSNR between the rendered and the copied frame 0 as a diagnostic (expected to reflect sampling noise only).
- Base scenes:
  - v2 grounded: 100 scenes, seeds 401 and up, from ~/e5_multibody/data/v2/out100.
  - v3float: 70 scenes.
  - Total: 170 x 3 new renders = 510.
  - Optional dose subset: 40 v2 scenes with camera and object motion scaled by {0, 0.5, 1}. This needs a --motion_scale_cam and --motion_scale_obj flag, and gives 40 x 8 new renders.
- Splits, by base scene:
  - v2 even seeds = fit, v2 odd seeds = test;
  - v3float = transfer test;
  - real data = test only.
  - Every fitted quantity (top-k components, edit directions, probe thresholds, gains) uses fit scenes only.
- GT per variant (gt_factorial.py; reuses the E5 GT code):
  - camera poses and intrinsics;
  - per-pixel frame-0 3D points;
  - per-frame object poses;
  - moving-object masks per view;
  - for view-0 query points at target view t: x_true, x_static, A = x_true - x_static, B = x_static - x1, and the world displacement.
  - The E5 query sets (e5_xm_eval.py, rng = default_rng(scene index)) are reused unchanged for D, and the same pixel indices are used in A, B and C.

### 2.2 View set
Use the E5 round-2 setting for Any4D and VGGT: 7 views at frames 0, 4, 8, 12, 16, 20 and 23, with target view 23 for slopes. Check the exact list in e5_xm_eval.py before freezing. OpenD4RT uses all 24 frames. TrackCraft3R uses frames [0, 2, ..., 20, 23].

## 3. Code modules (repo src/attr/ and src/p3/; queue tasks u*_ and p3_*)

1. render_factorial.py: a wrapper that calls kubric_e5_scene.py with the freeze flags and seeds, does the frame-0 copy, and writes renders to $BIG/attr/factorial/<split>/<seed>/<variant>/.
2. gt_factorial.py: GT export as in 2.1.
3. Adapters in queue/src/e5_models.py:
   - OpenD4RTAdapter, TrackCraft3RAdapter and LiftedTrackerAdapter (P3 v2);
   - FourRCAdapter: query frame 0 at target t; returns pts3d_0 (base geometry), w_disp (relative motion), cam_pose and K;
   - MoReAdapter: cam_pose, plus motion outputs if --predict_motion provides them.
   - All return the shared dict plus w_disp.
4. Metrics:
   - attribution.py: slope, alpha and beta (w2_summary definitions and identifiability filter), s_w, interaction.
   - pose.py:
     - ATE after Sim(3) Umeyama on camera centres;
     - RPE-T and RPE-R over consecutive views;
     - for frozen-camera variants (A and C), spurious ego-motion = predicted rotation angle (deg) and translation norm divided by the median frame-0 depth, per view, against A.
   - budget.py: Shapley motion budget (section 5.3).
5. Hooks and interchange (Any4D; extends e5_causal.py, whose head-ablation hooks already work):
   - Cache per-head attention outputs (the input of the attention output projection, 12 x 64 per block) for every token of every view, plus the residual stream after each block.
   - Storage: about 0.35 GB per run in fp16, kept on the GPU per scene.
   - patch(component, positions, source_cache): replaces the component's output at the given token positions with the source cache during the base forward.
   - Key-region attention masking for selected heads uses an explicit attention computation for those heads only; all other heads keep the fused kernel. The mask is a boolean over keys, renormalised per query.
   - Inference is two-pass when the mask comes from the model's own probe. Pass 1 runs to layer L* and computes the mask; pass 2 runs the full model with masking.
6. VGGT hooks (xm venv): the same cache and patch operations over aggregator frame and global blocks. Positions include the per-frame camera token (index 0 of each frame's token list) and the register tokens. Check the token layout in the vggt aggregator code at G0h.
7. motion_probe_allviews.py: a logistic probe "token lies on a moving object at its own view".
   - Trained on Any4D (and VGGT) residuals at L* in {13, 15, 17, 19, 21}, chosen by CV on fit scenes.
   - Training data: D and B variants of fit scenes. B gives hard negatives with the same objects static.
   - Threshold: 90% recall on fit scenes.
   - Outputs: the probe weights and the threshold.
8. interchange.py: runs the U4 screens and joint patches; computes tau, leakage, kappa, nulls and completeness curves.
9. repair.py: U5a and U5b edits and baselines.
10. summary_attr.py: tables, Holm correction, figures F1 to F5; writes results/attr_preregistration.json and results/attr_summary.md.

## 4. Gates (U0; automatic, written to results/attr_gates.json; a failure blocks only dependants)

| Gate | Check | Pass |
|---|---|---|
| G0a to G0e | as in P3 v2 (Wan VAE and parity, TrackCraft3R 12-frame, OpenD4RT 24-frame, CoTracker3_Kubric contents) | as in P3 v2 |
| G0f | factorial renders: frame-0 copy done; rendered-vs-copied frame-0 PSNR at least 35 dB; frozen camera gives GT relative pose exactly identity; frozen objects give GT object displacement exactly 0 | all four |
| G0g | 4RC on 5 D scenes: output keys found; static EPE within 2x of Any4D | yes |
| G0h | VGGT token layout: the camera token index is confirmed by checking that patching it from D into D changes nothing | patched output equals base to 1e-5 |
| G0i | interchange identity: patching every component from D into D reproduces D exactly; patching from B into B likewise | max abs output diff at most 1e-4 (fp32) |
| G0j | MoRe weights downloadable and the full model runs | yes, else mark MoRe as dropped |
| G0k | pose sets unpacked; 4RC's relpose evaluator reproduces its reported Sintel ATE for 4RC itself within 10% | yes, else use our pose.py only |

## 5. Experiments

### U2. Behaviour (must; Oct 21 to Nov 1; about 1.5 GPU-days)

5.1 Runs:
- Any4D, VGGT, OpenD4RT and 4RC on all factorial variants A to D (510 + 170 scenes) and on the real data.
- TrackCraft3R conditions (i) GT inputs and (ii) DA3 inputs, on C and D, and condition (ii) on KITTI.
- Lifted CoTracker3 and lifted RAFT on C and D.
- MoRe on B, D and the pose sets.
- CoTracker3 on KITTI and Sintel (P3 X0c, framing item 3).

5.2 Object attribution:
- Per scene: alpha, beta (D only, identifiable scenes), image slope and s_w.
- In C, B = 0, so alpha equals the image slope, and s_w is computed the same way.
- Interaction statistic I = alpha_D - alpha_C, paired by base scene (Any4D primary). Since alpha_D needs identifiability and alpha_C does not, I uses the scenes identifiable in D. The analysis is repeated with s_w, which needs no identifiability filter.

5.3 Motion budget (Any4D and 4RC):
- For each moving query point at target view t: x_pred = pi(K_t R_t^T (P0 + F_t - T_t)), with the model's P0, F_t and camera (R_t, T_t, K_t).
- GT is expressed in model scale by the median depth ratio on static view-0 points.
- Deficit along the object-motion direction: d = (x_pred - x_true) . A_hat, with A_hat = A / |A|.
- For components c in {geometry P0, flow F, camera (R, T, K)}, the Shapley value phi_c averages, over the 6 orders, the change in d when c is replaced by its GT.
- Report the scene-median of phi_c / sum(phi) on D and C.
- Predictions: account R makes phi_F dominant; account K makes phi_camera large in D and near 0 in C; account G makes phi_P0 large.

5.4 Contamination and spurious ego-motion:
- On factorial scenes: dRPE-R = RPE-R(D) - RPE-R(B) and dATE, paired by base scene.
- Spurious ego-motion in C against A, which controls for any baseline jitter of the model on a static video.
- On real pose sets:
  - Run windows of 8 consecutive frames at each set's standard stride (Sintel 14-sequence subset; TUM-dynamics first 90 frames, stride 3; Bonn as in the MonST3R protocol), with Sim(3) alignment per window.
  - Report ATE, RPE-T and RPE-R. These are the same windows for every model and edit, so comparisons within a model are fair.
  - If dynamic masks exist in the tarballs, stratify windows by dynamic-pixel fraction; otherwise use the U5b internal-mask fraction.

5.5 Readout contrasts (P3 X1b) for OpenD4RT (r1 to r4) and Any4D (scene-flow route vs per-frame pointmap route), on D and C.

### U3. Cause, controlled (must, with a decision point; P3 X2 unchanged)
- {P-abs, P-res} x {L3D, L2D} decoders on frozen Wan2.1-1.3B final-block features (t = 5, clean latents), trained on CoTracker3_Kubric windows, 2 seeds each, plus an optional capacity cell. Details are in P3 v2 sections 4.4 and X2.
- Addition: evaluate every cell also on factorial C and D, to get the interaction I for the controlled models. Account R predicts P-res has a lower alpha in both C and D, with I near 0.
- Decision point Oct 27: if the trainer is not reproducing static EPE at most 6 px on held-out windows by Oct 27, U3 is dropped. The paper's cause claim then rests on U2 (interaction, budget, 4RC and OpenD4RT readouts) and is worded "consistent with".

### U4a. Pathway maps in Any4D (must; code Oct 21 to 27; runs Oct 27 to Nov 1; about 0.7 GPU-day)

Screens on fit scenes (about 50 v2 even-seed scenes). Components are the 288 attention heads (24 blocks x 12) and the 24 residual-stream states. Each screen is one forward per component per scene.

1. Object screen, base D and source B, at view-0 tokens (pixel-aligned by frame-0 identity):
   - tau_O(k) = <F(y_k) - F(D), F(B) - F(D)> / |F(B) - F(D)|^2, with F = scene flow at moving-object query points (view 0 to view 23, model scale);
   - camera leakage lambda_O(k) = |pose(y_k) - pose(D)| / |pose(C) - pose(D)|, with pose = the stacked 6D rotation and normalised translation of views 1 to 6.
2. Camera screen, base D and source C:
   - positions: view-0 tokens plus the scale token (aligned);
   - tau_C(k) uses the poses;
   - flow leakage lambda_C(k) = |F(y_k) - F(D)| / |F(B) - F(D)|.
   - Secondary: view-t tokens aligned by GT-warping C's activations into D's view-t grid through the static-scene reprojection (nearest token). The misregistration floor is the same warp applied from D to D (round trip). Report only effects above that floor.
3. Contamination screen, base D, per head h:
   - block attention from all queries in views 1 to 6 to keys on moving objects (GT masks at the token grid, any view), then renormalise;
   - kappa(h) = (e(D) - e(D | h)) / (e(D) - e(B)), with e = RPE-R + RPE-T (normalised);
   - also report the change in F, which should be small.
4. Necessity: resample-ablate (source = a random other fit scene) the top components of each screen, as in E5's resample protocol. Report the effect on F and on pose.
5. Nulls: 40 random component sets matched by block, type (frame vs global block), and activation norm, for every top-k test.
6. Joint patching on test scenes: P_obj = top-k by tau_O, P_cam = top-k by tau_C, P_cont = top-k by kappa, with k in {1, 3, 5, 10, 20}. k* is chosen on fit scenes as the smallest k reaching 80% of the full-model effect (completeness). Report tau, leakage and kappa at k* on test scenes, with null percentiles.
7. Consistency with E5: report where the five supply heads (b2h1, b22h9, b17h10, b14h6, b8h11) rank in tau_O. Prediction H3: all five rank in the top 20.

Secondary distributed analysis (supplement only):
- Projection patching on the span of the top-2 principal directions of natural activation differences (D - B for object, D - C for camera), per head, estimated on fit scenes.
- Report the "dormant" check: the variance of natural test-scene activations along the span, relative to the head's total variance. Only spans with at least 5% natural variance count (Makelov et al.).

### U4b. Pathway maps in VGGT (should; Oct 29 to Nov 3; about 0.7 GPU-day)
- Same screens. The camera screen patches the per-frame camera and register tokens, which are position-free and therefore clean. This is the cross-architecture test of H1.
- Correspondence output: tracker_fixed tracks (E5). Camera: VGGT camera head.
- To fit time, restrict the screen to global blocks plus the camera-token positions in frame blocks, on 30 fit scenes.

### U5a. Object-gain edit (must; Oct 29 to Nov 3; about 0.3 GPU-day)
- Directions per head in P_obj ∪ {the five E5 heads}:
  - U_O = the top-2 principal directions of h(D) - h(B) on moving view-0 tokens of fit scenes;
  - U_C = the top-2 principal directions of h(D) - h(C) on all view-0 tokens.
  - Orthogonalise U_O against U_C, giving projector P_O; P_C is built symmetrically.
- Edit: h to h + (g - 1) P (h - m), with m = the mean head output over static tokens in fit scenes, and g in {1.5, 2} fixed in advance. Variants:
  - (i) P = P_O;
  - (ii) P = P_C;
  - (iii) 20 random 2D projectors of matched norm;
  - (iv) whole-head gain (E5);
  - (v) the E5 five-head LoRA checkpoint (existing).
- Evaluation: test v2, v3float, KITTI, Sintel; alpha, beta, s_w, moving and static EPE.
- Test T5' (section 7). Success is a double dissociation: (i) gives alpha up by at least 0.10, beta moves by at most 0.03, moving EPE by at most +1 px, static EPE by at most +5%; (ii) moves beta by at least 0.10 with alpha moving by at most 0.03.

### U5b. Contamination masking (must on Any4D; should on VGGT; Oct 30 to Nov 4; about 0.5 GPU-day)
- Mask: the model's own all-view motion probe at L*, thresholded on fit scenes (two-pass inference). No GT at test time.
- Edits:
  - (a) block attention from all queries to probe-flagged keys in P_cont only (k* heads);
  - (b) the same in all heads;
  - (c) the same in k* random heads (20 draws, matched block distribution);
  - (d) GT mask in all heads (oracle attention forcing, "MoRe-like");
  - (e) the MoRe model itself (if G0j passes).
- Evaluation:
  - Kubric D test scenes: pose error, and its recovery toward B, = (e(D) - e(edit)) / (e(D) - e(B));
  - Sintel, TUM-dynamics and Bonn: ATE, RPE-T, RPE-R;
  - object flow on Kubric test (alpha, moving EPE) and static EPE, to show specificity.
  - Also report the probe-mask IoU against GT masks on Kubric, and against dynamic masks on Sintel if available.
- Test T10 (section 7). Success: (a) beats (c) on real pose RPE-R (one-sided, paired over windows), reaches at least 50% of the oracle (d) improvement, and costs at most +1 px moving EPE.
- If contamination is negligible in Any4D (the D minus B pose error CI includes 0 on Kubric and real data), U5b moves to VGGT. A null contamination result is reported as support for account R.

## 6. Metric definitions (frozen in results/attr_preregistration.json)
- alpha, beta and the identifiability filter: w2_summary (condition number of [A B] at most 10; median |A| and |B| at least 2 px).
- s_w: through-origin slope of predicted on GT world displacement over moving query points, after per-scene median-depth scale alignment. Static residual: median |w_disp| on static points / median |GT disp| on moving points.
- Pose: ATE (Sim(3) Umeyama on camera centres), RPE-T, RPE-R (consecutive views). Spurious ego-motion: rotation angle and translation over median depth, for frozen-camera variants.
- tau, lambda and kappa: section U4a. Completeness: tau or kappa of the top-k joint patch divided by the full-model effect.
- Boundary rule: exclude points within 8 px (at 512) of moving-object boundaries in every Kubric metric; report with and without.

## 7. Statistics and pre-registration
- Confirmatory Holm family (7 tests), all directional or explicitly two-sided:
  - T1': s_w < 0.9 for OpenD4RT and TrackCraft3R (ii) (P3), one-sided scene bootstrap.
  - T3': X2 parameterisation effect on the pull index (P3), two-sided, hierarchical bootstrap over seeds then scenes.
  - T5': U5a (i) vs (iii) on the alpha change, one-sided, paired over test scenes.
  - T7: factorial interaction I = alpha_D - alpha_C for Any4D, two-sided paired scene bootstrap.
    - |I| at least 0.10 is an effect (a negative I supports account K);
    - |I| below 0.05 with a CI including 0 is "no interaction" (supports account R);
    - in between is inconclusive.
  - T8: contamination dRPE-R(D - B) > 0 for Any4D, one-sided paired bootstrap; VGGT reported alongside.
  - T9: H1 double dissociation at k*. P_obj tau_O and P_cam tau_C both exceed the 95th percentile of their matched nulls, and both leakages are at most 0.2. The p-value is the rank in the null, combined as the maximum of the two.
  - T10: U5b (a) vs (c) on real-data RPE-R, one-sided, paired over windows.
- Everything else is descriptive, with 95% CIs (2000 bootstrap resamples).
- The unit is the base scene (Kubric) or the window (real pose). Fitted quantities come from fit scenes only.
- Thresholds, k grid, g, L* candidates, view sets and the boundary rule are frozen after the smoke runs and before any test-split result. This happens on Oct 27, together with the P3 thresholds.

## 8. Schedule

| Dates | Track A: data and behaviour | Track B: cause (U3) | Track C: mechanism and repair | Decision |
|---|---|---|---|---|
| Oct 19 to 20 | envs; downloads (pose sets, 4RC, OpenD4RT, TrackCraft3R, MoRe, shards in background); freeze flags; first renders; X0c starts | p3-wan env; Wan parity | Any4D cache and patch code; G0i | G0f on 10 scenes |
| Oct 20 to 23 | all 510 renders (CPU, background); gt_factorial; adapters; attribution, pose, budget code; U2 smoke | decoder, trainer, loader | key-region masking; motion probe; VGGT layout (G0h) | gates G0a to G0k |
| Oct 23 to 27 | U2 runs on Any4D, VGGT, OpenD4RT, 4RC as renders land | feature extraction; X2 pilot | U4a smoke on 5 fit scenes; U5 code | Oct 27: engineering freeze, pre-registration, U3 go/no-go |
| Oct 27 to Nov 1 | U2 remaining rows (TrackCraft3R, lifted, MoRe, real pose) | X2 main (8 runs) | U4a screens and joint patches | T1', T7, T8 readouts |
| Nov 1 to 4 | | X2 eval incl. factorial | U4b (VGGT); U5a; U5b | T3', T5', T9, T10 |
| Nov 4 to 6 | summary_attr; figures F1 to F5 | | reruns for bugs only | Nov 6: results freeze |
| Nov 6 to 10 | draft (new spine) | | | Nov 10: register |
| Nov 10 to 16 | writing, internal review, supplement | | | Nov 16: submit |

The three tracks are independent, so they can run as three parallel sub-agents, as in E5 round 2. Each writes and smoke-tests its own code, enqueues tasks and stops. In line with the user's preference, nothing polls the queue; the user says when runs finish.

GPU estimate:

| Item | GPU-days |
|---|---|
| U2 (TrackCraft3R about 0.5 of this) | 1.5 |
| U3 | 1.5 |
| U4a | 0.7 |
| U4b | 0.7 |
| U5 | 0.8 |
| Total | about 5.2 |

That is about 3.5 days of wall time at 1.5 effective GPUs, which fits with margin for OOM back-offs. Rendering is CPU-only. Measure time per scene on the first 10 renders; at 5 minutes per scene with 8 parallel workers, 510 renders take about 5 hours.

## 9. Queue tasks (enqueue.py; dependencies on gate files)
- Setup and gates: u0_env_<name>, u0_dl_<asset>, u0_gate_<g>.
- Renders and GT: u1_render_<split>_<variant>_<chunk>, u1_gt_<split>.
- Behaviour: u2_<model>_<split>_<variant>, u2_real_<model>_<set>, u2_budget_<model>, u2_score.
- Cause: p3_feat_wan, p3_x2_pilot, p3_x2_<param>_<loss>_s<seed>, p3_x2_eval (with factorial C and D).
- Mechanism: u4a_screen_<obj|cam|cont>_<chunk>, u4a_null, u4a_joint, u4b_screen_*, u4b_joint.
- Repair and summary: u5_probe_<model>, u5a_edit_<variant>_g<g>, u5b_mask_<variant>_<set>, u_summary.
- TrackCraft3R tasks set GPU_NEED_GB = 20; others keep 16.
- Results go to $BIG/attr/results/<task> and are pushed to GitHub by push.sh. The dashboard gets an "Attribution" section.

## 10. Paper map (8 pages)

| Section | Content | Experiments |
|---|---|---|
| 1 Intro, Fig. 1 | the factorial 2x2 with the predicted vs true motion budget on one scene; alpha in C and D; contamination | U1, U2 |
| 3 Setup | MotionFactorial, metrics, models (three reconstruction families, trackers) | U1, E5 data |
| 4 Behaviour | cross-family world slope and image alpha/beta (E5 C2 numbers carried); contamination; budget | U2, E5 |
| 5 Cause | interaction T7; budget; readout contrasts; X2 parameterisation | U2, U3 |
| 6 Mechanism | interchange maps, double dissociation (Any4D, VGGT); five heads inside P_obj; contamination heads; internal motion signal exists early (E5 probes with controls) | U4, E5 |
| 7 Repair | object-gain edit vs whole-head, random and LoRA; contamination masking vs random, oracle and MoRe; real pose and flow | U5, E5 LoRA |
| Supplement | E5 view tests and size test, C3 grouping, LEACE and steering (non-specific), dose subset, warp-aligned camera interchange, distributed-subspace analysis | |

## 11. Risks and fallbacks
1. The frame-0 copy hides a render difference. G0f checks it, and the GT-identity checks catch freeze bugs.
2. Interchange on Any4D view-t tokens is misregistered. The primary analyses use aligned positions only, with the measured floor for the secondary analysis.
3. Contamination may be negligible in Any4D. Then U5b moves to VGGT, and the null result supports account R.
4. Distributed pathways: no small k reaches 80% completeness. Report the completeness curves and drop T9 to descriptive. Behaviour and cause carry the paper.
5. The internal probe mask may transfer poorly to real data. Report its IoU where masks exist. If it fails, show U5b on Kubric only and the oracle on real data, and say so.
6. MoRe or 4RC fail their gates. They are dropped from tables. The oracle mask stands in for MoRe.
7. Engineering slips past Oct 27. Apply the cut order in section 0. U3 has its own go/no-go.
8. Shared GPUs. pick_gpu waits for free memory; the supervisor's OOM back-off is in place.

## 12. Facts checked for this plan (2026-10-18)
- Any4D PoseHead reads the last dense features of each view (any4d/models/any4d/model.py, the pose_head call around line 1649); the model has a single learned scale token.
- 4RC: inference code and weights released (HF Luo-Yihang/4RC, plus 4RC_geofinetune); the camera decoder is at arc/models/arc/heads/cam_dec.py.
- MoRe: inference and weights released, with weights on Google Drive (README); built on VGGT.
- SM4RT: checkpoint on HF shjlin/sm4rt (CC BY-NC 4.0); not used in the must tier.
- Pose sets: HF datasets yslan/pointmap_regression_evalsets holds sintel, tum, bonn and davis tarballs, and is not gated.
- Methodology papers (arXiv 2301.04709, 2303.02536, 2309.16042, 2311.17030) and positioning papers (2512.11508, 2602.07050, 2603.05078, 2602.10094, 2512.10935, 2607.22534, 2609.32596, 2512.19949): records retrieved.
- Still to check during U0:
  - the VGGT camera-token index;
  - the 4RC output keys;
  - whether the MoRe Drive download works;
  - whether the pose tarballs include dynamic masks;
  - the E5 view list in e5_xm_eval.py;
  - render time per scene.
