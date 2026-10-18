# P3 plan (version 2): is the camera-motion pull shrinkage toward a static world, across 4D model families?

Revised 2026-10-18 after one independent review of version 1 (log in section 11). Target: CVPR 2027. Registration Nov 10 AoE and submission Nov 16 AoE are fixed. This plan extends the E5 paper ("Seen but Under-Reported"); it is not a separate paper. Execution runs on the shared 4090 + A5000 server (10.0.117.206) through the E5 queue, after the user's go-ahead.

Two hard dates drive the plan. The engineering freeze is Oct 27: all code is written and tested by then. The results freeze is Nov 6.

## 0. The question and what it adds

E5 shows that Any4D and VGGT keep the camera-induced part of image motion (beta about 1) and recover only part of the object's own motion (alpha 0.61 and 0.22). CoTracker3 (0.96) and chained RAFT (0.88) do not show this. P3 asks why, and whether it is general.

The working hypothesis comes out of the review. Any4D's scene flow is allocentric: a world-frame displacement whose zero means "static in the world". TrackCraft3R predicts a residual from the static frame-0 pointmap. Under such an output parameterisation, ordinary regression shrinkage of the residual toward zero produces exactly the observed signature: beta near 1 and alpha below 1. Image-space trackers have no static-world zero to shrink toward. They also use correlation volumes and iterative refinement.

This gives a sharper headline than v1: "the pull is shrinkage toward a static world, present across 4D reconstruction families and absent in image-space trackers". If the tests reject it, we report the opposite (section 9).

It is consistent with E5's dose-response result, where the slope rises with object area and falls with occlusion: less evidence for the object's own motion means more pull toward the prior. It sits in tension with E5's null effect of texture and confidence. The paper must say this, and X2 adjudicates.

New claims, if supported:
- C6 (generality): on one world-frame axis, three reconstruction families show the pull: GFM-based (Any4D, VGGT), D4RT-style (OpenD4RT) and video-generator-based (TrackCraft3R). Image-space trackers lifted to the same axis do not.
- C7 (cause): in a controlled model system on frozen Wan features, a static-residual output parameterisation produces the pull and a direct absolute parameterisation reduces it, with loss space held fixed and then varied.
- C8 (representation vs readout): object motion is linearly recoverable from encoder tokens about as well as camera-induced motion (R^2 and calibration), so the deficit enters after the encoder. Within one model, query readouts that differ only in parameterisation differ in alpha.
- C9 (mechanism and edit): in the five Any4D supply heads, editing the encoding pattern of object motion orthogonal to camera motion moves alpha and not beta. The camera-motion edit does the reverse (a double dissociation), without the EPE cost of whole-head amplification.

Framing-list items: item 3 (CoTracker3 on KITTI and Sintel) is X0c. Item 4 (one more correspondence-capable 4D model) is covered by OpenD4RT and TrackCraft3R in X1. Item 2 (appearance swap) is cut before the deadline: the v1 design was confounded by correspondence failure, and a fixed design is kept for the rebuttal. Item 1 (pseudo-label self-training) moves to the P2 phase, where it is P2's main baseline.

## 1. Accounts and predictions

Notation follows w2_summary. On moving points, x_pred - x1 = alpha A + beta B, with A = x_true - x_static and B = x_static - x1. The world-frame object-motion slope s_w is the through-origin regression slope of predicted world displacement on GT world displacement for moving points, both expressed in frame-0 camera coordinates.

| Account | Statement | Key predictions |
|---|---|---|
| R (residual shrinkage) | The output is a static hypothesis plus a residual, and the residual is shrunk toward zero. | s_w < 1 for all reconstruction models on the world axis; lifted trackers near 1. In X2, P-res has a larger pull index (beta - alpha) than P-abs at matched static EPE. Within OpenD4RT, world-frame readouts have lower alpha than camera-frame absolute readouts. |
| F (features) | Encoder tokens under-represent the object's own motion. | X3: R^2 for A clearly below R^2 for B at the encoder output, at matched SNR. In X2, both parameterisations show a similar pull. |
| L (loss space) | 3D losses dominated by depth error under-weight lateral object motion. | X2: the L3D vs L2D supervision effect exceeds the parameterisation effect. |
| H4b (under-supply, E5 reading) | A few heads carry object motion at low gain, separately from camera motion. | X5: the A-pattern edit moves alpha only and the B-pattern edit moves beta only. Pattern subspaces are near-orthogonal. |
| H4a (overlap) | Object motion is partly carried along camera-motion directions. | X5: the A edit also moves beta. Pattern subspaces have small principal angles. |

The accounts are not exclusive. X2 estimates the parameterisation and loss-space effects and their interaction.

## 2. Models and assets (verified on 2026-10-18 unless marked)

| Model | Role | Output used | Weights and facts |
|---|---|---|---|
| Any4D | GFM-style reference | allocentric scene flow; pointmaps | on server; internals and five heads known |
| VGGT-1B + VGGT_tracker_fixed | GFM reference | 2D tracks, lifted with VGGT's own depth and cameras | on server |
| OpenD4RT | D4RT family, new | uv head; 3D at any (t_tgt, t_cam) | public: OpenD4RT_32CLIP_9Dataset_NoAUG (primary) and 48CLIP_9Mix_NoCropAUG (check), about 14 GB each; ViT-g, 256x256, tubelet 2 (even frame count), clips shorter than the training length run as given |
| TrackCraft3R | video-generator family, new | frame-0-camera-space tracks P0(t), a residual from the input frame-0 pointmap | HF trackcraft3r/checkpoint (17.8 GB, includes umT5 and three VAEs). 12 frames per run, each frame VAE-encoded separately, default 480x832 (stretch or pad), needs depth + W2C extrinsics + intrinsics; queries are frame-0 pixels only |
| CoTracker3, RAFT | image-space trackers | 2D tracks; lifted rows use GT depth and cameras | on server |
| Wan2.1-T2V-1.3B | frozen features for X2 and X3 | final block and per-layer tokens | HF Wan-AI/Wan2.1-T2V-1.3B and the -Diffusers repo; 30 blocks, dim 1536, 12 heads of 128, patch (1,2,2), VAE stride (4,8,8) |
| DiTracker (optional) | video-DiT-feature tracker | 2D tracks | repo weights; CogVideoX-2B layer 17 head 2 plus ResNet features; 480x720; VAE sampling is stochastic, so fix the seed |
| Gen4P | would be ideal | | code not released; add only if released before Oct 27 |

Dropped from v1: VideoMAEv2-L (confounded by resolution, token count, capacity and position extrapolation), DVD as a must row (optional only; its files are a full DiT plus rank-512 LoRA and are read at t = 500), MapAnything and PAGE-4D in X3.

## 3. Data
- Evaluation, unchanged and held out:
  - Kubric v2 (100), v3float (70), v4 epi_accel and epi_const. These are 512x512, 24 frames, 12 fps, GT 3D available.
  - KITTI 2015 (200 pairs plus multiview frames). GT world displacement uses the ego-motion machinery already in e5_realdata_gt.py; check that it provides world displacement, not only image flow.
  - Sintel (217 samples).
- X2 training: facebook/CoTracker3_Kubric (MOVi-F). One shard was inspected. It holds 120 frames at 512x512 with 32768 2D tracks, per-track depth, visibility, uint16 depth with its range, segmentation and full camera (positions, quaternions, focal length, sensor width). 3D tracks therefore follow from the tracks, per-track depth and intrinsics. The RGB format was not reached in the inspected part, so it is checked in gate G0e.
  - Use about 400 shards (about 120 GB) and take 25-frame windows at strides 1 and 2, giving about 1600 to 2000 clips.
  - E5-generator scenes are never used for training. Training on CoTracker3's own data also means X2 decoders and CoTracker3 share training data, which controls the data account.
- One boundary rule for every row: exclude points within 8 px (at 512) of motion boundaries computed from GT segmentation. Kubric and real-data tables are reported with and without it (E5 review W7).

## 4. Shared implementation (src/p3/, queue tasks p3_*)

Outputs go under $BIG/p3/{models,venvs,data,feat,runs,results}, with $BIG = /media/workbench/zone_x/e5_multibody. Check free space before downloading: about 120 GB of shards, about 35 GB of checkpoints, about 40 GB of X2 features and about 30 GB of X3 internals.

1. wan_feats.py (X2, X3)
   - Load the diffusers WanTransformer3DModel and AutoencoderKLWan.
   - Latents: latent_dist.mode(), then (z - latents_mean) / latents_std from the VAE config.
   - Readout condition: clean latents, timestep t = 5 (TrackCraft3R's convention: the last step of a 1000-step shift-5 schedule, t = 4.98), and the empty-prompt context computed once with umT5 and cached. T5 is never loaded again.
   - Sensitivity check on 20 scenes at t in {0, 92.5, 500}.
   - Parity against the official Wan2.1 code on 2 clips: per-block relative error below 1e-2 and cosine above 0.999, in fp32 and bf16.
   - Hooks:
     - residual stream after the patch embedding and after each block (31 states);
     - per-head outputs through a forward pre-hook on attn1.to_out[0], reshaped to 12 x 128;
     - post-RoPE q and k only through a custom WanAttnProcessor that stores argmax and pmax maps from latent-0 queries to latent-6 keys. Verify on one clip that the recomputed attention output equals the hooked to_out input.
   - Frames: E5 clips have 24 frames, so frame 23 is repeated to make 25. Latent k holds frames 4k-3 to 4k, so frame 23 is in latent 6. X3 labels for latent 6 use the chunk-average displacement over frames 21 to 23 as primary, with frame-21 and frame-23 labels as sensitivity. A per-frame VAE encoding variant is run as a further sensitivity check only, because base Wan has never seen per-frame latents.
   - Resolution: 480x480, a 30x30 token grid, 16 px per token at 480.
2. Adapters in queue/src/e5_models.py, all returning the existing dict plus a new key w_disp (V,h,w,3): world displacement from frame 0, in frame-0 camera coordinates.
   - OpenD4RTAdapter, 24 frames at 256 px. Readouts for X1b:
     - (r1) uv head at t_tgt;
     - (r2) 3D with t_cam = t_tgt, projected with predicted intrinsics and with GT intrinsics;
     - (r3) world displacement = D(u,v,0,t,0) - D(u,v,0,0,0);
     - (r4) r3 moved into camera t with GT cameras and projected.
   - TrackCraft3RAdapter, 12-frame runs at 480x832 with --resize_mode pad (no stretching of 512x512):
     - Kubric uses frames [0,2,4,...,20,23], so frame 23 is the target.
     - KITTI uses frames 10 to 20 plus a repeat of frame 20, with target 11.
     - Sintel uses frames i to i+11, with target i+1.
     - Inputs via build_user_npz: z-depth, frame-0-normalised W2C extrinsics, and fx, fy, cx, cy. Three input conditions: (i) GT depth + GT cameras; (ii) DA3 depth + cameras, the paper setting; (iii) GT depth + identity cameras, exploratory cue conflict.
     - w_disp = P0(t) - P0(0). Patch the predictor to use the cached empty context and skip the T5 GPU transfer (about 11 GB transient). Queue setting: GPU_NEED_GB = 20, one process per GPU. CPU RAM is about 25 to 30 GB per process, so check server RAM in X0.
   - LiftedTrackerAdapter: 2D tracks plus depth and cameras give w_disp. GT depth and cameras for CoTracker3 and RAFT. VGGT's own depth and cameras for the VGGT tracker.
3. world_slope.py: scores s_w and the static residual (median |w_disp| on static points divided by median |GT disp| on moving points).
   - Kubric scale is aligned per scene by the median depth ratio on static points; Any4D's Kubric units differ by about 3.3x. KITTI and Sintel use metric or median scaling.
   - The identifiability filter of w2_summary applies to image-space alpha and beta. s_w needs only |GT disp| of at least 2 px-equivalent.
   - CIs are scene bootstrap with 2000 resamples.
4. p3_decoder.py and p3_train.py (X2)
   - Features: Wan final-block tokens, projected Linear(1536 to 512) + LayerNorm, plus a Fourier embedding of each token's (t, y, x). RoPE acts only inside attention, so the residual stream may lack absolute position.
   - Query: Fourier(u, v) + an MLP on the 9x9 RGB patch from the 512 px frame + Gen4P continuous time embeddings for t_src, t_tgt and t_cam.
   - Body: 4 cross-attention blocks, width 512, 8 heads, independent queries, about 15 to 20M parameters.
   - Parameterisations:
     - P-abs: head_X(z_q) gives X directly, as in D4RT.
     - P-res: X = S + head_d(z_q), where S = head_X(z_q') is the same decoder queried at q' = (u, v, t_src, t_tgt := t_src, t_cam), i.e. the static hypothesis "where this point would be in camera t_cam if it had not moved". There is no extra supervision. Only the composition of the output changes, and S queries are already part of the normal training distribution.
   - Supervision:
     - L3D is the D4RT normalised sign-log L1 on X, with confidence.
     - L2D is a Huber loss on the projection of X with GT intrinsics, in normalised image coordinates.
     - Both have weight 1.0 on the read-out quantity, plus identical auxiliaries: visibility 0.1 and confidence 0.2.
   - Query sampler: 2048 queries per clip, 30% near edges, t_cam = t_tgt with probability 0.4.
   - Optimisation: AdamW, lr 1e-4, 1k warmup, cosine to 1e-6, 20k steps, batch 8, bf16. Features are pre-extracted, and the data loader is the bottleneck (about 155 MB per step), so put features on the fastest local disk.
5. p3_probes.py (X3), p3_edit.py (X5) and p3_summary.py (tables, Holm, figures). Conventions follow e5_internal_analysis.py: frame-0 tokens only, 5-fold CV grouped by scene, and fold resampling (not "seeds") for deterministic ridge probes.

## 5. Experiments (must set first)

### X0. Setup and gates (must; Oct 19 to 22; about 0.3 GPU-day)

Environments: p3-wan (diffusers with Wan), trackcraft3r, da3 (pins from the TrackCraft3R README), opend4rt. Downloads start on day 1, in the background.

Gates are automatic and written to results/p3_gates.json. A failed gate blocks only its dependants:

| Gate | Check | Pass criterion |
|---|---|---|
| G0a | Wan VAE round trip on 10 clips, and diffusers vs official parity | PSNR of at least 30 dB; parity passes |
| G0b | TrackCraft3R condition (i), 12-frame protocol, on 10 v2 smoke scenes | median static EPE at most 10 px at 512. Fallback: the repo's frame_stride options |
| G0d | OpenD4RT at 24 frames, 32CLIP checkpoint | static EPE within 2x of Any4D on the same smoke scenes |
| G0e | CoTracker3_Kubric: RGB frames present and readable; 3D tracks from tracks + depth + intrinsics | reprojection error of the lifted tracks at most 0.5 px |

### X0c. CoTracker3 on KITTI and Sintel (must; framing item 3; Oct 19 to 21; about 0.3 GPU-day)
The existing adapter is run on the E5 real-data protocols, both 2 views and the multiview window. This can start on day 1 because no new code is needed.

### X1 and X1b. Cross-family behaviour on one world-frame axis (must; Oct 21 to 27; about 1 GPU-day)
- Rows: Any4D, VGGT tracker (lifted), OpenD4RT (readouts r1 to r4), TrackCraft3R (i) and (ii) (and (iii) as exploratory), lifted CoTracker3 and lifted RAFT (GT depth and cameras). DiTracker is optional.
- Data: v2, v3float, v4 epi_accel and epi_const, KITTI, Sintel. TrackCraft3R condition (ii) on KITTI is required, since it is the new model in its paper setting on real data.
- Metrics:
  - primary: s_w and the static residual;
  - secondary: image-space slope, alpha, beta, sep_ratio, static and moving EPE (existing definitions).
  - For TrackCraft3R, image-space beta is set by the input cameras. It is reported descriptively and labelled as input-determined.
- X1b readout contrasts, inference only:
  - OpenD4RT: alpha for r1, r2, r3 and r4 on the same points.
  - Any4D: scene-flow correspondence vs the per-frame pointmap route (pointmap at view t matched through GT, which isolates geometry).
  - Account R predicts that the world-frame readouts (r3, r4) have lower alpha than the camera-frame absolute readouts (r1, r2).
- Confirmatory test T1': s_w < 0.9 for OpenD4RT and for TrackCraft3R (ii), one-sided scene bootstrap on v2 plus v3float. The lifted trackers must give s_w of at least 0.9 for the world axis to count as valid.

### X2. Output parameterisation x loss space on frozen Wan features (must; code Oct 22 to 27; runs Oct 27 to Nov 3; about 2 GPU-days)
- Cells: {P-abs, P-res} x {L3D, L2D}, 2 seeds each, plus one capacity check (P-abs, L3D at width 1024). That is 9 runs at about 3 to 4 GPU-hours each.
- Pilot first: one P-abs L3D run calibrates gate G2. Thresholds are written to results/p3_preregistration.json before any main run starts.
- G2: every cell reaches static EPE within 10% of the best cell on held-out MOVi-F windows. A lagging cell may train up to 1.5x as many steps, and this is recorded.
- Evaluation: E5 Kubric splits, querying frame-0 points with t_tgt = 23, read out through the 2D projection of X. Endpoints: pull index PI = beta - alpha (primary), with alpha, beta and s_w all reported.
- Confirmatory test T3': the parameterisation main effect on PI (P-res minus P-abs).
  - Statistics: hierarchical bootstrap (resample seeds within cell, then scenes), plus sign consistency across all seed pairings.
  - Two-sided.
  - Reading: an effect of at least 0.10 is a finding. 0.05 to 0.10 is reported as inconclusive. Below 0.05 with a CI including 0 counts as no effect.
- The loss-space main effect and the interaction are reported with the same statistics, but are not in the Holm family.
- Kubric-trained decoders are not evaluated on real data. X2 is a controlled model system.

### X3. Decodability of object vs camera motion (must, reduced; Oct 27 to Nov 1; about 0.5 GPU-day)
- Models: Any4D (stored internals; add A and B labels), base Wan (t = 5), random-init Wan (official WanModel init) as a control. VGGT is optional, from stored internals.
- Per layer:
  - cross-validated R^2 for A, B and D = A + B, with lambda chosen by nested CV per target;
  - the calibration slope after inner-fold calibration;
  - an SNR-matched comparison (subsample patches so that var(A) = var(B));
  - the same identifiability filter as the output metric (condition number of [A B] at most 10, median |A| and |B| at least 2 px).
- Controls: the layer-0 input, raw VAE latents for Wan (latent k already carries temporal differences through the causal VAE), label shuffle and fold resampling.
- Reading for C8: at the encoder output, R^2(A) is at least 0.8 x R^2(B) at matched SNR, and the output alpha is below 0.8. That places the deficit in the readout, consistent with E5's path patching. Otherwise the result supports account F, and we report the layer where R^2(A) falls.
- Descriptive only.

### X5. Directional edits with encoding patterns (should; code Oct 24 to 27; runs Oct 29 to Nov 3; about 1 GPU-day)
- Patterns: for each of the five Any4D heads (b2h1, b22h9, b17h10, b14h6, b8h11), regress head outputs on [A, B] at view-0 tokens (scene-grouped CV on v2 training folds). The A and B columns give 2-dimensional pattern subspaces (encoding patterns, not decoder weights; Haufe et al. 2014). Principal angles between them are compared with 40 random matched heads.
- Check first which tokens and views the E5 ablations edited in e5_causal.py, and match them.
- Edits: h to h + (g - 1) P h, with g in {1.5, 2} and P being one of:
  - (i) the A pattern orthogonalised against B;
  - (ii) the B pattern;
  - (iii) 20 random 2-dimensional projectors of matched norm;
  - (iv) the identity (the E5 whole-head edit).
- Projectors are fit on training folds and tested on held-out v2 folds, v3float, KITTI and Sintel, with g fixed in advance.
- Confirmatory test T5: (i) vs (iii) on the alpha change, paired over scenes, one-sided.
- Success for C9 (double dissociation): (i) raises alpha by at least 0.10, beta moves by at most 0.03, moving EPE changes by at most +1 px, static EPE by at most +5%; and (ii) moves beta by at least 0.10 with alpha moving by at most 0.03.
- If (i) behaves like (iii) or (iv), C9 is dropped and reported next to E5 item 5.

### Cut before the deadline
- X6 camera-routed readout: a new architecture, and account R predicts it would reproduce the pull. Its question is answered more cleanly by P-res vs P-abs.
- X4 on Wan heads, X3 on TrackCraft3R, MapAnything and PAGE-4D, the VideoMAE arm, and DVD as a must row.
- Appearance swap (framing item 2). Kept for the rebuttal in a fixed design: swap textures from frame 1 onward, against a matched texture-perturbation control.
- Pseudo-label self-training (framing item 1) moves to the P2 phase.

## 6. Schedule (about 1.5 effective GPUs; GPU need about 4 to 6 GPU-days in total)

| Dates | Work | Decision |
|---|---|---|
| Oct 19 to 21 | Environments and downloads (shards in background); X0c; G0a, G0d | Drop failing models |
| Oct 21 to 24 | OpenD4RT, TrackCraft3R and lifted adapters; world_slope.py; G0b, G0e; first X1 rows (OpenD4RT first, as the highest value per GPU-hour) | |
| Oct 22 to 27 | X2 decoder, trainer and data loader; Wan feature extraction; X2 pilot; X5 pattern code; preregistration JSON | Engineering freeze Oct 27 |
| Oct 27 to Nov 3 | X2 main runs (9); remaining X1 rows; X3; X5 | T1', T3', T5 readouts |
| Nov 3 to 6 | p3_summary, figures; reruns only for bugs | Results freeze Nov 6 |
| Nov 6 to 10 | Draft of the new sections; paper registration Nov 10 | |
| Nov 10 to 16 | Writing, internal review, supplementary | Submit Nov 16 |

Cut order if late: DiTracker, then TrackCraft3R condition (iii), then X3 on VGGT, then X5 on KITTI and Sintel, then the X2 capacity cell, then the whole of X5. Never cut OpenD4RT or the world-frame re-scoring.

P2 (counterfactual-camera supervision) starts after submission. Its ReCamMaster fidelity gate (15 to 30 GPU-hours) could run in the background on the 4x A6000 server before then, but only if the user agrees to use that machine.

## 7. Queue integration
- Task names:
  - setup and gates: p3_setup_<env>, p3_dl_<asset>, p3_g0{a,b,d,e};
  - X0c and X1: p3_x0c_cotracker_{kitti,sintel}, p3_x1_<model>_<cond>_<split>, p3_x1_score;
  - X2: p3_feat_wan, p3_x2_pilot, p3_x2_<param>_<loss>_s<seed>, p3_x2_cap, p3_x2_eval;
  - X3, X5 and summary: p3_x3_<model>, p3_x5_patterns, p3_x5_edit_<proj>_g<g>, p3_summary.
- Dependencies point at gate files. TrackCraft3R tasks set GPU_NEED_GB = 20. Training checkpoints every 1k steps and resumes after OOM back-off. A P3 section is added to the dashboard.
- As before, the agent enqueues everything and stops polling. The user says when runs finish, and analysis starts then.

## 8. Statistics
- Confirmatory Holm family, all directional p-values with effect sizes:
  - T1': world-frame slope below 0.9 for OpenD4RT and TrackCraft3R (ii);
  - T3': the X2 parameterisation effect on PI;
  - T5: the directional edit vs random projectors.
  Everything else is descriptive with 95% CIs.
- Units: the scene for inference-only work. For trained decoders, run then scene, through the hierarchical bootstrap.
- Thresholds, gains and the boundary rule are frozen in results/p3_preregistration.json after the X2 pilot and before main runs. Round-1 E5 results stay labelled exploratory.

## 9. How outcomes read

| Outcome | Reading in the paper |
|---|---|
| T1' holds for both new models, lifted trackers near 1, P-res > P-abs on PI | Main result: the pull is residual shrinkage toward a static world. It is shared by GFM, D4RT and video-generator reconstruction models, and parameterisation alone reproduces it. |
| T1' holds, but X2 shows no parameterisation effect | Generality holds and the cause is not parameterisation. Check the loss-space effect and X3 (account F). |
| TrackCraft3R or OpenD4RT shows no pull | The pull is family-dependent. Say which and why (their readouts in X1b). |
| X5 double dissociation | Mechanism plus a specific edit that fixes magnitude without the accuracy cost. |
| X5 non-specific | Report it honestly next to E5 item 5. The paper rests on C6 to C8. |

The E5 claim "not uncertainty-driven shrinkage" must be reworded either way. Under account R it becomes "shrinkage toward a static-world prior that depends on the evidence for object motion (area, occlusion), not on texture or confidence".

## 10. Risks
1. Two assumptions fail without G0b: that TrackCraft3R behaves on 12 non-uniformly spaced Kubric frames, and that OpenD4RT behaves at 24 frames. The gates catch this early, and OpenD4RT can use its 48-frame checkpoint.
2. World-frame GT on KITTI depends on the existing ego-motion estimate. If it is not available, KITTI is reported in image space only.
3. Small frozen-feature decoders may not mirror large models. G2 and the capacity cell guard this, and X2 is presented as a controlled model system.
4. Engineering time, not GPU time, is the binding constraint. Hence the Oct 27 freeze and the cut order.
5. Strong-accept odds depend on T1' holding on real data, T3' being large and seed-consistent, and X5 dissociating. The reviewer's estimate for all three by the deadline is about 20 to 25%. The reduced plan gives a coherent accept-level paper with good probability.

## 11. Review log (one review pass of version 1)

The review (P3_plan_review.md) found 4 critical, 11 major and 9 minor issues. What changed:
- TrackCraft3R facts were wrong in v1. It uses 12-frame runs with per-frame VAE encoding at 480x832; there is no 4N+1 rule and no 81-frame clips. Fixed in the adapter and gate G0b.
- TrackCraft3R's beta is circular, because its output is a residual from input cameras. The primary cross-family metric is now the world-frame slope with lifted-tracker controls, and image-space beta for TrackCraft3R is descriptive only.
- The features x objective 2x2 did not isolate the objective (head weights, target space and capacity varied together, and alpha-only could not separate pull from uniform shrinkage). It is replaced by parameterisation x loss space on Wan features, with equal read-out weights, matched static EPE, a capacity cell and the pull-index endpoint.
- Scope: cut X6, X7, the Wan part of X4, VideoMAE and the extra X3 models. Added an engineering freeze on Oct 27.
- The probe ratio a/b is not shrinkage-invariant, so it is replaced by R^2, calibration and SNR-matched comparisons.
- The readout timestep is fixed (t = 5, with a sensitivity check). The latent-6 label ambiguity is handled.
- Hierarchical bootstrap for trained runs; a three-test Holm family; gray-zone readings.
- DiTracker's T2 was confounded and underpowered, so it was removed from the confirmatory family.
- Training data: CoTracker3_Kubric instead of 2000 new renders.
- X5 uses encoding patterns and a double dissociation.
- Added X1b readout contrasts (OpenD4RT, Any4D) and the boundary rule.
- Implementation fixes: VAE normalisation, hook points (q and k need a custom processor), the T5 transient in TrackCraft3R, DVD weight format, the DiTracker seed, and OpenD4RT resolution and frame parity.

Where this plan does not follow the review: the review suggested per-frame VAE encoding as the main way to make the X3 target frame unambiguous. Base Wan was never trained on per-frame latents, so its features would be out of distribution. Temporal compression with chunk-average labels stays primary, and per-frame encoding is a sensitivity check.

Open items checked only during execution: RGB format in the CoTracker3_Kubric shards (G0e), real TrackCraft3R memory use (estimated 10 to 13 GB steady), server RAM for the 17.8 GB TrackCraft3R checkpoint, and KITTI world-frame GT.
