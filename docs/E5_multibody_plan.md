# E5: Multibody Structure in 4D Feed-Forward Models: Working Plan

Companion to "Probing 4D Feed-Forward Reconstruction Models: An Epipolar-Geometry Analysis". Mathematical reference: Vidal, Ma, Soatto and Sastry, "Two-View Multibody Structure from Motion", IJCV 68(1), 7-25, 2006.

Status: draft for editing.

## 1. Goal

Test whether 4D feed-forward models (Any4D first, then VGGT as a control, then D4RT when weights allow) represent a scene with several independently moving rigid bodies as a multibody structure. Concretely: can the number of rigid motions, and the assignment of points to motions, be recovered from (a) the model's outputs, (b) its attention-derived correspondences, layer by layer, and (c) its token features, beyond what semantic clustering alone explains?

## 2. What Vidal et al. provides

- Veronese embedding. Each image point x is lifted to nu_n(x), a vector of dimension M_n = (n+1)(n+2)/2. For n = 1, 2, 3, 4 this is 3, 6, 10, 15.
- Multibody epipolar constraint. Every correspondence satisfies nu_n(x2)^T F_mb nu_n(x1) = 0 (Theorem 1), where F_mb is the M_n x M_n multibody fundamental matrix.
- Number of motions (Theorem 3). Build V_i with rows nu_i(x2) kron nu_i(x1), which has M_i^2 columns. Then n = min{ i : rank(V_i) = M_i^2 - 1 }. This requires N >= M_n^2 - 1 correspondences (8, 35, 99, 224 for n = 1 to 4) and at least 8 points per motion.
- Recovery of individual motions. Epipolar lines come from derivatives of the multibody constraint; epipoles come from GPCA on those lines (Theorem 2 links the null space of F_mb to the epipoles); individual F_i then follow linearly; segmentation of points follows from the F_i.
- Stated conditions. Independent motions with nonzero translation, distinct epipoles, and 3D points in general configuration. Open cases named by the authors: pure rotation, repeated left and right epipoles, planar objects.
- Stated limitation. The method is "mostly designed for noise-free correspondences"; the linear step is noise-sensitive and statistically sub-optimal.

## 3. Implications for the probe

1. Motions are counted relative to the camera. With a moving camera the static background is motion 1, so a scene with 2 moving objects should give n = 3.
2. The exact rank test will not survive model-derived correspondences. Patch-level attention matches and decoded correspondences are noisy, so V_i is full rank at every degree. E5 needs a noise-tolerant counting rule (normalized singular-value gap, or a model-selection criterion), calibrated on ground truth before any model result is interpreted. This is the central methodological decision of E5.

## 4. Proposed sequence

### Step 1. Multibody toolkit and noise-floor calibration (no model needed)

Implement Hartley normalization, Veronese embedding, V_i construction, rank and spectral-gap motion counting, F_mb estimation, epipole recovery via GPCA, and per-point segmentation. Reproduce Vidal's synthetic setting (n = 2, N = 100n points). Then add noise in steps up to roughly patch size (14 or 16 px) and record the noise level at which the motion count fails. This curve is the noise floor used to interpret every later step.

Deliverables: toolkit code, unit tests on GT, noise-floor figure.

### Step 2. Controlled dataset

Kubric (MOVi-style) rigid multi-object scenes with a moving camera and 0 to 3 moving objects, with GT poses, object identities and correspondences. Stratified splits:

- generic motions;
- Vidal degeneracies: rotation-only objects, shared epipoles, planar objects;
- motion along epipolar lines (blind spot already flagged in the main document);
- semantic confounds: identical-looking objects on different motions, and different-looking objects on the same rigid motion (glued together, or both static).

The confound split is what supports the claim "clustering beyond what semantic clustering explains".

### Step 3. E5-C: behavioural test on Any4D output correspondences

For a pixel in view 0, take its predicted 3D point, add the allocentric scene flow, and project into view t with the predicted pose and intrinsics. Run the Step 1 pipeline on these correspondences.

Questions: is the estimated n correct, and does the F_mb segmentation match the GT rigid bodies? This tests whether the model's output is multibody-consistent. It needs only inference, so it is the cheapest first result. Choose frame pairs with enough baseline to avoid near pure rotation.

### Step 4. E5-A: attention-derived correspondences, layer by layer

Per layer and head, take each source patch's argmax match in QK^T (Bratulic-style), sharing infrastructure with E3. Run the multibody count per layer and locate the layer, if any, where the correct n first becomes recoverable. Compare it with the E1 motion transition and with the layer where the single-body F emerges. Interpret against the Step 1 noise floor.

### Step 5. E5-F: training-free feature clustering

Subspace clustering (SSC, LRR, or GPCA on PCA-reduced tokens) of per-token features at every layer. Score against GT rigid-body labels (ARI, NMI). Semantic baselines: frozen DINOv2 features and the encoder output. Key quantity: per-layer gain over the semantic baseline on the confound split.

### Step 6. E5-P: trained SE(3)-mixture probe

A small probe outputting K rigid transforms plus soft per-token assignments (as in SM4RT), supervised only through decoded scene flow or correspondences, never through GT labels. Sweep K or use a sparsity prior on assignments; compare the K actually used with the GT number of motions. Linear and MLP versions, following Bratulic et al.

### Step 7. Synthesis and follow-on

One per-layer figure combining n accuracy (E5-A), clustering gain over semantics (E5-F) and probe K (E5-P), with E1 and the F-emergence layer marked. Use any standout layer as the target for E3 and E8 causal knockouts. Then repeat Steps 3 to 6 on VGGT (no 4D training) to separate what 4D training adds, and on D4RT once weights are available.

## 5. Practical notes and open decisions

- Order. Steps 1 and 3 first. Step 1 is pure maths; Step 3 needs only Any4D inference (should fit on an 8 GB RTX 4060 at moderate resolution with a few views). Steps 4 to 6 need hooks into attention and the residual stream.
- Two-view versus multi-view. Vidal's method is two-view; Any4D is multi-view. Use pairs (view 0, view t), consistent with Any4D's view-0-referenced scene flow. Multi-view trajectory subspaces belong to E6.
- Noise-tolerant counting rule. Start with a spectral-gap rule calibrated in Step 1; keep a model-selection criterion as a robustness check.
- To decide: Kubric scene budget and resolution; frame-pair selection rule; patch-to-pixel mapping for attention correspondences; which subspace clustering method is primary.

## 6. Notes

For compute I will be using the ssh server at 10.0.117.206. For the controlled dataset that is generated suing Kubric, First I want to visually validate few samples form this controlled dataset so that I can give a go ahead. Only after that use the experiments that involves the controlled dataset. Currently the server is running some other jobs in the GPU. Once done I will update here the GPU status. We can start the compute then. In the meantime we can do the rest of the setting up needed.

## 7. Step 1 results (toolkit and noise floor)

Status: done (synthetic, CPU only). Files: e5_multibody.py (toolkit), e5_step1_calibration.py (sweep), e5_step1_calibration_raw.csv, e5_step1_summary_by_snr.csv, e5_step1_noise_floor.png.

Setup. Two views, 512 px image, f = 500 px, depth 4 to 10, rotations up to 15 deg, unit translation, 120 points per motion, 30 trials per condition. Noise: Gaussian sigma 0.25 to 8 px, and patch-grid quantisation at 14 and 16 px (models argmax attention matches). SNR = cross-motion separation (median Sampson residual of one motion's clean points under another motion's true F, minimum over pairs) divided by effective epipolar noise std.

Findings.

1. Noise-free: exact Theorem 3 count and segmentation are correct in 30 of 30 scenes for n = 1, 2, 3.
2. Eq. (19) as printed cannot be used with the paper's mu = 5e-3. The complexity term at i = 2 (0.175) already exceeds the largest possible i = 1 cost (at most 0.165), so it always returns n = 1. Kept in code but flagged; replaced by a thresholded rank-drop statistic d_i = s_last^2 / sum of the other s_k^2 (kappa = 2.5e-6, calibrated globally).
3. Two motions: segmentation with n known reaches median error of about 1% once SNR exceeds about 16; the algebraic count is correct in at least 93% of trials in that range (sequential RANSAC 61 to 100%). Below SNR about 10, segmentation is close to random.
4. Three motions: neither the algebraic method nor sequential RANSAC is reliable at any SNR tested (median segmentation error 0.16 to 0.32 for SNR above 16). The degree-3 embedding (100 columns) is very noise sensitive. The RANSAC baseline is untuned (threshold 2.5 x noise, 20 minimum inliers) and over-splits.
5. Patch-level matches (14 to 16 px) give median SNR of about 2.7 to 2.9 in these scenes (10th to 90th percentile 1.6 to 9.8), far below the 2-motion threshold.
6. Caveat: at low SNR the count accuracy of the algebraic rule is inflated, because noise biases it toward n of 2 or more rather than toward the correct n. Count accuracy is only meaningful together with segmentation error.
7. Minor: the paper's text lists 225 as the minimum N for n = 4, while Theorem 3 gives M_4^2 - 1 = 224.

Proposed plan changes (for your review).

- Step 2 (Kubric): compute SNR per scene from GT and stratify by it. Make 2 total motions (camera-moving background plus one moving object) the primary condition, and 3 motions secondary. Design scenes so that SNR at the expected correspondence noise exceeds about 16.
- Step 3 (E5-C): dense decoded outputs should have roughly pixel-level error, so this remains the most feasible variant for 2 motions.
- Step 4 (E5-A): the pure Vidal count on patch argmax matches is not expected to work. Options: (a) sub-patch correspondences via soft-argmax over attention; (b) replace counting with a GT-conditioned statistic, for example how much better the matches are explained by the per-body true F's than by a single F; (c) restrict to high-separation 2-motion scenes.
- Add a tuned robust multi-model baseline (for example T-linkage or Progressive-X) before drawing conclusions for 3 motions.

## 8. Step 2 results (controlled Kubric dataset)

Status: done. 70 scenes, 10 per split (seeds 101-114 pilot, 201-256 scale-up), 512 x 512 px, 24 frames, 32 spp, scripted rigid SE(3) motion (no physics; moving objects float). Stored on 10.0.117.206 in ~/e5_multibody/data/pilot_v1/out. Generator: kubric_e5_scene.py; GT post-processing: e5_kubric_post.py.

- GT correspondences (from depth, segmentation and per-frame world matrices) match Kubric's rendered optical flow with median error 0.000 px in all scenes.
- Separation target 25 px (analytic check in the generator, skipped for camrot and epi_aligned). The measured separation (visible surface points) is often lower than the analytic estimate, so the measured value is used for stratification.
- User validated pilot samples and approved scaling to 10 per split.

## 9. Step 3 results (E5-C: Any4D output correspondences)

Status: done. Any4D (any4d_4v_combined.pth, images only), 7 views per scene (frames 0,4,8,12,16,20,23, view 0 = reference), 518 x 518. Predicted correspondence: x_t = K_t R_t^T(pts3d_0 + scene_flow_t - T_t). Main numbers use the pair (frame 0, frame 23). Files: e5_any4d_infer.py, e5_step3_analysis.py, step3_any4d_7v.json, e5_step3_any4d_per_scene.csv, e5_step3_any4d_multibody.png.

Findings (medians over 70 scenes unless stated).

1. Correspondence error: static background 3.7 px, moving bodies 32.0 px at frame 23; moving error grows with temporal distance (6.3 px at frame 4).
2. Any4D's correspondences are less multibody-separated than GT: excluding epi_aligned, separation ratio (Any4D / GT) median 0.61 (IQR 0.23 to 0.95), lower than GT in 77% of 60 scenes (Wilcoxon p = 1.7e-5). Interpretation (hypothesis): predicted motion of moving bodies is pulled toward the camera-induced (background) epipolar geometry.
3. Each predicted body is nearly rigid in two-view terms (median Sampson residual under its own fitted F: 1.1 px), yet predicted SNR is low (median 6.0; only 13 of 60 non-epi scenes exceed 16).
4. Vidal segmentation with n known: GT correspondences give 0 error in every non-degenerate scene; Any4D correspondences give median error 0.32 (random labels = 0.5 for 2 motions). Error falls as predicted SNR rises (Spearman rho = -0.48, p = 9e-5), but even scenes with predicted SNR above 16 have median error 0.29, worse than the synthetic Step 1 expectation; a likely cause is non-iid error such as flow smoothing across object boundaries (untested).
5. Motion count: not interpreted. The rank-drop rule is biased toward 2 or more motions under noise (Step 1 caveat), so the Any4D count "accuracy" for 2-motion splits is not evidence; for n3 it is 0.2 for both GT and Any4D.
6. Any4D's own motion mask (|scene flow| > 0.1 m, the demo threshold) has median IoU 0.64 with GT moving pixels, and is similar in the epi_aligned split (median 0.58), where two-view geometry gives zero separation. Caveat: in epi_aligned the object translates parallel to the camera, which is also a depth/scale ambiguity across views, so detection there must come from non-geometric cues (priors such as object size or the missing ground contact of floating objects). This connects to E2's double control and E9.

Caveats: 10 scenes per split; one Any4D view set (7 views); one model; scripted floating objects are out of distribution for Any4D's training data.

Suggested next steps: (a) erode GT masks to test the boundary-smoothing explanation; (b) measure, per moving point, the fraction of object motion recovered relative to the static hypothesis (links to E3 rigid-hypothesis hits); (c) repeat with 4 views (the checkpoint's training setting); (d) add objects resting on the floor to remove the ground-contact cue in epi_aligned.

## 10. Steps 4 to 6 results (Any4D internals)

Status: done on all 70 scenes. Any4D's info-sharing transformer has 24 blocks (even = global attention across views, odd = within-frame), 12 heads of size 64, no positional rotary encoding, 37 x 37 patch tokens per view. Extracted: residual stream (25 states: transformer input plus each block output) for view 0 and view 6 (frame 23); for every global block and head, view-0 query to view-6 key attention argmax and soft-argmax. Files: e5_any4d_extract.py, e5_gt_patches.py, e5_internal_analysis.py, e5_adjacency_control.py, e5_steps456_per_layer.csv, e5_evidence_matrix.csv, e5_steps456_internals.png.

### Adjustments to Steps 4 to 6 (so each claim has more than one line of evidence)

- Findings are organised as four claims (C1 to C4), each tested by at least two of: output correspondences (Step 3), attention (E5-A), clustering (E5-F) and trained probes (E5-P).
- E5-A: Vidal's exact count is not used on attention matches (Step 1 noise floor rules it out). Replaced by a cross-fitted multibody gain: residual of a moving body under the background F divided by its held-out residual under its own F (1 = no multibody structure). Added E3-style hit rates for the true moved location and the static-hypothesis location.
- E5-F: k-means on PCA-32 tokens, with an objects-only variant (background texture otherwise dominates), instance labels as the semantic baseline, and confound-pair measures.
- E5-P: the SE(3)-mixture probe was replaced by a low-rank bilinear same-rigid-body pair probe with an average-linkage estimate of K, plus a ridge probe of the object-motion component and a view-0-only moving-vs-static probe. 5-fold cross-validation grouped by scene; transformer input (encoder features) is the appearance baseline.
- Added an adjacency control (pairs of object instances, same vs different body, adjacent vs far) because glued objects touch in the image.

### Claim-level conclusions

1. C1, rigid-body structure with depth: supported, but as a gradual increase, not a sharp transition. Attention correspondences already separate bodies at the second global block (layer 2, multibody gain about 8x) and stay lower afterwards. In the residual stream, the pair-probe AUC on object pairs rises from 0.85 to 0.93, k-means ARI against rigid bodies from 0.49 to 0.61, and probe-implied K accuracy from 0.31 to 0.50 (K is under-counted).
2. C2, moving bodies pulled toward the camera-induced motion: supported at the output (separation 0.61 of GT; 0.79 of object motion recovered), but not in attention, where heads land on the true moved location (hit rate up to 0.44) far more than on the static-hypothesis location (at most 0.05). The pull therefore arises after cross-view matching (hypothesis: in the scene-flow decoder or late blocks), which is a concrete target for E8 knockouts.
3. C3, grouping beyond appearance: partial and preliminary. The pair probe separates the same-asset moving and static copies twice as often at layer 24 as at the input (0.26 to 0.52, 10 scenes), and k-means joins look-alike moving objects that share a motion (0.69) but not those that move differently (0.13) at layer 24, with no difference at the input (7 and 8 pairs). However, across all scenes the gain in body ARI is matched by an equal gain in instance ARI, and k-means never splits the same-asset copies.
4. C4, epipolar-aligned motion detected: not attributable to E5. Only the output mask supports it; the internal probe is at ceiling (AUC 0.99) at the input layer because moving objects are larger and float, so the dataset cannot separate motion from these appearance cues.

### Caveats

- 70 scenes, one model, one view set (7 views), one frame pair (0, 23) for internals.
- Confound splits have 10 scenes each; moving-object pair contrasts use 7 and 8 pairs.
- Moving objects are larger than static ones and float: this shortcut affects any moving-vs-static readout and the same-asset confound.
- The ridge slope is attenuated by regularisation, so it does not measure under-recovery by the model.

### Suggested next steps

- Regenerate a confound-free dataset: same size distribution for moving and static objects, moving objects resting on or sliding along the floor, more glued and n3 scenes. Then rerun C3 and C4.
- E8: knock out late blocks or the scene-flow head inputs to localise where the C2 pull arises; the attention heads with highest track hits (global blocks at layers 2, 6, 10) are candidates for the opposite test.

## 11. E8 results (causal ablations on Any4D)

### First pass: blunt knockouts (v1, 70 scenes, 12 conditions)

- Skipping windows of 4 blocks, or zeroing a set of 12 heads (top tracking heads, top static-hypothesis heads, random heads), broke the model in a non-specific way: static and moving errors both rose sharply. The tracking and static head sets also shared 6 of 12 heads, so these conditions could not separate the two hypotheses. We therefore moved to single-unit mean ablation.

### Single-unit mean-ablation screen (v1, 18 scenes, 3 per non-degenerate split)

- Each of the 288 attention heads was mean-ablated on its own (its output replaced with the dataset mean before the output projection), and each of the 24 blocks had its residual update replaced by its token mean. Read-outs: object-motion slope (predicted vs true object motion, 1 = full recovery), static EPE, moving EPE, separation ratio.
- Base: slope 0.606, static EPE 3.64 px, moving EPE 34.7 px, separation ratio 0.50.
- No single unit removes the pull toward camera motion. The largest slope increase that leaves static EPE within 1 px is +0.021.
- Five heads supply object motion. Ablating any one of them lowers the slope and raises moving error a lot, while static error barely changes:

| Head | Block type | d slope | d static (px) | d moving (px) | Scenes where slope fell |
|---|---|---|---|---|---|
| b2 h1 | global | -0.287 | +0.56 | +15.8 | 18/18 |
| b22 h9 | global | -0.270 | -0.06 | +14.5 | 18/18 |
| b17 h10 | frame | -0.248 | -0.09 | +12.2 | 18/18 |
| b14 h6 | global | -0.181 | +0.14 | +7.8 | 18/18 |
| b8 h11 | global | -0.171 | +0.36 | +7.2 | 16/18 |

- Across global heads, a head's attention tracking hit rate predicts its ablation effect (Spearman -0.21, p = 0.012). b2 h1 has the 4th highest tracking hit rate (0.35 vs a mean of 0.07).
- Blocks whose ablation is motion specific (d slope < -0.1 with d static < 3 px): 2, 8, 13, 15, 16, 17.

### Replication on v2 (30 new scenes, 5 per non-degenerate split)

| Head | d slope | d static (px) | d moving (px) | Scenes where slope fell |
|---|---|---|---|---|
| b2 h1 | -0.297 | +0.94 | +27.1 | 29/30 |
| b22 h9 | -0.273 | -0.10 | +21.2 | 30/30 |
| b17 h10 | -0.300 | -0.06 | +24.3 | 30/30 |
| b14 h6 | -0.163 | +0.21 | +10.5 | 30/30 |
| b8 h11 | -0.185 | +0.45 | +16.2 | 29/30 |

Base slope on these scenes is 0.561. All five heads replicate on scenes they were not selected on, with effect sizes as large as or larger than in v1 (sign test p < 1e-7 for each).

### Interpretation

- The camera-motion pull reflects too little object motion being supplied by a small set of units. It does not come from a unit that actively pushes objects toward camera motion. Removing a supplier makes the pull worse; no single removal makes it better.
- Caveat: mean ablation removes a unit's scene-specific contribution but keeps its average. Combined ablations and path patching into the scene-flow decoder have not been run yet.

## 12. v2 replication on a confound-free dataset

### Dataset

- 100 Kubric scenes. Moving objects now slide along the floor (xy translation plus yaw), and every object, moving or static, is 1.4 to 2.2 m in size. n3, confound_same and confound_glued have 20 scenes each; the other splits have 10. Step 3 analysed 98 scenes; probe statistics use the scenes where each metric is defined (n per row below).
- At the input layer, the view-0 moving-vs-static probe is now near chance (AUC 0.57; 0.38 on epipolar-aligned scenes), compared with 0.99 in v1. Object size and floating no longer give motion away.

### Output level (Step 3)

| Metric | v1 | v2 |
|---|---|---|
| Separation ratio (pred/GT) | 0.61 | 0.75 |
| Fraction of scenes below GT | 0.77 | 0.78 |
| EPE static / moving (px) | 3.70 / 31.99 | 3.82 / 30.72 |
| Vidal segmentation error on Any4D matches | 0.32 | 0.32 |
| Motion-mask IoU all / epi | 0.64 / 0.58 | 0.65 / 0.53 |

### Internals, layer 0 vs layer 24 (scene-level paired; 95% bootstrap CI; Wilcoxon)

| Read-out | L0 | L24 | Delta [95% CI] | n | p |
|---|---|---|---|---|---|
| Same-body pair AUC, object pairs | 0.770 | 0.914 | +0.144 [0.116, 0.173] | 93 | < 1e-6 |
| Moving vs static AUC, view-0 objects | 0.569 | 0.998 | +0.429 [0.37, 0.49] | 93 | < 1e-6 |
| Same, epipolar-aligned only | 0.379 | 0.998 | +0.619 [0.44, 0.78] | 9 | 0.004 |
| Same-asset copies separated (probe) | 0.115 | 0.583 | +0.467 [0.33, 0.59] | 20 | 4e-5 |
| K estimated correctly | 0.19 | 0.52 | +0.33 [0.22, 0.44] | 100 | 1e-6 |
| k-means ARI, rigid body | 0.336 | 0.519 | +0.182 [0.12, 0.25] | 94 | < 1e-6 |
| k-means ARI, instance | 0.527 | 0.662 | +0.136 [0.09, 0.19] | 94 | < 1e-6 |
| ARI body minus instance | -0.190 | -0.143 | +0.047 [-0.01, 0.10] | 94 | 0.32 |
| k-means copies split | 0.450 | 0.619 | +0.169 [0.06, 0.28] | 20 | 0.012 |

- Motion decodability in view-0 tokens rises slowly through the stack. It passes 0.9 at layer 17 on epipolar-aligned scenes and at layer 19 overall.
- Moving-object pairs (probe join rate, L0 / 8 / 16 / 24): same motion (glued, 18 pairs) 0.89 / 0.88 / 0.84 / 0.88; different motion (17 pairs) 0.90 / 0.75 / 0.50 / 0.60.
- Attention: tracking hit rate peaks at 0.52 and static-hypothesis hit rate stays at or below 0.044. Multibody gain peaks at layer 2 (0.80).

### Claim-level update

- C1 (rigid-body grouping is built gradually): replicated. Pair AUC and K accuracy both rise with depth, and from a lower start than in v1.
- C2 (output pull toward camera motion): replicated but smaller (separation 0.75 of GT, in 78% of scenes below GT; output slope 0.77). Attention still shows no static-hypothesis routing, and E8 points to too little supply of object motion.
- C3 (identical copies are split by motion): the probe evidence replicates and is stronger. Copy separation rises from 0.12 to 0.58, and different-motion look-alike movers are pulled apart while same-motion pairs stay joined. The v1 k-means result for glued pairs does not replicate: in v2, glued pairs are already joined at layer 0 (0.66 vs 0.19 for different-motion pairs), probably because glued objects are adjacent (median gap 2.9 patches vs 7). Unsupervised clustering therefore does not show a motion effect beyond what instance identity explains (body minus instance ARI change +0.05, p = 0.32). C3 rests on the supervised probe, not on clustering.
- C4 (motion is inferred beyond two-view epipolar geometry): now supported internally. With the input shortcut gone, the transformer makes object motion linearly decodable even for epipolar-aligned motion (0.38 to 0.998). The output-level mask IoU on epi scenes (0.53) is lower than overall, so the decoder uses this information only partly.

### Remaining caveats

- Epipolar-aligned scenes are few (n = 9 with defined probe AUC). The confound_same and confound_glued subsets have 20 scenes each.
- Probes are linear and trained across scenes; a decodable signal does not prove the model uses it. E8 gives causal evidence for supply heads, not for the probe directions.
- Grounded objects can reveal motion through floor contact over 7 views. This is beyond two-view geometry by design, but it is a cue the model could learn without multibody reasoning.

### Next steps

- Ablate the five supply heads together, and patch their outputs into the scene-flow decoder, to test whether they jointly explain the slope.
- Steer along the probe motion direction at layers 17 to 21 to test whether the decodable motion signal is causally used.
- Increase the number of epipolar-aligned scenes to tighten C4.

## 13. CVPR experiment queue (items 1, 2, 3, 4, 5, 7, 8)

The user selected items 1, 2, 3, 4, 5, 7 and 8 from their observation list, plus the next steps in section 12. They run as a sequential, crash-safe queue on the GPU server. Each finished task is pushed to GitHub.

### Queue mechanics

- `~/e5_multibody/queue/supervisor.py` runs `tasks.json` one task at a time on the less-used GPU.
- Cron restarts the supervisor every 5 minutes and at boot. A lock keeps one instance alive.
- After a crash, the interrupted task is re-queued. Every task script skips per-scene work already on disk.
- When a task finishes, `push.sh` commits these to the repo under `results/queue/<task>/`, `scripts/queue/` and `src/`:
  - the task's results folder (files under 50 MB, no arrays)
  - its log
  - the task scripts and the source
- Pushes use a write deploy key scoped to this repo only. Failed pushes retry every 10 minutes.
- New data goes on `/media/workbench/zone_x/e5_multibody` (1.2 TB free). v1 and v2 stay where they are.
- New datasets wait behind a gate (`approvals/v3_visual`) until the user approves the pilot contact sheets.

### Tasks

| # | Task | Item | What it does |
|---|---|---|---|
| 1 | v3_pilot | 1, 2 | Pilot renders of the two new scene types, with contact sheets and GIFs for visual approval |
| 2 | t4_causal_v2 | 4 | Five supply heads ablated together; amplification x1.5, x2, x3; matched random-head controls; path patching (scene-flow decoder only vs geometry and pose decoders only, and per decoder tap) |
| 3 | t5_subspace_v2 | 5 | LEACE erasure of the linear motion subspace at layers 17 to 21 (single layers and all five), rank-matched random erasure, steering movers toward static and static objects toward moving, random-direction controls |
| 4 | t3_viewsweep_v2 | 3 | 2, 3, 4 and 7 input views: outputs, probes (layers 0, 4, ..., 24) and attention |
| 5 | v3_scale | 1, 2 | After approval: 20 far-glued scenes, 70 floating shadowless scenes, 20 extra grounded epipolar-aligned scenes in v2 |
| 6 | v3_pipeline | 1, 2 | Steps 3 to 6 plus the adjacency control on the new datasets, and on v2 with 30 epipolar-aligned scenes |
| 7 | v3_viewsweep | 2, 3 | View-count sweep on the floating set |
| 8+ | t7_* | 7 | Cross-model: MapAnything (static, same architecture), VGGT-1B (static), plus one or two dynamic models with public weights. Same output metrics, probes and attention hits |
| 8+ | t8_* | 8 | Real data at the output level: KITTI scene flow 2015 and MPI Sintel, 2-view and multi-view; slope, EPE, separation ratio |

### New scene types

- **confound_farglued (C3 adjacency control).**
  - Objects A and B are different assets, rigidly coupled 2.4 to 3.2 m apart through an invisible offset (one motion group).
  - C is an identical copy of A with its own independent motion.
  - There is one static object.
  - Appearance says A and C belong together, motion says A and B, and A and B are never adjacent.
- **v3float (C4 without floor cues).**
  - Every object, moving or static, floats 0.6 to 1.1 m above the floor.
  - Motion is the same planar motion as v2.
  - No object casts a shadow, so neither floor contact nor a moving shadow reveals motion.
  - Sizes are matched to v2.
  - Epipolar-aligned scenes rise to 30.

### Smoke-test notes (1 to 4 scenes, not results)

- The causal script runs all 19 conditions. Tap 0 (encoder features) reproduces the base run exactly, which is the expected sanity check.
- LEACE fits remove most of the linear "moving" information (training-fold R^2 from about 0.93 down to 0.01 to 0.31). Rank-matched random erasure leaves the outputs unchanged.

## 14. Final results and interpretation (all queue runs complete, 2026-10-16)

All 18 queue tasks finished. There were five failures. The floating pipeline (one rerun) and three cross-model runs (VGGT, MapAnything, PAGE-4D) were re-run. The Any4D cross-model reference had silently included the 20 new scenes and was rerun on the original 100. On top of that, 40 scenes per model needed an index-mismatch fix and a re-evaluation; after it, every model has 100 of 100 scenes. Figures: fig_c2_pull_everywhere.png, fig_c4_beyond_epipolar.png, fig_c3_farglued.png, fig_xmodel_internals.png. Numbers: final_summary.json, xm_output_table.json.

### C2 behaviour: the camera-motion pull is general

"Slope" is the fraction of true object motion recovered (1 = full motion, 0 = the object is treated as static).

| Data (Any4D) | Slope, mean [95% CI, from final_summary.json] | Scenes with slope < 1 | Separation ratio |
|---|---|---|---|
| Kubric v2 grounded (94) | 0.75 [0.71, 0.78] | 95% | mean 0.76, 75% below GT |
| Kubric floating (51) | 0.76 [0.69, 0.82] | 92% | median 0.71, 74% below GT |
| Kubric far-glued (19) | 0.73 [0.65, 0.79] | 100% | median 0.55, 84% below GT |
| KITTI 2015, 2 views, wide (195) | 0.58 [0.51, 0.63] | 84% | undefined (sep_gt < 1 px) |
| Sintel clean, 2 views, wide (217) | 0.31 [0.23, 0.39] | 88% | median 0.54 |
| Sintel final, 2 views, wide (217) | 0.33 [0.24, 0.41] | 89% | median 0.51 |

- The pull is present on every dataset and is stronger on real footage than on Kubric.
- It is also present in the static-trained VGGT-1B, read out from its tracking head: slope 0.34, separation ratio 0.23 (99% of scenes below GT). Any4D on the same 100 scenes gives 0.61 and 0.83. Dynamic training therefore reduces the pull but does not remove it.
- The pull lives in the correspondence and motion outputs, not in per-frame geometry. The 3D slope computed from each frame's own pointmap is close to 1 for every model (Any4D 0.99, MapAnything 0.92, VGGT 1.02, PAGE-4D 1.02). Each frame places the object where it really is; the error appears only when the model links points across frames.

### C2 mechanism: five heads, one route, a partial fix

- **Joint ablation.** Mean-ablating the five supply heads together drops the slope from 0.61 to 0.01, while static error rises only from 4.1 to 7.1 px. Two of three matched random 5-head sets have no effect. The third broke the model (static EPE 87 px), probably because it contains the early frame-block head b1 h2.
- **Route.** Path patching shows the whole effect passes through the scene-flow decoder: slope 0.01 when only that decoder sees the ablated features, with static error unchanged. It enters through the block-17 and final-layer taps; the block-11 tap contributes little.
- **Amplification.** Scaling the five heads by 1.5 raises the slope to 0.95 and the separation ratio from 0.83 to 0.93, with static error +0.6 px. But moving error rises from 35 to 42 px, and a scale of 2 overshoots (slope 1.17). Amplification fixes the magnitude of object motion, not its accuracy. It is evidence that the pull is under-supply, not yet a usable fix.

### Item 5: is the decodable motion subspace used? Weakly, and not specifically

- **Erasure.** LEACE erasure at layers 17 to 21 lowers the slope (0.61 to 0.21 to 0.46 per layer, 0.08 when all five layers are erased), and rank-matched random erasure does nothing. However, erasure also raises static error by 7 to 17 px. Motion information in the residual stream is entangled with geometry, so removing it is not a clean, mover-specific intervention.
- **Steering.** Steering movers toward "static" is not direction-specific (a random direction gives the same slope drop). Steering static objects toward "moving" is only weakly specific.
- **Verdict.** The probe direction is necessary for good outputs, but we cannot claim the model reads motion along that particular linear direction.

### C4: motion is inferred beyond two-view epipolar geometry

- **Floating, shadowless scenes.** These remove floor contact and shadows. View-0 motion AUC on epipolar-aligned scenes rises from 0.52 at the input (chance) to 0.999 at layer 24 (n = 26). It passes 0.9 at layer 17.
- **Two views.** With only two input views, AUC still rises from 0.51 to 0.995. At mid depth (layer 12), two views actually give higher AUC than seven (0.87 vs 0.75).
- **The predicted two-view collapse does not happen.** Vidal's ambiguity is a statement about pure two-view geometry. A learned model can break it with monocular priors. An object moving along an epipolar line produces a triangulated depth that conflicts with its single-image depth and size, and this conflict exposes the motion. This is our proposed mechanism; it was not tested directly.
- **The decoder uses this only partly.** The output motion-mask IoU on floating epipolar-aligned scenes is 0.31, against 0.56 on the other splits, and it stays at 0.29 to 0.32 for every view count. The transformer knows these objects move, but the output mostly fails to say so.

### C3: reframe as instance grouping plus motion splitting

Far-glued control (object A coupled to a distant, different-looking B; C an identical copy of A with its own motion):

- **Pair probe, A to B.** The join rate rises from 0.80 to 0.95 (change +0.15 [0.02, 0.28]). At the input, A-B joins less than A-C, by 0.18 (p = 0.001). By layers 16 to 24 the two are equal (about 0.94).
- **Pair probe, A to C.** Identical copies with different motion stay joined (0.98 to 0.94). The probe does not split them.
- **k-means.** A-C joining falls from 0.44 to 0.19 (p = 0.065), so identical copies with different motion are split. A-B stays low throughout (0.20 to 0.19): distant coupled objects are never joined.
- **Clustering follows instances, not rigid bodies.** The body-minus-instance ARI change is +0.05 (p = 0.73) on v2, +0.05 (p = 0.12) on the floating set, and -0.03 (p = 0.035) on the far-glued set.
- **Conclusion.** By the criterion in the user's observations (item 1), C3 should be reframed. The transformer reduces appearance-driven grouping and adds motion evidence, and identical copies are separated (v2 probe copy separation 0.13 to 0.65, n = 20). But it does not build rigid-body groups that span distant, different-looking parts.

### C1 and the cross-model view (item 7)

| Model (100 v2 scenes) | Motion AUC, input to last | Epi AUC, last | Pair AUC, last | Copies separated, last | Probe object-motion slope, last |
|---|---|---|---|---|---|
| Any4D (dynamic) | 0.56 to 0.999 | 1.000 | 0.921 | 0.649 | 0.493 |
| MapAnything (static base of Any4D) | 0.55 to 0.992 | 0.997 | 0.863 | 0.326 | 0.381 |
| VGGT-1B (static) | 0.55 to 0.999 | 0.998 | 0.939 | 0.832 | 0.368 |
| PAGE-4D (dynamic, VGGT base) | 0.55 to 0.996 | 0.979 | 0.909 | 0.599 | 0.334 |

- **Motion detection does not need dynamic training.** Every model, including the two static-trained ones, makes motion linearly decodable, with epipolar-aligned motion included. It looks like a by-product of multi-view reconstruction: the model has to detect points that are inconsistent with a rigid scene.
- **The effect of dynamic fine-tuning depends on the model family.**
  - MapAnything to Any4D improves rigid grouping (0.86 to 0.92), copy separation (0.33 to 0.65) and the probe object-motion slope (0.38 to 0.49).
  - VGGT to PAGE-4D does not: it lowers pair AUC (0.94 to 0.91) and copy separation (0.83 to 0.60).
  - Static VGGT has the strongest grouping of all four models.
  - With one pair per family we cannot separate the effect of dynamic training from the effect of each fine-tuning recipe.
- **Attention.** In every model, attention follows the moving object (tracking hit rate about 0.5) and almost never routes to the static-scene location (at most 0.11). The pull is not caused by attention ignoring motion.

### Revised paper story

1. **Behaviour.** 4D feed-forward models under-recover object motion. This holds on Kubric, KITTI and Sintel, and in both a dynamic and a static model. The deficit is in cross-frame correspondence, not per-frame geometry.
2. **Representation.** Motion is computed inside the transformer from near-chance input. This happens even for epipolar-aligned motion and with only two views, so these models go beyond the two-view multibody limit of Vidal et al. Grouping follows instances and splits by motion, rather than forming full rigid bodies.
3. **Mechanism.** Five heads supply object motion to the scene-flow decoder through two taps. Removing them erases object motion; amplifying them restores its magnitude but not its accuracy.
4. **The gap.** The transformer knows more than the output says (epipolar-aligned mask IoU 0.31 despite near-perfect internal decodability). This motivates a decoder-side or head-targeted fix (item 10).

### Remaining caveats

- Probes are linear and cross-scene. Item 5 could not show that the decodable direction is used specifically.
- Only one dynamic and one static model per family. MapAnything and PAGE-4D have no 2D motion output, so the pull can be compared only between Any4D and VGGT.
- Real data:
  - KITTI separation ratio is undefined for most scenes.
  - Sintel motion is mostly non-rigid, and only every 4th frame was used.
  - The square input sees a centre crop, so the wide input is the reference.
- Far-glued set: 19 to 20 scenes, with 13 to 16 usable pairs per type.
- The monocular-prior explanation for C4 is a hypothesis.

## 15. Round 2 results (self-review items 1 to 4, plan item 10), 2026-10-18

All 34 round-2 tasks finished.

- **Failures before success.** Eight tasks first failed because another user's job filled GPU 0. All succeeded on retry: three on their second or third try, and five after a manual re-queue once their four attempts were used up.
- **Checkpoint-selection bug.** The first heads, control and gain fine-tunes had a checkpoint-selection bug; they were re-run as *_sel tasks. Every fine-tune number below comes from the corrected runs.

Figures:
- fig_w2_summary.png, fig_w2_alpha_beta.png (shrinkage);
- fig_v4_accel_viewcount.png, fig_v4_sizecue.png (view test);
- fig_probe_controls.png, fig_heads_null.png (controls);
- ft_comparison.png (method).

### 15.1 Is the pull just regression shrinkage? No. It is specific to feed-forward reconstruction models (review W2)

Decomposition on moving points: x_pred - x1 = alpha (x_true - x_static) + beta (x_static - x1).
- A pure camera-motion pull gives beta = 1 and alpha < 1.
- Shrinkage toward zero image motion gives alpha = beta < 1.

All models were scored on the same scenes and query points.

| Model | Kubric v2 slope | alpha | beta | Real-data slope (2 views) |
|---|---|---|---|---|
| Any4D | 0.61 [0.56, 0.65] | 0.61 | 1.03 | KITTI 0.58, Sintel 0.31 to 0.33 |
| VGGT-1B tracker | 0.34 [0.29, 0.41] | 0.22 | 1.02 | not run |
| CoTracker3 (24 frames) | 0.96 [0.93, 0.97] | 0.95 | 0.99 | not run |
| RAFT chained over 7 frames | 0.88 [0.84, 0.90] | 0.85 | 0.98 | KITTI 0.95, Sintel 0.90 to 0.91 (2-view RAFT) |
| RAFT direct 0 to 23 | 0.59 [0.48, 0.69] | 0.61 | 0.68 | n/a |

- **The pattern in both reconstruction models.** Any4D and VGGT keep the camera-induced motion exactly (beta about 1, also on static points) while under-recovering the object's own motion (alpha well below beta; Any4D alpha minus beta -0.42, p = 8e-11).
- **Dedicated correspondence models barely show it.** CoTracker3 and chained RAFT on the same inputs reach 0.85 to 0.97.
- **The only baseline that looks like Any4D fails differently.** RAFT run directly over the large 0-to-23 gap has a slope similar to Any4D, but it shrinks both components (beta 0.68 to 0.78), which is ordinary large-displacement failure.
- **The real-data gap is large.** RAFT recovers 0.90 to 0.95 of object motion on KITTI and Sintel, against Any4D's 0.31 to 0.58.

Dose-response (per-object slope, standardised regression):
- **Size.** Any4D loses more motion on small objects (log image area +0.07 per SD on v2, +0.26 on the floating set).
- **Occlusion.** It also loses more on occluded ones (-0.05 and -0.11).
- **No dependence on texture or the model's own confidence.**
- **Edges.** Object boundaries are recovered worse than interiors in every model.

So the pull is a structural, camera-preserving bias of these models. It is stronger for small and partly hidden objects, consistent with motion evidence from few patch tokens being outvoted by the surrounding static scene. Generic regression shrinkage would also have shrunk the camera component and depended on texture and confidence, and it does neither.

### 15.2 The non-degenerate view test: a third view does not help (review W3)

New floating, shadowless scenes, verified from ground truth:
- **epi_accel and epi_const (30 seed pairs).** The two-view multibody separation of the 0 and 23 frames is 0.00 px in every scene. Frames 0 and 23 are pixel-identical within a pair. With accelerated motion, a third view exposes the mover by 31 to 61 px.
- **epi_sizerand and epi_sizematch (30 each).** These decouple the mover's apparent size from its depth.

Results:
- **The extra views add nothing.** Mid-depth motion AUC is 0.974 with 2 views and 0.973 to 0.978 with 3, 4 or 7. The difference-in-differences against the constant-velocity control is 0.000 to 0.003 (CIs span 0; 0 of 3 view counts pass the pre-specified test). Output mask IoU and slope are also unchanged.
  - Any4D flags epipolar-aligned movers from two views, before the geometry allows it.
  - It does not use the multi-view constraint even when a third view makes the motion geometrically resolvable.
- **Apparent size is not the cue.** Randomising size does not lower decodability (0 of 4 view counts pass the pre-specified test, Holm-corrected). Matching the size to the static reading lowers it somewhat with a probe trained only on round-1 floating scenes (-0.16 to -0.20, uncorrected p = 0.02 to 0.04). A probe trained on these scenes still decodes them at 0.98.
- **The cue remains unidentified.** It is some monocular or appearance-level cue that we did not isolate.
- **Size drives the output, not the detection.** On the same scenes the output slope and mask IoU rise strongly with the mover's size (Spearman 0.6 to 0.7, p < 0.001). This matches the dose-response in 15.1.

### 15.3 Probe controls: the rise is computed by the trained transformer (review W4, item 6)

Five seeds per probe; linear and 2-layer MLP probes; patch-level and instance-level label shuffles.

- **Motion is not decodable without the trained transformer.**
  - In a fully random Any4D, or a pretrained DINOv2 encoder followed by a random transformer, view-0 motion AUC stays at chance at every layer. At layer 24 the trained model exceeds them by +0.51 (v2) and +0.48 (floating), p < 1e-15.
  - Label-shuffle selectivity at layer 24 is +0.49 (v2) and +0.48 (floating).
- **Layer 0 holds only a little nonlinear motion information.**
  - The MLP finds some at layer 0 on v2 (selectivity +0.13, p = 0.001), but not on the floating set (+0.04, not significant).
  - Raw public DINOv2 tokens carry less (trained layer 0 minus DINOv2 +0.14 on v2).
  - The near-chance input claim holds for the confound-free floating set and is weaker on grounded v2.
- **Grouping beyond instance identity exists but is small.**
  - Pair-probe selectivity against the instance-level shuffle is +0.11 at layer 24 (p = 1e-10) and about 0 at layer 0.
  - At layer 24, the trained model exceeds the random-init models by +0.26 and +0.14.
- **Seeded probes reproduce round 1** (v2 motion AUC 0.569 to 0.998; pair AUC 0.770 to 0.918).

### 15.4 Causal specificity of the five heads (review W5)

There are 40 matched random sets (4 global-attention heads + 1 frame-attention head). Three of them, all containing an early frame-attention head, break the model (static EPE above +10 px).

| Intervention | Five heads: change in slope | Null median | Five heads: change in static EPE (px) | Null median | Empirical p |
|---|---|---|---|---|---|
| Mean ablation | -0.60 | -0.01 | +3.0 | -0.01 | 0.073 (z = -3.4, robust z = -22.5) |
| Resample ablation (10 sets) | -0.53 | -0.01 | +4.5 | +0.13 | 0.18 (floor 0.09) |
| Amplification x1.5 (20 sets) | +0.35 | 0.00 | +0.6 | +0.05 | 0.048 (floor; z = 15) |

- **The p-values are limited by the null itself.** Only the three null sets that break the whole model reach the five heads' slope effect (slope -0.57 to -0.61, but static EPE +32 to +83 px). Among the 37 non-breaking sets, the largest slope drop is -0.12, about a fifth of the five heads' effect. Resample ablation reproduces the effect (-0.53), so it is not an artefact of mean ablation.
- **Effects on static error are within the null.** Under mean and resample ablation, the five heads' static-EPE change sits at the 90th percentile of the null (two-sided p 0.12 to 0.18).
- **Single resampled heads.** Block 17 head 10 (-0.32) and block 22 head 9 (-0.27) carry the most.
- **Amplification fixes magnitude, not accuracy.** At x1.5 the slope rises by +0.35 while moving EPE also rises (+6.7 px).

### 15.5 Method: a 72k-parameter edit of the five heads (plan item 10)

Setup:
- LoRA (rank 8, 71,680 parameters) restricted to the five heads' attention slices, trained on 400 new Kubric scenes (disjoint seeds).
- Loss: a reimplemented Any4D scene-flow loss plus distillation to the frozen model. Any4D's training code is not public, and its training data already includes Kubric.

| Variant | Kubric v2 slope | v2 moving EPE (px) | v2 static EPE (px) | KITTI 2-view slope | Sintel final 2-view slope |
|---|---|---|---|---|---|
| Base | 0.607 | 35.0 | 4.10 | 0.575 | 0.327 |
| (a) five-head LoRA | 0.716 (+0.11, p = 3e-26) | 28.4 (-6.6) | 4.14 (+0.04) | 0.598 (+0.022, p = 1e-10) | 0.333 (+0.007, p = 4e-07) |
| (b) scene-flow decoder only | 0.767 (+0.16) | 29.5 | 4.14 | 0.542 (-0.034) | 0.171 (-0.156) |
| (c) heads + decoder | 0.776 (+0.17) | 28.9 | 4.14 | 0.512 (-0.063) | 0.173 (-0.153) |
| (d) LoRA on 5 random matched heads | 0.615 (+0.008) | 34.6 | 4.10 | 0.578 (+0.003) | 0.327 (+0.001) |
| (e) 5 learned head gains | 0.620 (+0.013) | 34.4 | 4.11 | 0.590 (+0.015) | 0.332 (+0.005) |

- **The five-head edit is specific.** It gains +0.10 slope over the matched control. Unlike amplification, it also lowers moving error (-6.6 px), and it leaves geometry (depth, rotation and translation errors) and static error unchanged. The same gains appear on the floating (+0.11) and far-glued (+0.09) sets.
- **It transfers to real footage in the right direction, but only a little.** The gain is +0.022 to +0.025 on KITTI and +0.007 to +0.011 on Sintel, all p < 1e-4. That closes about 6% of the gap to RAFT on KITTI.
- **Fine-tuning the scene-flow decoder overfits to Kubric.** It gives the largest Kubric gains but makes real data clearly worse (Sintel slope -0.10 to -0.16, moving EPE +2.6 to +3.8 px).
- **Separation ratio and mask IoU do not improve.** The output motion is larger but not more cleanly separated.
- **The "gap" is mostly magnitude.** On floating epipolar-aligned scenes, the base model's output scene-flow magnitude already separates movers from static points with AUC 0.94, while its thresholded motion-mask IoU is only 0.31. The output carries the motion signal but under-scaled, which is the pull itself.

### 15.6 Updated claims

- **C2 (pull):** strengthened and redefined. Feed-forward reconstruction models (Any4D, VGGT) keep camera-induced motion and under-report object motion; dedicated trackers and chained flow do not. It is worse for small and occluded objects, and not explained by generic shrinkage.
- **C4 (beyond epipolar geometry):** reworded. Motion is detected from two views even when two-view geometry is ambiguous, and the model gains nothing from views that resolve it geometrically. So detection relies on a learned two-view or appearance cue, not multi-view consistency. Size is not the cue.
- **Mechanism:** five heads, now with null distributions; specific under mean and resample ablation; editable with a 72k-parameter LoRA.
- **Probe claims** survive random-init and shuffle controls.

### 15.7 Remaining limitations

- The unidentified two-view cue (C4).
- The real-data gain of the method is small. Training was on Kubric only, and the loss is a reimplementation.
- There is still one dynamic and one static model per family.
- Head-null p-values are floored by the number of null sets.
- CoTracker3 was not run on real data; RAFT is the real-data baseline.
