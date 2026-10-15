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
