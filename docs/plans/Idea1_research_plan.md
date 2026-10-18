# Research plan: the causal geometry of camera and object motion attribution in 4D feed-forward models

Version 1, 2026-10-18. This document folds Idea 1 ("the causal geometry of camera and object motion disentanglement", from the pasted research agenda) into the E5 paper together with P3. The result is one coherent CVPR 2027 submission. The implementation and experiment plan that follows from it is in a separate file, Unified_E5_P3_Idea1_implementation_plan.md.

## 1. One narrative for the whole paper

Every frame-to-frame image displacement of a scene point is the sum of a camera-induced part and the point's own motion. With u_t = pi(K_t T_t X_t), a change in u_t can come from the camera transform T_t, from the point position X_t, or from both. A 4D reconstruction model must therefore attribute image motion to ego-motion and to object motion. Classically this is the multibody and ego-motion ambiguity (Vidal et al. 2006), which E5 already uses as its motivation.

The E5 paper becomes a study of how feed-forward 4D models make this attribution, why they get it wrong, which internal pathways carry each part, and how those pathways can be repaired. The four parts of the story map onto the three work streams:

| Question | Answer we test | Stream | Status |
|---|---|---|---|
| What goes wrong? | Object motion is under-attributed (E5 pull: beta about 1, alpha 0.61 for Any4D, 0.22 for VGGT). Object motion may also leak into the camera estimate. | E5 behaviour + P3 X1 + Idea 1 behaviour level | E5 done; P3 and Idea 1 parts new |
| Why? | Shrinkage of a static-world residual (P3 account R), versus attribution of object motion to the camera (account K). | P3 X2 + Idea 1 factorial interaction and motion budget | new |
| Where in the network? | Partly separable pathways: an object-motion supply pathway (the five E5 heads) and a camera pathway, with identifiable cross-talk. | E5 mechanism + Idea 1 interchange interventions | E5 heads done; interchange new |
| Can it be fixed without retraining? | Two GT-free, circuit-guided edits: an object-gain edit for under-attribution, and contamination masking for camera pose. Both use the model's own internal motion signal. | E5 LoRA + P3 X5 + Idea 1 repair level | partly done; new edits |

The coherence comes from one observation that E5 already established: these models "see" independent motion internally (motion is linearly decodable at layers 17 to 24 with AUC up to 0.999 even for epipolar-aligned motion) but "under-report" it in their outputs. Idea 1 asks what the network does with what it sees. The repair section closes the loop by using the internally seen motion as the mask and the gain signal.

Working title: "Where Does Object Motion Go? Motion Attribution in Feed-Forward 4D Reconstruction". The E5 framing listed it as an alternative. The current "Seen but Under-Reported" can stay as the subtitle.

## 2. Research lineage and positioning (all papers below were retrieved and checked on 2026-10-18)

1. Classical: multibody structure from motion (Vidal, Ma, Soatto, Sastry, IJCV 2006) and ego-motion versus independent motion. It defines the attribution problem, and E5 uses it to build the controlled scenes.
2. Feed-forward geometry and its mechanisms. Bratulić et al., "On Geometric Understanding and Learned Priors in Feed-forward 3D Reconstruction Models" (arXiv 2512.11508), show that epipolar geometry emerges in intermediate layers of DUSt3R, VGGT and DA3 and is causally tied to correspondence heads. That work covers static scenes. Our step is the dynamic case: how matching becomes motion attribution.
3. Feed-forward 4D models that disentangle by design:
   - MoRe (arXiv 2603.05078, CVPR 2026) is motivated by moving objects corrupting camera pose and adds attention forcing on a VGGT backbone.
   - 4RC (arXiv 2602.10094, ICML 2026) factorises outputs into base geometry plus relative motion.
   - Any4D (arXiv 2512.10935) predicts allocentric scene flow.
   - SM4RT (arXiv 2607.22534) uses structured motion bases.

   These build separation into the architecture or the loss. We ask whether separation emerges inside networks trained to solve the task, and what failure follows when it does not.
4. Representation studies of video and 4D models: How Much 3D Do Video Foundation Models Encode? (arXiv 2512.19949) and Interpreting Physics in Video World Models (arXiv 2602.07050). These are probing and steering studies. As the pasted agenda notes, another probe study would be weak, so probes enter our paper only as the source of the internal motion signal and as controls.
5. Post-hoc correction of feed-forward 4D trackers: GAUGE (arXiv 2609.32596) corrects group-wise radial errors with metric anchors. It addresses a different error mode, with a different method. It is a related-work and baseline citation.
6. Causal-interpretability methodology:
   - Causal abstraction (Geiger et al., arXiv 2301.04709) and distributed interchange interventions (Geiger et al., arXiv 2303.02536) provide the formal frame: a high-level causal model with variables Camera and Object is aligned to network components through interchange interventions.
   - Towards Best Practices of Activation Patching (Zhang and Nanda, arXiv 2309.16042) supplies the metric and corruption choices.
   - Is This the Subspace You Are Looking For? (Makelov et al., arXiv 2311.17030) is the main pitfall: subspace patching can activate dormant pathways. Our design avoids learned subspaces in the primary analysis for this reason.

Novelty gap, stated carefully. A keyword search on arXiv found no 2025 or 2026 paper that runs interchange interventions on pretrained 4D reconstruction transformers with camera and object motion as causal variables. This is weak evidence (keyword search), not a guarantee. The closest risks are MoRe (the failure mode, solved architecturally) and Bratulić et al. (the methodology, on static scenes). The paper must show three things that neither has: a factorial counterfactual design with exact counterfactual targets, a double dissociation of pathways in models not built to separate motion, and a GT-free repair that beats matched random and oracle-mask baselines.

## 3. The causal model we test

High-level variables per scene: C (camera trajectory), O (independent object motion), S (scene structure and appearance at frame 0). In a correct allocentric 4D model:
- predicted camera poses depend on C only;
- predicted object scene flow depends on O only;
- static-background flow is zero;
- image correspondence x_t = pi(K_t T_t (X_0 + F_t)) depends on both.

Two failure modes break this:
- Under-attribution (E5): predicted object flow F_obj is a shrunk version of O.
- Contamination (Idea 1): predicted camera depends on O.

A third possibility links them, and is the most interesting to test. The missing object motion could be absorbed by the camera or by depth instead of vanishing. Then the two failures are one failure seen from two sides.

Alignment hypotheses for the network:
- H1 (separable pathways): there exist small component sets P_obj and P_cam such that interchanging P_obj between inputs that differ only in O changes object flow and not camera, and interchanging P_cam between inputs that differ only in C changes camera and not object flow (double dissociation).
- H2 (contamination pathway): object information reaches the camera head through a small set of attention heads that read moving-object tokens. Blocking those reads restores the camera without changing object flow.
- H3 (under-supply): the five E5 supply heads belong to P_obj, and their output along the counterfactual object-motion direction sets the gain alpha.

Competing accounts for why alpha is below 1, with their predictions on the factorial design:

| Account | Prediction |
|---|---|
| R, residual shrinkage toward a static world (P3) | alpha is the same with a static and a moving camera (no C x O interaction). The image-space deficit sits in the flow term of the motion budget. Spurious ego-motion is near zero with a static camera. |
| K, attribution to camera | alpha is lower when the camera moves (negative interaction). The budget shows the camera term absorbing part of the object motion. A static camera with a moving object produces spurious ego-motion. P_obj and P_cam share components. |
| G, attribution to geometry (links E5's monocular-depth-prior conflict) | The budget shows the depth term absorbing the deficit. The object's frame-0 depth is biased in the direction that explains its image motion as parallax. |

The accounts are not exclusive. The factorial design measures each term with exact counterfactual ground truth.

## 4. Design: four levels, one counterfactual benchmark

### 4.1 The factorial counterfactual set ("MotionFactorial")
For each base scene from the E5 Kubric generator, we render four variants from the same seed:

| Variant | Camera | Objects |
|---|---|---|
| A | static | static |
| B | moving | static |
| C | static | moving |
| D | moving | moving |

- D is the existing E5 scene.
- "Static camera" holds the camera at its frame-0 pose.
- "Static objects" holds them at their frame-0 pose.
- Geometry, textures and lighting are unchanged.
- Frame 0 is identical across the four variants by construction. The plan copies D's frame 0 into A, B and C, and checks that the rendered frame 0 differs only by sampling noise.

This one property solves the token-alignment problem that the pasted agenda warns about. At layer 0, every view-0 token sees the same input in all four variants. Any difference at a later layer is information routed from later frames, so view-0 activations can be swapped between variants without misregistration. Any4D's scene flow is defined on view-0 pixels, so the object pathway is read exactly where alignment is perfect.

Base scenes are the E5 v2 grounded set (100) and the v3float set (70). An optional 3 x 3 dose subset (camera and object motion scaled by 0, 0.5 and 1) covers 40 scenes.

### 4.2 Level 1, behaviour (what goes wrong, and where the motion goes)
Measures per model on A to D and on real data:
- Object-motion attribution: image slope, alpha and beta (E5 definitions), and the world-frame slope s_w (P3). Report these in C versus D, which gives the C x O interaction.
- Contamination: relative pose error in D minus B, paired by scene. In C, report spurious ego-motion (rotation angle, and translation divided by median scene depth) against A.
- Motion budget: a Shapley attribution of the image-space deficit along the object-motion direction to three components, geometry X_0, flow F_t and camera (K_t, T_t). Each component is replaced by its GT value in all 6 orders. This works for models whose correspondence is composed from these parts (Any4D, 4RC).
- Real data: camera pose on Sintel, TUM-dynamics and Bonn (preprocessed sets from HF yslan/pointmap_regression_evalsets), stratified by the fraction of dynamic pixels. Also the E5 KITTI and Sintel flow protocols.

Models:
- Any4D and VGGT (E5 references);
- OpenD4RT and TrackCraft3R (P3);
- 4RC: a third reconstruction family member with an explicit base-plus-relative-motion output, a natural in-the-wild instance of account R;
- MoRe: an architectural answer to contamination, so it is the baseline the repair must be compared with;
- CoTracker3 and RAFT lifted to the world axis (P3 controls).

TrackCraft3R takes cameras as input, so it enters the object side only.

### 4.3 Level 2, localisation (which components matter)
On Any4D, which has known internals and the five E5 heads, and on VGGT:
- Object pathway: interchange with base D and source B (only O differs) at view-0 tokens.
  - Per-head output patching, all 24 x 12 heads in Any4D, plus the residual stream per layer.
  - Score tau_O, the fraction of the object-flow change toward B that is transferred.
  - Leakage, the camera change caused, normalised by the size of the camera effect between C and D.
- Camera pathway: interchange with base D and source C (only C differs).
  - Positions: view-0 tokens and position-free tokens, which are aligned. In VGGT these are the per-frame camera tokens, so the camera interchange is clean there; in Any4D, the scale token.
  - Score tau_C and the object-flow leakage.
  - For Any4D view-t tokens, a GT-warp alignment is used only as a secondary analysis, with a measured misregistration floor: a warp-and-unwarp round trip of D into itself.
- Contamination pathway, per head:
  - At view-t query tokens, remove attention to moving-object keys (GT mask; renormalise).
  - Score kappa, the fraction of the excess camera error of D over B that is removed, and its effect on object flow.
- Necessity: E5 already showed it for the object side (mean and resample ablation of the five heads). It is repeated for the top camera and contamination components.
- Nulls: random component sets matched by block, type and activation norm (40 sets).

### 4.4 Level 3, causal interchange of the full pathway (sufficiency and specificity)
- Take the top-k components by tau_O, tau_C and kappa on a fit split of scenes, and patch them jointly on a held-out split.
- The H1 double dissociation holds if the following both hold, with each value set before analysis (implementation plan section 8):
  - patching P_obj from B into D transfers at least half of the object-flow effect with camera leakage under a set bound;
  - patching P_cam from C into D transfers at least half of the camera effect with object-flow leakage under the same bound.
- Completeness: the fraction of the total effect explained by k components, as k grows.
- Primary interventions are on whole components (heads, tokens), not on learned subspaces (Makelov et al.).
- One distributed variant is allowed as a secondary analysis: a projection onto the span of natural counterfactual activation differences (D minus B and D minus C). These directions are on-manifold by construction, which addresses the dormant-direction illusion.

### 4.5 Level 4, mechanism-guided repair (GT-free)
1. Under-attribution: an object-gain edit in the five heads. The direction is the counterfactual object-motion direction (the mean of head output in D minus B on moving view-0 tokens), orthogonalised against the camera direction (D minus C). This replaces P3 X5's regression patterns with natural counterfactual differences.
   - Success is a double dissociation: alpha moves and beta does not; the camera edit does the reverse.
   - Baselines: whole-head gain (E5: slope 0.95 at x1.5 but moving EPE 35 to 42 px), random directions, and the E5 five-head LoRA.
2. Contamination: in the top contamination heads only, block attention from tokens that the model's own internal motion probe flags as moving.
   - The probe is trained on Kubric labels for all views (E5 probes: layers 17 to 19). At test time no GT mask is used.
   - Baselines: GT mask in all heads (an oracle attention forcing), internal mask in all heads, internal mask in k random heads, and the MoRe model.
   - Evaluated on factorial held-out scenes and on Sintel, TUM-dynamics and Bonn pose. Object flow and static EPE are reported, to show the edit is specific.

## 5. How the parts depend on each other

- E5 to Idea 1. E5 supplies the controlled scene generator, the alpha/beta and slope metrics, the five supply heads (a candidate P_obj, prediction H3), the internal motion probes (the repair mask), and the KITTI and Sintel protocols. E5's non-specific LEACE and steering result (item 5) is the cautionary precedent. It motivates whole-component interchange with exact counterfactual targets, rather than probe-direction steering.
- P3 to Idea 1. P3's account R predicts no C x O interaction and a flow-dominated budget. Idea 1's factorial design is the cleanest test of that prediction, cleaner than P3's cross-model comparison, because the camera can be switched off with everything else fixed. P3's world-frame slope and lifted-tracker controls are reused unchanged. P3 X2 (output parameterisation on frozen Wan features) remains the controlled-training proof of the cause. 4RC's base-plus-relative-motion output adds an in-the-wild residual model.
- Idea 1 to P2. The factorial C and D pairs are exact camera-counterfactual pairs with GT. They give the oracle upper bound for P2's counterfactual-camera supervision, where ReCamMaster pairs are the approximate real-video version. P2 starts after submission, and the factorial set is its synthetic testbed.

## 6. Contributions as they would read

1. MotionFactorial: a factorial counterfactual benchmark that switches camera motion and object motion independently, with frame-0 identity for token-aligned interventions, together with attribution metrics (alpha and beta, world slope, contamination, motion budget).
2. Behaviour across families: under-attribution of object motion in GFM, D4RT-style and video-generator reconstruction models but not in trackers; camera contamination by object motion; and a measured answer to "where does object motion go".
3. Cause: whether under-attribution is shrinkage toward a static world or attribution to camera or depth, tested by the factorial interaction, the motion budget, within-model readout contrasts and a controlled parameterisation experiment.
4. Mechanism: interchange-intervention maps of object, camera and contamination pathways, with a double-dissociation test against matched nulls, in two architectures (Any4D, VGGT).
5. Repair: two GT-free, circuit-guided edits that use the model's own internal motion signal, compared with random-component, oracle-mask and architectural (MoRe) baselines on synthetic and real data.

The E5 representational results (motion detected without two-view geometry; probe controls) move to the support of contribution 4 and the repair mask. The E5 view tests and the C3 grouping result go to the supplement.

## 7. What would make it strong, and what would sink it

Strong-accept profile:
- The pull is present across families, and the factorial interaction gives a clear answer (either R or K).
- A double dissociation holds in both Any4D and VGGT.
- At least one repair beats random components and approaches the oracle mask on real pose benchmarks, without hurting flow.

Accept profile:
- Behaviour plus cause plus localisation maps, with partial specificity, and the repair reported honestly.

Main threats:
1. Pathways may be distributed, so that no small set dissociates. Report the completeness curves; the cause and behaviour results still stand.
2. Contamination may be small in Any4D on Kubric (its camera may already ignore objects). Then the contamination story shifts to VGGT or real data. Either outcome is reportable, and a null contamination result combined with a strong pull supports account R.
3. Interchange artefacts from misregistration. They are handled by keeping primary interventions on aligned tokens and by measuring a misregistration floor.
4. MoRe weights may be unreachable from the server (Google Drive). Then the comparison is the oracle GT-mask attention forcing only, labelled as MoRe-like.
5. Time. The implementation plan sets the cut order: localisation and repair on VGGT go first, then 4RC, then the dose subset.

## 8. Decisions this plan asks of the user
1. Approve the paper's new spine and title: the attribution narrative, with E5's view tests moved to the supplement.
2. Approve the factorial renders (about 510 Kubric renders on the server CPU, same generator, same seeds as v2 and v3float).
3. Confirm that MoRe may be downloaded from Google Drive on the server (needs gdown and drive.google.com access). Otherwise, the oracle attention-forcing baseline only.
