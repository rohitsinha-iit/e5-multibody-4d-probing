# Framing E5 for CVPR

Basis: plan sections 14 and 15 (all round-1 and round-2 runs). Everything below rests on numbers already in the plan.

## 1. The paper in one sentence

Feed-forward 4D reconstruction models detect independently moving objects internally, but report only a fraction of their motion. The camera-induced part of the motion is kept and the object's own part is shrunk. This behaviour is absent in dedicated trackers, is carried by a handful of attention heads that feed the scene-flow decoder, and can be partly corrected by editing those heads.

## 2. Suggested title

Primary: **"Seen but Under-Reported: How 4D Feed-Forward Reconstruction Models Handle Independently Moving Objects"**

Alternatives:
- "Where Does Object Motion Go? Probing Multibody Structure in 4D Feed-Forward Reconstruction"
- "Feed-Forward 4D Models Keep the Camera and Shrink the Object"

Do not put Vidal or "multibody structure from motion" in the title. Use Vidal's theory as motivation and as the tool that defines the controlled scenes and the two-view ambiguity.

## 3. Contributions (as they should read in the introduction)

1. **A diagnostic toolkit and controlled data.** Kubric scene families that isolate the classical multibody ambiguities:
   - grounded and floating, size-matched motion;
   - far-apart rigid pairs;
   - epipolar-aligned motion with constant and accelerated velocity (zero two-view multibody separation by construction);
   - randomised size.

   Plus a decomposition of predicted motion into camera-induced and object components (alpha, beta), applied identically to reconstruction models and to trackers.
2. **A behavioural finding that holds on real footage.**
   - Any4D recovers 0.61 of object motion on Kubric, 0.58 on KITTI and 0.31 to 0.33 on Sintel, while keeping the camera component exactly (beta about 1).
   - VGGT-1B shows the same pattern (0.34).
   - CoTracker3 (0.96) and chained RAFT (0.88; 0.90 to 0.95 on KITTI and Sintel) do not.
   - Each frame's own 3D reconstruction is correct (pointmap 3D slope about 1 for all four models), so the deficit lies in cross-frame correspondence.
3. **A representational finding.**
   - Motion is not linearly decodable at the input, and not at all in random-init or DINOv2-plus-random models. The trained transformer makes it decodable (AUC about 0.5 to 0.999, selectivity +0.48).
   - This happens even for epipolar-aligned motion and with two views.
   - Views that make accelerated motion geometrically resolvable add nothing (difference-in-differences 0.000 to 0.003).
   - So motion detection relies on a learned two-view cue rather than on multi-view geometric consistency. Apparent size is ruled out as that cue.
4. **A mechanism.**
   - Five attention heads carry object motion into the scene-flow decoder through its block-17 and final-layer taps.
   - Mean or resample ablation removes almost all object motion (slope -0.60 and -0.53). No non-breaking random matched set exceeds -0.12.
   - Static error stays within the null distribution.
5. **A targeted edit.**
   - A 72k-parameter LoRA on only those five heads raises the slope by +0.11 and lowers moving error by 6.6 px on held-out Kubric, with geometry unchanged.
   - The same edit on matched random heads does nothing (+0.008).
   - It transfers to KITTI and Sintel in the right direction (+0.02 and +0.01).
   - Fine-tuning the scene-flow decoder instead overfits to Kubric and hurts real data.

## 4. Paper structure (8 pages)

1. **Introduction.**
   - Lead with figure F1: the alpha-beta plot of Any4D and VGGT vs CoTracker3 and RAFT, with the KITTI and Sintel slopes.
   - The question is "do 4D feed-forward models represent independently moving objects, and if so, why do their outputs under-report them?"
2. **Related work.** 3D/4D feed-forward models (DUSt3R, VGGT, MapAnything, Any4D, PAGE-4D, and the training-free dynamic wrappers); point tracking and scene flow; multibody SfM (Vidal) and motion segmentation; interpretability of vision transformers and 3D foundation models (Bratulic et al.).
3. **Setup.**
   - Models and controlled scenes.
   - Metrics: slope, alpha and beta, separation ratio.
   - Why epipolar-aligned and accelerated scenes isolate the two-view ambiguity (cite Vidal).
   - The Step 1 noise floor goes to the supplement and justifies why we do not run multibody segmentation on patch-level matches.
4. **Behaviour: the camera-preserving pull.**
   - Kubric and real data.
   - Comparison against trackers and flow.
   - Dose-response: worse for small and occluded objects, independent of texture and confidence.
   - Geometry vs correspondence dissociation.
5. **Representation: motion is detected early, without multi-view geometry.**
   - Layerwise probes with random-init, DINOv2-only and shuffle controls.
   - The accelerated view test.
   - The size test.
   - Grouping follows instances, plus a small motion component; identical copies are split. This is the reframed C3, kept as a secondary result.
   - Cross-model: motion decodability also appears in static models.
6. **Mechanism: five heads feed the scene-flow decoder.**
   - Screen, then null distributions.
   - Resample ablation and path patching.
   - Amplification fixes magnitude but not accuracy.
   - Erasure and steering (non-specific) go to the supplement with an honest one-line summary.
7. **Editing the mechanism.** Five-head LoRA vs controls, Kubric and real data, geometry preserved. Why decoder fine-tuning overfits.
8. **Discussion and limitations.**

## 5. Main figures

| Fig. | Content | Source |
|---|---|---|
| 1 | Teaser: alpha vs beta for all models on Kubric, KITTI and Sintel; inset of an example scene with the predicted vs true motion of one object | fig_w2_alpha_beta.png, fig_c2_pull_everywhere.png |
| 2 | Controlled scene families and the two-view ambiguity (contact sheets plus the 3-view residual) | v4_scale_epi.png, pilot sheets |
| 3 | Probe curves with controls; accelerated view test | fig_probe_controls.png, fig_v4_accel_viewcount.png, fig_c4_beyond_epipolar.png |
| 4 | Mechanism: head null histograms, path-patching taps | fig_heads_null.png, causal results |
| 5 | Edit: variants on Kubric vs real data | ft_comparison.png |

## 6. What to claim and what not to claim

**Claim:**
- The camera-preserving pull in feed-forward reconstruction models, with numbers on Kubric, KITTI and Sintel.
- Its absence in trackers and chained flow.
- The internal detection without multi-view geometry.
- The five-head mechanism.
- The targeted edit and its specificity.

**Do not claim:**
- That models do multibody structure from motion.
- That the edit is a practical fix. The real-data gain is small: about 6% of the KITTI gap to RAFT.
- That dynamic training causes multibody structure. The cross-model results go in both directions.
- That we know the two-view cue.
- That the decodable direction is the one used. LEACE and steering were non-specific.

**Reword C3** as "grouping by instance plus motion splitting".

## 7. What reviewers will still attack, and the prepared answer

1. **"The fix is weak on real data."** Present it as a validation of the mechanism and a controlled comparison (five heads vs random heads vs decoder), not as a method contribution. The decoder baseline overfitting while the head edit transfers is itself a finding. If time allows, train on a mix of Kubric and PointOdyssey-like data (if a non-gated source exists), or self-train with RAFT or CoTracker pseudo-labels on real video. That is the most promising way to turn the edit into a real-data gain.
2. **"What is the two-view cue?"** Say plainly that it is unidentified, and list what was ruled out: floor contact, shadows, apparent size, and extra views. A cheap follow-up is a texture or illumination swap between frames, to test appearance change versus monocular depth.
3. **"Only one model per family."** Add one more dynamic model with a correspondence output if any public one exists. Otherwise, frame the claim as "two reconstruction models with correspondence heads vs two correspondence models", which the data supports.
4. **"Head p-values are not small."** Show the histograms. The only random sets that match also destroy static geometry (static EPE +32 to +83 px). The non-breaking maximum is -0.12 against -0.60.
5. **"Sintel is non-rigid."** It is used only for the behavioural slope. KITTI covers rigid real motion.
6. **"Mixed statistics and many tests."** Use Holm correction across the pre-registered round-2 tests (the view and size tests already use it), report effect sizes with CIs, and label round-1 exploratory results as such.

## 8. Expected standing

- **Before round 2:** borderline reject. The main doubts were shrinkage, a degenerate view test, missing probe controls and weak specificity.
- **After round 2:** these four doubts are addressed with pre-specified tests, and the paper has a clear behavioural finding with a baseline contrast, a mechanism with null distributions, and a specific edit.
- **Outlook:** a solid weak accept to accept as an analysis paper at CVPR. The main residual risks are the small real-data gain of the edit and the unidentified two-view cue.

## 9. Short list of remaining experiments, if time allows

1. Pseudo-label self-training of the five-head LoRA on real video (RAFT or CoTracker targets), evaluated on KITTI and Sintel. This is the best chance of a stronger method result.
2. An appearance-swap test for the two-view cue.
3. CoTracker3 on KITTI and Sintel, to complete the baseline table.
4. One more correspondence-capable 4D model, if public weights exist.
