# CVPR review: "Probing 4D Feed-Forward Reconstruction Models: Multibody Structure"

Reviewer role: area expert in multi-view geometry and interpretability. Basis: plan sections 1 to 14 (E5_multibody_plan.md), the four final figures, the queue results and the public repository.

## Summary of the submission

The paper asks whether 4D feed-forward reconstruction models (mainly Any4D) represent multibody structure in the sense of Vidal et al. (IJCV 2006). The authors build controlled Kubric scenes (v1, a confound-reduced v2, and two control sets: floating shadowless scenes and "far-glued" rigid pairs). They then test four claims:

- **C1, grouping:** rigid-body grouping is built gradually through the layers.
- **C2, the pull:** moving objects are pulled toward the camera-induced motion.
- **C3, copy splitting:** identical copies are split by motion.
- **C4, beyond epipolar geometry:** motion is inferred beyond two-view epipolar geometry.

Evidence comes from:

- output metrics (an object-motion "slope", multibody separation ratio, Vidal segmentation);
- linear probes on the residual stream;
- attention hit rates;
- a single-unit mean-ablation screen that finds five "supply" heads;
- path patching into the scene-flow decoder;
- LEACE erasure and steering;
- a view-count sweep;
- a cross-model comparison (MapAnything, VGGT-1B, PAGE-4D);
- an output-level evaluation on KITTI 2015 and Sintel.

Main reported results:

- Any4D recovers only 0.3 to 0.76 of true object motion on every dataset. VGGT-1B recovers even less (0.34).
- Five heads carry almost all object motion to the scene-flow decoder: ablating them gives slope 0.01, and amplifying them by 1.5 gives slope 0.95.
- Object motion becomes linearly decodable inside the transformer from a near-chance input, including epipolar-aligned motion and with only two views.
- Grouping follows object instances rather than rigid bodies, so C3 is reframed.

## Strengths

1. **A clear, testable behavioural finding with real-data support.** The under-recovery of object motion holds on Kubric (0.73 to 0.76), KITTI (0.58, 84% of 195 scenes below 1) and Sintel (0.31 to 0.33). It is a concrete, falsifiable property of these models, and practitioners will care about it.
2. **Good experimental hygiene on confounds.** The authors found that v1 leaked motion through object size and floating (input probe AUC 0.99). They rebuilt the dataset, then added floating shadowless and far-glued controls. Few interpretability papers re-run their own positive results after removing a shortcut, and this one also reports the result that did not replicate (the v1 glued-pair clustering contrast).
3. **Causal analysis that goes beyond correlation.**
   - The five heads were selected on v1 and validated on unseen v2 scenes (29 or 30 of 30 in the expected direction).
   - Path patching localises the effect to the scene-flow decoder, through its layer-17 and final taps.
   - The encoder tap reproduces the base run exactly, which is a good sanity check.
4. **A useful dissociation between geometry and correspondence.** The 3D slope from each frame's own pointmap is about 1 for all four models; the deficit appears only in cross-frame correspondence. This sharpens what "the pull" is and where a fix should go.
5. **Honest reporting of null and mixed results.** LEACE and steering are reported as non-specific, C3 is reframed, and the amplification "fix" is reported with its moving-error cost.
6. **Reproducibility.** Public code, pinned model revisions, scripts for every experiment, and per-scene results.

## Weaknesses

### W1. The Vidal framing is mostly decorative, and the authors' own calibration undermines it

The title and introduction promise an analysis grounded in two-view multibody structure from motion. In practice:

- **The Vidal machinery does not hold up on this data.**
  - Eq. 19 had to be replaced because, as printed, it always returns one motion.
  - Three motions are unreliable at every SNR tested.
  - Reliable two-motion segmentation needs SNR above about 16, while patch-level correspondences have a median SNR of about 2.8.
  - So the multibody segmentation of internal correspondences runs in a regime the paper's own Step 1 shows to be unreliable.
- **The main evidence never uses multibody geometry.** C1, C3 and C4 rest on linear probes, k-means and attention hit rates. C2 rests on a slope against a static hypothesis, not on the multibody fundamental matrix.

The paper should either make the Vidal toolkit carry real weight (for example, report multibody separation and rank-based motion counts on GT-quality correspondences extracted from the model) or present it as background motivation and retitle accordingly.

### W2. The "camera-motion pull" may be ordinary regression shrinkage

A model trained with an L1 or L2 loss on scene flow, under uncertainty, will predict something closer to the conditional mean. Since most training pixels are static, that shrinks object motion toward zero, which is exactly toward the static (camera-only) hypothesis. The paper treats the pull as a structural property of multibody reasoning but does not separate it from this generic attenuation. Needed:

- **A dose-response test.** Does the pull grow with predictive uncertainty (low texture, small objects, large motion, occlusion)? This is item 9 in the authors' own list, which was not run.
- **A non-reconstruction baseline.** Is the pull present in a dedicated optical-flow or point-tracking model (RAFT, CoTracker) on the same scenes? If CoTracker also recovers about 0.6 of motion, the pull is a property of dense correspondence under uncertainty, not of 4D reconstruction.
- **Care with the VGGT number.** The VGGT result (slope 0.34) uses a separately trained tracker head ("tracker_fixed"), so its pull may reflect that head's training rather than the reconstruction backbone.

### W3. The view-count sweep tests an ill-posed hypothesis

In the epipolar-aligned split, the camera translates sideways at a fixed orientation and the object translates parallel to the camera.

- **Why the view count can't matter.** Relative to the camera, a constant-velocity object moving along the baseline is indistinguishable from a static object at a different depth, whatever the number of views. Vidal's two-view ambiguity therefore does not go away with 3, 4 or 7 views in this configuration.
- **What the result does show.** Equal decodability for 2 and 7 views is expected, not evidence of "going beyond two-view geometry". It shows the model uses non-geometric cues (single-image depth, size priors), which the authors propose but do not test.
- **Two fixes are possible:**
  - use a configuration that is ambiguous with two views but resolvable with more, such as an object whose velocity changes between frames, or a camera that rotates after the first pair;
  - or test the monocular-depth hypothesis directly: scale objects to break the size prior, or swap textures to break the depth prior.

### W4. Decodability is not shown to be used, and the C4 claim leans on it

Item 5 found that LEACE erasure at layers 17 to 21 reduces the slope but also raises static error by 7 to 17 px, and that steering along the motion direction does no better than a random direction. So the key C4 statement ("the transformer computes motion beyond two-view geometry") rests on linear decodability alone. Two further problems:

- **The output barely reflects it.** The motion-mask IoU on floating epipolar-aligned scenes is 0.31, against 0.56 elsewhere, so the decoded information is mostly not expressed.
- **Probe controls are missing.** There is no randomly initialised Any4D, no DINOv2-only baseline beyond layer 0, and no label-shuffle selectivity (item 6 of the authors' list was dropped). A near-chance linear probe at layer 0 does not rule out nonlinearly available information in the encoder features.

I would ask for item 6 and for a nonlinear probe at layer 0.

### W5. Causal specificity is weaker than the text suggests

- **Static error rises too.** Ablating the five heads raises static error from 4.1 to 7.1 px (73%). That is far from "sparing static geometry".
- **The random control is too small.** Only three random 5-head sets were drawn, and one broke the model (static error 87 px). Before calling these five heads specific, the paper needs a distribution over at least 20 matched random sets, plus a resample-ablation variant (patching activations from another scene instead of the token mean).
- **Amplification fixes magnitude only.** At a scale of 1.5, moving-object error rises from 35 to 42 px even as the slope reaches 0.95, so the direction or location of motion gets worse while its magnitude improves. The paper should say this plainly.
- **The screen cannot see distributed computation.** Single-unit screening found no unit that removes the pull. A set-level search (greedy or learned masks) would strengthen the "under-supply" interpretation.

### W6. Cross-model conclusions are under-powered and confounded

- **One pair per family.** There is one static-to-dynamic pair per family (MapAnything to Any4D, VGGT to PAGE-4D), and the two pairs point in opposite directions. The paper cannot separate "dynamic training" from each team's fine-tuning recipe and data.
- **The pull is compared on only two models.** MapAnything and PAGE-4D have no 2D motion output, so the headline behaviour is compared only between Any4D and VGGT.
- **One metric doesn't belong in the table.** As the authors themselves show, the pointmap 3D slope cannot measure the pull, so it should not appear as a cross-model behavioural metric.
- **Relative depth is not comparable across architectures.** Layer depth is normalised (0 to 1), but VGGT and PAGE-4D have 48 blocks with interleaved frame and global attention, while Any4D has 24. The "rise at 0.6 relative depth" comparison is not obviously meaningful.

### W7. The real-data evaluation is useful but narrow

- **KITTI.** The separation ratio is undefined for most scenes, because cars move along the ego direction, so KITTI supports only the slope.
- **Sintel.** Motion is mostly non-rigid (characters), which violates the rigid multibody assumption the paper builds on. "Instances" there are connected moving regions.
- **Input mode.** The square 518 px input sees only about 28% of KITTI's moving pixels, and results differ substantially between square and wide input. That is worth a sentence in itself.
- **Protocol mismatch.** The real-data protocol excludes motion-boundary pixels and the Kubric protocol does not. Harmonise them before comparing numbers side by side.
- **No rigid multi-object real benchmark.** Dynamic Replica and PointOdyssey were blocked, which is understandable but leaves a gap.

### W8. Statistics and risk of post-hoc framing

- **Small subsets, many tests.** Many conclusions rest on small subsets: 9 epipolar-aligned scenes in v2, 26 in the floating set, and 13 to 16 usable pairs per type in far-glued. Dozens of comparisons are reported without correction for multiple testing.
- **Unseeded probes.** The pair probe is not seeded, and the authors acknowledge run-to-run variation. Probe results should be averaged over seeds and reported with intervals.
- **Hypotheses revised along the way.** The claims evolved across three dataset versions, C3 was reframed after the far-glued result, and the v2 view sweep was reinterpreted after it contradicted the prediction. The paper should state which hypotheses were pre-specified and which were revised, or split results into confirmatory and exploratory.

## Questions for the authors

1. How does the slope depend on predictive uncertainty, object speed and object size? Can the pull be distinguished from regression attenuation?
2. Does a dedicated tracker (CoTracker) or flow model (RAFT) on the same Kubric scenes show the same 0.6 slope?
3. In the epipolar-aligned configuration, what geometric information do 7 views add over 2? If none, what does the view sweep test?
4. What happens to decodability on floating epipolar-aligned scenes if object size is randomised independently of depth, breaking the single-image size cue?
5. How specific are the five heads when compared against at least 20 random matched sets, and under resample rather than mean ablation?
6. Can a decoder-side intervention (fine-tuning only the scene-flow head, or only the five heads) close the gap between internal decodability and output IoU? That would turn the analysis into a method.
7. Do the C1 and C3 probe curves survive a randomly initialised Any4D and a label-shuffle control?

## Missing comparisons to consider

- Dense trackers and scene-flow methods as non-reconstruction baselines (CoTracker, RAFT-3D).
- Classical multibody structure-from-motion and motion-segmentation baselines on the same scenes, to show what an explicit geometric method achieves at the same SNR.
- Prior interpretability work on 3D foundation models beyond Bratulic et al., for example probing depth and correspondence in DUSt3R and VGGT.

## Overall assessment

The paper has one solid, practically relevant finding: 4D feed-forward models under-recover object motion. This holds across synthetic and real data and across two models, and it is localised to cross-frame correspondence rather than per-frame geometry. There is also a credible partial mechanism (five heads feeding the scene-flow decoder), and the care taken over confounds is above average.

Against this:

- the theoretical framing (Vidal multibody SfM) does little work;
- the C4 view-sweep result is uninformative by construction;
- the step from "decodable" to "computed" is unsupported after item 5;
- the pull lacks a test against generic regression shrinkage;
- the cross-model story has one pair per family, and the two pairs disagree.

**Rating: borderline reject (3 of 6).** Addressing W2, W3 and W4 with the experiments listed would move this to weak accept.

**Confidence: 4 of 5.**

## What would most change my score (in priority order)

1. **Regression-shrinkage control (W2).** Dose-response of the slope against uncertainty, speed and size, plus CoTracker and RAFT on the same scenes. This is cheap and decides whether C2 is a property of 4D models.
2. **A non-degenerate view-count test and a test of the monocular cue (W3).** Varying-velocity objects or a rotating camera, and size randomisation on floating epipolar-aligned scenes.
3. **Probe controls (W4, item 6).** Random-init Any4D, DINOv2-only features, label-shuffle selectivity, and seeded probes with intervals.
4. **Stronger causal specificity (W5).** At least 20 random matched head sets, resample ablation, and an explicit statement that amplification fixes magnitude only.
5. **A method angle.** Fine-tune only the five heads, or only the scene-flow decoder, and report gains on KITTI and Sintel. This turns a diagnostic paper into one with an improvement.
6. **Reframe the title and introduction** around the behavioural finding and the mechanism, presenting Vidal as motivation and as a tool for building the controlled scenes.
