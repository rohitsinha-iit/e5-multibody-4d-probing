# E5 / P3 / Idea 1: results analysis and interpretation

Written 2026-10-25 (IST). Sources: `results/u2/confirmatory_u2.json`, `results/u2/summary_u2.md` (written 2026-10-25 02:33), `results/u4u5_confirmatory.json`, `results/u4u5_summary.json` (written 2026-10-24 16:36), `results/u4a/rank_*.json`, and the frozen pre-registration (`attr_preregistration.json`, sha256 69bdf016...). All numbers are scene-bootstrap means with 95% CIs (2000 resamples, seed 0) unless marked. The boundary rule (8 px exclusion) is primary throughout. Splits: test = v2 odd seeds (48 scenes), transfer = v3float (69 scenes), fit = v2 even seeds (50 scenes).

## 0. Bottom line

1. **The pull is real and spans families.** All five confirmatory tests reject after Holm correction. OpenD4RT and TrackCraft3R under-report world-frame object motion (s_w 0.80 and 0.49), while lifted 2D trackers on the same scenes reach 0.99 and 0.98, so the deficit belongs to the models and not to the metric. 4RC is the clear exception (s_w about 1.0).
2. **The cause is mostly shrinkage, with a smaller camera-specific part.** Any4D already loses about 17% of object motion with the camera frozen (alpha_C 0.83). Moving the camera removes a further 0.10 (T7 I = -0.098 [-0.173, -0.023]). The interaction is significant but falls just short of the pre-registered 0.10 effect size, so by rule it is **inconclusive**. The motion budget assigns the deficit to the flow channel (share about 1.0), with a camera share of only 0.03. The best-supported reading is account R as the main term, plus a modest K component.
3. **Contamination of camera pose by object motion exists but is very small.** Any4D's dRPE-R is +0.008 deg, about 4% of its RPE-R, and VGGT shows none. Masking it with the probe does not help (U5b fails).
4. **Object motion runs through a narrow bottleneck; camera motion is distributed.** One head (b14h5) carries 80% of the object-motion interchange effect on fit scenes (73% on test), with almost no pose leakage. No head set up to k = 20 reaches 80% completeness for camera motion. The double dissociation passes its rule descriptively (p = 0.024), but it is **not confirmatory**, because k* is undefined for the camera screen.
5. **H3 fails as pre-registered.** Three of the five E5 heads rank in the top 5 (b17h10 2, b22h9 3, b2h1 5), b14h6 is 17th and b8h11 is 114th on test. The E5 story needs a rewrite: the five heads are not one homogeneous supply set.
6. **Repairs are half successful.** The GT-free object-gain edit is specific against random projectors (T5p +0.105, p = 0.0005, survives Holm) and raises alpha by 0.12. It fails the full U5a success rule because static EPE rises 14% and moving EPE rises 1.1 px, and the camera-direction edit (ii) does not move beta. Contamination masking (U5b) fails.

## 1. Confirmatory family (Holm, alpha = 0.05)

| Test | Hypothesis | Estimate [95% CI] | n | raw p | Holm p | Holm decision | Pre-registered reading |
|---|---|---|---|---|---|---|---|
| T1a | OpenD4RT r3 s_w < 0.9 | 0.804 [0.749, 0.856] | 103 | 0.0005 | 0.0025 | reject H0 | pull present (T1 readable: lifted CoTracker3 0.992, RAFT 0.982) |
| T1b | TrackCraft3R (ii) s_w < 0.9 | 0.486 [0.426, 0.554] | 103 | 0.0005 | 0.0025 | reject H0 | pull present |
| T5p | object-gain edit (i), g = 1.5, beats random projectors on d alpha | +0.105 [0.073, 0.139] | 39 | 0.0005 | 0.0025 | reject H0 | edit is direction-specific |
| T7 | Any4D interaction I = alpha_D - alpha_C, two-sided | -0.098 [-0.173, -0.023] | 39 | 0.0080 | 0.0080 | reject I = 0 | **inconclusive** (0.05 <= abs(I) < 0.10); sign favours K |
| T8 | Any4D dRPE-R(D - B) > 0 | +0.0083 deg [0.0020, 0.0146] | 48 | 0.0035 | 0.0070 | reject H0 | contamination present, small |

Notes:
- 0.0005 is the floor of a 2000-resample bootstrap p (1/2001). The three smallest p-values are floor values.
- T7 is a two-level decision. Holm establishes that I is not 0, but the magnitude rule (abs(I) >= 0.10 for "effect") is not met, so the reading stays "inconclusive". The s_w repeat gives -0.076 [-0.126, -0.027]. Without the boundary rule, I = -0.123 [-0.190, -0.061], which would count as an "effect" for account K. That variant is secondary and must not replace the primary result.
- T9 (double dissociation) and T10 (masking vs random) are descriptive, as frozen (cam and cont have no k*; plan risk 4). T9 p = 0.024 passes its rule; T10 p = 0.181 fails.

## 2. Interpretation by question

### 2.1 Does the pull hold across model families? (P3, T1, cross-family s_w)

World-frame object slope s_w in factorial D, mean [95% CI]:

| model | test (v2) | transfer (v3float) | s_w in C, test / transfer | I_sw (D - C), test |
|---|---|---|---|---|
| Any4D | 0.73 [0.65, 0.81] | 0.78 [0.70, 0.86] | 0.82 / 0.85 | -0.076 [-0.126, -0.027] |
| VGGT | 0.48 [0.30, 0.65] | 0.68 [0.48, 0.85] | 0.08 / 0.07 (CI spans 0) | +0.382 [0.196, 0.579] |
| OpenD4RT r3 | 0.90 [0.86, 0.93] | 0.73 [0.65, 0.81] | 0.98 / 1.05 | -0.079 [-0.121, -0.040] |
| 4RC | 1.01 [0.99, 1.03] | 1.10 [1.04, 1.18] | 0.99 / 1.15 | +0.022 [0.005, 0.040] |
| TrackCraft3R (i) GT inputs | 0.65 [0.59, 0.71] | 0.59 [0.54, 0.63] | 0.72 / 0.73 | -0.063 [-0.114, -0.014] |
| TrackCraft3R (ii) DA3 inputs | 0.59 [0.50, 0.70] | 0.40 [0.34, 0.48] | 0.76 / 0.76 | -0.158 [-0.236, -0.079] |
| CoTracker3, lifted | 0.99 [0.97, 1.00] | 1.00 [0.99, 1.00] | 0.99 / 1.01 | -0.007 [-0.025, 0.007] |
| RAFT chained, lifted | 0.98 [0.95, 1.00] | 0.99 [0.96, 1.01] | 0.98 / 1.01 | 0.000 [-0.022, 0.024] |

Reading:
- Four of the five reconstruction families under-report world-frame object motion. For every one of them that reports object motion with a frozen camera, s_w falls when the camera moves (negative I_sw). The lifted trackers show neither effect, so the deficit is not produced by lifting, scale alignment or the scenes.
- **4RC is a clean negative control from inside the model class.** Its s_w is 1.0 on v2 and slightly above 1 on v3float, its interaction is 0 or positive, and its contamination is 0. Whatever 4RC does differently (its separate relative-motion head and camera decoder) is the architectural counterexample the paper needs. This deserves a paragraph, not just a table row.
- **OpenD4RT is the literal "seen but under-reported" case.** Its 2D attribution alpha is about 0.93 to 0.96 in D, close to the trackers, but its world-frame readout s_w is 0.90 (test) and 0.73 (transfer). The motion is present in the image-space tracks and shrinks in the world-frame readout.
- **Caveat on T1a.** The pooled pre-registered test passes, but on v2 test alone s_w = 0.898 [0.859, 0.933], p = 0.46. The OpenD4RT deficit is driven by v3float (0.73). The camera-induced drop is still present on v2 (C 0.98 to D 0.90). Report the pooled result as specified and show the split values next to it.
- **VGGT is a different regime.** With the camera frozen it reports almost no object motion (alpha_C 0.12, s_w CI spans 0). In D it gains object motion (alpha_D 0.31, I = +0.20). VGGT has no flow head, so its point maps encode a mostly static world, and the "object motion" in D partly rides on the camera solution. VGGT should be presented as a static-world baseline, not as another instance of the pull.
- **Boundary sensitivity.** Without the 8 px exclusion, lifted RAFT drops to s_w 0.872 [0.842, 0.900] and T1 would not be readable. Boundary pixels contaminate every model's s_w. The primary rule was set in advance, but reviewers will ask, so report both.

Real data (E5 protocol) agrees with the ordering. Object slopes on KITTI and Sintel: 4RC 0.73 to 0.92, Any4D 0.31 to 0.76, OpenD4RT 0.41 to 0.63, VGGT -0.01 to 0.20, with CoTracker3 at 0.81 to 0.90 and RAFT at 0.90 to 0.95 as references.

### 2.2 Why? Shrinkage (account R) vs camera attribution (account K)

Evidence for Any4D on test scenes:

| Signal | Value | Account R predicts | Account K predicts |
|---|---|---|---|
| alpha with camera frozen (C) | 0.83 [0.76, 0.89] | deficit without camera motion | alpha_C near 1 |
| interaction I (T7) | -0.098 [-0.173, -0.023] | about 0 | clearly negative |
| motion budget, flow share (D) | 1.01 [0.98, 1.05] | flow dominant | camera share large |
| motion budget, camera share (D / C) | 0.03 / 0.03 (transfer D 0.11) | about 0 | large in D, about 0 in C |
| spurious ego-motion, C - A rotation | +0.016 deg [0.008, 0.025] | about 0 | positive |
| contamination dRPE-R(D - B) (T8) | +0.008 deg [0.002, 0.015] | about 0 | positive |

Reading:
- Most of the deficit is already present without camera motion. Any4D recovers about 83% of object motion in C, which is a shrinkage-like loss of roughly 17 points out of the roughly 23 points lost in D (alpha_D 0.77). The budget attributes the deficit to the flow channel rather than to the camera or geometry channels. Both observations point to account R as the main term.
- A real camera-specific component sits on top: a negative interaction of about 0.07 to 0.10 on test, transfer and the s_w repeat (smaller on the fit split, -0.045 with a CI spanning 0), plus small but significant contamination and spurious ego-motion. This is the account K term, and it is second order in size.
- Paper wording: "Object-motion under-reporting in Any4D is mainly a shrinkage of the residual toward a static world (present with a frozen camera). Camera motion adds a smaller, significant penalty (I = -0.10, inconclusive against the pre-registered 0.10 threshold). Object motion leaks into the camera estimate only marginally (+0.008 deg RPE-R)." Because X2 was paused, the cause claim stays at "consistent with".
- This also corrects the E5 wording "not uncertainty-driven shrinkage". The factorial data show a shrinkage-like deficit with a frozen camera. The defensible claim is narrower: the deficit is not explained by camera attribution alone.
- The 4RC motion budget is unstable (for example, transfer C geometry share -1.76) because 4RC's deficit is near zero and the budget is a ratio over the deficit. Do not show 4RC budget bars (F1c currently does).

### 2.3 Where in the network? (U4a, Any4D)

Single-head interchange on fit scenes. Ranks below count heads only; the 24 whole-residual-stream components, which trivially carry everything, are excluded.

| head | objall tau_O (rank) | cam tau_C | cont kappa (rank of 288) |
|---|---|---|---|
| b14h5 | 0.80 (1) | 0.055 (top camera head) | 0.80 (15) |
| b17h10 | 0.62 (2) | 0.001 | 0.79 (116) |
| b22h9 | 0.54 (3) | 0.000 | 0.79 (120) |
| b2h1 | 0.18 (5) | -0.001 | 0.82 (4) |
| b14h6 | 0.024 (12) | -0.003 | 0.80 (22) |
| b8h11 | 0.004 (40) | 0.000 | 0.84 (1) |

Completeness (scene median, test): objall 0.73 at k = 1, 0.99 at k = 3. Camera 0.56 at k = 1, 0.48 at k = 3, 0.76 at k = 20. Contamination 0.10 at k = 20.

Reading:
- **Object motion is concentrated.** One head, b14h5, transfers 80% of the B-to-D object-motion difference on fit scenes and 73% on test, with pose leakage of only 0.004. The second-best head in block 14 reaches 0.024. Three heads complete the effect (0.99). The single-head scores are far from additive (0.80 + 0.62 + 0.54 > 1). This fits a serial pathway in which b2h1, b14h5, b17h10 and b22h9 are successive stages, but that is a hypothesis and has not been tested.
- **Camera motion is distributed.** No single head transfers more than 0.055 of the camera effect, the completeness curve is non-monotonic (0.56 at k = 1, 0.48 at k = 3), and it plateaus at about 0.75. In the camwarp screen, patching the whole residual stream transfers 0.99 of the camera effect at early blocks, falling to 0.78 by blocks 21 to 23. Camera information sits in the residual stream and is not routed through a few heads.
- **Double dissociation (T9, descriptive):** P_obj (b14h5) tau_O 0.73 vs a null 95th percentile of 0.013, leakage 0.004. P_cam (top 20) tau_C 0.078 vs a null 95th percentile of -0.002, leakage 0.079. Both pass the rule; transfer replicates (tau_O 0.77). Two caveats:
  - The camera side is weak in absolute terms. Twenty heads transfer under 8% of the camera effect, so "dissociation" really means "the camera pathway does not run through the object heads".
  - The matched null for k = 1 draws one head from block 14, which has 11 alternatives, and the 40 draws produced only 6 distinct values. The p-value of 1/41 overstates the effective null size. The clearer statement is that b14h5 beats every other head in its block by more than 30 times (0.80 vs at most 0.024 on fit).
- **b14h5 is a hub, not an object-only head.** Its interchange content is object-specific, but ablating it (the necessity test) destroys object motion (tau_O 0.79), camera motion (tau_C 0.65) and pose (RPE-R 3.7 deg vs a baseline of about 0.2 deg). It is also the top head on the camera screen. The paper should say that the B-to-D difference at b14h5 encodes object motion, not that b14h5 "is" an object head.
- **The contamination screen is uninformative.** Every head scores kappa of about 0.79 to 0.84, and the joint top-20 patch reaches only 0.10 completeness. The denominator e(D) - e(B) is about 4% of the pose error, so the ratio is dominated by noise. b8h11 ranks 1st by 0.844 vs 0.789 at rank 120, so the ranking carries no information. Contamination is too small to localise at head level.
- **H3 (all five E5 heads in the top 20) fails.** Test ranks are 2, 3, 5, 17 and 114. Rewrite the E5 claim: three of the five LoRA heads (b17h10, b22h9, b2h1) lie on the object pathway found independently by interchange. b14h6 is a weak neighbour of the hub b14h5 (b14h5 was not in the E5 set). b8h11 is not on the pathway. This partial convergence of a gradient-selected LoRA set and a causal screen is a reasonable result to report, as long as it is framed as partial.
- **The joint view-0 object screen (obj, k* = 10)** gives tau_O 0.24 on test. The all-view screen (objall) is the right primary, as decided.

### 2.4 Can it be fixed without GT? (U5a, U5b)

**U5a, object-gain edit, Kubric v2 test (identifiable, n = 39), changes vs unedited:**

| variant | d alpha | d beta | d moving EPE (px) | static EPE ratio - 1 | d RPE-R (deg) |
|---|---|---|---|---|---|
| (i) P_O, g = 1.5 (primary) | +0.121 [0.089, 0.157] | -0.038 [-0.126, 0.064] | +1.14 [-1.22, 3.33] | +0.137 [0.027, 0.254] | +0.060 |
| (i) P_O, g = 2 | +0.152 [0.093, 0.211] | -0.093 | +7.57 | +1.19 | +0.198 |
| (ii) P_C, g = 1.5 | +0.042 [0.002, 0.081] | -0.033 [-0.141, 0.066] | +1.17 | +0.025 | +0.011 |
| (iii) random, g = 1.5 | +0.016 [0.013, 0.019] | +0.004 | -0.19 | -0.003 | +0.002 |
| (iv) whole-head gain (E5), g = 1.5 | +0.244 [0.195, 0.292] | +0.031 | +6.20 | +1.03 | +0.225 |
| (v) E5 LoRA checkpoint | +0.064 [0.028, 0.101] | -0.006 | -2.62 [-4.87, -0.81] | +0.007 | +0.000 |

Success rule at g = 1.5, as specified:
- (i) d alpha >= 0.10: **pass**.
- abs(d beta) <= 0.03: **fail** (-0.038, CI spans 0).
- moving EPE <= +1 px: **fail** (+1.14, CI spans 0).
- static EPE <= +5%: **fail** (+13.7%).
- (ii) d beta >= 0.10: **fail**.
- Overall: **U5a fails its success rule**, while its specificity test T5p passes.

Reading:
- The projector edit is specific. It moves alpha 7.5 times more than matched random projectors (0.121 vs 0.016), and its alpha gain per unit of static-EPE cost is about 3.7 times that of the E5 whole-head gain (0.88 vs 0.24). It is still not clean, and the fine-tuned LoRA (v) remains the most selective edit: a smaller alpha gain (+0.064) with improved moving EPE and no static or pose cost.
- With the boundary rule off, (i) improves moving EPE by 4.0 px [-7.4, -1.0]. Under the primary rule it is +1.1 px. Most of the edit's EPE benefit therefore sits near object boundaries.
- **Why (ii) fails is informative.** The fitted object and camera directions inside these heads are nearly collinear: cos_OC is 0.91 to 0.99 (b2h1 0.994). The B-direction edit cannot move beta independently, because the heads hardly separate the two motions in their 2D output subspace. This is direct mechanistic evidence for a shared motion code, and it explains why amplifying object motion also perturbs static points.
- **Real data (exploratory, not pre-registered for a decision):**
  - KITTI: (i) g = 1.5 slope +0.069 [0.048, 0.091] at +2.6 px moving EPE. Unexpectedly, (ii) g = 1.5 gives slope +0.219 [0.196, 0.241] with moving EPE -0.28 px (CI spans 0) and static EPE +2.7%.
  - Sintel: changes are small, (i) -0.02 and (ii) +0.04.
  - LoRA (v): KITTI +0.023, Sintel +0.007.
  - The KITTI result for (ii) matches the collinearity finding: on forward-driving footage the "camera" direction carries the object signal. Report it as exploratory. It suggests an "object-motion subspace" edit (span of P_O and P_C) as follow-up work, but that would be a new analysis after the results freeze and must be labelled exploratory.

**U5b, contamination masking (Any4D):**
- Kubric test: every masking variant makes pose error worse.
  - (a) probe mask in the 20 contamination heads: dRPE-R +0.0038 deg, moving EPE +11.4 px.
  - (b) probe mask in all heads: +0.0098 deg, moving EPE +54 px, alpha -0.73.
  - (c) random heads: +0.0007 deg.
  - (d) oracle GT mask in all heads: +0.0101 deg, moving EPE +55 px, alpha -0.76.
  - The probe itself works well (median IoU with GT masks 0.77, recall 0.95, about 2.3% of tokens masked).
- Real pose sets: T10 (a) vs (c) d = +0.0046 deg [-0.005, 0.014], p = 0.18, so (a) does not beat random heads. Base vs (a): +0.0002 deg on Bonn and TUM, -0.003 deg on Sintel. There is no improvement.
- The plan's "(a) reaches half of the oracle" is formally true but meaningless, because the oracle itself makes things worse. The recovery ratio has an unusable CI ([-0.46, 1.06] for (a)) because its denominator e(D) - e(B) is near 0.
- **U5b fails.** Together with T8's small effect and the flat contamination screen, the conclusion is that Any4D's camera estimate is nearly immune to object motion, so there is little to repair. The plan says that a negligible contamination result supports account R; here contamination is small but not null (CI excludes 0), so state it as "small and not repairable by attention masking". Masking moving keys in all heads mainly removes the object signal (alpha -0.76), which confirms that object motion reaches the flow output through attention to moving tokens.

## 3. Data-quality checks done for this analysis

1. **"Mean of empty slice" warnings** (u4u5_summary.py lines 144 and 183) are harmless. They come from all-NaN lists: two Kubric test scenes have no moving pixels left after the 8 px exclusion (moving-EPE n = 46 of 48), and non-identifiable scenes have no alpha. These are excluded from the bootstrap, which is correct.
2. **Any4D Sintel pose: 0.953 deg (U2) vs 0.371 deg (U5b).** Every window agrees exactly (maximum per-window difference 0.0). U2 reports means; the U5b real-data table and figure F5b report medians. Any4D Sintel is heavy-tailed (worst windows 4.9, 6.5 and 9.2 deg). The F5b title says "median"; every table in the paper must say which statistic it uses.
3. **T9 null pool at k = 1** is effectively the 11 other block-14 heads (6 distinct null values across the 40 draws). The conclusion does not depend on this, because b14h5 exceeds all of them by a wide margin, but describe the null honestly.
4. **Contamination-screen ranks are flat** (kappa 0.79 to 0.84 for almost all heads). Do not cite b8h11 as a "contamination head".
5. **Rank files mix heads and whole-residual components.** H3 and k* use head-only ranks, which is correct; anyone reading rank_*.json directly must filter out the `b*r` entries.
6. **Pending (not run):** U4b VGGT interchange and MoRe variant (e). MoRe appears only through the U2 pose rows; its contamination is +0.014 deg [-0.020, 0.050] on test, CI spans 0.

## 4. Implications for the paper

Recommended claims, in order of strength:

1. **C1 (behaviour, strong).** Feed-forward 4D models under-report object motion in the world frame across families (Any4D, OpenD4RT, TrackCraft3R; VGGT is a static-world extreme). Lifted trackers do not, and 4RC does not. T1a and T1b survive Holm.
2. **C2 (cause, moderate).** The deficit is mainly shrinkage toward a static world, present with a frozen camera. Camera motion adds a smaller, significant penalty, inconclusive against the 0.10 threshold. Contamination of the pose is tiny (T7, T8, budget, spurious ego-motion). Worded "consistent with".
3. **C3 (mechanism, moderate to strong).** In Any4D the object-motion difference passes through a narrow head pathway led by b14h5 (80% at k = 1, 99% at k = 3), while camera motion is carried diffusely in the residual stream. Within the pathway heads, object and camera directions are nearly collinear (cos 0.91 to 0.99), a shared code that explains both the pull and why separate repairs fail. The double dissociation is descriptive.
4. **C4 (repair, weak to moderate).** A GT-free projector edit on the pathway heads is direction-specific (T5p) and raises alpha by 0.12, but it does not meet the clean-repair criteria. Contamination masking does not help.

Subtitle check: "Seen but Under-Reported" is directly supported by OpenD4RT (2D alpha about 0.95 vs world s_w 0.73 to 0.90) and by the interchange result (the motion signal exists at b14h5 and is under-amplified downstream). Keep it.

What to drop or reword:
- "Five supply heads" becomes "three of the five E5 heads lie on the causally identified pathway".
- "Not uncertainty-driven shrinkage" becomes "not explained by camera attribution alone".
- Any claim that contamination masking repairs pose: report U5b as a negative result in the supplement.
- Present 4RC as a counterexample, not as a model that "also shows the pull".

Suggested next steps before the results freeze (Nov 6). These need GPUs; all four are currently busy with other jobs.
1. U4b VGGT interchange (should). It would test whether a b14h5-like bottleneck exists in a second architecture, which is the biggest generality gap in C3.
2. An exploratory "motion-subspace" edit (span of P_O and P_C) on v2 test and KITTI, labelled post hoc. It follows directly from the collinearity and KITTI findings.
3. A path-patching check of the serial-pathway hypothesis (b2h1, b14h5, b17h10, b22h9), CPU-light if cached activations exist.
4. Figure fixes: drop the 4RC bars from F1c, label medians in F5b, and add the tracker reference band to F2a.

## 5. Figures (drafts generated by the pipelines)

- F1 factorial: alpha in C vs D, RPE-R in B vs D, motion budget (4RC bars to be removed).
- F2 cross-family: s_w per model with lifted trackers; contamination per model.
- F4a pathway maps (fit), F4b double dissociation (test), F4c completeness curves.
- F5a U5a edits (d alpha, d beta per variant), F5b U5b masking (real pose sets, medians).
