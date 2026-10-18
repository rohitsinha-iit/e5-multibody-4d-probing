# Review of P3_plan.md (version 1): area-chair and experimentalist pass

Reviewed: P3_plan.md v1 (2026-10-18), against E5_CVPR_framing.md, w2_summary_summary.md, E5_CVPR_review.md and sections P2/P3 of gen4p_alternatives_proposals.md. Implementation facts were checked on 2026-10-18 against the GitHub sources (Wan-Video/Wan2.1, huggingface/diffusers main, cvlab-kaist/TrackCraft3r, cvlab-kaist/DiTracker, Lijiaxin0111/Open-d4rt, EnVision-Research/DVD, facebookresearch/co-tracker) and the Hugging Face API, including safetensors headers read by HTTP range request and one CoTracker3_Kubric shard streamed to 181 MB. No GPU job was run and the user's servers were not contacted. Run-time and memory figures are FLOP-based estimates, not measurements.

## 1. Summary judgement

The plan asks the right three questions (generality, cause, mechanism) and reuses E5 infrastructure sensibly. Its weaknesses are concentrated in four places:

1. The TrackCraft3R protocol rests on wrong facts (12-frame runs with per-frame latents, not 4N+1).
2. TrackCraft3R's beta is fixed by its input cameras, and its output is a residual from the static world. A positive T1 is therefore nearly guaranteed and does not test the family claim.
3. The X2 2x2 varies the objective together with loss weighting, target space and readout, and it omits the variable most likely to produce the pull: an output parameterised relative to a static world.
4. The scope is about twice what 18 days and an 8-page paper can carry.

The more interesting reading the data may support is that the camera-preserving pull is L1 shrinkage expressed in a static-world-residual output space. That is a sharper and more general claim than C6 or C7 as written, and the redesigned X1 and X2 below test it directly.

## 2.1 Critical issues

**C1. [X1, section 3 (Wan conventions), G0b, Risk 5]** The TrackCraft3R protocol is built on wrong facts. The plan pads E5 clips to 25 = 4N+1 frames, places frame 23 in latent 6, uses a 9-frame KITTI window 'because 9 = 4x2+1', calls the model 'trained on 81-frame clips', and offers '49 frames by temporal upsampling' as the G0b fallback. In the released code the model always processes exactly 12 frames per run, every frame is VAE-encoded on its own (vae_temporal_compression = False, so there is no 4N+1 constraint and no latent mixing), longer spans are covered by interleaved 12-frame sub-sequences that share the frame-0 anchor, and the default input is 480x832 (stretch or pad). As written, X1's TrackCraft3R rows would be run out of distribution and G0b's fallback cannot work.

*Fix.* Rewrite the TrackCraft3R adapter around 12-frame runs: for Kubric use the reference frame plus 11 frames that include the target (for example [0, 2, 4, ..., 20, 23] or the repo's interleaved mode), --resize_mode pad at 480x832 so the 512x512 frames are not stretched, and the native z-depth plus W2C extrinsics in build_user_npz. For KITTI use frames 10 to 20 (11 frames) plus one repeated frame, or interleaved runs anchored at frame 10 with target 11. Replace the G0b fallback by the repo's frame_stride sweep. Delete the 4N+1 and 81-frame statements for TrackCraft3R; the 4N+1 rule applies only to the X2/X3 Wan feature extractor, where you choose the VAE mode yourself.

**C2. [X1 (T1), C6, Risk 1]** TrackCraft3R's beta is circular and T1 cannot separate a 4D-reconstruction pull from generic shrinkage. The model outputs P0(tj) = P0(t0) + predicted world-frame displacement in frame-0 camera space, i.e. a residual from the static hypothesis, and depth plus cameras are required inputs. Projecting its world tracks with the same cameras that were fed in makes beta = 1 for any point predicted static, so beta measures the input cameras (GT or DA3), not the model. Any shrinkage of the predicted world displacement toward zero then yields alpha < beta mechanically. The same argument applies to Any4D-style scene flow: 'keep the camera, shrink the object' is what L1 shrinkage looks like for any model whose output zero is 'static in the world'. As planned, a positive T1 is close to guaranteed and says nothing about families.

*Fix.* Make the primary TrackCraft3R endpoint a 3D world-frame object-motion slope: regress predicted world displacement on GT world displacement for moving points (plus the static-point residual as a sanity check), with GT cameras only used to express GT in frame-0 space. Compute the same world-frame slope for Any4D, OpenD4RT (query with t_cam fixed to t_src) and a 'lifted tracker' baseline (CoTracker3 2D tracks lifted with GT target depth and GT cameras). Report image-space alpha/beta for TrackCraft3R only descriptively and state that beta is input-determined. Reword C6 as a claim about output space and parameterisation, unless the lifted-tracker row shows no shrinkage in the same world-frame metric.

**C3. [X2 (C7, T3)]** The features x objective 2x2 does not isolate 'the 4D objective'. O2D and O4D differ simultaneously in target space (2D vs 3D), loss scale and weighting (the read-out 2D head gets weight 1.0 in O2D and 0.1 in O4D), t_cam conditioning, and how the 15 to 20M decoder spends capacity. A lower alpha in O4D readout (a) can come from an under-trained 2D head, and in readout (b) from depth-error coupling. Neither factor varies the variable that most plausibly produces the pull (output parameterised as a residual from the static world, see the TrackCraft3R and Any4D heads). The alpha-only endpoint also cannot tell a camera-preserving pull (beta near 1, alpha low) from uniform shrinkage (alpha and beta both low), which a cost-volume-free regressor may show under either objective (compare direct RAFT 0 to 23: beta 0.68).

*Fix.* Redesign X2 around Wan features only (VideoMAE as an optional secondary row): factor 1 = output parameterisation {absolute 3D position in the target camera, world-frame residual displacement from the static back-projection}; factor 2 = supervision {3D, 2D}, with equal weight on whichever head is read out and an identical query sampler. Gate every cell on matched static EPE (within 10% of each other, not only <= 6 px) and run one cell at 2x width as a capacity check. Make the pre-registered endpoint the pull index (beta - alpha) with alpha and beta both reported, and keep the prediction two-sided. This costs the same 8 to 12 runs and answers the question reviewers will actually ask.

**C4. [Section 6 (schedule), Section 0 (scope)]** Scope is not achievable by a Nov 6 freeze and does not fit an 8-page paper. The plan needs four new environments, three new model adapters, a Wan hook extractor with parity checks, a new D4RT-style decoder and trainer, 2000 new renders, probes, subspace geometry, edits and a new architecture (X6), in 18 calendar days on about 1.5 shared GPUs, while the E5 round-2 reruns are still being completed. GPU-hours are not the binding constraint (the plan's own total is about 8 GPU-days); engineering and debugging time is. Adding four claims (C6 to C9) to a paper that already carries five contributions will read as diffuse to an area chair.

*Fix.* Fix an engineering freeze of Oct 27 and cut to: X0 (reduced gates), X0c, X1 with OpenD4RT and corrected TrackCraft3R (DiTracker optional), redesigned X2 (Wan only, 2 seeds per cell, add a third seed only for cells near a threshold), X3 restricted to Any4D (existing internals) and base Wan, and X5 on Any4D only. Drop X6, X7a, X7b, the Wan part of X4, X5 on the Wan decoder, and X3 on TrackCraft3R, MapAnything and PAGE-4D. Target one new main-paper section (generality plus cause) and move the rest to the supplement.

## 2.2 Major issues

**M1. [X3 probe (2), C8]** The probe ratio a/b is neither invariant to ridge shrinkage nor well defined. Ridge shrinks each principal direction of the feature covariance by sigma_i^2/(sigma_i^2 + lambda), so if A and B are carried by directions of different variance the shrinkage is anisotropic and a/b moves with lambda. Even without shrinkage, the cross-validated prediction attenuates the lower-SNR component more, so a/b measures relative decodability, not a 'gain' (a linear readout can rescale any linearly present signal). When A and B are correlated (epipolar-aligned scenes, objects moving with the camera), regressing D_hat on [A, B] is ill-conditioned and a and b trade off. The C8 statement 'similar gain at the encoder output' is therefore not testable with this estimator.

*Fix.* Report, per layer, cross-validated R^2 for A, for B and for D with lambda chosen by nested CV per target; add the calibration slope of each probe after inner-fold calibration, and an SNR-matched comparison (subsample patches so var(A) and var(B) match). Apply the same identifiability filter as the output metric (condition number of [A B] <= 10, median |A|, |B| >= 2 px). Rephrase C8 as 'object motion is linearly recoverable from encoder tokens with R^2 comparable to camera-induced motion, so the deficit is introduced after the encoder', which E5's path patching already supports.

**M2. [Section 4.1 (wan_feats.py), X3, X4]** The readout timestep is undefined. 'The final scheduler step (Gen4P tau_last)' is not a fixed value: Wan's own 50-step UniPC or DPM sampler with shift 5 ends at t = 92.5 (sigma about 0.093), TrackCraft3R reads clean latents at scheduler.timesteps[-1] of a 1000-step shift-5 schedule, t = 4.98, and DVD reads clean latents at t = 500 (denoise_step 0.5). Gen4P's value cannot be checked because its code is not released. Features and probe curves change with the AdaLN timestep, so Wan, Wan+DVD and TrackCraft3R rows would be read at inconsistent conditions.

*Fix.* Use noise-free clean latents with t = 5 (TrackCraft3R's convention) as the primary Wan readout, read Wan+DVD at its native t = 500, and add a one-scene-set sensitivity check at t in {0, 92.5, 500}. State the value in the preregistration file.

**M3. [Section 3 (Wan conventions), X3, X4]** Frame-0-token probing is valid in principle for Wan (latent 0 holds only frame 0 at layer 0; the bidirectional DiT then lets frame-0 tokens attend to all latents, which is what a 'where does this patch go' probe needs), but the target is ill-posed: frame 23 shares latent 6 with frames 21, 22 and the duplicated frame 24, so GT displacement and attention hit-rate labels at 'frame 23' are ambiguous by up to two frames of motion. Latent-k tokens also carry temporal differences at layer 0 through the causal VAE convolutions, so the layer-0 baseline for the pooled pose probe is not a 'no-motion' baseline.

*Fix.* Label latent-6 targets with the chunk-average displacement and report frame-21 and frame-23 labels as a sensitivity check; or, preferably, add a per-frame VAE encoding variant (each frame encoded alone, as TrackCraft3R and DiTracker do) so the target frame is unambiguous. For the pose probe add the raw VAE latents and a per-frame-encoded layer-0 input as baselines.

**M4. [X2 (T3), Section 8]** Pseudo-replication: the objective and feature effects are properties of trained runs (3 seeds per cell), but T3 bootstraps scenes over seed-averaged alpha, which ignores between-run variance and gives intervals that are too narrow. There is also an uncovered gray zone between the kill rule (effects < 0.05) and the success rule (effects >= 0.1).

*Fix.* Use a hierarchical bootstrap (resample seeds within cell, then scenes) or a mixed model with a seed random effect; additionally require the sign of the effect to hold in every seed pairing. Pre-register the reading of 0.05 to 0.1 effects ('inconclusive, reported as such').

**M5. [X1 (T2), DiTracker row]** T2 is confounded and has little power. DiTracker fuses CogVideoX-2B layer-17 head-2 query/key features with CoTracker's ResNet features (resnet_fuse_mode concat) and is trained on Kubric MOVi-F only, while the public CoTracker3 checkpoint used in E5 adds a real-video pseudo-label stage. Both are correlation-volume trackers with iterative refinement, so both are expected near slope 1 whatever the features do.

*Fix.* Remove T2 from the confirmatory family. If DiTracker is run, compare it against a Kubric-only CoTracker3 checkpoint if one is available and describe it as 'a video-DiT-feature tracker also shows no pull'. Spend the saved effort on OpenD4RT.

**M6. [Section 3 (P3-train), G0e]** Rendering 2000 new scenes is the riskiest CPU item and duplicates an available resource. One CoTracker3_Kubric shard (inspected) holds 120 frames at 512x512 with 32768 2D tracks, per-track depth (so 3D tracks via the intrinsics), visibility, uint16 depth with its range, instance segmentation and full camera (positions, quaternions, focal length, sensor width). Training the X2 decoders on exactly CoTracker3's data also turns the 'data' account into a controlled comparison, and testing on E5-generator scenes avoids in-distribution artefacts.

*Fix.* Use CoTracker3_Kubric windows (for example 25-frame windows at stride 1 or 2 from about 400 shards, about 120 GB) as the primary X2 training set, keep E5-generator scenes for evaluation only, and render new scenes only if a second data condition is wanted. Budget the download (about 300 MB per shard) instead of render time.

**M7. [X2 feature family 2 (VideoMAEv2-L)]** The VideoMAE arm is confounded with resolution, token count, capacity and position extrapolation: 224 px input gives 14x14 tokens (one token is 36.6 px at 512 against 17 px for Wan), 12 temporal tokens against 7 latents, 0.3B against 1.3B parameters, and positions beyond the 1568 pretrained rows of the fixed 1D sin-cos table. A feature main effect would not be interpretable as 'generator vs MAE features'. D4RT and OpenD4RT use a ViT-g VideoMAEv2 encoder, not ViT-L.

*Fix.* Demote VideoMAE to a secondary row with an explicit confound statement, or drop it in favour of the parameterisation factor. If kept, match the token footprint (four 224 px tiles per frame) and describe it as 'a non-generative video encoder', not 'D4RT's encoder'.

**M8. [X4, X5]** Subspace angles and edit projectors are built from ridge decoding weights. Decoding directions are not encoding directions when the head-output covariance is anisotropic or when A and B are correlated (Haufe et al., NeuroImage 2014), so small or large principal angles between decoders do not test H4a against H4b, and amplifying a decoding direction need not amplify the represented A component.

*Fix.* Estimate encoding patterns by regressing head outputs on [A, B] (scene-grouped CV), build the projectors from those patterns (orthogonalised as planned), and compute principal angles between pattern subspaces against the random-head null. Use the B projector (ii) as a double dissociation in T5: success requires (i) to move alpha and not beta, and (ii) to move beta and not alpha.

**M9. [Section 8 (Holm family), X1 to X6 success rules]** The confirmatory family mixes tests of different kinds (T1 is a CI-threshold rule, T3 a bootstrap effect, T5 a paired difference), several tests lack a stated direction, and success thresholds (G2 static EPE <= 6 px, alpha +0.10, EPE +1 px) are uncalibrated.

*Fix.* Define every confirmatory test as a directional p-value (one-sided bootstrap or sign test) with its effect size; the family becomes T1' (world-frame slope < 0.9 for TrackCraft3R and OpenD4RT), T3' (parameterisation effect on the pull index) and T5 (directional edit vs random). Calibrate G2 on one pilot run, then freeze thresholds in p3_preregistration.json before the main runs.

**M10. [Missing experiments for C6 to C9]** Reviewers will ask for evidence the plan does not produce: (1) cheap within-model readout contrasts that separate representation from readout without training anything (OpenD4RT exposes a 2D uv head, a 3D head and the t_cam query; Any4D exposes per-frame pointmaps and scene flow); (2) a non-reconstruction model placed on the same world-frame axis (lifted tracker); (3) at least one new model on real data with its paper-setting inputs (TrackCraft3R with DA3 on KITTI); (4) the protocol harmonisation already requested in the E5 review (motion-boundary exclusion on Kubric vs real data).

*Fix.* Add X1b: OpenD4RT readout contrasts (uv head; 3D with t_cam = t_tgt projected; 3D with t_cam = t_src transformed by GT cameras) and the Any4D equivalent; add the lifted-CoTracker3 row; run TrackCraft3R condition (ii) on KITTI; and re-score all rows with one boundary rule. These are inference-only and cost under 0.5 GPU-day together.

## 2.3 Minor issues

**M1. [Section 4.1 (parity check)]** A max-abs feature difference below 1e-2 in bf16 is unrealistic for late Wan residual streams, whose magnitudes grow with depth.

*Fix.* Check parity per block with relative error (< 1e-2) and cosine similarity (> 0.999) on 2 clips in fp32 and bf16.

**M2. [Section 4.1 (hooks)]** In both implementations the RMSNorm on q and k acts over the full 1536-dim vector (across heads), and the post-RoPE q and k are computed inline inside the attention call, so they cannot be captured with a module forward hook.

*Fix.* Per-head outputs: forward pre-hook on self_attn.o (official) or attn1.to_out[0] (diffusers), reshaped to 12 x 128. Post-RoPE q/k: a custom WanAttnProcessor (diffusers) or a patched WanSelfAttention.forward (official). Verify on one clip that the recomputed attention output equals the hooked o-input.

**M3. [Section 4.1 (VAE)]** The official WanVAE.encode returns the deterministic mean normalised by the 16-channel mean and std; diffusers' AutoencoderKLWan does not normalise (the pipeline applies latents_mean and latents_std).

*Fix.* In the diffusers path use latent_dist.mode(), then apply (z - latents_mean) / latents_std from the VAE config; include this in the G0a check.

**M4. [DiTracker adapter]** DiTracker encodes frames with latent_dist.sample(), so features are stochastic between runs.

*Fix.* Fix the torch seed per clip or patch to the distribution mean; record it.

**M5. [X3 (Wan + DVD)]** The DVD files are not LoRA-only: each 4.49 GB bf16 file holds a full DiT (with PEFT rank-512 LoRA tensors) and a VAE, and the official loader merges after loading the shipped DiT strictly.

*Fix.* Merge from DVD's own DiT weights (merge_lora_layer) rather than onto the public Wan weights, and read it at t = 500.

**M6. [TrackCraft3R memory and queue]** The 17.8 GB checkpoint includes the umT5-XXL encoder, and the predictor moves the text encoder to the GPU at initialisation to encode the empty prompt (about 11.4 GB transient). CPU RAM per process is about 25 to 30 GB.

*Fix.* Cache the empty-prompt context once and patch the predictor to skip the T5 transfer; set GPU_NEED_GB to about 20 for TrackCraft3R tasks and run them one at a time per GPU.

**M7. [OpenD4RT adapter]** OpenD4RT runs at 256x256 (one token covers 32 px at 512), needs an even frame count (tubelet 2), and its 48-frame checkpoint has only seen 48-frame clips.

*Fix.* Use the 32-frame checkpoint as primary and the 48-frame one as a check; keep 24 frames; report the resolution caveat next to its row.

**M8. [X3 controls]** '5 seeds' for a deterministic ridge probe only resamples folds; the random-init Wan control needs the Wan init (modulation and zero-init details) to be meaningful.

*Fix.* Call it fold resampling; initialise with the official WanModel init; keep the label-shuffle controls.

**M9. [X7a]** Swapping textures between frames 0 and 23 breaks appearance-based correspondence itself, so a drop in detection is expected under every account.

*Fix.* Cut X7a before the deadline; if kept for the rebuttal, swap textures from frame 1 onward and compare against a matched texture-perturbation control.

**M10. [Storage]** About 64 GB of X3 internals, 39 GB of X2 features and either 2000 renders or about 120 GB of MOVi-F shards are needed on $BIG.

*Fix.* Check free space on /media/workbench/zone_x before enqueueing and delete per-head tensors once X4 statistics are computed.

## 3. Answers to the specific probes in the brief

1. **Does the X2 2x2 separate objective from decoder capacity or readout?** No. O2D and O4D differ in head weighting (1.0 vs 0.1 on the read-out 2D head), target space, t_cam conditioning and capacity allocation, and the alpha-only endpoint confounds pull with uniform shrinkage. See C3 for the redesign (parameterisation x supervision, matched static EPE, pull index endpoint, one 2x-width cell).
2. **Is the probe a/b invariant to ridge shrinkage, and valid with correlated A and B?** No on both counts. Ridge shrinkage is anisotropic across feature principal directions, CV attenuation hits the lower-SNR target harder, and with correlated A and B the second-stage regression is ill-conditioned. See M1.
3. **Is frame-0-token probing valid for Wan?** Yes in principle: latent 0 contains only frame 0 at layer 0, and the bidirectional DiT lets frame-0 tokens gather information from all latents, which is what the probe needs. Two caveats: the frame-23 target is blurred by the 4-frame latent 6, and latent-k tokens already carry motion at layer 0 through the causal VAE. See M3.
4. **Is TrackCraft3R projection with input cameras circular for beta?** Yes. Beta equals 1 for any point predicted static under the input cameras, and the residual output makes alpha < beta follow from any shrinkage. See C2.
5. **Are the X5 projectors fit and tested on separate data?** Yes, if folds are grouped by scene: P3-train plus v2 training folds for fitting, held-out v2, v3float, KITTI and Sintel for testing, with g fixed in advance. The remaining problem is that the projectors come from decoding weights rather than encoding patterns (M8), and that the B projector should serve as a double dissociation.

## 4. Statistics

- Unit of analysis: the scene is correct for inference-only comparisons (X1, X5 on Any4D). For trained decoders (X2, X6) the training run is a second unit, and the plan's scene bootstrap of seed-averaged alpha is pseudo-replicated (M4).
- Multiple comparisons: a Holm family is appropriate, but its members must be directional p-value tests of one kind. T1 as written is a CI-threshold rule, T2 is uninformative, and T6 belongs to a cut experiment. A family of three (T1' world-frame slope, T3' parameterisation effect on the pull index, T5 directional edit) is easier to defend (M9).
- Seeds: ridge probes are deterministic, so '5 seeds' means fold resampling. Three seeds per X2 cell are too few for a run-level effect without a sign-consistency requirement.
- Identifiability: the alpha/beta identifiability filter (condition number <= 10, median |A|, |B| >= 2 px) must also apply to probe-level estimates and to the TrackCraft3R world-frame decomposition.
- Gray zones: every success or kill rule needs a stated reading of intermediate outcomes (for example, effects between 0.05 and 0.1 in X2).

## 5. Implementation facts

| Item | Status | Fact | Source |
|---|---|---|---|
| 4a Wan2.1 block structure | verified | WanAttentionBlock = self-attention (q, k, v, o Linear; WanRMSNorm on q and k over the full dim 1536; 3D RoPE split 22/21/21 frequency pairs for t/h/w applied to q and k), then cross-attention to the text context (with its own q/k RMSNorm and a LayerNorm norm3; no timestep modulation), then FFN 1536 to 8960. AdaLN: each block has a learned modulation (1,6,1536) added to time_projection(e), giving shift, scale and gate for self-attention and for the FFN. 1.3B config: 30 layers, dim 1536, 12 heads of 128, patch (1,2,2), VAE stride (4,8,8), text_len 512. | github.com/Wan-Video/Wan2.1 wan/modules/model.py; wan/configs/wan_t2v_1_3B.py, shared_config.py |
| 4a Hook points (per-head output, q/k) | verified (with correction) | Per-head outputs: input of self_attn.o (official) or attn1.to_out[0] (diffusers WanTransformerBlock), shape [B, S, 12x128]. q/k after RoPE are not module outputs: official code calls rope_apply inside the flash_attention call; diffusers applies RoPE inside WanAttnProcessor. Capturing them needs a patched forward or a custom attention processor. Pre-RoPE q/k can be hooked at norm_q/norm_k outputs. | wan/modules/model.py lines 130 to 158; diffusers src/diffusers/models/transformers/transformer_wan.py (WanAttnProcessor, WanAttention) |
| 4a Diffusers-format repo | verified | Wan-AI/Wan2.1-T2V-1.3B-Diffusers exists (31 files: model_index.json, scheduler, text_encoder in 5 shards, tokenizer, transformer, vae). Class WanTransformer3DModel; _keep_in_fp32_modules includes time_embedder, scale_shift_table and the norms. | HF API models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers; diffusers transformer_wan.py |
| 4a VAE latent normalisation | verified | Per-channel mean and std (16 values each). Official WanVAE.encode returns the posterior mean mu normalised as (mu - mean) * (1/std); encoder processes frame 0 alone then chunks of 4 frames (iter_ = 1 + (T-1)//4), so latent k covers frames 4k-3 to 4k with causal caching. Diffusers AutoencoderKLWan stores latents_mean/latents_std in its config and the pipeline applies them; the VAE itself does not. | wan/modules/vae.py (WanVAE, WanVAE_.encode); diffusers autoencoder_kl_wan.py, pipelines/wan/pipeline_wan.py |
| 4a Timestep convention for a clean readout | verified (plan's term is ambiguous) | Flow matching with t = 1000 x sigma; clean data is sigma = 0. Wan's 50-step sampler with shift 5 ends at t = 92.5 (not clean). TrackCraft3R uses scheduler.timesteps[-1] of FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True).set_timesteps(1000) = t 4.98 on clean latents; DVD uses t = 500 (denoise_step 0.5) on clean latents. Gen4P's tau_last could not be verified (no code). | wan/text2video.py, wan/utils/fm_solvers_unipc.py; TrackCraft3r diffsynth/schedulers/flow_match.py and evaluation/wan_scene_flow_predictor.py; DVD diffsynth/schedulers/flow_match.py, ckpt/model_config.yaml (values computed from the code) |
| 4b TrackCraft3R inputs and frame count | wrong in plan | Requires RGB, depth and per-frame camera (assert in predict). NPZ from build_user_npz.py: images_jpeg_bytes, depth_map (T,H,W) z-depth, extrinsics_w2c (T,4,4) frame-0-normalised, fx_fy_cx_cy (4,). The model always processes 12 frames per run (frame_stride selectable; interleaved 12-frame sub-sequences with a shared frame-0 anchor for longer spans). Each frame is VAE-encoded separately (vae_temporal_compression False). Default resolution 480x832, resize stretch or pad. The plan's 4N+1, 25-frame and 81-frame statements are wrong for this model. | cvlab-kaist/TrackCraft3r README.md, scripts/inference_user_video.py, scripts/build_user_npz.py, evaluation/README.md, evaluation/scripts/eval_stride.sh |
| 4b TrackCraft3R output frame and query frames | verified | Outputs track_map (T,H,W,3) in frame-0 camera space, reconstructed as P0(tj) = (delta + P0(t0)_norm) x scale + mean, i.e. a residual from the frame-0 pointmap built from the input depth and cameras. Queries are frame-0 pixels only (scene_flow_query_mode 'frame0_latent'). Token layout: a 'diagonal' stream [RGB / pointmap of frame j] and a 'row' stream [RGB0 / pointmap0] concatenated in time, so 2 x 12 latent frames. | evaluation/wan_scene_flow_predictor.py; diffsynth/pipelines/wan_video_new.py (model_fn_wan_video) |
| 4b TrackCraft3R GPU memory | could not verify (estimate given) | Repo states only training on 8x H200. Checkpoint: 17.8 GB bf16 with DiT, rank-1024 LoRA on self/cross attention q,k,v,o and ffn.0/ffn.2, umT5 encoder and three VAEs (vae, vae_pj, vae_vis). Estimate: about 10 to 13 GB steady at 480x832 x 12 frames (37,440 tokens, about 460 TFLOP per forward with unmerged LoRA), with an about 11 GB transient at initialisation when T5 is moved to the GPU. | HF trackcraft3r/checkpoint safetensors header (read via range request); predictor code |
| 4c DiTracker inference API and resolution | verified | CoTracker3-style interface: video (B,T,3,H,W) and queries (B,N,3) as (t, x, y), so arbitrary query frames and points are supported; EvaluationPredictor wraps it. Model resolution [480, 720]. Features: CogVideoX-2B layer 17 head 2 query/key, concatenated with ResNet features; frames VAE-encoded one by one with latent_dist.sample() at timestep 19; frames processed in chunks of 13 with frame 0 prepended. Weights in ./checkpoint; trained on Kubric MOVi-F. | cvlab-kaist/DiTracker README.md, cotracker/models/core/cotracker/ditracker.py, feature_extractor.py, cotracker/models/evaluation_predictor.py |
| 4d OpenD4RT weights and frame handling | verified | Weights public: checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt and OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt (about 13.95 GB each). Input 256x256, ViT-g encoder (40 layers, dim 1408, tubelet 2, interleaved frame-wise and global attention, 1D sinusoidal position over flattened tokens), learned discrete timestep embeddings, heads xyz, uv, visibility, displacement, normal, confidence. For clips with num_frames <= clip_frames the code uses all frames as given (no padding), so 24 frames run; behaviour at 24 frames is untested. Training code released 2026-06-04. | github.com/Lijiaxin0111/Open-d4rt README.md, checkpoints/*/model.yaml, infer_track_3d.py (_make_anchor_clip_indices), src/model/encoder.py; HF Lijiaxin0111/OpenD4RT file list |
| 4e CoTracker3_Kubric shard contents | verified (partial inspection) | Shard 0000.tar.gz (307 MB) streamed until 181 MB: per sequence 0000/0000.npy is a pickled dict with coords (32768,120,2), coords_depth (32768,120), visibility (32768,120), depth (120,512,512,1) uint16 plus metadata depth_range, segmentations (120,512,512,1), camera {positions (120,3), quaternions (120,4), focal_length, sensor_width, field_of_view}; plus per-frame depths/*.npy, *_trajs_2d.npy, *_visibility.npy, *_with_rank.npz. Depth, camera and 3D tracks (2D tracks + per-track depth + intrinsics) are all present. RGB frame files were not reached in the first 60 members. 5871 shards. | HF datasets/facebook/CoTracker3_Kubric (0000.tar.gz streamed); co-tracker cotracker/datasets/kubric_movif_dataset.py |
| 4f VideoMAEv2-Large positional embedding | verified (with correction) | use_learnable_pos_emb false: a fixed 1D sin-cos table over the flattened (T/2 x 14 x 14) token index, built at init from num_frames = 16 (1568 rows). 24 frames require constructing the model with num_frames = 24 (2352 rows; the first 1568 are unchanged, the rest are extrapolated). PatchEmbed asserts 224x224, so the spatial size cannot change without breaking the position semantics. forward returns a mean-pooled fc_norm vector; token features need block hooks. Frozen 24-frame use runs mechanically; quality at extrapolated positions is untested. Licence CC-BY-NC-4.0. | HF OpenGVLab/VideoMAEv2-Large modeling_videomaev2.py, config.json, README.md |
| 4g DVD weights | verified (with correction) | Released files model.safetensors (v1.0) and dvd_1.1.safetensors are 4.49 GB bf16 each and contain the full DiT plus PEFT LoRA tensors (rank 512 on self- and cross-attention q,k,v,o and ffn.0/ffn.2) and a VAE; not LoRA-only. Official loader loads the DiT state dict strictly then calls merge_lora_layer(), so merging works. Base model Wan2.1-T2V-1.3B; regression at t = 500. | HF FayeHongfeiZhang/DVD safetensors headers; EnVision-Research/DVD ckpt/model_config.yaml, test_script/test_single_video.py, diffsynth/schedulers/flow_match.py |

Also checked (plan section 2): trackcraft3r/checkpoint (1 file, model.safetensors, 17.8 GB) and trackcraft3r/trackcraft3r-eval (448 files) exist; THUDM/CogVideoX-2b exists; Wan-AI/Wan2.1-T2V-1.3B holds diffusion_pytorch_model.safetensors (5.68 GB), Wan2.1_VAE.pth and models_t5_umt5-xxl-enc-bf16.pth (11.4 GB).

## 6. Feasibility (estimates; Nov 6 results freeze, about 1.5 effective 24 GB GPUs)

| Workload | Time | GPU memory | Total for the plan | Verdict |
|---|---|---|---|---|
| Wan2.1-1.3B forward, 7x30x30 = 6300 tokens, hooks on 31 residual states + per-head outputs + q/k maps | about 23.7 TFLOP; 0.3 s (4090) to 0.5 s (A5000) compute; about 1 to 2 s per clip with VAE and I/O | about 6 to 8 GB (2.84 GB bf16 weights, VAE encode of 25 frames at 480 px dominates); no T5 if the empty context is cached | 250 E5 scenes: under 0.5 h; 4 Wan variants under 2 h. Disk writes (64 GB) likely the bottleneck | Feasible |
| Wan per-frame VAE variant (25 latent frames, 22,500 tokens) | about 152 TFLOP; 2 to 3.5 s per clip | about 8 to 10 GB | 250 scenes: about 0.3 h | Feasible; recommended for X3/X4 label clarity |
| Wan final-block features for X2 training (2000 clips) | about 1.5 s per clip | as above | about 1 h; 39 GB disk | Feasible |
| TrackCraft3R inference (12 frames, 480x832, 2 x 12 x 1560 = 37,440 tokens, unmerged rank-1024 LoRA) | about 460 TFLOP; 6 s (4090) to 10 s (A5000) DiT, plus 3 VAE passes; about 20 to 30 s per run | about 10 to 13 GB steady; about 11 GB extra transient if T5 is moved to GPU at init; 25 to 30 GB CPU RAM to load the checkpoint | 3 conditions x (about 330 Kubric + 195 KITTI + 217 Sintel) runs: about 15 to 20 GPU-h, plus DA3 preprocessing | Feasible if T5 transfer is patched; one process per GPU |
| DiTracker (CogVideoX-2B, 480x720, chunks of 1+13 frames) | about 130 TFLOP per chunk if all 30 layers run; about 10 to 20 s per 24-frame clip | estimate 12 to 16 GB (transformer may load in fp32 by default); T5 (about 4.7B params) encoded once on CPU | about 330 Kubric clips: 1 to 2 GPU-h | Feasible but optional |
| OpenD4RT (ViT-g, 24 frames at 256 px, 3072 tokens) | under 1 s per clip | about 4 to 6 GB; 14 GB checkpoint load in CPU RAM | under 1 GPU-h for all eval sets | Feasible; highest value per GPU-hour |
| VideoMAEv2-L (24 frames, 224 px, 2352 tokens) | about 1.5 TFLOP; under 0.1 s per clip | under 3 GB | negligible | Feasible but confounded |
| X2 decoder training (20k steps, batch 8, 2048 queries, pre-extracted features) | compute about 0.05 to 0.1 s per step; data loading of 155 MB per step is the bottleneck | 4 to 8 GB | 3 to 4 GPU-h per run if features are page-cached or on NVMe; 8 to 12 runs: 30 to 50 GPU-h | Feasible if reduced to 2 seeds x 4 cells first |
| P3-train rendering (2000 Kubric scenes) | unknown per-scene time; at 5 min per scene and 16 workers about 10 h | CPU only | 1 to 2 days with retries | Risky; replace by CoTracker3_Kubric windows |

GPU time is not the binding constraint: the reduced plan needs roughly 4 to 6 GPU-days (TrackCraft3R 15 to 20 GPU-h, X2 30 to 50 GPU-h, everything else under 10 GPU-h), which fits 1.5 effective GPUs over 18 days even with OOM back-offs. Engineering is the constraint: four new environments, three adapters, a hook extractor, a decoder trainer and a data pipeline. Cut order: (1) X6, (2) X7a and X7b, (3) X5 on the Wan decoder and the Wan part of X4, (4) X3 on TrackCraft3R, MapAnything and PAGE-4D, (5) the VideoMAE arm of X2, (6) DiTracker, (7) the third X2 seed. Never cut OpenD4RT (cheapest new reconstruction model with a correspondence output) or the world-frame re-scoring. All per-run numbers are FLOP-based estimates from the verified architectures (effective 80 TFLOP/s on the 4090 and 45 TFLOP/s on the A5000), not measurements; no GPU job was run.

## 7. Experiments a reviewer will demand, and experiments not worth their cost

**Will be demanded (for C6 to C9):**

1. World-frame object-motion slope for every model with a 3D output, together with a lifted-tracker row on the same axis. Without it C6 is open to the 'generic shrinkage in your output space' objection.
2. Readout contrasts inside OpenD4RT (uv head, 3D projected with t_cam = t_tgt, 3D with t_cam = t_src plus GT cameras) and Any4D. These test 'representation vs readout' (C8) without training.
3. At least one new model on real data in its paper setting: TrackCraft3R with DA3 on KITTI.
4. A parameterisation manipulation (absolute vs static-residual output) as the causal test of C7.
5. For C9, a double dissociation (the A-edit moves alpha only; the B-edit moves beta only), set against 20 random projectors and the whole-head edit, with static and moving EPE.
6. One boundary-exclusion rule applied to all Kubric and real-data rows (carried over from E5 review W7).

**Not worth their cost before Nov 16:**

- X6 camera-routed readout. It is a new architecture, and the shrinkage account predicts that a factorised 'camera plus object residual' readout reproduces the pull rather than removing it. If any of it is kept, keep only the parameterisation contrast inside X2.
- X7b pseudo-label self-training. It is a P2 baseline, not evidence for C6 to C9.
- X7a appearance swap. It is confounded by correspondence failure (minor issue m10).
- X4 on Wan heads and X3 on TrackCraft3R's DiT (pointmap tokens mixed in), MapAnything and PAGE-4D. These add rows but no new claim.
- DiTracker as a confirmatory test (T2 has no power; M5).
- 2000 new renders, when CoTracker3_Kubric already contains depth, cameras and 3D-liftable tracks (M6).

## 8. Recommended reduced must-set (for orientation, not a rewrite)

1. X0: environments, G0a (VAE plus parity), G0b (TrackCraft3R 12-frame protocol), G0d (OpenD4RT at 24 frames).
2. X0c: CoTracker3 on KITTI and Sintel.
3. X1 plus X1b: OpenD4RT, TrackCraft3R (i) and (ii), plus the lifted CoTracker3 row, all scored by the world-frame slope and the readout contrasts.
4. X2': Wan features, parameterisation x supervision, 2 seeds, hierarchical statistics.
5. X3': decodability (R^2, calibration) on Any4D and base Wan.
6. X5 on Any4D with encoding-pattern projectors and the double dissociation.

## 9. Overall verdict

As written, the plan cannot support a strong-accept-level paper by Nov 16. Two of its central tests are invalid as designed (TrackCraft3R's beta is set by its input cameras, and the features x objective 2x2 does not isolate the objective). The TrackCraft3R protocol rests on wrong frame and resolution facts, and the scope exceeds both the 18 days to the results freeze and the 8-page limit. Once fixed (world-frame slope as the cross-family metric, a lifted-tracker baseline, OpenD4RT readout contrasts, X2 redesigned around output parameterisation with hierarchical statistics, and X6, X7 and most of X4 cut), the extension can turn the E5 paper from a solid weak accept or accept into a clear accept. The most likely new headline is 'the pull is shrinkage in a static-world-residual output space, present across reconstruction families and absent in image-space trackers'. A strong accept needs that result to hold for OpenD4RT and TrackCraft3R on real data and the parameterisation effect to be large and seed-consistent. I put the probability of that within the deadline at about 20 to 25%. The probability of a coherent, defensible accept-level paper with the reduced plan is good, provided the engineering freeze is held on Oct 27.
