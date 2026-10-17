# ft_train_gain_sel

Fine-tuning variant **gain** of Any4D from any4d_4v_combined.pth.

- Trainable parameters: 5; heads: [[2, 1], [22, 9], [17, 10], [14, 6], [8, 11]]; LoRA rank 8, alpha 16.
- Learning rates: LoRA 0.0002, scene-flow head 2e-05, gain 0.005; AdamW, warmup 100, cosine to 10%; bf16, activation checkpointing.
- Loss: Any4D scene-flow loss (scale-invariant, log-norm, x10.0 on moving pixels) + 1.0 x distillation to the frozen model (pointmaps, depth, pose, metric scale).
- Stopped: early stop at step 800 (best 400). Best validation step: 400. Merge check (adapter vs merged weights): {'d_sf': 0.0005878806114196777, 'd_pts': 0.007144927978515625}.

| validation (held-out training scenes, frames 0 to 23) | step 0 (base) | best step |
|---|---|---|
| loss | 0.0116 | 0.0121 |
| slope (mean) | 0.6757 | 0.6896 |
| EPE moving (median px) | 36.1928 | 35.5865 |
| EPE static (median px) | 3.9417 | 3.9739 |
| motion-mask IoU (median) | 0.5076 | 0.5033 |

These are training-distribution numbers; the held-out evaluation is in the matching ft_eval task.
