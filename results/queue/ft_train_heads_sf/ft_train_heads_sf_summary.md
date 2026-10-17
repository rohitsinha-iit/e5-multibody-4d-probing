# ft_train_heads_sf

Fine-tuning variant **heads_sf** of Any4D from any4d_4v_combined.pth.

- Trainable parameters: 19,093,059; heads: [[2, 1], [22, 9], [17, 10], [14, 6], [8, 11]]; LoRA rank 8, alpha 16.
- Learning rates: LoRA 0.0002, scene-flow head 2e-05, gain 0.005; AdamW, warmup 100, cosine to 10%; bf16, activation checkpointing.
- Loss: Any4D scene-flow loss (scale-invariant, log-norm, x10.0 on moving pixels) + 1.0 x distillation to the frozen model (pointmaps, depth, pose, metric scale).
- Stopped: max_steps 1200. Best validation step: 1200. Merge check (adapter vs merged weights): {'d_sf': 0.0011048614978790283, 'd_pts': 0.008114814758300781}.

| validation (held-out training scenes, frames 0 to 23) | step 0 (base) | best step |
|---|---|---|
| loss | 0.0116 | 0.0098 |
| slope (mean) | 0.6757 | 0.7991 |
| EPE moving (median px) | 36.1928 | 37.8953 |
| EPE static (median px) | 3.9417 | 3.9488 |
| motion-mask IoU (median) | 0.5076 | 0.5308 |

These are training-distribution numbers; the held-out evaluation is in the matching ft_eval task.
