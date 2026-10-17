# ft_train_sf

Fine-tuning variant **sf** of Any4D from any4d_4v_combined.pth.

- Trainable parameters: 19,021,379; heads: none (scene-flow head only); LoRA rank 8, alpha 16.
- Learning rates: LoRA 0.0002, scene-flow head 2e-05, gain 0.005; AdamW, warmup 100, cosine to 10%; bf16, activation checkpointing.
- Loss: Any4D scene-flow loss (scale-invariant, log-norm, x10.0 on moving pixels) (no distillation needed: transformer and geometry heads are frozen).
- Stopped: max_steps 1200. Best validation step: 1200. Merge check (adapter vs merged weights): {'d_sf': 0.0, 'd_pts': 0.0}.

| validation (held-out training scenes, frames 0 to 23) | step 0 (base) | best step |
|---|---|---|
| loss | 0.0116 | 0.0092 |
| slope (mean) | 0.6757 | 0.7964 |
| EPE moving (median px) | 36.1928 | 37.7783 |
| EPE static (median px) | 3.9417 | 3.9732 |
| motion-mask IoU (median) | 0.5076 | 0.5232 |

These are training-distribution numbers; the held-out evaluation is in the matching ft_eval task.
