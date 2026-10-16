# ft_pilot

Pilot run to choose the number of fine-tuning steps (variant heads_sf = five-head LoRA + scene-flow head, the largest variant).

- Validation every 100 steps on 38 held-out training scenes (10%, never used for gradients).
- Validation loss 0.0116 at step 0, minimum 0.0099 at step 700; 90% of the reduction reached at step 600.
- Speed: 5.36 s per optimizer step (2 samples of 7 views).
- Chosen max_steps for every variant: **1200** (rule: 2 x step-at-90%, rounded up to 100, within [400, 2500] and 3 h). Each run still early-stops on its own validation loss.

| step | loss | slope | EPE moving | EPE static | mask IoU |
|---|---|---|---|---|---|
| 0 | 0.0116 | 0.676 | 36.2 | 3.94 | 0.508 |
| 100 | 0.0110 | 0.804 | 31.5 | 4.04 | 0.471 |
| 200 | 0.0105 | 0.742 | 36.7 | 3.96 | 0.514 |
| 300 | 0.0108 | 0.860 | 33.4 | 4.04 | 0.460 |
| 400 | 0.0102 | 0.772 | 39.6 | 4.03 | 0.458 |
| 500 | 0.0102 | 0.774 | 34.2 | 3.98 | 0.504 |
| 600 | 0.0101 | 0.769 | 35.6 | 3.96 | 0.538 |
| 700 | 0.0099 | 0.791 | 35.4 | 3.96 | 0.535 |
| 800 | 0.0100 | 0.795 | 36.5 | 3.96 | 0.520 |
| 900 | 0.0100 | 0.804 | 36.0 | 3.99 | 0.532 |
