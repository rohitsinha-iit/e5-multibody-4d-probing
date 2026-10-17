# r2_probe_extract: control-model internals for the probe controls (review W4)

Variants: randtf = public DINOv2 ViT-L/14 encoder + Any4D projection, fusion norm, transformer and heads at their own seeded random init; randall = everything random (encoder re-initialised with the dinov2 constructor, pretrained=False); enc_any4d / enc_dino = raw encoder patch tokens of the trained Any4D encoder / the public DINOv2 encoder (1024-d, before fusion norm, projection and view PE); trained = round-1 Any4D internals (symlinks). Layers 0,4,...,24 for the transformer variants (0 = transformer input).

| dataset | source | scenes | stored layers | non-finite values | encoder max abs param diff to public DINOv2 |
|---|---|---|---|---|---|
| v2 | trained | 100 | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 | 0 (first 20 scenes) |  |
| v2 | randtf | 100 | 0,4,8,12,16,20,24 | 0 (first 20 scenes) | 0 |
| v2 | randall | 100 | 0,4,8,12,16,20,24 | 0 (first 20 scenes) | 72.27 |
| v2 | enc_any4d | 100 | 0 | 0 (first 20 scenes) | 0.0617 |
| v2 | enc_dino | 100 | 0 | 0 (first 20 scenes) | 0 |
| v3float | trained | 69 | 0,2,4,6,8,10,12,14,16,18,20,22,24,17,19,21 | 0 (first 20 scenes) |  |
| v3float | randtf | 69 | 0,4,8,12,16,20,24 | 0 (first 20 scenes) | 0 |
| v3float | randall | 69 | 0,4,8,12,16,20,24 | 0 (first 20 scenes) | 72.27 |
| v3float | enc_any4d | 69 | 0 | 0 (first 20 scenes) | 0.0617 |
| v3float | enc_dino | 69 | 0 | 0 (first 20 scenes) | 0 |

The trained Any4D encoder differs from the public DINOv2 weights (it was fine-tuned), so enc_any4d and enc_dino are two different baselines.

