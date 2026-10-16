# v4_pilot: pilot of the new epipolar scene types (review item W3)

Two scenes per new split, rendered with the v3float settings (floating, shadowless, sizes 1.4 to 2.2 m for static objects).
Contact sheets and GIFs are in sheets/. The scaled task v4_scale_epi waits for the gate approvals/v4_visual.

What to check visually:
- epi_accel vs epi_const with the same seed: frames 0 and 23 look the same; in the GIF the accelerated object starts or stops while the constant one glides.
- epi_sizerand: the moving object can be much smaller or larger than the static objects.
- epi_sizematch: the moving object is large (2.2 to 4.4 m), so read as a static object at the closer ambiguous depth it would be of typical size.
- Nothing touches the floor, no shadows, every object stays in view.


Scenes checked: 8; scenes with errors: 0.

All numbers come from Kubric ground truth (depth, segmentation, camera and object matrices).

| split | n | sep GT 0 to 23 (px), median [min, max] | 3-view residual, moving, frame 12 (px), median [min, max] | 3-view residual, background, frame 12 (px) | max centre gap from static reading (px) | static objects per scene, mean |
|---|---|---|---|---|---|---|
| epi_accel | 2 | 0.00 [0.00, 0.00] | 43.1 [38.7, 47.6] | 0.00 | 43.7 | 2.50 |
| epi_const | 2 | 0.00 [0.00, 0.00] | 0.0 [0.0, 0.0] | 0.00 | 0.0 | 2.50 |
| epi_sizerand | 2 | 0.00 [0.00, 0.00] | 0.0 [0.0, 0.0] | 0.00 | 0.0 | 2.00 |
| epi_sizematch | 2 | 0.00 [0.00, 0.00] | 0.0 [0.0, 0.0] | 0.00 | 0.0 | 2.00 |

## Pairing check (epi_accel vs epi_const, same seed)

- Pairs: 2. Largest pixel difference (0 to 255) in frame 0: 0; in frame 23: 0. Mean absolute difference in frame 12: 1.24 (only the moving object differs there).
- A difference of 0 in frames 0 and 23 means the two-view input (frames 0 and 23) is identical within a pair.

## Size cue

| split | n | true moving size (m), median [min, max] | size if read as static (m), median [min, max] | max abs error of static scale vs 1/(1+k) |
|---|---|---|---|---|
| epi_accel | 2 | 1.74 [1.51, 1.98] | 1.03 [0.92, 1.15] | 5.8e-15 |
| epi_const | 2 | 1.74 [1.51, 1.98] | 1.03 [0.92, 1.15] | 5.8e-15 |
| epi_sizerand | 2 | 3.33 [3.29, 3.36] | 2.01 [1.98, 2.04] | 2.9e-15 |
| epi_sizematch | 2 | 3.42 [3.19, 3.64] | 1.94 [1.89, 1.99] | 2.8e-15 |

Static objects in every split are 1.4 to 2.2 m (the v3float range).

![GT checks](v4_pilot.png)

How to read the figure: (a) for epi_accel the static fit from frames 0 and 23 misses the object in a third view by tens of pixels, while for epi_const and for the background it fits to about 0 px. (b) Two-view separation is about 0 in every split, so frames 0 and 23 alone are ambiguous. (c) The accelerated object's image track leaves the track of its static reading (the constant-velocity object stays on it). (d) Size randomisation decouples apparent size from the static interpretation.
