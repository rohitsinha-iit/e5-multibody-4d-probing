# w2_dose: dose-response of the object-motion slope (review item 1, W2)

Per-object slope (objects with more than 20 moving query points), winsorised to [-0.5, 1.5]. Bins are quintiles of the Any4D objects (number of motions: its values). Regression: OLS of per-object slope on all factors, standardised (coefficient = change in slope per 1 SD), scene-cluster bootstrap 95% CI. Confidence is model-specific and is added in a separate fit.

What each pattern implies:
- Regression shrinkage (the prediction moves toward the training-set mean under uncertainty) predicts a LOWER slope at low texture, small area, high occlusion and low confidence, so positive coefficients for texture, area and confidence and a negative one for occlusion.
- A structural pull toward the static (camera-only) hypothesis that does not depend on how well the object is seen predicts flat curves for texture, area, occlusion and confidence, with dependence on the geometry instead (camera rotation, camera-induced displacement, number of motions).
- Speed: a lower slope at high speed is expected under both readings (large motions are rarer and harder), so it does not separate them; a lower slope at LOW speed points to a fixed offset or a detection threshold.

| model | data | objects | log speed | log area | texture | occlusion | cam rotation | n motions | log cam disp | confidence (added to the same fit) |
|---|---|---|---|---|---|---|---|---|---|---|
| Any4D | v2 | 138 | +0.018 [-0.031, +0.062] | +0.071 [+0.030, +0.113] | -0.025 [-0.067, +0.020] | -0.050 [-0.088, -0.014] | +0.021 [-0.019, +0.059] | -0.010 [-0.040, +0.020] | -0.030 [-0.065, +0.004] | -0.018 [-0.056, +0.024] |
| VGGT-1B tracker | v2 | 138 | +0.016 [-0.029, +0.058] | -0.016 [-0.073, +0.042] | -0.025 [-0.077, +0.027] | +0.060 [+0.011, +0.113] | -0.092 [-0.151, -0.033] | -0.036 [-0.090, +0.015] | +0.064 [+0.000, +0.121] | -0.090 [-0.127, -0.048] |
| CoTracker3, 24 fr | v2 | 138 | +0.019 [+0.004, +0.034] | +0.031 [-0.008, +0.091] | -0.000 [-0.012, +0.012] | -0.022 [-0.043, -0.003] | -0.002 [-0.019, +0.010] | -0.007 [-0.022, +0.007] | -0.021 [-0.045, -0.002] | +0.026 [+0.002, +0.044] |
| CoTracker3, 7 fr | v2 | 138 | +0.019 [+0.000, +0.041] | +0.031 [-0.009, +0.089] | -0.001 [-0.016, +0.016] | -0.037 [-0.068, -0.010] | +0.005 [-0.016, +0.023] | -0.009 [-0.025, +0.008] | -0.027 [-0.053, -0.005] | +0.020 [-0.001, +0.037] |
| RAFT, 0 to 23 | v2 | 138 | -0.059 [-0.139, +0.016] | +0.128 [+0.054, +0.207] | +0.012 [-0.070, +0.100] | -0.117 [-0.205, -0.028] | -0.026 [-0.141, +0.076] | +0.012 [-0.061, +0.086] | +0.010 [-0.087, +0.098] | +0.146 [+0.059, +0.239] |
| RAFT chain, 7 fr | v2 | 138 | +0.007 [-0.016, +0.029] | +0.058 [+0.017, +0.111] | +0.003 [-0.022, +0.027] | -0.043 [-0.073, -0.016] | +0.004 [-0.018, +0.025] | -0.003 [-0.031, +0.021] | -0.007 [-0.032, +0.017] | n/a |
| RAFT chain, 24 fr | v2 | 138 | +0.021 [-0.004, +0.045] | +0.072 [+0.025, +0.126] | +0.022 [-0.010, +0.051] | -0.031 [-0.064, +0.000] | +0.001 [-0.025, +0.028] | -0.004 [-0.034, +0.025] | +0.001 [-0.026, +0.029] | n/a |
| Any4D | v3float | 84 | +0.009 [-0.035, +0.051] | +0.255 [+0.169, +0.332] | +0.008 [-0.047, +0.065] | -0.107 [-0.152, -0.056] | -0.054 [-0.150, +0.041] | +0.010 [-0.043, +0.063] | -0.090 [-0.169, -0.006] | +0.020 [-0.022, +0.068] |
| VGGT-1B tracker | v3float | 84 | -0.017 [-0.103, +0.073] | +0.108 [+0.006, +0.202] | +0.010 [-0.095, +0.108] | -0.023 [-0.129, +0.073] | -0.113 [-0.243, -0.003] | +0.000 [-0.087, +0.094] | +0.119 [-0.004, +0.222] | +0.028 [-0.082, +0.121] |
| CoTracker3, 24 fr | v3float | 84 | +0.005 [-0.001, +0.014] | +0.017 [+0.001, +0.038] | +0.001 [-0.015, +0.013] | -0.008 [-0.019, +0.005] | -0.000 [-0.018, +0.014] | +0.002 [-0.006, +0.010] | +0.008 [-0.001, +0.018] | +0.009 [+0.000, +0.020] |
| CoTracker3, 7 fr | v3float | 84 | -0.009 [-0.030, +0.008] | +0.023 [-0.002, +0.052] | +0.011 [-0.010, +0.030] | -0.022 [-0.040, -0.003] | -0.003 [-0.021, +0.013] | +0.006 [-0.005, +0.017] | +0.005 [-0.010, +0.021] | +0.042 [+0.010, +0.069] |
| RAFT, 0 to 23 | v3float | 84 | -0.143 [-0.232, -0.051] | +0.193 [+0.062, +0.331] | +0.023 [-0.105, +0.152] | -0.247 [-0.352, -0.122] | +0.026 [-0.153, +0.190] | +0.008 [-0.067, +0.089] | -0.034 [-0.193, +0.111] | +0.056 [-0.047, +0.173] |
| RAFT chain, 7 fr | v3float | 84 | -0.016 [-0.053, +0.014] | +0.056 [+0.024, +0.091] | +0.021 [-0.004, +0.045] | -0.045 [-0.075, -0.015] | +0.003 [-0.024, +0.027] | +0.010 [-0.012, +0.031] | -0.004 [-0.032, +0.025] | n/a |
| RAFT chain, 24 fr | v3float | 84 | -0.013 [-0.055, +0.023] | +0.055 [+0.026, +0.084] | +0.023 [-0.002, +0.048] | -0.035 [-0.061, -0.009] | +0.004 [-0.026, +0.032] | +0.014 [-0.009, +0.033] | +0.001 [-0.030, +0.034] | n/a |

![v2](fig_w2_dose_v2.png)

![floating](fig_w2_dose_v3float.png)

![regression](fig_w2_dose_regression.png)
