# Synthetic-mask parameter settings

The mask design was informed by observations of experimental crack morphology and width scale. The detailed numerical settings prescribe controlled geometric variations; they should not be interpreted as statistical estimates of each morphology parameter's population distribution.

| Component | Setting | Role |
|---|---|---|
| Canvas | Width 800, height 600 | Match the benchmark image format |
| Physical scale | 0.01278 mm/pixel | Convert design references into millimeters |
| Centerline interpolation points | Random integer from 4, 5, 6 | Cubic interpolation with limited path complexity |
| B-spline | Degree 3, smoothing 0, 500 samples | Smooth curved centerline |
| Control-point x positions | Spread across 12%–88% of canvas width; jitter ±4% of usable width; sorted | Maintain x ordering |
| Initial control-point y | Mid-height plus ±15% of usable height | Vary vertical placement |
| Vertical walk | Increments ±12% of usable height; first increment zero; y clipped to 18%–82% of height | Vary curvature while remaining inside the image |
| Width components | One or two multiplicative sinusoids | Introduce longitudinal width variation |
| Each width component | Amplitude 0.03–0.08; frequency 0.8–2.5; phase 0–2π | Moderate smooth variation |
| End taper | First and last 15% of samples; cosine ramp; minimum tip width max(1.5, 0.08w₀) pixels | Narrow the tips |
| Left fraction | Base 0.38–0.62; sinusoidal amplitude 0.05–0.15; frequency 0.5–2.0; random phase; clipped to 0.30–0.70 | Introduce bounded asymmetry |
| Boundary-noise standard deviation | clip(0.03w₀, 0.5, 2.0) pixels before smoothing | Scale irregularity with nominal width |
| Actual preset noise range | 0.5–1.32 pixels before smoothing | Applies to the nominal widths 6–44 pixels |
| Noise smoothing | Five-point moving average, same-length output | Suppress isolated spikes |
| Offset floor | 0.5 pixels after addition of noise | Keep offsets positive |
| Rasterization | Clip coordinates; convert to int32; fill polygon | Produce a 0/255 binary mask |
| Reference location | Maximum unperturbed width over sample indices 100–399 | Exclude the tapered ends |

Here w₀ is the nominal pixel width. The noise sequences, rather than the complete offsets, are smoothed. Each side receives an independent Gaussian noise sequence.

The 3%–8% amplitude applies to each sinusoidal factor. Two multiplicative factors can produce a total variation larger than ±8%.

## Preset configurations

`configs/balanced160.csv` contains the deterministic width-and-seed settings required to regenerate the 160 samples. Each row specifies the sample identifier, nominal width, and deterministic seed.

| Nominal group | Nominal pixel widths | Samples |
|---|---|---:|
| L1 | 6, 8, 10, 12 | 40 |
| L2 | 14, 16, 18 | 40 |
| L3 | 22, 26, 30 | 40 |
| L4 | 34, 36, 38, 40, 42, 44 | 40 |

The samples per individual nominal width need not be equal within a group. Group membership follows generation settings, not thresholds applied to the modulated design-reference widths.
