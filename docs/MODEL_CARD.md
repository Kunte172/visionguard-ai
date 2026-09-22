# Model card: patch-statistics-0.1.0

## Intended use

This model is a transparent, CPU-friendly baseline for demonstrating normal-only visual anomaly
detection. It is intended for learning, portfolio review, API integration, and initial system
prototyping.

## Not intended for

- autonomous accept/reject decisions on a live production line;
- safety-critical, medical, identity, or surveillance decisions;
- estimating performance on a real factory from synthetic benchmark metrics;
- images captured under an uncalibrated camera, viewpoint, exposure, or lighting setup.

## Training data

Only synthetic normal images from the `train` split are used for fitting. Synthetic normal
validation images calibrate the decision threshold and heatmap scale. Defect images and masks are
held out for evaluation. The generator produces brushed grayscale surfaces with controlled noise,
then adds scratch, circular spot, or branching crack patterns to test images.

## Method

Each overlapping patch is represented by intensity mean/standard deviation, horizontal and
vertical absolute gradients, 10th/90th percentiles, and center-versus-border contrast. Features
are standardized and stored in a bounded normal memory bank. A patch score is the mean distance
to its nearest normal neighbours. The image score averages the largest 5% of patch scores.

## Metrics

The evaluated values live in `artifacts/metrics.json` and are reproduced by `visionguard train`.
They cover image AUROC, average precision, threshold accuracy, and pixel AUROC. Results apply only
to the committed synthetic generator and fixed seed.

## Known limitations and risks

- Hand-crafted statistics cannot express complex semantic defects.
- Synthetic anomalies are cleaner than most real defects and may inflate results.
- Resizing to 96×96 can erase small defects or distort aspect ratio.
- The validation threshold is sensitive to changes in camera and surface distribution.
- Heatmaps use overlapping patch aggregation and are approximate localization, not segmentation.
- No confidence calibration, abstention band, drift detector, or per-product profile exists.

## Human oversight

A real operator should review anomaly images and heatmaps. The system should default to manual
review when its input validation, camera health, model loading, or drift checks fail. Threshold
changes should require an evaluated dataset version and approval trail.
