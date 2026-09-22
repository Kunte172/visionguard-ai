# Evaluation protocol

## Split policy

| Split | Labels | Used for |
|---|---|---|
| train | normal only | scaler and patch memory |
| validation | normal only | image threshold and heatmap scale |
| test | normal + three defects | image and pixel metrics |

The validator fails if any anomalous row appears in training. Masks are never consumed by the
trainer; they are used only for pixel-level evaluation.

## Metrics

- **Image AUROC** assesses ranking across all held-out normal and anomalous images.
- **Average precision** summarizes precision/recall across score thresholds.
- **Accuracy at threshold** checks the deployed threshold calibrated only on validation normals.
- **Pixel AUROC** compares the aggregated patch heatmap with generated defect masks.

## Reproduce

```bash
uv run visionguard generate-sample --output data/sample --seed 42
uv run visionguard validate --manifest data/sample/manifest.csv
uv run visionguard train --manifest data/sample/manifest.csv
```

The command writes `artifacts/metrics.json` and `artifacts/test_predictions.csv`. The latter is
ignored because it is a derived artifact. Do not compare these synthetic numbers directly with
published industrial anomaly-detection benchmarks.

## Acceptance criteria for a real pilot

Before a line trial, define defect-size-weighted recall, false reject rate, throughput, p95/p99
latency, camera failure detection, and an abstention/manual-review policy with the quality team.
Measure each product and camera configuration separately, including temporal and batch splits.
