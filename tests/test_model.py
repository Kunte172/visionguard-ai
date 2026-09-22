from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from visionguard_ai.features import load_grayscale
from visionguard_ai.model import DetectorBundle, load_bundle, predict_array, train_detector


def test_training_outputs_metrics_and_artifacts(
    trained_project: dict[str, Path | object],
) -> None:
    result = trained_project["result"]
    assert hasattr(result, "metrics")
    metrics = result.metrics  # type: ignore[union-attr]
    assert 0 <= metrics["image"]["auroc"] <= 1
    assert 0 <= metrics["pixel"]["auroc"] <= 1
    assert metrics["dataset"]["train_normal"] == 12
    assert json.loads(Path(trained_project["metrics"]).read_text())["dataset"]["synthetic"]
    assert Path(trained_project["predictions"]).is_file()


def test_bundle_round_trip_and_prediction(trained_project: dict[str, Path | object]) -> None:
    bundle = load_bundle(Path(trained_project["model"]))
    assert isinstance(bundle, DetectorBundle)
    sample = next((Path(trained_project["root"]) / "sample/test/anomaly").glob("*.png"))
    prediction = predict_array(bundle, load_grayscale(sample))
    assert prediction.anomaly_score >= 0
    assert prediction.heatmap.shape == (96, 96)
    assert float(prediction.heatmap.max()) <= 1


def test_load_bundle_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "missing.joblib")
    invalid = tmp_path / "invalid.joblib"
    joblib.dump({"not": "a model"}, invalid)
    with pytest.raises(TypeError, match="DetectorBundle"):
        load_bundle(invalid)


def test_training_validates_hyperparameters(trained_project: dict[str, Path | object]) -> None:
    manifest = Path(trained_project["manifest"])
    with pytest.raises(ValueError, match="threshold_quantile"):
        train_detector(manifest, threshold_quantile=0.5)
    with pytest.raises(ValueError, match="memory_limit"):
        train_detector(manifest, neighbors=5, memory_limit=2)


def test_prediction_rejects_too_small_image(trained_project: dict[str, Path | object]) -> None:
    bundle = load_bundle(Path(trained_project["model"]))
    with pytest.raises(ValueError, match="smaller"):
        predict_array(bundle, np.zeros((8, 8), dtype=np.float32))
