from __future__ import annotations

from pathlib import Path

import pytest

from visionguard_ai.model import TrainingResult, save_training_result, train_detector
from visionguard_ai.sample_data import generate_sample_dataset


@pytest.fixture(scope="session")
def trained_project(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path | TrainingResult]:
    root = tmp_path_factory.mktemp("visionguard")
    manifest = generate_sample_dataset(
        root / "sample",
        train_normal=12,
        validation_normal=6,
        test_normal=6,
        test_per_defect=4,
        seed=17,
    )
    result = train_detector(manifest, memory_limit=1_500)
    model = root / "detector.joblib"
    metrics = root / "metrics.json"
    predictions = root / "predictions.csv"
    save_training_result(
        result,
        model_path=model,
        metrics_path=metrics,
        predictions_path=predictions,
    )
    return {
        "root": root,
        "manifest": manifest,
        "result": result,
        "model": model,
        "metrics": metrics,
        "predictions": predictions,
    }
