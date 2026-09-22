from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from visionguard_ai.data import load_manifest, resolve_paths
from visionguard_ai.features import (
    PatchGrid,
    extract_patch_grid,
    load_grayscale,
    patch_scores_to_heatmap,
)


@dataclass(frozen=True)
class DetectorBundle:
    scaler: StandardScaler
    memory_features: np.ndarray
    threshold: float
    heatmap_scale: float
    patch_size: int
    stride: int
    neighbors: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ImagePrediction:
    anomaly_score: float
    threshold: float
    is_anomaly: bool
    heatmap: np.ndarray


@dataclass(frozen=True)
class TrainingResult:
    bundle: DetectorBundle
    metrics: dict[str, Any]
    predictions: pd.DataFrame


def train_detector(
    manifest_path: str | Path,
    *,
    patch_size: int = 16,
    stride: int = 8,
    neighbors: int = 5,
    memory_limit: int = 4_000,
    threshold_quantile: float = 0.99,
    seed: int = 42,
) -> TrainingResult:
    data = resolve_paths(manifest_path, load_manifest(manifest_path))
    if not 0.9 <= threshold_quantile < 1:
        raise ValueError("threshold_quantile must be in [0.9, 1.0)")
    if neighbors < 1 or memory_limit < neighbors:
        raise ValueError("memory_limit must be at least the number of neighbors")

    train = data[data["split"] == "train"]
    validation = data[data["split"] == "validation"]
    test = data[data["split"] == "test"]
    train_features = np.concatenate(
        [
            extract_patch_grid(load_grayscale(path), patch_size=patch_size, stride=stride).features
            for path in train["image_path"]
        ]
    )
    scaler = StandardScaler().fit(train_features)
    scaled_train = scaler.transform(train_features).astype(np.float32)
    rng = np.random.default_rng(seed)
    if len(scaled_train) > memory_limit:
        indices = rng.choice(len(scaled_train), size=memory_limit, replace=False)
        memory = scaled_train[indices]
    else:
        memory = scaled_train

    provisional = DetectorBundle(
        scaler=scaler,
        memory_features=memory,
        threshold=0,
        heatmap_scale=1,
        patch_size=patch_size,
        stride=stride,
        neighbors=neighbors,
        metadata={},
    )
    validation_image_scores: list[float] = []
    validation_patch_scores: list[np.ndarray] = []
    for path in validation["image_path"]:
        grid, patch_scores = _score_patches(provisional, load_grayscale(path))
        validation_image_scores.append(_image_score(patch_scores))
        validation_patch_scores.append(patch_scores)
        assert grid.features.shape[0] == patch_scores.shape[0]

    threshold = float(np.quantile(validation_image_scores, threshold_quantile))
    heatmap_scale = float(np.quantile(np.concatenate(validation_patch_scores), threshold_quantile))
    metadata = {
        "model_version": "patch-statistics-0.1.0",
        "trained_at": datetime.now(UTC).isoformat(),
        "algorithm": "scaled-patch-features-knn-distance",
        "training_mode": "normal-only",
        "patch_size": patch_size,
        "stride": stride,
        "neighbors": neighbors,
        "memory_patches": int(len(memory)),
        "synthetic_data": True,
    }
    bundle = DetectorBundle(
        scaler=scaler,
        memory_features=memory,
        threshold=threshold,
        heatmap_scale=heatmap_scale,
        patch_size=patch_size,
        stride=stride,
        neighbors=neighbors,
        metadata=metadata,
    )
    predictions, metrics = evaluate_detector(bundle, test)
    metrics["dataset"] = {
        "rows": int(len(data)),
        "train_normal": int(len(train)),
        "validation_normal": int(len(validation)),
        "test_rows": int(len(test)),
        "synthetic": True,
    }
    return TrainingResult(bundle=bundle, metrics=metrics, predictions=predictions)


def evaluate_detector(
    bundle: DetectorBundle, test: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pixel_truth: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    for row in test.itertuples(index=False):
        prediction = predict_array(bundle, load_grayscale(str(row.image_path)))
        with Image.open(str(row.mask_path)) as mask_image:
            mask = np.asarray(mask_image.convert("L"), dtype=np.float32) / 255.0
        pixel_truth.append((mask > 0.5).ravel())
        pixel_scores.append(prediction.heatmap.ravel())
        rows.append(
            {
                "image_id": row.image_id,
                "label": row.label,
                "defect_type": row.defect_type,
                "anomaly_score": prediction.anomaly_score,
                "threshold": prediction.threshold,
                "predicted_label": "anomaly" if prediction.is_anomaly else "normal",
            }
        )
    predictions = pd.DataFrame(rows)
    truth = (predictions["label"] == "anomaly").astype(int)
    predicted = (predictions["predicted_label"] == "anomaly").astype(int)
    all_pixel_truth = np.concatenate(pixel_truth).astype(int)
    all_pixel_scores = np.concatenate(pixel_scores)
    metrics = {
        "image": {
            "auroc": float(roc_auc_score(truth, predictions["anomaly_score"])),
            "average_precision": float(
                average_precision_score(truth, predictions["anomaly_score"])
            ),
            "accuracy_at_threshold": float(accuracy_score(truth, predicted)),
            "threshold": bundle.threshold,
        },
        "pixel": {
            "auroc": float(roc_auc_score(all_pixel_truth, all_pixel_scores)),
        },
    }
    return predictions, metrics


def predict_array(bundle: DetectorBundle, image: np.ndarray) -> ImagePrediction:
    grid, patch_scores = _score_patches(bundle, image)
    image_score = _image_score(patch_scores)
    heatmap = patch_scores_to_heatmap(grid, patch_scores, bundle.heatmap_scale)
    return ImagePrediction(
        anomaly_score=image_score,
        threshold=bundle.threshold,
        is_anomaly=image_score > bundle.threshold,
        heatmap=heatmap,
    )


def save_training_result(
    result: TrainingResult,
    *,
    model_path: str | Path,
    metrics_path: str | Path,
    predictions_path: str | Path,
) -> None:
    destinations = [Path(model_path), Path(metrics_path), Path(predictions_path)]
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.bundle, destinations[0])
    destinations[1].write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result.predictions.to_csv(destinations[2], index=False)


def load_bundle(path: str | Path) -> DetectorBundle:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Model artifact does not exist: {source}")
    bundle = joblib.load(source)
    if not isinstance(bundle, DetectorBundle):
        raise TypeError("Model artifact is not a VisionGuard DetectorBundle")
    return bundle


def _score_patches(bundle: DetectorBundle, image: np.ndarray) -> tuple[PatchGrid, np.ndarray]:
    grid = extract_patch_grid(image, patch_size=bundle.patch_size, stride=bundle.stride)
    query = bundle.scaler.transform(grid.features).astype(np.float32)
    squared = (
        np.sum(query**2, axis=1, keepdims=True)
        + np.sum(bundle.memory_features**2, axis=1)
        - 2 * query @ bundle.memory_features.T
    )
    distances = np.sqrt(np.maximum(squared, 0))
    nearest = np.partition(distances, bundle.neighbors - 1, axis=1)[:, : bundle.neighbors]
    return grid, nearest.mean(axis=1)


def _image_score(patch_scores: np.ndarray) -> float:
    top_count = max(1, int(math.ceil(len(patch_scores) * 0.05)))
    return float(np.sort(patch_scores)[-top_count:].mean())
