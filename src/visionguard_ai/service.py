from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from visionguard_ai.features import load_grayscale
from visionguard_ai.model import DetectorBundle, load_bundle, predict_array
from visionguard_ai.visualization import image_to_data_url, render_overlay


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a supported image."""


@dataclass(frozen=True)
class VisualPrediction:
    anomaly_score: float
    threshold: float
    is_anomaly: bool
    model_version: str
    heatmap_data_url: str


class VisualInspectionService:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._bundle: DetectorBundle | None = None

    @property
    def ready(self) -> bool:
        return self._bundle is not None

    @property
    def metadata(self) -> dict[str, Any]:
        return {} if self._bundle is None else dict(self._bundle.metadata)

    def load(self) -> None:
        self._bundle = load_bundle(self.model_path)

    def predict_bytes(self, payload: bytes) -> VisualPrediction:
        if self._bundle is None:
            self.load()
        assert self._bundle is not None
        try:
            with Image.open(io.BytesIO(payload)) as uploaded:
                uploaded.load()
                image = uploaded.convert("RGB").resize((96, 96))
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("Uploaded file is not a valid image") from exc
        prediction = predict_array(self._bundle, load_grayscale(image))
        overlay = render_overlay(image, prediction.heatmap)
        return VisualPrediction(
            anomaly_score=prediction.anomaly_score,
            threshold=prediction.threshold,
            is_anomaly=prediction.is_anomaly,
            model_version=str(self._bundle.metadata["model_version"]),
            heatmap_data_url=image_to_data_url(overlay),
        )
