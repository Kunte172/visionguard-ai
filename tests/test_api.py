from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from visionguard_ai.api import create_app


def image_bytes(fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (96, 96), color=(120, 125, 122)).save(buffer, format=fmt)
    return buffer.getvalue()


def test_health_and_model_info(trained_project: dict[str, Path | object]) -> None:
    with TestClient(create_app(Path(trained_project["model"]))) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_ready"] is True
        model = client.get("/v1/model")
        assert model.status_code == 200
        assert model.json()["training_mode"] == "normal-only"


def test_prediction_endpoint(trained_project: dict[str, Path | object]) -> None:
    with TestClient(create_app(Path(trained_project["model"]))) as client:
        response = client.post(
            "/v1/predict",
            files={"file": ("surface.png", image_bytes(), "image/png")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "patch-statistics-0.1.0"
    assert body["heatmap_data_url"].startswith("data:image/png;base64,")


def test_prediction_validation_errors(trained_project: dict[str, Path | object]) -> None:
    with TestClient(create_app(Path(trained_project["model"]))) as client:
        unsupported = client.post(
            "/v1/predict", files={"file": ("notes.txt", b"hello", "text/plain")}
        )
        invalid = client.post(
            "/v1/predict", files={"file": ("broken.png", b"not-an-image", "image/png")}
        )
        oversized = client.post(
            "/v1/predict",
            files={"file": ("large.png", b"x" * (5 * 1024 * 1024 + 1), "image/png")},
        )
    assert unsupported.status_code == 415
    assert invalid.status_code == 422
    assert oversized.status_code == 413


def test_missing_model_reports_degraded(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path / "missing.joblib")) as client:
        assert client.get("/health").json()["status"] == "degraded"
        assert client.get("/v1/model").status_code == 503
        response = client.post(
            "/v1/predict", files={"file": ("surface.jpg", image_bytes("JPEG"), "image/jpeg")}
        )
        assert response.status_code == 503
