from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from visionguard_ai import __version__
from visionguard_ai.schemas import HealthResponse, PredictionResponse
from visionguard_ai.service import InvalidImageError, VisualInspectionService

MAX_FILE_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg"}


def create_app(model_path: str | Path | None = None) -> FastAPI:
    resolved_model_path = (
        Path(model_path)
        if model_path is not None
        else Path(os.getenv("VISIONGUARD_MODEL_PATH", "artifacts/detector.joblib"))
    )
    service = VisualInspectionService(resolved_model_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if resolved_model_path.exists():
            service.load()
        yield

    app = FastAPI(
        title="VisionGuard AI API",
        description="Normal-only visual anomaly detection with localized heatmaps.",
        version=__version__,
        lifespan=lifespan,
    )
    origins = [
        origin.strip()
        for origin in os.getenv("VISIONGUARD_CORS_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        model_available = service.ready or resolved_model_path.exists()
        return HealthResponse(
            status="ok" if model_available else "degraded",
            model_ready=model_available,
            model_path=str(resolved_model_path),
        )

    @app.get("/v1/model", tags=["model"])
    def model_info() -> dict[str, object]:
        try:
            if not service.ready:
                service.load()
        except (FileNotFoundError, TypeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return service.metadata

    @app.post("/v1/predict", response_model=PredictionResponse, tags=["inference"])
    async def predict(file: Annotated[UploadFile, File()]) -> PredictionResponse:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail="Only PNG and JPEG images are supported")
        payload = await file.read(MAX_FILE_BYTES + 1)
        if len(payload) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 5 MB limit")
        try:
            result = service.predict_bytes(payload)
        except (FileNotFoundError, TypeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except InvalidImageError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictionResponse.model_validate(result, from_attributes=True)

    return app


app = create_app()
