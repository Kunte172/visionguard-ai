from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_path: str


class PredictionResponse(BaseModel):
    anomaly_score: float = Field(ge=0)
    threshold: float = Field(ge=0)
    is_anomaly: bool
    model_version: str
    heatmap_data_url: str
