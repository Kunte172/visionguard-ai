from __future__ import annotations

from enum import StrEnum


class ImageLabel(StrEnum):
    NORMAL = "normal"
    ANOMALY = "anomaly"


class DefectType(StrEnum):
    NONE = "none"
    SCRATCH = "scratch"
    SPOT = "spot"
    CRACK = "crack"
