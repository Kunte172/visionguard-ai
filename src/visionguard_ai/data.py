from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from visionguard_ai.domain import DefectType, ImageLabel

REQUIRED_COLUMNS = (
    "image_id",
    "image_path",
    "mask_path",
    "split",
    "label",
    "defect_type",
)


class DatasetValidationError(ValueError):
    """Raised when an image dataset violates the VisionGuard contract."""


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest = Path(path)
    if not manifest.exists():
        raise DatasetValidationError(f"Manifest does not exist: {manifest}")
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise DatasetValidationError(f"Missing required columns: {', '.join(missing)}")
    data = frame.loc[:, REQUIRED_COLUMNS].copy()
    if data.empty:
        raise DatasetValidationError("Manifest must contain at least one image")
    if data["image_id"].duplicated().any():
        raise DatasetValidationError("image_id values must be unique")

    _validate_values(data, "split", {"train", "validation", "test"})
    _validate_values(data, "label", {item.value for item in ImageLabel})
    _validate_values(data, "defect_type", {item.value for item in DefectType})
    if (data.loc[data["split"] == "train", "label"] != ImageLabel.NORMAL.value).any():
        raise DatasetValidationError("Training split must contain only normal images")

    root = manifest.parent
    dimensions: set[tuple[int, int]] = set()
    for row in data.itertuples(index=False):
        image_path = root / str(row.image_path)
        mask_path = root / str(row.mask_path)
        if not image_path.exists() or not mask_path.exists():
            raise DatasetValidationError(f"Missing image or mask for {row.image_id}")
        with Image.open(image_path) as image:
            image_size = image.size
            dimensions.add(image_size)
        with Image.open(mask_path) as mask:
            if mask.size != image_size:
                raise DatasetValidationError(f"Mask size does not match image for {row.image_id}")
    if len(dimensions) != 1:
        raise DatasetValidationError("All images must use the same dimensions")
    return data


def resolve_paths(manifest_path: str | Path, data: pd.DataFrame) -> pd.DataFrame:
    root = Path(manifest_path).parent
    resolved = data.copy()
    resolved["image_path"] = resolved["image_path"].map(lambda value: str(root / value))
    resolved["mask_path"] = resolved["mask_path"].map(lambda value: str(root / value))
    return resolved


def _validate_values(data: pd.DataFrame, column: str, allowed: set[str]) -> None:
    invalid = sorted(set(data[column]) - allowed)
    if invalid:
        raise DatasetValidationError(f"Invalid {column} values: {', '.join(invalid)}")
