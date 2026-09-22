from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from visionguard_ai.data import DatasetValidationError, load_manifest, resolve_paths
from visionguard_ai.sample_data import generate_sample_dataset


def test_generate_and_validate_dataset(tmp_path: Path) -> None:
    manifest = generate_sample_dataset(
        tmp_path / "sample",
        train_normal=4,
        validation_normal=4,
        test_normal=4,
        test_per_defect=4,
        seed=2,
    )
    data = load_manifest(manifest)
    assert len(data) == 24
    assert set(data.loc[data["split"] == "train", "label"]) == {"normal"}
    assert set(data["defect_type"]) == {"none", "scratch", "spot", "crack"}
    resolved = resolve_paths(manifest, data)
    assert all(Path(path).is_file() for path in resolved["image_path"])


def test_generator_rejects_invalid_sizes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 64"):
        generate_sample_dataset(tmp_path, image_size=32)
    with pytest.raises(ValueError, match="at least 4"):
        generate_sample_dataset(tmp_path, train_normal=2)


def test_manifest_contract_errors(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="does not exist"):
        load_manifest(tmp_path / "missing.csv")

    empty = tmp_path / "empty.csv"
    empty.write_text("image_id,image_path,mask_path,split,label,defect_type\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="at least one"):
        load_manifest(empty)

    missing_column = tmp_path / "missing-column.csv"
    missing_column.write_text("image_id,image_path\n1,x.png\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Missing required columns"):
        load_manifest(missing_column)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("split", "future", "Invalid split"),
        ("label", "maybe", "Invalid label"),
        ("defect_type", "dent", "Invalid defect_type"),
    ],
)
def test_manifest_rejects_invalid_values(
    tmp_path: Path, column: str, value: str, message: str
) -> None:
    manifest = generate_sample_dataset(
        tmp_path / column,
        train_normal=4,
        validation_normal=4,
        test_normal=4,
        test_per_defect=4,
    )
    data = pd.read_csv(manifest, dtype=str)
    data.loc[0, column] = value
    data.to_csv(manifest, index=False)
    with pytest.raises(DatasetValidationError, match=message):
        load_manifest(manifest)


def test_manifest_rejects_anomaly_in_training_and_duplicates(tmp_path: Path) -> None:
    manifest = generate_sample_dataset(
        tmp_path / "training",
        train_normal=4,
        validation_normal=4,
        test_normal=4,
        test_per_defect=4,
    )
    data = pd.read_csv(manifest, dtype=str)
    data.loc[0, "label"] = "anomaly"
    data.loc[0, "defect_type"] = "scratch"
    data.to_csv(manifest, index=False)
    with pytest.raises(DatasetValidationError, match="only normal"):
        load_manifest(manifest)

    data.loc[0, ["label", "defect_type"]] = ["normal", "none"]
    data.loc[1, "image_id"] = data.loc[0, "image_id"]
    data.to_csv(manifest, index=False)
    with pytest.raises(DatasetValidationError, match="unique"):
        load_manifest(manifest)


def test_manifest_rejects_missing_file_and_wrong_mask_shape(tmp_path: Path) -> None:
    manifest = generate_sample_dataset(
        tmp_path / "files",
        train_normal=4,
        validation_normal=4,
        test_normal=4,
        test_per_defect=4,
    )
    data = pd.read_csv(manifest, dtype=str)
    missing = manifest.parent / data.loc[0, "image_path"]
    missing.unlink()
    with pytest.raises(DatasetValidationError, match="Missing image or mask"):
        load_manifest(manifest)

    manifest = generate_sample_dataset(
        tmp_path / "shape",
        train_normal=4,
        validation_normal=4,
        test_normal=4,
        test_per_defect=4,
    )
    data = pd.read_csv(manifest, dtype=str)
    mask = manifest.parent / data.loc[0, "mask_path"]
    Image.new("L", (80, 80)).save(mask)
    with pytest.raises(DatasetValidationError, match="Mask size"):
        load_manifest(manifest)


def test_manifest_rejects_inconsistent_image_dimensions(tmp_path: Path) -> None:
    manifest = generate_sample_dataset(
        tmp_path / "dimensions",
        train_normal=4,
        validation_normal=4,
        test_normal=4,
        test_per_defect=4,
    )
    data = pd.read_csv(manifest, dtype=str)
    image_path = manifest.parent / data.loc[0, "image_path"]
    mask_path = manifest.parent / data.loc[0, "mask_path"]
    Image.new("L", (80, 80)).save(image_path)
    Image.new("L", (80, 80)).save(mask_path)
    with pytest.raises(DatasetValidationError, match="same dimensions"):
        load_manifest(manifest)
