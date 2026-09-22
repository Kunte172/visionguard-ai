from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from visionguard_ai.features import extract_patch_grid, load_grayscale, patch_scores_to_heatmap


def test_extract_grid_and_heatmap() -> None:
    image = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    grid = extract_patch_grid(image, patch_size=16, stride=8)
    assert grid.features.shape == (9, 7)
    assert grid.image_shape == (32, 32)
    heatmap = patch_scores_to_heatmap(grid, np.arange(9, dtype=np.float32), scale=8)
    assert heatmap.shape == image.shape
    assert float(heatmap.min()) >= 0
    assert float(heatmap.max()) <= 1


def test_load_grayscale_from_pillow() -> None:
    image = Image.new("RGB", (12, 10), color=(255, 0, 0))
    array = load_grayscale(image)
    assert array.shape == (10, 12)
    assert array.dtype == np.float32


def test_extract_grid_validates_arguments() -> None:
    image = np.zeros((12, 12), dtype=np.float32)
    with pytest.raises(ValueError, match="smaller"):
        extract_patch_grid(image, patch_size=16)
    with pytest.raises(ValueError, match="patch_size"):
        extract_patch_grid(image, patch_size=3)
