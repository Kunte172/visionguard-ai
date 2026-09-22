from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PatchGrid:
    features: np.ndarray
    positions: list[tuple[int, int, int, int]]
    image_shape: tuple[int, int]


def load_grayscale(source: str | Path | Image.Image) -> np.ndarray:
    image = source if isinstance(source, Image.Image) else Image.open(source)
    array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if array.ndim != 2:
        raise ValueError("Expected a two-dimensional grayscale image")
    return array


def extract_patch_grid(image: np.ndarray, *, patch_size: int = 16, stride: int = 8) -> PatchGrid:
    height, width = image.shape
    if height < patch_size or width < patch_size:
        raise ValueError("Image is smaller than patch_size")
    if patch_size < 4 or stride < 1:
        raise ValueError("patch_size must be >= 4 and stride must be positive")

    features: list[list[float]] = []
    positions: list[tuple[int, int, int, int]] = []
    for top in range(0, height - patch_size + 1, stride):
        for left in range(0, width - patch_size + 1, stride):
            patch = image[top : top + patch_size, left : left + patch_size]
            dx = np.diff(patch, axis=1)
            dy = np.diff(patch, axis=0)
            center = patch[patch_size // 4 : -patch_size // 4, patch_size // 4 : -patch_size // 4]
            border_mean = (
                float(patch.mean()) * patch.size - float(center.mean()) * center.size
            ) / (patch.size - center.size)
            features.append(
                [
                    float(patch.mean()),
                    float(patch.std()),
                    float(np.abs(dx).mean()),
                    float(np.abs(dy).mean()),
                    float(np.quantile(patch, 0.1)),
                    float(np.quantile(patch, 0.9)),
                    float(center.mean() - border_mean),
                ]
            )
            positions.append((top, left, top + patch_size, left + patch_size))
    return PatchGrid(
        features=np.asarray(features, dtype=np.float32),
        positions=positions,
        image_shape=(height, width),
    )


def patch_scores_to_heatmap(grid: PatchGrid, scores: np.ndarray, scale: float) -> np.ndarray:
    heat_sum = np.zeros(grid.image_shape, dtype=np.float32)
    heat_count = np.zeros(grid.image_shape, dtype=np.float32)
    safe_scale = max(float(scale), 1e-8)
    normalized = np.clip(scores / safe_scale, 0, 1)
    for score, (top, left, bottom, right) in zip(normalized, grid.positions, strict=True):
        heat_sum[top:bottom, left:right] += float(score)
        heat_count[top:bottom, left:right] += 1
    return heat_sum / np.maximum(heat_count, 1)
