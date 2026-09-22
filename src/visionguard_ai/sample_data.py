from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from visionguard_ai.domain import DefectType, ImageLabel


def generate_sample_dataset(
    output_dir: str | Path,
    *,
    image_size: int = 96,
    train_normal: int = 48,
    validation_normal: int = 16,
    test_normal: int = 18,
    test_per_defect: int = 10,
    seed: int = 42,
) -> Path:
    if image_size < 64:
        raise ValueError("image_size must be at least 64")
    if min(train_normal, validation_normal, test_normal, test_per_defect) < 4:
        raise ValueError("each dataset group must contain at least 4 images")

    root = Path(output_dir)
    rows: list[dict[str, str]] = []
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    groups = [
        ("train", ImageLabel.NORMAL, DefectType.NONE, train_normal),
        ("validation", ImageLabel.NORMAL, DefectType.NONE, validation_normal),
        ("test", ImageLabel.NORMAL, DefectType.NONE, test_normal),
        ("test", ImageLabel.ANOMALY, DefectType.SCRATCH, test_per_defect),
        ("test", ImageLabel.ANOMALY, DefectType.SPOT, test_per_defect),
        ("test", ImageLabel.ANOMALY, DefectType.CRACK, test_per_defect),
    ]

    sequence = 1
    for split, label, defect_type, count in groups:
        for _ in range(count):
            image = _normal_surface(image_size, rng, np_rng)
            mask = Image.new("L", (image_size, image_size), color=0)
            if label is ImageLabel.ANOMALY:
                image, mask = _add_defect(image, mask, defect_type, rng)

            stem = f"VG-{sequence:05d}"
            image_relative = Path(split) / label.value / f"{stem}.png"
            mask_relative = Path("masks") / f"{stem}.png"
            image_path = root / image_relative
            mask_path = root / mask_relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path)
            mask.save(mask_path)
            rows.append(
                {
                    "image_id": stem,
                    "image_path": image_relative.as_posix(),
                    "mask_path": mask_relative.as_posix(),
                    "split": split,
                    "label": label.value,
                    "defect_type": defect_type.value,
                }
            )
            sequence += 1

    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _normal_surface(size: int, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    base = rng.uniform(0.48, 0.62)
    amplitude = rng.uniform(0.025, 0.06)
    frequency = rng.uniform(2.2, 4.8)
    phase = rng.uniform(0, math.tau)
    y, x = np.mgrid[0:size, 0:size]
    brushed = amplitude * np.sin((x / size) * math.tau * frequency + phase)
    vertical = 0.015 * np.cos((y / size) * math.tau * (frequency / 2))
    noise = np_rng.normal(0, 0.018, size=(size, size))
    array = np.clip(base + brushed + vertical + noise, 0, 1)
    image = Image.fromarray((array * 255).astype(np.uint8), mode="L")
    return image.filter(ImageFilter.GaussianBlur(radius=0.35))


def _add_defect(
    image: Image.Image, mask: Image.Image, defect_type: DefectType, rng: random.Random
) -> tuple[Image.Image, Image.Image]:
    image_draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)
    size = image.width
    color = rng.choice([rng.randint(20, 65), rng.randint(210, 245)])

    if defect_type is DefectType.SCRATCH:
        start = (rng.randint(12, size // 2), rng.randint(12, size - 12))
        end = (rng.randint(size // 2, size - 12), rng.randint(12, size - 12))
        width = rng.randint(2, 4)
        image_draw.line([start, end], fill=color, width=width)
        mask_draw.line([start, end], fill=255, width=width + 3)
    elif defect_type is DefectType.SPOT:
        radius = rng.randint(6, 12)
        cx = rng.randint(radius + 8, size - radius - 8)
        cy = rng.randint(radius + 8, size - radius - 8)
        box = (cx - radius, cy - radius, cx + radius, cy + radius)
        image_draw.ellipse(box, fill=color)
        mask_draw.ellipse(box, fill=255)
    elif defect_type is DefectType.CRACK:
        points = [(rng.randint(10, 24), rng.randint(18, size - 18))]
        for index in range(1, 5):
            points.append(
                (
                    int(10 + index * (size - 20) / 4),
                    max(8, min(size - 8, points[-1][1] + rng.randint(-14, 14))),
                )
            )
        image_draw.line(points, fill=color, width=2, joint="curve")
        mask_draw.line(points, fill=255, width=6, joint="curve")
    else:
        raise ValueError(f"Unsupported defect type: {defect_type}")

    return image.filter(ImageFilter.GaussianBlur(radius=0.25)), mask
