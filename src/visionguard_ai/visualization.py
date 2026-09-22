from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageFilter


def render_overlay(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    base = image.convert("RGB")
    heat = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8), mode="L")
    heat = heat.filter(ImageFilter.GaussianBlur(radius=2.2))
    heat_array = np.asarray(heat, dtype=np.float32) / 255.0
    red = np.full_like(heat_array, 255)
    green = np.clip(255 * (1 - heat_array * 0.65), 0, 255)
    blue = np.clip(70 * (1 - heat_array), 0, 255)
    color = Image.fromarray(np.stack([red, green, blue], axis=-1).astype(np.uint8), mode="RGB")
    alpha = Image.fromarray((heat_array * 175).astype(np.uint8), mode="L")
    return Image.composite(color, base, alpha)


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
