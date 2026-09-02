"""Transformaciones compartidas para pipelines de preprocessing."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class TransformRecord:
    """Registro de una transformación aplicada."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


class ImageTransformer:
    """Transformaciones base para imágenes."""

    def __init__(self, target_size: tuple[int, int] = (224, 224)) -> None:
        self.target_size = target_size
        self.records: list[TransformRecord] = []

    def resize(self, image: Image.Image) -> Image.Image:
        """Redimensionar imagen al tamaño objetivo."""
        resized = image.resize(self.target_size, Image.Resampling.LANCZOS)
        self.records.append(
            TransformRecord("resize", {"target_size": self.target_size})
        )
        return resized

    def normalize_color(self, image: Image.Image) -> Image.Image:
        """Convertir a formato RGB uniforme."""
        if image.mode != "RGB":
            converted = image.convert("RGB")
            self.records.append(
                TransformRecord("normalize_color", {"from_mode": image.mode})
            )
            return converted
        self.records.append(TransformRecord("normalize_color", {"no_change": True}))
        return image

    def to_numpy(self, image: Image.Image) -> np.ndarray:
        """Convertir imagen a array numpy float32 normalizado."""
        arr = np.array(image, dtype=np.float32) / 255.0
        self.records.append(TransformRecord("to_numpy", {"normalized": True}))
        return arr

    def clear_records(self) -> None:
        """Limpiar registro de transformaciones."""
        self.records.clear()

    def get_records(self) -> list[TransformRecord]:
        """Obtener registro de transformaciones."""
        return self.records.copy()


def load_image(image_path: Path) -> Image.Image:
    """Cargar imagen desde disco."""
    return Image.open(image_path)
