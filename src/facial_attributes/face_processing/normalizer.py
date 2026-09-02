"""Normalización de rostros extraídos."""

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class NormalizerConfig:
    """Configuración del normalizador."""

    target_size: tuple[int, int] = (224, 224)
    normalize_pixels: bool = True


class FaceNormalizer:
    """Normalizador de rostros para modelo."""

    def __init__(self, config: NormalizerConfig | None = None) -> None:
        self.config = config or NormalizerConfig()

    def normalize(self, face: Image.Image) -> np.ndarray:
        """Normalizar un rostro extraído.

        Args:
            face: Imagen del rostro.

        Returns:
            Array numpy normalizado.
        """
        resized = face.resize(self.config.target_size, Image.Resampling.LANCZOS)

        arr = np.array(resized, dtype=np.float32)

        if self.config.normalize_pixels:
            arr = arr / 255.0

        return arr

    def normalize_batch(self, faces: list[Image.Image]) -> np.ndarray:
        """Normalizar un lote de rostros.

        Args:
            faces: Lista de imágenes de rostros.

        Returns:
            Array numpy con lote de rostros normalizados.
        """
        normalized = [self.normalize(face) for face in faces]
        return np.stack(normalized)

    def get_output_shape(self) -> tuple[int, int, int]:
        """Obtener forma de salida esperada."""
        return (*self.config.target_size, 3)
