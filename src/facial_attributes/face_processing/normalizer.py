"""Normalización de rostros extraídos."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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

        Aplica escalado [0,1] y normalización con mean/std de ImageNet,
        consistente con el preprocesamiento de entrenamiento/evaluación.

        Args:
            face: Imagen del rostro.

        Returns:
            Array numpy normalizado (HWC).
        """
        resized = face.resize(self.config.target_size, Image.Resampling.LANCZOS)

        arr: np.ndarray = np.array(resized, dtype=np.float32)

        if self.config.normalize_pixels:
            arr = arr / 255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

        return arr

    def normalize_batch(self, faces: list[Image.Image]) -> np.ndarray:
        """Normalizar un lote de rostros.

        Args:
            faces: Lista de imágenes de rostros.

        Returns:
            Array numpy con lote de rostros normalizados.
        """
        normalized: list[np.ndarray] = [self.normalize(face) for face in faces]
        result: np.ndarray = np.stack(normalized)
        return result

    def get_output_shape(self) -> tuple[int, int, int]:
        """Obtener forma de salida esperada."""
        return (*self.config.target_size, 3)
