"""Pipeline de preprocessing para entrenamiento."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from facial_attributes.preprocessing.transforms import (
    ImageTransformer,
    TransformRecord,
    load_image,
)


@dataclass
class AugmentationConfig:
    """Configuración de data augmentation."""

    enabled: bool = True
    horizontal_flip_prob: float = 0.5
    rotation_degrees: float = 10.0
    brightness_factor: float = 0.2
    contrast_factor: float = 0.2


@dataclass
class TrainingConfig:
    """Configuración del pipeline de entrenamiento."""

    target_size: tuple[int, int] = (224, 224)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    seed: int = 42


class TrainingPreprocessor:
    """Pipeline de preprocessing para entrenamiento con augmentation."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.transformer = ImageTransformer(self.config.target_size)
        self._rng = np.random.default_rng(self.config.seed)

    def process_image(
        self, image_path: Path, apply_augmentation: bool = True
    ) -> tuple[np.ndarray, list[TransformRecord]]:
        """Procesar una imagen para entrenamiento.

        Args:
            image_path: Ruta a la imagen.
            apply_augmentation: Si aplicar augmentation.

        Returns:
            Tupla de (imagen procesada como array, registro de transformaciones).
        """
        self.transformer.clear_records()

        image = load_image(image_path)
        image = self.transformer.normalize_color(image)

        if apply_augmentation and self.config.augmentation.enabled:
            image = self._apply_augmentation(image)

        image = self.transformer.resize(image)
        arr = self.transformer.to_numpy(image)

        return arr, self.transformer.get_records()

    def process_batch(
        self,
        image_paths: list[Path],
        apply_augmentation: bool = True,
    ) -> tuple[np.ndarray, list[list[TransformRecord]]]:
        """Procesar un lote de imágenes.

        Args:
            image_paths: Lista de rutas a imágenes.
            apply_augmentation: Si aplicar augmentation.

        Returns:
            Tupla de (lote de imágenes, lista de registros).
        """
        batch = []
        all_records = []

        for path in image_paths:
            arr, records = self.process_image(path, apply_augmentation)
            batch.append(arr)
            all_records.append(records)

        return np.stack(batch), all_records

    def _apply_augmentation(self, image: Image.Image) -> Image.Image:
        """Aplicar augmentation aleatoria a la imagen."""
        if self._rng.random() < self.config.augmentation.horizontal_flip_prob:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            self.transformer.records.append(
                TransformRecord("horizontal_flip", {"applied": True})
            )

        if self.config.augmentation.rotation_degrees > 0:
            angle = self._rng.uniform(
                -self.config.augmentation.rotation_degrees,
                self.config.augmentation.rotation_degrees,
            )
            image = image.rotate(
                angle, resample=Image.Resampling.BILINEAR, expand=False
            )
            self.transformer.records.append(
                TransformRecord("rotation", {"degrees": float(angle)})
            )

        if self.config.augmentation.brightness_factor > 0:
            factor = 1.0 + self._rng.uniform(
                -self.config.augmentation.brightness_factor,
                self.config.augmentation.brightness_factor,
            )
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)
            self.transformer.records.append(
                TransformRecord("brightness", {"factor": float(factor)})
            )

        if self.config.augmentation.contrast_factor > 0:
            factor = 1.0 + self._rng.uniform(
                -self.config.augmentation.contrast_factor,
                self.config.augmentation.contrast_factor,
            )
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)
            self.transformer.records.append(
                TransformRecord("contrast", {"factor": float(factor)})
            )

        return image

    def set_seed(self, seed: int) -> None:
        """Establecer semilla para reproducibilidad."""
        self._rng = np.random.default_rng(seed)
