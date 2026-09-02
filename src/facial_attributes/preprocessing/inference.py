"""Pipeline de preprocessing para inferencia."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from facial_attributes.preprocessing.transforms import (
    ImageTransformer,
    TransformRecord,
    load_image,
)


@dataclass
class InferenceConfig:
    """Configuración del pipeline de inferencia."""

    target_size: tuple[int, int] = (224, 224)


class InferencePreprocessor:
    """Pipeline de preprocessing para inferencia optimizado para latencia."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()
        self.transformer = ImageTransformer(self.config.target_size)

    def process_image(
        self, image_path: Path
    ) -> tuple[np.ndarray, list[TransformRecord]]:
        """Procesar una imagen para inferencia.

        Args:
            image_path: Ruta a la imagen.

        Returns:
            Tupla de (imagen procesada como array, registro de transformaciones).
        """
        self.transformer.clear_records()

        image = load_image(image_path)
        image = self.transformer.normalize_color(image)
        image = self.transformer.resize(image)
        arr = self.transformer.to_numpy(image)

        return arr, self.transformer.get_records()

    def process_pil_image(
        self, image: Image.Image
    ) -> tuple[np.ndarray, list[TransformRecord]]:
        """Procesar una imagen PIL ya cargada.

        Args:
            image: Imagen PIL.

        Returns:
            Tupla de (imagen procesada como array, registro de transformaciones).
        """
        self.transformer.clear_records()

        image = self.transformer.normalize_color(image)
        image = self.transformer.resize(image)
        arr = self.transformer.to_numpy(image)

        return arr, self.transformer.get_records()

    def process_batch(
        self, image_paths: list[Path]
    ) -> tuple[np.ndarray, list[list[TransformRecord]]]:
        """Procesar un lote de imágenes para inferencia.

        Args:
            image_paths: Lista de rutas a imágenes.

        Returns:
            Tupla de (lote de imágenes, lista de registros).
        """
        batch = []
        all_records = []

        for path in image_paths:
            arr, records = self.process_image(path)
            batch.append(arr)
            all_records.append(records)

        return np.stack(batch), all_records
