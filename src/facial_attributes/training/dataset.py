"""Dataset para entrenamiento de atributos faciales."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FacialAttributeDataset(Dataset):
    """Dataset de atributos faciales para PyTorch."""

    def __init__(
        self,
        annotations_file: Path,
        images_dir: Path,
        attribute_columns: list[str] | None = None,
        transform=None,
    ) -> None:
        """Inicializar dataset.

        Args:
            annotations_file: Ruta al CSV de anotaciones.
            images_dir: Directorio de imágenes.
            attribute_columns: Columnas de atributos a utilizar.
            transform: Transformaciones a aplicar.
        """
        self.df = pd.read_csv(annotations_file)
        self.images_dir = images_dir
        self.transform = transform

        if attribute_columns is None:
            self.attribute_columns = [
                col for col in self.df.columns if col.startswith("Atr_")
            ]
        else:
            self.attribute_columns = attribute_columns

        if "image_id" not in self.df.columns:
            raise ValueError("El CSV debe tener una columna 'image_id'")

    def __len__(self) -> int:
        """Longitud del dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Obtener elemento del dataset.

        Args:
            idx: Índice del elemento.

        Returns:
            Tupla de (imagen, atributos).
        """
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        image_path = self.images_dir / f"{image_id:06d}.jpg"
        if not image_path.exists():
            image_path = self.images_dir / f"{image_id}.jpg"

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        attributes = torch.tensor(
            row[self.attribute_columns].values.astype(np.float32),
            dtype=torch.float32,
        )

        return image, attributes

    def get_attribute_columns(self) -> list[str]:
        """Obtener columnas de atributos."""
        return self.attribute_columns

    def get_num_attributes(self) -> int:
        """Obtener número de atributos."""
        return len(self.attribute_columns)

    def get_class_weights(self) -> torch.Tensor:
        """Calcular pesos de clase para manejo de desbalance.

        Returns:
            Tensor con pesos por atributo.
        """
        weights = []
        for col in self.attribute_columns:
            pos_count = self.df[col].sum()
            total = len(self.df)
            neg_count = total - pos_count
            weight = neg_count / (pos_count + 1e-6)
            weights.append(min(weight, 50.0))
        return torch.tensor(weights, dtype=torch.float32)
