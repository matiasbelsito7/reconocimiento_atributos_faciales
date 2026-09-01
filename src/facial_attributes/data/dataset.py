"""Gestión de datasets para reconocimiento de atributos faciales."""

from pathlib import Path
from typing import Any

import pandas as pd


class DatasetManager:
    """Gestor de datasets con soporte para múltiples fuentes."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        self.images_dir = self.raw_dir / "images"
        self.annotations_dir = self.raw_dir / "annotations"

    def load_annotations(self, dataset_name: str = "celeba") -> pd.DataFrame:
        """Cargar anotaciones de un dataset."""
        annotations_file = self.annotations_dir / f"{dataset_name}_attributes.csv"

        if not annotations_file.exists():
            raise FileNotFoundError(f"Anotaciones no encontradas: {annotations_file}")

        return pd.read_csv(annotations_file)

    def get_attribute_columns(self, df: pd.DataFrame) -> list[str]:
        """Obtener columnas de atributos del DataFrame."""
        return [col for col in df.columns if col.startswith("Atr_")]

    def filter_observable_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtrar solo atributos visualmente observables."""
        observable_attrs = [
            "Atr_eyeglasses",
            "Atr_hat",
            "Atr_earrings",
            "Atr_necklace",
            "Atr_necktie",
            "Atr_lipstick",
            "Atr_smiling",
            "Atr_mouth_slightly_open",
            "Atr_goatee",
            "Atr_mustache",
            "Atr_no_beard",
            "Atr_sideburns",
            "Atr_bangs",
            "Atr_receding_hairline",
            "Atr_straight_hair",
            "Atr_wavy_hair",
            "Atr_bald",
            "Atr_black_hair",
            "Atr_blond_hair",
            "Atr_brown_hair",
            "Atr_gray_hair",
            "Atr_5_o_clock_shadow",
            "Atr_blurry",
            "Atr_heavy_makeup",
        ]

        available_attrs = [attr for attr in observable_attrs if attr in df.columns]
        return df[["image_id"] + available_attrs]

    def split_dataset(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> dict[str, pd.DataFrame]:
        """Dividir dataset en train/val/test."""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Los ratios deben sumar 1.0")

        shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        return {
            "train": shuffled[:n_train],
            "val": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        }

    def save_split(
        self, splits: dict[str, pd.DataFrame], output_dir: Path | None = None
    ) -> None:
        """Guardar splits en archivos CSV."""
        if output_dir is None:
            output_dir = self.processed_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, df in splits.items():
            output_file = output_dir / f"{split_name}_attributes.csv"
            df.to_csv(output_file, index=False)

    def get_dataset_info(self, dataset_name: str = "celeba") -> dict[str, Any]:
        """Obtener información del dataset."""
        df = self.load_annotations(dataset_name)
        attribute_cols = self.get_attribute_columns(df)

        return {
            "name": dataset_name,
            "total_images": len(df),
            "total_attributes": len(attribute_cols),
            "attribute_columns": attribute_cols,
        }
