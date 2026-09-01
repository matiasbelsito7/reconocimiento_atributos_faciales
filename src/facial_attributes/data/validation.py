"""Validación de datasets de imágenes y anotaciones."""

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


class DataValidator:
    """Validador de integridad y formato de datasets."""

    def __init__(self, images_dir: Path, annotations_file: Path) -> None:
        self.images_dir = images_dir
        self.annotations_file = annotations_file
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate(self) -> dict[str, Any]:
        """Ejecutar todas las validaciones y retornar reporte."""
        self.errors = []
        self.warnings = []

        self._validate_annotations_format()
        self._validate_images_exist()
        self._validate_images_readable()
        self._validate_no_duplicates()
        self._validate_attribute_distribution()

        return {
            "status": "PASS" if not self.errors else "FAIL",
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self._get_stats(),
        }

    def _validate_annotations_format(self) -> None:
        """Verificar que el CSV de anotaciones tenga el formato correcto."""
        if not self.annotations_file.exists():
            self.errors.append(
                f"Archivo de anotaciones no encontrado: {self.annotations_file}"
            )
            return

        try:
            df = pd.read_csv(self.annotations_file)
        except Exception as e:
            self.errors.append(f"Error al leer CSV: {e}")
            return

        if "image_id" not in df.columns:
            self.errors.append("Columna 'image_id' no encontrada en el CSV")

        attribute_cols = [col for col in df.columns if col.startswith("Atr_")]
        if not attribute_cols:
            self.errors.append(
                "No se encontraron columnas de atributos (prefijo 'Atr_')"
            )

        for col in attribute_cols:
            unique_vals = df[col].dropna().unique()
            if not set(unique_vals).issubset({0, 1}):
                self.warnings.append(f"Columna {col} contiene valores diferentes a 0/1")

    def _validate_images_exist(self) -> None:
        """Verificar que todas las imágenes referenciadas existan."""
        if not self.annotations_file.exists():
            return

        df = pd.read_csv(self.annotations_file)
        missing = []

        for image_id in df["image_id"]:
            image_path = self.images_dir / f"{image_id}.jpg"
            if not image_path.exists():
                missing.append(image_id)

        if missing:
            self.errors.append(f"{len(missing)} imágenes no encontradas")

    def _validate_images_readable(self) -> None:
        """Verificar que las imágenes sean legibles."""
        if not self.annotations_file.exists():
            return

        df = pd.read_csv(self.annotations_file)
        unreadable = []

        for image_id in df["image_id"].head(100):  # Muestrear primeras 100
            image_path = self.images_dir / f"{image_id}.jpg"
            if image_path.exists():
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                except Exception:
                    unreadable.append(image_id)

        if unreadable:
            self.errors.append(
                f"{len(unreadable)} imágenes ilegibles (de 100 muestreadas)"
            )

    def _validate_no_duplicates(self) -> None:
        """Detectar imágenes duplicadas."""
        if not self.annotations_file.exists():
            return

        df = pd.read_csv(self.annotations_file)
        duplicates = df[df["image_id"].duplicated()]

        if not duplicates.empty:
            self.warnings.append(f"{len(duplicates)} entradas duplicadas encontradas")

    def _validate_attribute_distribution(self) -> None:
        """Reportar distribución de atributos para identificar desbalance."""
        if not self.annotations_file.exists():
            return

        df = pd.read_csv(self.annotations_file)
        attribute_cols = [col for col in df.columns if col.startswith("Atr_")]

        for col in attribute_cols:
            dist = df[col].value_counts(normalize=True)
            if len(dist) > 0:
                minority_ratio = dist.min()
                if minority_ratio < 0.1:
                    self.warnings.append(
                        f"Atributo {col} imbalanceado: {minority_ratio:.1%} en clase minoritaria"
                    )

    def _get_stats(self) -> dict[str, Any]:
        """Obtener estadísticas generales del dataset."""
        if not self.annotations_file.exists():
            return {}

        df = pd.read_csv(self.annotations_file)
        attribute_cols = [col for col in df.columns if col.startswith("Atr_")]

        return {
            "total_images": len(df),
            "total_attributes": len(attribute_cols),
            "attribute_columns": attribute_cols,
        }
