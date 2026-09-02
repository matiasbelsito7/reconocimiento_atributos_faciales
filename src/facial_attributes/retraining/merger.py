"""Combinación controlada de datasets."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class MergeResult:
    """Resultado de la combinación de datasets."""

    success: bool
    total_samples: int = 0
    samples_from_existing: int = 0
    samples_from_new: int = 0
    merged_path: str = ""
    error: str | None = None


class DatasetMerger:
    """Combinador controlado de datasets."""

    def __init__(self, output_dir: str | Path = "data/processed") -> None:
        """Inicializar combinador de datasets.

        Args:
            output_dir: Directorio de salida para datasets combinados.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def merge_datasets(
        self,
        existing_annotations_path: str | Path,
        new_annotations_path: str | Path,
        existing_images_dir: str | Path,
        new_images_dir: str | Path,
        output_name: str = "merged",
        validate_schema: bool = True,
    ) -> MergeResult:
        """Combinar datasets existentes con nuevos datos.

        Args:
            existing_annotations_path: Ruta a anotaciones existentes.
            new_annotations_path: Ruta a nuevas anotaciones.
            existing_images_dir: Directorio de imágenes existentes.
            new_images_dir: Directorio de nuevas imágenes.
            output_name: Nombre del dataset combinado.
            validate_schema: Si validar que los schemas coinciden.

        Returns:
            Resultado de la combinación.
        """
        try:
            existing_df = pd.read_csv(existing_annotations_path)
            new_df = pd.read_csv(new_annotations_path)

            if validate_schema:
                schema_valid = self._validate_schema(existing_df, new_df)
                if not schema_valid:
                    return MergeResult(
                        success=False,
                        error="Schema mismatch between existing and new datasets",
                    )

            existing_columns = set(existing_df.columns)
            new_columns = set(new_df.columns)

            if existing_columns != new_columns:
                common_columns = existing_columns.intersection(new_columns)
                existing_df = existing_df[list(common_columns)]
                new_df = new_df[list(common_columns)]

            merged_df = pd.concat([existing_df, new_df], ignore_index=True)
            merged_df = merged_df.drop_duplicates()

            output_path = self.output_dir / f"{output_name}_annotations.csv"
            merged_df.to_csv(output_path, index=False)

            self._copy_new_images(new_images_dir, existing_images_dir)

            return MergeResult(
                success=True,
                total_samples=len(merged_df),
                samples_from_existing=len(existing_df),
                samples_from_new=len(new_df),
                merged_path=str(output_path),
            )

        except Exception as e:
            return MergeResult(
                success=False,
                error=f"Merge failed: {str(e)}",
            )

    def _validate_schema(self, existing_df: pd.DataFrame, new_df: pd.DataFrame) -> bool:
        """Validar que los schemas coincidan.

        Args:
            existing_df: DataFrame existente.
            new_df: DataFrame nuevo.

        Returns:
            True si los schemas son compatibles.
        """
        existing_cols = set(existing_df.columns)
        new_cols = set(new_df.columns)

        if not new_cols.issubset(existing_cols):
            missing = new_cols - existing_cols
            print(f"Warning: New dataset has columns not in existing: {missing}")

        return True

    def _copy_new_images(
        self, new_images_dir: str | Path, target_dir: str | Path
    ) -> None:
        """Copiar nuevas imágenes al directorio objetivo.

        Args:
            new_images_dir: Directorio de nuevas imágenes.
            target_dir: Directorio objetivo.
        """
        new_images_dir = Path(new_images_dir)
        target_dir = Path(target_dir)

        if not new_images_dir.exists():
            return

        for image_file in new_images_dir.glob("*"):
            if image_file.is_file():
                target_path = target_dir / image_file.name
                if not target_path.exists():
                    import shutil

                    shutil.copy2(image_file, target_path)

    def validate_new_data(
        self,
        annotations_path: str | Path,
        images_dir: str | Path,
        required_columns: list[str] | None = None,
    ) -> dict[str, object]:
        """Validar nuevos datos antes de la combinación.

        Args:
            annotations_path: Ruta a anotaciones.
            images_dir: Directorio de imágenes.
            required_columns: Columnas requeridas.

        Returns:
            Diccionario con resultado de validación.
        """
        issues = []
        warnings = []

        annotations_path = Path(annotations_path)
        images_dir = Path(images_dir)

        if not annotations_path.exists():
            issues.append(f"Annotations file not found: {annotations_path}")
            return {"valid": False, "issues": issues, "warnings": warnings}

        if not images_dir.exists():
            issues.append(f"Images directory not found: {images_dir}")
            return {"valid": False, "issues": issues, "warnings": warnings}

        try:
            df = pd.read_csv(annotations_path)
        except Exception as e:
            issues.append(f"Failed to read annotations: {str(e)}")
            return {"valid": False, "issues": issues, "warnings": warnings}

        if len(df) == 0:
            issues.append("Annotations file is empty")
            return {"valid": False, "issues": issues, "warnings": warnings}

        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                issues.append(f"Missing required columns: {missing_cols}")

        image_files = list(images_dir.glob("*.{jpg,jpeg,png}"))
        if len(image_files) == 0:
            warnings.append("No images found in directory")

        if "image_id" in df.columns:
            missing_images = []
            for img_id in df["image_id"]:
                if not any(img_id in f.name for f in image_files):
                    missing_images.append(img_id)
            if missing_images:
                warnings.append(
                    f"{len(missing_images)} images referenced but not found"
                )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "num_samples": len(df),
            "num_images": len(image_files),
        }
