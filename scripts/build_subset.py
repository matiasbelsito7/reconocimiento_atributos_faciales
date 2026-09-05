"""Preparar subconjunto representativo de CelebA para entrenamiento en CPU."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_subset(
    annotations_file: Path,
    output_file: Path,
    sample_size: int = 5000,
    stratify_col: str = "Atr_Male",
    random_state: int = 42,
) -> None:
    """Crear subconjunto estratificado por un atributo.

    Args:
        annotations_file: Ruta al CSV completo de anotaciones.
        output_file: Ruta del CSV de salida.
        sample_size: Tamaño del subconjunto.
        stratify_col: Columna para estratificar.
        random_state: Semilla para reproducibilidad.
    """
    df = pd.read_csv(annotations_file)

    if stratify_col not in df.columns:
        raise ValueError(f"Columna de estratificación no encontrada: {stratify_col}")

    class_counts = df[stratify_col].value_counts()
    n_classes = len(class_counts)
    per_class = max(1, sample_size // n_classes)

    samples = []
    for cls, count in class_counts.items():
        cls_df = df[df[stratify_col] == cls]
        n_sample = min(per_class, len(cls_df))
        samples.append(
            cls_df.sample(n=n_sample, random_state=random_state)
        )

    subset = pd.concat(samples).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_file, index=False)

    print(f"Subconjunto creado: {len(subset)} filas")
    print(f"Guarado en: {output_file}")
    print(f"Distribución de {stratify_col}:")
    print(subset[stratify_col].value_counts())


if __name__ == "__main__":
    build_subset(
        annotations_file=Path("data/raw/annotations/celeba_attributes.csv"),
        output_file=Path("data/processed/celeba_subset_5000.csv"),
        sample_size=5000,
        stratify_col="Atr_Male",
    )
