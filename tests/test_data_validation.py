"""Tests para el módulo de validación de datos."""

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from facial_attributes.data.validation import DataValidator


@pytest.fixture
def sample_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Crear dataset de ejemplo para tests."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    annotations_file = tmp_path / "annotations.csv"

    # Crear imágenes de ejemplo
    for i in range(5):
        img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
        img.save(images_dir / f"img_{i:04d}.jpg")

    # Crear anotaciones de ejemplo
    df = pd.DataFrame(
        {
            "image_id": [f"img_{i:04d}" for i in range(5)],
            "Atr_smiling": [1, 0, 1, 0, 1],
            "Atr_eyeglasses": [0, 1, 0, 1, 0],
        }
    )
    df.to_csv(annotations_file, index=False)

    return images_dir, annotations_file


def test_validate_valid_dataset(sample_dataset: tuple[Path, Path]) -> None:
    """Test de validación con dataset válido."""
    images_dir, annotations_file = sample_dataset
    validator = DataValidator(images_dir, annotations_file)
    report = validator.validate()

    assert report["status"] == "PASS"
    assert len(report["errors"]) == 0


def test_validate_missing_annotations(tmp_path: Path) -> None:
    """Test de validación con anotaciones faltantes."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    annotations_file = tmp_path / "nonexistent.csv"

    validator = DataValidator(images_dir, annotations_file)
    report = validator.validate()

    assert report["status"] == "FAIL"
    assert any("no encontrado" in error for error in report["errors"])


def test_validate_missing_images(tmp_path: Path) -> None:
    """Test de validación con imágenes faltantes."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    annotations_file = tmp_path / "annotations.csv"
    df = pd.DataFrame(
        {
            "image_id": ["img_0001", "img_0002"],
            "Atr_smiling": [1, 0],
        }
    )
    df.to_csv(annotations_file, index=False)

    validator = DataValidator(images_dir, annotations_file)
    report = validator.validate()

    assert report["status"] == "FAIL"
    assert any("imágenes no encontradas" in error for error in report["errors"])


def test_validate_unbalanced_attributes(tmp_path: Path) -> None:
    """Test de validación con atributos imbalanceados."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Crear imágenes
    for i in range(100):
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        img.save(images_dir / f"img_{i:04d}.jpg")

    annotations_file = tmp_path / "annotations.csv"
    df = pd.DataFrame(
        {
            "image_id": [f"img_{i:04d}" for i in range(100)],
            "Atr_smiling": [1] * 95 + [0] * 5,  # Muy imbalanceado
            "Atr_eyeglasses": [0] * 50 + [1] * 50,  # Balanceado
        }
    )
    df.to_csv(annotations_file, index=False)

    validator = DataValidator(images_dir, annotations_file)
    report = validator.validate()

    assert report["status"] == "PASS"
    assert len(report["warnings"]) > 0
