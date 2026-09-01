"""Tests para el módulo de gestión de datasets."""

from pathlib import Path

import pandas as pd
import pytest

from facial_attributes.data.dataset import DatasetManager


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Crear directorio de datos de ejemplo."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    images_dir = raw_dir / "images"
    annotations_dir = raw_dir / "annotations"

    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    # Crear anotaciones de ejemplo
    df = pd.DataFrame(
        {
            "image_id": [f"img_{i:04d}" for i in range(10)],
            "Atr_smiling": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "Atr_eyeglasses": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "Atr_hat": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        }
    )
    df.to_csv(annotations_dir / "celeba_attributes.csv", index=False)

    return data_dir


def test_load_annotations(sample_data_dir: Path) -> None:
    """Test de carga de anotaciones."""
    manager = DatasetManager(sample_data_dir)
    df = manager.load_annotations()

    assert len(df) == 10
    assert "image_id" in df.columns
    assert "Atr_smiling" in df.columns


def test_get_attribute_columns(sample_data_dir: Path) -> None:
    """Test de obtención de columnas de atributos."""
    manager = DatasetManager(sample_data_dir)
    df = manager.load_annotations()
    attrs = manager.get_attribute_columns(df)

    assert len(attrs) == 3
    assert all(attr.startswith("Atr_") for attr in attrs)


def test_filter_observable_attributes(sample_data_dir: Path) -> None:
    """Test de filtrado de atributos observables."""
    manager = DatasetManager(sample_data_dir)
    df = manager.load_annotations()
    filtered = manager.filter_observable_attributes(df)

    assert "image_id" in filtered.columns
    assert "Atr_smiling" in filtered.columns
    assert "Atr_hat" in filtered.columns


def test_split_dataset(sample_data_dir: Path) -> None:
    """Test de división del dataset."""
    manager = DatasetManager(sample_data_dir)
    df = manager.load_annotations()
    splits = manager.split_dataset(df)

    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(df)


def test_split_dataset_invalid_ratios(sample_data_dir: Path) -> None:
    """Test de división con ratios inválidos."""
    manager = DatasetManager(sample_data_dir)
    df = manager.load_annotations()

    with pytest.raises(ValueError):
        manager.split_dataset(df, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)


def test_save_split(sample_data_dir: Path, tmp_path: Path) -> None:
    """Test de guardado de splits."""
    manager = DatasetManager(sample_data_dir)
    df = manager.load_annotations()
    splits = manager.split_dataset(df)

    output_dir = tmp_path / "splits"
    manager.save_split(splits, output_dir)

    assert (output_dir / "train_attributes.csv").exists()
    assert (output_dir / "val_attributes.csv").exists()
    assert (output_dir / "test_attributes.csv").exists()


def test_get_dataset_info(sample_data_dir: Path) -> None:
    """Test de obtención de información del dataset."""
    manager = DatasetManager(sample_data_dir)
    info = manager.get_dataset_info()

    assert info["name"] == "celeba"
    assert info["total_images"] == 10
    assert info["total_attributes"] == 3
