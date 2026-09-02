"""Tests para el módulo de preprocessing."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from facial_attributes.preprocessing.inference import (
    InferenceConfig,
    InferencePreprocessor,
)
from facial_attributes.preprocessing.training import (
    AugmentationConfig,
    TrainingConfig,
    TrainingPreprocessor,
)
from facial_attributes.preprocessing.transforms import ImageTransformer, load_image


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Crear imagen de ejemplo para tests."""
    img = Image.new("RGB", (300, 200), color=(128, 64, 32))
    image_path = tmp_path / "test_image.jpg"
    img.save(image_path)
    return image_path


@pytest.fixture
def sample_rgb_image(tmp_path: Path) -> Path:
    """Crear imagen RGB de ejemplo."""
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    image_path = tmp_path / "rgb_image.jpg"
    img.save(image_path)
    return image_path


@pytest.fixture
def sample_rgba_image(tmp_path: Path) -> Path:
    """Crear imagen RGBA de ejemplo."""
    img = Image.new("RGBA", (224, 224), color=(100, 150, 200, 255))
    image_path = tmp_path / "rgba_image.png"
    img.save(image_path)
    return image_path


class TestImageTransformer:
    """Tests para ImageTransformer."""

    def test_resize(self, sample_image: Path) -> None:
        """Test de redimensionamiento."""
        transformer = ImageTransformer(target_size=(224, 224))
        image = load_image(sample_image)
        resized = transformer.resize(image)

        assert resized.size == (224, 224)

    def test_normalize_color_rgb(self, sample_rgb_image: Path) -> None:
        """Test de normalización de color en imagen RGB."""
        transformer = ImageTransformer()
        image = load_image(sample_rgb_image)
        normalized = transformer.normalize_color(image)

        assert normalized.mode == "RGB"

    def test_normalize_color_rgba(self, sample_rgba_image: Path) -> None:
        """Test de normalización de color en imagen RGBA."""
        transformer = ImageTransformer()
        image = load_image(sample_rgba_image)
        normalized = transformer.normalize_color(image)

        assert normalized.mode == "RGB"

    def test_to_numpy(self, sample_rgb_image: Path) -> None:
        """Test de conversión a numpy."""
        transformer = ImageTransformer()
        image = load_image(sample_rgb_image)
        arr = transformer.to_numpy(image)

        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_records_tracking(self, sample_image: Path) -> None:
        """Test de registro de transformaciones."""
        transformer = ImageTransformer(target_size=(224, 224))
        image = load_image(sample_image)

        transformer.normalize_color(image)
        transformer.resize(image)

        records = transformer.get_records()
        assert len(records) == 2
        assert records[0].name == "normalize_color"
        assert records[1].name == "resize"

    def test_clear_records(self, sample_image: Path) -> None:
        """Test de limpieza de registros."""
        transformer = ImageTransformer()
        image = load_image(sample_image)

        transformer.normalize_color(image)
        assert len(transformer.get_records()) == 1

        transformer.clear_records()
        assert len(transformer.get_records()) == 0


class TestTrainingPreprocessor:
    """Tests para TrainingPreprocessor."""

    def test_process_image_without_augmentation(self, sample_image: Path) -> None:
        """Test de procesamiento sin augmentation."""
        config = TrainingConfig(
            target_size=(224, 224),
            augmentation=AugmentationConfig(enabled=False),
        )
        preprocessor = TrainingPreprocessor(config)

        arr, records = preprocessor.process_image(
            sample_image, apply_augmentation=False
        )

        assert arr.shape == (224, 224, 3)
        assert arr.dtype == np.float32
        assert len(records) > 0

    def test_process_image_with_augmentation(self, sample_image: Path) -> None:
        """Test de procesamiento con augmentation."""
        config = TrainingConfig(
            target_size=(224, 224),
            augmentation=AugmentationConfig(enabled=True),
        )
        preprocessor = TrainingPreprocessor(config)

        arr, records = preprocessor.process_image(sample_image, apply_augmentation=True)

        assert arr.shape == (224, 224, 3)
        assert arr.dtype == np.float32

    def test_process_batch(self, sample_image: Path, tmp_path: Path) -> None:
        """Test de procesamiento de lote."""
        img = Image.new("RGB", (200, 150), color=(50, 100, 150))
        image_path2 = tmp_path / "test_image2.jpg"
        img.save(image_path2)

        config = TrainingConfig(
            target_size=(224, 224),
            augmentation=AugmentationConfig(enabled=False),
        )
        preprocessor = TrainingPreprocessor(config)

        batch, all_records = preprocessor.process_batch(
            [sample_image, image_path2], apply_augmentation=False
        )

        assert batch.shape == (2, 224, 224, 3)
        assert len(all_records) == 2

    def test_set_seed_reproducibility(self, sample_image: Path) -> None:
        """Test de reproducibilidad con semilla."""
        config = TrainingConfig(
            target_size=(224, 224),
            augmentation=AugmentationConfig(enabled=True),
        )

        preprocessor1 = TrainingPreprocessor(config)
        preprocessor1.set_seed(42)
        arr1, _ = preprocessor1.process_image(sample_image, apply_augmentation=True)

        preprocessor2 = TrainingPreprocessor(config)
        preprocessor2.set_seed(42)
        arr2, _ = preprocessor2.process_image(sample_image, apply_augmentation=True)

        np.testing.assert_array_equal(arr1, arr2)


class TestInferencePreprocessor:
    """Tests para InferencePreprocessor."""

    def test_process_image(self, sample_image: Path) -> None:
        """Test de procesamiento para inferencia."""
        config = InferenceConfig(target_size=(224, 224))
        preprocessor = InferencePreprocessor(config)

        arr, records = preprocessor.process_image(sample_image)

        assert arr.shape == (224, 224, 3)
        assert arr.dtype == np.float32
        assert len(records) > 0

    def test_process_pil_image(self, sample_rgb_image: Path) -> None:
        """Test de procesamiento de imagen PIL."""
        config = InferenceConfig(target_size=(224, 224))
        preprocessor = InferencePreprocessor(config)

        image = load_image(sample_rgb_image)
        arr, records = preprocessor.process_pil_image(image)

        assert arr.shape == (224, 224, 3)
        assert arr.dtype == np.float32

    def test_process_batch(self, sample_image: Path, tmp_path: Path) -> None:
        """Test de procesamiento de lote para inferencia."""
        img = Image.new("RGB", (200, 150), color=(50, 100, 150))
        image_path2 = tmp_path / "test_image2.jpg"
        img.save(image_path2)

        config = InferenceConfig(target_size=(224, 224))
        preprocessor = InferencePreprocessor(config)

        batch, all_records = preprocessor.process_batch([sample_image, image_path2])

        assert batch.shape == (2, 224, 224, 3)
        assert len(all_records) == 2

    def test_no_augmentation_applied(self, sample_image: Path) -> None:
        """Test de que no se aplica augmentation en inferencia."""
        config = InferenceConfig(target_size=(224, 224))
        preprocessor = InferencePreprocessor(config)

        _, records = preprocessor.process_image(sample_image)

        augmentation_transforms = [
            "horizontal_flip",
            "rotation",
            "brightness",
            "contrast",
        ]
        for record in records:
            assert record.name not in augmentation_transforms


class TestPreprocessingReproducibility:
    """Tests de reproducibilidad entre pipelines."""

    def test_same_output_without_augmentation(self, sample_image: Path) -> None:
        """Test de que ambos pipelines producen el mismo resultado sin augmentation."""
        training_config = TrainingConfig(
            target_size=(224, 224),
            augmentation=AugmentationConfig(enabled=False),
        )
        inference_config = InferenceConfig(target_size=(224, 224))

        training_preprocessor = TrainingPreprocessor(training_config)
        inference_preprocessor = InferencePreprocessor(inference_config)

        train_arr, _ = training_preprocessor.process_image(
            sample_image, apply_augmentation=False
        )
        inference_arr, _ = inference_preprocessor.process_image(sample_image)

        np.testing.assert_array_almost_equal(train_arr, inference_arr, decimal=5)
