"""Tests para el módulo de inferencia."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image, ImageDraw

from facial_attributes.inference.pipeline import (
    FacePrediction,
    InferenceConfig,
    InferencePipeline,
    InferenceResult,
)


@pytest.fixture
def sample_image_with_face(tmp_path: Path) -> Path:
    """Crear imagen con un rostro simple para tests."""
    img = Image.new("RGB", (300, 300), color=(200, 180, 160))
    draw = ImageDraw.Draw(img)

    draw.ellipse([100, 80, 200, 220], fill=(180, 150, 130))
    draw.ellipse([130, 120, 150, 140], fill=(50, 50, 50))
    draw.ellipse([160, 120, 180, 140], fill=(50, 50, 50))
    draw.arc([135, 160, 175, 190], 0, 180, fill=(150, 100, 100), width=2)

    image_path = tmp_path / "face_image.jpg"
    img.save(image_path)
    return image_path


@pytest.fixture
def sample_image_without_face(tmp_path: Path) -> Path:
    """Crear imagen sin rostro."""
    img = Image.new("RGB", (300, 300), color=(100, 150, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 250, 250], fill=(150, 200, 100))

    image_path = tmp_path / "no_face_image.jpg"
    img.save(image_path)
    return image_path


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Crear imagen PIL de ejemplo."""
    img = Image.new("RGB", (300, 300), color=(200, 180, 160))
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 80, 200, 220], fill=(180, 150, 130))
    return img


class TestInferenceConfig:
    """Tests para InferenceConfig."""

    def test_default_config(self) -> None:
        """Test de configuración por defecto."""
        config = InferenceConfig()

        assert config.threshold == 0.5
        assert config.device == "cpu" or config.device == "auto"
        assert config.num_attributes == 40

    def test_custom_config(self) -> None:
        """Test de configuración personalizada."""
        config = InferenceConfig(
            threshold=0.7,
            device="cpu",
            num_attributes=20,
            backbone="resnet34",
        )

        assert config.threshold == 0.7
        assert config.device == "cpu"
        assert config.num_attributes == 20
        assert config.backbone == "resnet34"


class TestFacePrediction:
    """Tests para FacePrediction."""

    def test_face_prediction_creation(self) -> None:
        """Test de creación de FacePrediction."""
        prediction = FacePrediction(
            bbox={"x": 10, "y": 20, "w": 100, "h": 150},
            attributes={"smiling": 0.92, "glasses": 0.15},
            confidence=0.95,
        )

        assert prediction.bbox["x"] == 10
        assert prediction.attributes["smiling"] == 0.92
        assert prediction.confidence == 0.95


class TestInferenceResult:
    """Tests para InferenceResult."""

    def test_inference_result_creation(self) -> None:
        """Test de creación de InferenceResult."""
        result = InferenceResult(
            faces=[],
            num_faces_detected=0,
            num_faces_with_predictions=0,
            inference_time_ms=10.5,
            image_size=(300, 300),
        )

        assert result.num_faces_detected == 0
        assert result.inference_time_ms == 10.5

    def test_inference_result_with_error(self) -> None:
        """Test de creación de InferenceResult con error."""
        result = InferenceResult(
            faces=[],
            num_faces_detected=0,
            num_faces_with_predictions=0,
            inference_time_ms=5.0,
            image_size=(300, 300),
            error="No se detectaron rostros",
        )

        assert result.error == "No se detectaron rostros"


class TestInferencePipeline:
    """Tests para InferencePipeline."""

    def test_pipeline_initialization(self) -> None:
        """Test de inicialización del pipeline."""
        config = InferenceConfig(device="cpu")
        pipeline = InferencePipeline(config)

        assert pipeline is not None

    def test_pipeline_without_model(self, sample_pil_image: Image.Image) -> None:
        """Test de pipeline sin modelo cargado."""
        config = InferenceConfig(device="cpu")
        pipeline = InferencePipeline(config)

        result = pipeline.predict(sample_pil_image)

        assert isinstance(result, InferenceResult)
        assert result.error is not None
        assert "Modelo no cargado" in result.error

    def test_predict_from_path(self, sample_image_with_face: Path) -> None:
        """Test de predicción desde ruta."""
        config = InferenceConfig(device="cpu")
        pipeline = InferencePipeline(config)

        result = pipeline.predict_from_path(sample_image_with_face)

        assert isinstance(result, InferenceResult)
        assert result.image_size == (300, 300)

    def test_predict_batch(self, sample_pil_image: Image.Image) -> None:
        """Test de predicción en lote."""
        config = InferenceConfig(device="cpu")
        pipeline = InferencePipeline(config)

        images = [sample_pil_image, sample_pil_image]
        results = pipeline.predict_batch(images)

        assert len(results) == 2
        assert all(isinstance(r, InferenceResult) for r in results)

    def test_pipeline_with_mock_model(self, sample_pil_image: Image.Image) -> None:
        """Test de pipeline con modelo mockeado."""
        config = InferenceConfig(device="cpu")
        pipeline = InferencePipeline(config)

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = torch.tensor([[0.9, 0.1, 0.8]])
        pipeline._model = mock_model

        with patch.object(pipeline._face_detector, "detect") as mock_detect:
            mock_detect.return_value = MagicMock(
                faces=[MagicMock(x=100, y=80, width=100, height=140, confidence=0.95)],
                image_size=(300, 300),
                num_faces=1,
            )

            with patch.object(
                pipeline._face_extractor, "extract_faces"
            ) as mock_extract:
                mock_face = MagicMock()
                mock_face.bounding_box = MagicMock(
                    x=100, y=80, width=100, height=140, confidence=0.95
                )
                mock_face.image = Image.new("RGB", (224, 224), color=(128, 128, 128))
                mock_extract.return_value = [mock_face]

                result = pipeline.predict(sample_pil_image)

                assert result.error is None
                assert result.num_faces_detected == 1


class TestInferenceIntegration:
    """Tests de integración del pipeline de inferencia."""

    def test_full_pipeline_without_model(self, sample_pil_image: Image.Image) -> None:
        """Test del pipeline completo sin modelo."""
        config = InferenceConfig(device="cpu")
        pipeline = InferencePipeline(config)

        result = pipeline.predict(sample_pil_image)

        assert result.error is not None
        assert result.inference_time_ms >= 0

    def test_result_structure(self, sample_pil_image: Image.Image) -> None:
        """Test de la estructura del resultado."""
        config = InferenceConfig(device="cpu")
        pipeline = InferencePipeline(config)

        result = pipeline.predict(sample_pil_image)

        assert hasattr(result, "faces")
        assert hasattr(result, "num_faces_detected")
        assert hasattr(result, "num_faces_with_predictions")
        assert hasattr(result, "inference_time_ms")
        assert hasattr(result, "image_size")
        assert hasattr(result, "error")
