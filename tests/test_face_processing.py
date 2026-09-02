"""Tests para el módulo de face processing."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image, ImageDraw

from facial_attributes.face_processing.detector import (
    BoundingBox,
    DetectionResult,
    DetectorConfig,
    FaceDetector,
)
from facial_attributes.face_processing.extractor import (
    ExtractedFace,
    ExtractorConfig,
    FaceExtractor,
)
from facial_attributes.face_processing.normalizer import (
    FaceNormalizer,
    NormalizerConfig,
)
from facial_attributes.face_processing.pipeline import (
    FaceProcessingPipeline,
    PipelineConfig,
    PipelineResult,
    ProcessedFace,
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


@pytest.fixture
def mock_detector() -> MagicMock:
    """Mock del detector de rostros."""
    detector = MagicMock(spec=FaceDetector)
    detector.detect.return_value = DetectionResult(
        faces=[BoundingBox(x=100, y=80, width=100, height=140, confidence=0.95)],
        image_size=(300, 300),
        num_faces=1,
    )
    return detector


class TestBoundingBox:
    """Tests para BoundingBox."""

    def test_bounding_box_creation(self) -> None:
        """Test de creación de BoundingBox."""
        bbox = BoundingBox(x=10, y=20, width=100, height=150, confidence=0.95)

        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 150
        assert bbox.confidence == 0.95


class TestDetectionResult:
    """Tests para DetectionResult."""

    def test_detection_result_creation(self) -> None:
        """Test de creación de DetectionResult."""
        faces = [BoundingBox(x=10, y=20, width=100, height=150, confidence=0.95)]
        result = DetectionResult(faces=faces, image_size=(300, 300), num_faces=1)

        assert result.num_faces == 1
        assert result.image_size == (300, 300)


class TestFaceDetector:
    """Tests para FaceDetector."""

    def test_detector_initialization(self) -> None:
        """Test de inicialización del detector."""
        config = DetectorConfig(confidence_threshold=0.6, nms_threshold=0.4)
        detector = FaceDetector(config)

        assert detector is not None

    def test_detector_default_config(self) -> None:
        """Test de configuración por defecto."""
        detector = FaceDetector()

        assert detector.config.confidence_threshold == 0.7
        assert detector.config.nms_threshold == 0.3

    def test_detector_model_not_available(self) -> None:
        """Test de detector cuando modelo no está disponible."""
        config = DetectorConfig(model_dir="/nonexistent/path")
        detector = FaceDetector(config)

        assert detector._check_model_available() is False

    def test_detector_raises_without_model(self, sample_pil_image: Image.Image) -> None:
        """Test de que detector lanza error sin modelo."""
        config = DetectorConfig(model_dir="/nonexistent/path")
        detector = FaceDetector(config)

        with pytest.raises(RuntimeError, match="Modelo de detección no disponible"):
            detector.detect(sample_pil_image)


class TestFaceExtractor:
    """Tests para FaceExtractor."""

    def test_extractor_initialization(self) -> None:
        """Test de inicialización del extractor."""
        config = ExtractorConfig(margin_percent=0.3, min_face_size=40)
        extractor = FaceExtractor(config)

        assert extractor.config.margin_percent == 0.3
        assert extractor.config.min_face_size == 40

    def test_extract_faces_empty_detection(self, sample_pil_image: Image.Image) -> None:
        """Test de extracción con detección vacía."""
        extractor = FaceExtractor()
        detection = DetectionResult(faces=[], image_size=(300, 300), num_faces=0)

        faces = extractor.extract_faces(sample_pil_image, detection)

        assert len(faces) == 0

    def test_extract_faces_with_detection(self, sample_pil_image: Image.Image) -> None:
        """Test de extracción con detección."""
        extractor = FaceExtractor()
        face = BoundingBox(x=50, y=50, width=200, height=200, confidence=0.95)
        detection = DetectionResult(faces=[face], image_size=(300, 300), num_faces=1)

        faces = extractor.extract_faces(sample_pil_image, detection)

        assert len(faces) == 1
        assert isinstance(faces[0], ExtractedFace)

    def test_extract_largest_face(self, sample_pil_image: Image.Image) -> None:
        """Test de extracción del rostro más grande."""
        extractor = FaceExtractor()
        face1 = BoundingBox(x=50, y=50, width=100, height=100, confidence=0.9)
        face2 = BoundingBox(x=150, y=150, width=150, height=150, confidence=0.95)
        detection = DetectionResult(
            faces=[face1, face2], image_size=(300, 300), num_faces=2
        )

        largest = extractor.extract_largest_face(sample_pil_image, detection)

        assert largest is not None
        assert largest.bounding_box.width == 150

    def test_extract_largest_face_empty(self, sample_pil_image: Image.Image) -> None:
        """Test de extracción del rostro más grande sin rostros."""
        extractor = FaceExtractor()
        detection = DetectionResult(faces=[], image_size=(300, 300), num_faces=0)

        largest = extractor.extract_largest_face(sample_pil_image, detection)

        assert largest is None


class TestFaceNormalizer:
    """Tests para FaceNormalizer."""

    def test_normalizer_initialization(self) -> None:
        """Test de inicialización del normalizador."""
        config = NormalizerConfig(target_size=(160, 160), normalize_pixels=True)
        normalizer = FaceNormalizer(config)

        assert normalizer.config.target_size == (160, 160)

    def test_normalize_face(self) -> None:
        """Test de normalización de rostro."""
        normalizer = FaceNormalizer()
        face = Image.new("RGB", (200, 200), color=(128, 128, 128))

        normalized = normalizer.normalize(face)

        assert normalized.shape == (224, 224, 3)
        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_normalize_batch(self) -> None:
        """Test de normalización de lote."""
        normalizer = FaceNormalizer()
        faces = [Image.new("RGB", (200, 200), color=(128, 128, 128)) for _ in range(3)]

        batch = normalizer.normalize_batch(faces)

        assert batch.shape == (3, 224, 224, 3)

    def test_get_output_shape(self) -> None:
        """Test de obtención de forma de salida."""
        normalizer = FaceNormalizer()
        shape = normalizer.get_output_shape()

        assert shape == (224, 224, 3)


class TestFaceProcessingPipeline:
    """Tests para FaceProcessingPipeline."""

    def test_pipeline_initialization(self) -> None:
        """Test de inicialización del pipeline."""
        with patch.object(FaceDetector, "_load_model", return_value=None):
            pipeline = FaceProcessingPipeline()

            assert pipeline is not None

    def test_pipeline_with_custom_config(self) -> None:
        """Test de pipeline con configuración personalizada."""
        config = PipelineConfig(
            use_largest_face_only=False,
            min_confidence=0.8,
        )
        with patch.object(FaceDetector, "_load_model", return_value=None):
            pipeline = FaceProcessingPipeline(config)

            assert pipeline.config.use_largest_face_only is False
            assert pipeline.config.min_confidence == 0.8

    def test_process_image_returns_pipeline_result(
        self, sample_pil_image: Image.Image
    ) -> None:
        """Test de que process_image retorna PipelineResult."""
        with patch.object(FaceDetector, "_load_model", return_value=None):
            pipeline = FaceProcessingPipeline()

            with patch.object(pipeline._detector, "detect") as mock_detect:
                mock_detect.return_value = DetectionResult(
                    faces=[
                        BoundingBox(x=100, y=80, width=100, height=140, confidence=0.95)
                    ],
                    image_size=(300, 300),
                    num_faces=1,
                )
                result = pipeline.process_image(sample_pil_image)

                assert isinstance(result, PipelineResult)
                assert isinstance(result.num_faces_detected, int)
                assert isinstance(result.num_faces_processed, int)

    def test_get_output_shape(self) -> None:
        """Test de obtención de forma de salida."""
        with patch.object(FaceDetector, "_load_model", return_value=None):
            pipeline = FaceProcessingPipeline()
            shape = pipeline.get_output_shape()

            assert shape == (224, 224, 3)


class TestProcessedFace:
    """Tests para ProcessedFace."""

    def test_processed_face_creation(self) -> None:
        """Test de creación de ProcessedFace."""
        normalized = np.random.rand(224, 224, 3).astype(np.float32)
        face = ProcessedFace(
            normalized_face=normalized,
            bounding_box={"x": 10, "y": 20, "width": 100, "height": 150},
            confidence=0.95,
            original_size=(300, 300),
        )

        assert face.normalized_face.shape == (224, 224, 3)
        assert face.confidence == 0.95
