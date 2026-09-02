"""Pipeline completo de inferencia."""

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from facial_attributes.face_processing.detector import DetectorConfig, FaceDetector
from facial_attributes.face_processing.extractor import ExtractorConfig, FaceExtractor
from facial_attributes.face_processing.normalizer import (
    FaceNormalizer,
    NormalizerConfig,
)
from facial_attributes.model.classifier import FacialAttributeClassifier, ModelConfig


@dataclass
class InferenceConfig:
    """Configuración de inferencia."""

    threshold: float = 0.5
    device: str = "auto"
    model_path: str | None = None
    num_attributes: int = 40
    backbone: str = "resnet18"
    attribute_names: list[str] | None = None


@dataclass
class FacePrediction:
    """Predicción para un rostro."""

    bbox: dict[str, int]
    attributes: dict[str, float]
    confidence: float


@dataclass
class InferenceResult:
    """Resultado completo de inferencia."""

    faces: list[FacePrediction]
    num_faces_detected: int
    num_faces_with_predictions: int
    inference_time_ms: float
    image_size: tuple[int, int]
    error: str | None = None


class InferencePipeline:
    """Pipeline completo de inferencia: imagen → atributos faciales."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        """Inicializar pipeline de inferencia.

        Args:
            config: Configuración de inferencia.
        """
        self.config = config or InferenceConfig()
        self._device = self._get_device()

        self._face_detector = FaceDetector(DetectorConfig())
        self._face_extractor = FaceExtractor(ExtractorConfig())
        self._face_normalizer = FaceNormalizer(NormalizerConfig(target_size=(224, 224)))

        self._model: FacialAttributeClassifier | None = None
        if self.config.model_path:
            self._load_model(self.config.model_path)

    def _get_device(self) -> torch.device:
        """Obtener dispositivo disponible."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.config.device)

    def _load_model(self, model_path: str) -> None:
        """Cargar modelo desde checkpoint.

        Args:
            model_path: Ruta al checkpoint del modelo.
        """
        model_config = ModelConfig(
            num_attributes=self.config.num_attributes,
            backbone=self.config.backbone,
            pretrained=False,
        )
        self._model = FacialAttributeClassifier(model_config).to(self._device)

        checkpoint = torch.load(
            model_path, map_location=self._device, weights_only=False
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()

    def predict(self, image: Image.Image) -> InferenceResult:
        """Realizar inferencia en una imagen.

        Args:
            image: Imagen PIL.

        Returns:
            Resultado de inferencia.
        """
        start_time = time.time()

        try:
            if self._model is None:
                return InferenceResult(
                    faces=[],
                    num_faces_detected=0,
                    num_faces_with_predictions=0,
                    inference_time_ms=0.0,
                    image_size=(image.width, image.height),
                    error="Modelo no cargado. Especifica model_path en la configuración.",
                )

            detection = self._face_detector.detect(image)

            if detection.num_faces == 0:
                return InferenceResult(
                    faces=[],
                    num_faces_detected=0,
                    num_faces_with_predictions=0,
                    inference_time_ms=(time.time() - start_time) * 1000,
                    image_size=(image.width, image.height),
                    error="No se detectaron rostros en la imagen.",
                )

            extracted_faces = self._face_extractor.extract_faces(image, detection)

            predictions: list[FacePrediction] = []
            for face in extracted_faces:
                normalized = self._face_normalizer.normalize(face.image)
                tensor = torch.tensor(normalized, dtype=torch.float32).permute(2, 0, 1)
                tensor = tensor.unsqueeze(0).to(self._device)

                with torch.no_grad():
                    scores = self._model.predict_proba(tensor)
                    scores_np = scores.cpu().numpy()[0]

                attributes = {}
                if self.config.attribute_names:
                    for i, name in enumerate(self.config.attribute_names):
                        attributes[name] = float(scores_np[i])
                else:
                    for i, score in enumerate(scores_np):
                        attributes[f"attr_{i}"] = float(score)

                predictions.append(
                    FacePrediction(
                        bbox={
                            "x": face.bounding_box.x,
                            "y": face.bounding_box.y,
                            "w": face.bounding_box.width,
                            "h": face.bounding_box.height,
                        },
                        attributes=attributes,
                        confidence=face.bounding_box.confidence,
                    )
                )

            inference_time = (time.time() - start_time) * 1000

            return InferenceResult(
                faces=predictions,
                num_faces_detected=detection.num_faces,
                num_faces_with_predictions=len(predictions),
                inference_time_ms=inference_time,
                image_size=(image.width, image.height),
            )

        except Exception as e:
            return InferenceResult(
                faces=[],
                num_faces_detected=0,
                num_faces_with_predictions=0,
                inference_time_ms=(time.time() - start_time) * 1000,
                image_size=(image.width, image.height),
                error=f"Error durante la inferencia: {str(e)}",
            )

    def predict_from_path(self, image_path: Path) -> InferenceResult:
        """Realizar inferencia desde una ruta de archivo.

        Args:
            image_path: Ruta a la imagen.

        Returns:
            Resultado de inferencia.
        """
        image = Image.open(image_path)
        return self.predict(image)

    def predict_batch(self, images: list[Image.Image]) -> list[InferenceResult]:
        """Realizar inferencia en un lote de imágenes.

        Args:
            images: Lista de imágenes PIL.

        Returns:
            Lista de resultados de inferencia.
        """
        return [self.predict(img) for img in images]
