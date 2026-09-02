"""Pipeline completo de face processing."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from facial_attributes.face_processing.detector import (
    DetectorConfig,
    FaceDetector,
)
from facial_attributes.face_processing.extractor import (
    ExtractorConfig,
    FaceExtractor,
)
from facial_attributes.face_processing.normalizer import (
    FaceNormalizer,
    NormalizerConfig,
)


@dataclass
class PipelineConfig:
    """Configuración del pipeline de face processing."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)
    use_largest_face_only: bool = True
    min_confidence: float = 0.9


@dataclass
class ProcessedFace:
    """Rostro procesado listo para inferencia."""

    normalized_face: np.ndarray
    bounding_box: dict
    confidence: float
    original_size: tuple[int, int]
    source_path: Path | None = None


@dataclass
class PipelineResult:
    """Resultado del pipeline de face processing."""

    processed_faces: list[ProcessedFace]
    num_faces_detected: int
    num_faces_processed: int
    image_path: Path | None = None


class FaceProcessingPipeline:
    """Pipeline completo de face processing."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._detector = FaceDetector(self.config.detector)
        self._extractor = FaceExtractor(self.config.extractor)
        self._normalizer = FaceNormalizer(self.config.normalizer)

    def process_image(
        self, image: Image.Image, source_path: Path | None = None
    ) -> PipelineResult:
        """Procesar una imagen completa.

        Args:
            image: Imagen PIL.
            source_path: Ruta origen de la imagen.

        Returns:
            Resultado del pipeline con rostros procesados.
        """
        detection = self._detector.detect(image)

        if self.config.use_largest_face_only:
            largest_face = self._extractor.extract_largest_face(
                image, detection, source_path
            )
            if largest_face is None:
                return PipelineResult(
                    processed_faces=[],
                    num_faces_detected=detection.num_faces,
                    num_faces_processed=0,
                    image_path=source_path,
                )

            if largest_face.bounding_box.confidence < self.config.min_confidence:
                return PipelineResult(
                    processed_faces=[],
                    num_faces_detected=detection.num_faces,
                    num_faces_processed=0,
                    image_path=source_path,
                )

            normalized = self._normalizer.normalize(largest_face.image)
            return PipelineResult(
                processed_faces=[
                    ProcessedFace(
                        normalized_face=normalized,
                        bounding_box={
                            "x": largest_face.bounding_box.x,
                            "y": largest_face.bounding_box.y,
                            "width": largest_face.bounding_box.width,
                            "height": largest_face.bounding_box.height,
                        },
                        confidence=largest_face.bounding_box.confidence,
                        original_size=largest_face.original_size,
                        source_path=source_path,
                    )
                ],
                num_faces_detected=detection.num_faces,
                num_faces_processed=1,
                image_path=source_path,
            )

        extracted_faces = self._extractor.extract_faces(image, detection, source_path)

        processed: list[ProcessedFace] = []
        for face in extracted_faces:
            if face.bounding_box.confidence < self.config.min_confidence:
                continue

            normalized = self._normalizer.normalize(face.image)
            processed.append(
                ProcessedFace(
                    normalized_face=normalized,
                    bounding_box={
                        "x": face.bounding_box.x,
                        "y": face.bounding_box.y,
                        "width": face.bounding_box.width,
                        "height": face.bounding_box.height,
                    },
                    confidence=face.bounding_box.confidence,
                    original_size=face.original_size,
                    source_path=source_path,
                )
            )

        return PipelineResult(
            processed_faces=processed,
            num_faces_detected=detection.num_faces,
            num_faces_processed=len(processed),
            image_path=source_path,
        )

    def process_batch(self, images: list[Image.Image]) -> list[PipelineResult]:
        """Procesar un lote de imágenes.

        Args:
            images: Lista de imágenes PIL.

        Returns:
            Lista de resultados del pipeline.
        """
        return [self.process_image(img) for img in images]

    def get_output_shape(self) -> tuple[int, int, int]:
        """Obtener forma de salida esperada."""
        return self._normalizer.get_output_shape()
