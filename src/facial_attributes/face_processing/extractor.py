"""Extracción de rostros detectados."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from facial_attributes.face_processing.detector import BoundingBox, DetectionResult


@dataclass
class ExtractedFace:
    """Rostro extraído con metadata."""

    image: Image.Image
    bounding_box: BoundingBox
    original_size: tuple[int, int]
    source_path: Path | None = None


@dataclass
class ExtractorConfig:
    """Configuración del extractor."""

    margin_percent: float = 0.2
    min_face_size: int = 30


class FaceExtractor:
    """Extractor de rostros desde imágenes."""

    def __init__(self, config: ExtractorConfig | None = None) -> None:
        self.config = config or ExtractorConfig()

    def extract_faces(
        self,
        image: Image.Image,
        detection: DetectionResult,
        source_path: Path | None = None,
    ) -> list[ExtractedFace]:
        """Extraer rostros de una imagen.

        Args:
            image: Imagen original.
            detection: Resultado de detección.
            source_path: Ruta origen de la imagen.

        Returns:
            Lista de rostros extraídos.
        """
        extracted: list[ExtractedFace] = []

        for face in detection.faces:
            if (
                face.width < self.config.min_face_size
                or face.height < self.config.min_face_size
            ):
                continue

            margin_x = int(face.width * self.config.margin_percent)
            margin_y = int(face.height * self.config.margin_percent)

            x1 = max(0, face.x - margin_x)
            y1 = max(0, face.y - margin_y)
            x2 = min(image.width, face.x + face.width + margin_x)
            y2 = min(image.height, face.y + face.height + margin_y)

            face_crop = image.crop((x1, y1, x2, y2))

            extracted.append(
                ExtractedFace(
                    image=face_crop,
                    bounding_box=face,
                    original_size=(image.width, image.height),
                    source_path=source_path,
                )
            )

        return extracted

    def extract_largest_face(
        self,
        image: Image.Image,
        detection: DetectionResult,
        source_path: Path | None = None,
    ) -> ExtractedFace | None:
        """Extraer el rostro más grande de la imagen.

        Args:
            image: Imagen original.
            detection: Resultado de detección.
            source_path: Ruta origen de la imagen.

        Returns:
            El rostro más grande o None si no hay rostros.
        """
        if not detection.faces:
            return None

        largest = max(detection.faces, key=lambda f: f.width * f.height)
        faces = self.extract_faces(
            image,
            DetectionResult(
                faces=[largest],
                image_size=detection.image_size,
                num_faces=1,
            ),
            source_path,
        )

        return faces[0] if faces else None

    def extract_from_path(
        self, image_path: Path, detection: DetectionResult
    ) -> list[ExtractedFace]:
        """Extraer rostros desde una ruta de archivo.

        Args:
            image_path: Ruta a la imagen.
            detection: Resultado de detección.

        Returns:
            Lista de rostros extraídos.
        """
        image = Image.open(image_path)
        return self.extract_faces(image, detection, image_path)
