"""Detector de rostros usando OpenCV DNN."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class BoundingBox:
    """Bounding box de un rostro detectado."""

    x: int
    y: int
    width: int
    height: int
    confidence: float


@dataclass
class DetectionResult:
    """Resultado de detección de rostros."""

    faces: list[BoundingBox]
    image_size: tuple[int, int]
    num_faces: int


@dataclass
class DetectorConfig:
    """Configuración del detector de rostros."""

    confidence_threshold: float = 0.7
    nms_threshold: float = 0.3
    input_size: tuple[int, int] = (300, 300)
    model_dir: str = "models/face_detection"


class FaceDetector:
    """Detector de rostros usando OpenCV DNN con modelo Caffe.

    Para usar este detector, descarga los siguientes archivos y colócalos en models/face_detection/:
    - deploy.prototxt: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
    - res10_300x300_ssd_iter_140000.caffemodel: https://dl.opencv.org/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self._model_dir = Path(self.config.model_dir)
        self._net = self._load_model()

    def _load_model(self) -> cv2.dnn_Net | None:
        """Cargar modelo de detección."""
        proto_path = self._model_dir / "deploy.prototxt"
        weights_path = self._model_dir / "res10_300x300_ssd_iter_140000.caffemodel"

        if not proto_path.exists() or not weights_path.exists():
            return None

        net = cv2.dnn.readNetFromCaffe(str(proto_path), str(weights_path))
        return net

    def _check_model_available(self) -> bool:
        """Verificar si el modelo está disponible."""
        return self._net is not None

    def detect(self, image: Image.Image) -> DetectionResult:
        """Detectar rostros en una imagen.

        Args:
            image: Imagen PIL.

        Returns:
            Resultado de detección con bounding boxes.

        Raises:
            RuntimeError: Si el modelo no está disponible.
        """
        if not self._check_model_available():
            raise RuntimeError(
                "Modelo de detección no disponible. "
                "Descarga los archivos necesarios y colócalos en "
                f"{self._model_dir}/. "
                "Consulta la documentación para más detalles."
            )

        if image.mode != "RGB":
            image = image.convert("RGB")

        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        h, w = img_array.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(img_array, self.config.input_size),
            1.0,
            self.config.input_size,
            (104.0, 177.0, 123.0),
        )

        self._net.setInput(blob)
        detections = self._net.forward()

        faces: list[BoundingBox] = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.config.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)

                faces.append(
                    BoundingBox(
                        x=max(0, x1),
                        y=max(0, y1),
                        width=min(w, x2) - max(0, x1),
                        height=min(h, y2) - max(0, y1),
                        confidence=float(confidence),
                    )
                )

        return DetectionResult(
            faces=faces,
            image_size=(image.width, image.height),
            num_faces=len(faces),
        )

    def detect_from_path(self, image_path: Path) -> DetectionResult:
        """Detectar rostros desde una ruta de archivo.

        Args:
            image_path: Ruta a la imagen.

        Returns:
            Resultado de detección.
        """
        image = Image.open(image_path)
        return self.detect(image)
