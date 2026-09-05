"""Modelos Pydantic para request/response de la API."""

from pydantic import BaseModel, Field


class BoundingBoxResponse(BaseModel):
    """Bounding box de un rostro detectado."""

    x: int = Field(..., description="Coordenada X del borde izquierdo")
    y: int = Field(..., description="Coordenada Y del borde superior")
    w: int = Field(..., description="Ancho del bounding box")
    h: int = Field(..., description="Alto del bounding box")


class FaceResult(BaseModel):
    """Predicción para un rostro individual."""

    bbox: BoundingBoxResponse = Field(..., description="Bounding box del rostro")
    attributes: dict[str, float] = Field(
        ..., description="Scores por atributo (0.0 - 1.0)"
    )
    confidence: float = Field(..., description="Confianza de la detección")


class PredictResponse(BaseModel):
    """Respuesta completa de predicción."""

    faces: list[FaceResult] = Field(
        default_factory=list, description="Predicciones por rostro"
    )
    num_faces_detected: int = Field(..., description="Número de rostros detectados")
    num_faces_with_predictions: int = Field(
        ..., description="Número de rostros con predicciones"
    )
    inference_time_ms: float = Field(
        ..., description="Tiempo de inferencia en milisegundos"
    )
    image_size: list[int] = Field(
        ..., description="Dimensiones de la imagen [width, height]"
    )
    error: str | None = Field(default=None, description="Mensaje de error si existe")


class AttributeInfo(BaseModel):
    """Información de un atributo facial."""

    name: str = Field(..., description="Nombre del atributo")
    display_name: str = Field(..., description="Nombre legible para el usuario")
    index: int = Field(..., description="Índice del atributo en el vector de salida")


class AttributesListResponse(BaseModel):
    """Lista de atributos faciales disponibles."""

    attributes: list[AttributeInfo] = Field(
        ..., description="Lista de atributos soportados"
    )
    num_attributes: int = Field(..., description="Número total de atributos")


class HealthResponse(BaseModel):
    """Respuesta del health check."""

    status: str = Field(..., description="Estado del servicio")
    device: str = Field(..., description="Dispositivo de inferencia")
    model_loaded: bool = Field(..., description="Si el modelo está cargado")
    num_attributes: int = Field(..., description="Número de atributos del modelo")
    face_detector_available: bool = Field(
        ..., description="Si el detector de rostros está disponible"
    )
