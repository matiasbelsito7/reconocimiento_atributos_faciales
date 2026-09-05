"""Endpoints de la API de inferencia."""

import io

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from facial_attributes.api.dependencies import (
    ATTRIBUTE_DISPLAY_NAMES,
    CELEBA_ATTRIBUTE_NAMES,
    get_pipeline,
)
from facial_attributes.api.schemas import (
    AttributeInfo,
    AttributesListResponse,
    BoundingBoxResponse,
    FaceResult,
    HealthResponse,
    PredictResponse,
)

router = APIRouter(prefix="/api")

MAX_FILE_SIZE_MB = 10
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Verificar estado del servicio."""
    try:
        pipeline = get_pipeline()
        model_loaded = pipeline._model is not None
        device = str(pipeline._device)
        num_attributes = pipeline.config.num_attributes
        face_detector_available = pipeline._face_detector._check_model_available()
    except RuntimeError:
        model_loaded = False
        device = "unknown"
        num_attributes = 40
        face_detector_available = False

    return HealthResponse(
        status="ok",
        device=device,
        model_loaded=model_loaded,
        num_attributes=num_attributes,
        face_detector_available=face_detector_available,
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:  # noqa: B008
    """Realizar predicción de atributos faciales en una imagen.

    Acepta imágenes JPEG, PNG, WebP o BMP. El archivo no debe
    superar los 10MB.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Tipo de archivo no soportado: {file.content_type}. "
                f"Usa: {', '.join(ALLOWED_CONTENT_TYPES)}"
            ),
        )

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"El archivo supera el tamaño máximo de {MAX_FILE_SIZE_MB}MB.",
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="No se pudo procesar la imagen. Asegúrate de que sea un archivo de imagen válido.",
        ) from None

    try:
        pipeline = get_pipeline()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="El servicio de inferencia no está disponible.",
        ) from None

    result = pipeline.predict(image)

    faces = []
    for face in result.faces:
        faces.append(
            FaceResult(
                bbox=BoundingBoxResponse(
                    x=face.bbox["x"],
                    y=face.bbox["y"],
                    w=face.bbox["w"],
                    h=face.bbox["h"],
                ),
                attributes=face.attributes,
                confidence=face.confidence,
            )
        )

    return PredictResponse(
        faces=faces,
        num_faces_detected=result.num_faces_detected,
        num_faces_with_predictions=result.num_faces_with_predictions,
        inference_time_ms=result.inference_time_ms,
        image_size=list(result.image_size),
        error=result.error,
    )


@router.get("/attributes", response_model=AttributesListResponse)
async def list_attributes() -> AttributesListResponse:
    """Listar los atributos faciales disponibles."""
    attributes = [
        AttributeInfo(
            name=name,
            display_name=ATTRIBUTE_DISPLAY_NAMES.get(name, name),
            index=i,
        )
        for i, name in enumerate(CELEBA_ATTRIBUTE_NAMES)
    ]

    return AttributesListResponse(
        attributes=attributes,
        num_attributes=len(attributes),
    )
