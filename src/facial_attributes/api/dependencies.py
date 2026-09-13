"""Dependencias inyectables para la API."""

from __future__ import annotations

import logging
from pathlib import Path

from facial_attributes.config.loader import ConfigLoader
from facial_attributes.inference.pipeline import InferenceConfig, InferencePipeline

logger = logging.getLogger(__name__)

_PIPELINE: InferencePipeline | None = None

CELEBA_ATTRIBUTE_NAMES: list[str] = [
    "5_o_Clock_Shadow",
    "Bald",
    "Bangs",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "Mouth_Slightly_Open",
    "Mustache",
    "No_Beard",
    "Receding_Hairline",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
]

ATTRIBUTE_DISPLAY_NAMES: dict[str, str] = {
    "5_o_Clock_Shadow": "Barba de 5 minutos",
    "Bald": "Calvo/a",
    "Bangs": "Flequillo",
    "Black_Hair": "Cabello negro",
    "Blond_Hair": "Cabello rubio",
    "Blurry": "Borroso",
    "Brown_Hair": "Cabello castaño",
    "Eyeglasses": "Lentes",
    "Goatee": "Perilla",
    "Gray_Hair": "Cabello canoso",
    "Heavy_Makeup": "Maquillaje pesado",
    "Mouth_Slightly_Open": "Boca ligeramente abierta",
    "Mustache": "Bigote",
    "No_Beard": "Sin barba",
    "Receding_Hairline": "Entrada del cabello",
    "Sideburns": "Patillas",
    "Smiling": "Sonriente",
    "Straight_Hair": "Cabello liso",
    "Wavy_Hair": "Cabello ondulado",
    "Wearing_Earrings": "Usando aretes",
    "Wearing_Hat": "Usando sombrero",
    "Wearing_Lipstick": "Usando lápiz labial",
    "Wearing_Necklace": "Usando collar",
    "Wearing_Necktie": "Usando corbata",
}


def get_pipeline() -> InferencePipeline:
    """Obtener la instancia del pipeline de inferencia.

    Returns:
        Instancia del pipeline inicializada.

    Raises:
        RuntimeError: Si el pipeline no ha sido inicializado.
    """
    if _PIPELINE is None:
        raise RuntimeError(
            "Pipeline no inicializado. "
            "Llama a init_pipeline() antes de usar get_pipeline()."
        )
    return _PIPELINE


def _load_thresholds_config() -> (
    tuple[float, dict[str, float], dict[str, float], float, float]
):
    """Cargar thresholds, márgenes y margen facial desde config/inference.yaml.

    Returns:
        Tupla (default, per_attribute, margins_per_attribute, margin_default,
        face_margin).
    """
    try:
        inference_config = ConfigLoader().load_inference()
    except Exception:
        logger.warning(
            "No se pudo cargar config/inference.yaml. Usando thresholds por defecto."
        )
        return 0.5, {}, {}, 0.0, 0.3

    thresholds = inference_config.thresholds
    return (
        thresholds.default,
        thresholds.per_attribute,
        thresholds.margins_per_attribute,
        thresholds.margin_default,
        inference_config.face_extraction.margin,
    )


def init_pipeline(model_path: str | None = None) -> InferencePipeline:
    """Inicializar el pipeline de inferencia como singleton.

    Args:
        model_path: Ruta al checkpoint del modelo. Si es None, se usa
            la ruta por defecto de inference.yaml.

    Returns:
        Instancia del pipeline inicializada.
    """
    global _PIPELINE  # noqa: PLW0603

    if model_path is None:
        default_path = Path("checkpoints_40k_obs24/best_model.pt")
        if default_path.exists():
            model_path = str(default_path)
        else:
            logger.warning(
                "No se encontró modelo en %s. " "La inferencia no estará disponible.",
                default_path,
            )

    (
        default_threshold,
        per_attribute_thresholds,
        per_attribute_margins,
        margin_default,
        face_margin,
    ) = _load_thresholds_config()

    config = InferenceConfig(
        device="auto",
        model_path=model_path,
        num_attributes=24,
        attribute_names=CELEBA_ATTRIBUTE_NAMES,
        threshold=default_threshold,
        per_attribute_thresholds=per_attribute_thresholds,
        per_attribute_margins=per_attribute_margins,
        margin_default=margin_default,
        face_margin=face_margin,
    )

    _PIPELINE = InferencePipeline(config)
    logger.info(
        "Pipeline inicializado. Modelo cargado: %s",
        model_path is not None,
    )
    return _PIPELINE
