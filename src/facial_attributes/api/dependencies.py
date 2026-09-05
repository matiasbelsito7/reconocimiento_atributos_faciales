"""Dependencias inyectables para la API."""

from __future__ import annotations

import logging
from pathlib import Path

from facial_attributes.inference.pipeline import InferenceConfig, InferencePipeline

logger = logging.getLogger(__name__)

_PIPELINE: InferencePipeline | None = None

CELEBA_ATTRIBUTE_NAMES: list[str] = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]

ATTRIBUTE_DISPLAY_NAMES: dict[str, str] = {
    "5_o_Clock_Shadow": "Barba de 5 minutos",
    "Arched_Eyebrows": "Cejas arqueadas",
    "Attractive": "Atractivo/a",
    "Bags_Under_Eyes": "Bolsas bajo los ojos",
    "Bald": "Calvo/a",
    "Bangs": "Flequillo",
    "Big_Lips": "Labios grandes",
    "Big_Nose": "Nariz grande",
    "Black_Hair": "Cabello negro",
    "Blond_Hair": "Cabello rubio",
    "Blurry": "Borroso",
    "Brown_Hair": "Cabello castaño",
    "Bushy_Eyebrows": "Cejas gruesas",
    "Chubby": "Gordito/a",
    "Double_Chin": "Papada",
    "Eyeglasses": "Lentes",
    "Goatee": "Perilla",
    "Gray_Hair": "Cabello canoso",
    "Heavy_Makeup": "Maquillaje pesado",
    "High_Cheekbones": "Pómulos altos",
    "Male": "Masculino",
    "Mouth_Slightly_Open": "Boca ligeramente abierta",
    "Mustache": "Bigote",
    "Narrow_Eyes": "Ojos estrechos",
    "No_Beard": "Sin barba",
    "Oval_Face": "Rostro ovalado",
    "Pale_Skin": "Piel pálida",
    "Pointy_Nose": "Nariz puntiaguda",
    "Receding_Hairline": "Entrada del cabello",
    "Rosy_Cheeks": "Mejillas sonrojadas",
    "Sideburns": "Patillas",
    "Smiling": "Sonriente",
    "Straight_Hair": "Cabello liso",
    "Wavy_Hair": "Cabello ondulado",
    "Wearing_Earrings": "Usando aretes",
    "Wearing_Hat": "Usando sombrero",
    "Wearing_Lipstick": "Usando lápiz labial",
    "Wearing_Necklace": "Usando collar",
    "Wearing_Necktie": "Usando corbata",
    "Young": "Joven",
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
        default_path = Path("checkpoints/best_model.pt")
        if default_path.exists():
            model_path = str(default_path)
        else:
            logger.warning(
                "No se encontró modelo en %s. " "La inferencia no estará disponible.",
                default_path,
            )

    config = InferenceConfig(
        device="auto",
        model_path=model_path,
        num_attributes=40,
        attribute_names=CELEBA_ATTRIBUTE_NAMES,
    )

    _PIPELINE = InferencePipeline(config)
    logger.info(
        "Pipeline inicializado. Modelo cargado: %s",
        model_path is not None,
    )
    return _PIPELINE
