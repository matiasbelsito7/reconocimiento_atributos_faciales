"""Módulo de registro de modelos para reconocimiento de atributos faciales."""

from facial_attributes.model_registry.registry import ModelRegistry
from facial_attributes.model_registry.schemas import (
    ModelMetadata,
    ModelState,
    ModelVersion,
)

__all__ = [
    "ModelMetadata",
    "ModelRegistry",
    "ModelState",
    "ModelVersion",
]
