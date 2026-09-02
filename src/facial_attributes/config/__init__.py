"""Módulo de configuración para reconocimiento de atributos faciales."""

from facial_attributes.config.loader import ConfigLoader
from facial_attributes.config.schemas import (
    InferenceConfig,
    ModelConfig,
    PipelineConfig,
    TrainingConfig,
)

__all__ = [
    "ConfigLoader",
    "InferenceConfig",
    "ModelConfig",
    "PipelineConfig",
    "TrainingConfig",
]
