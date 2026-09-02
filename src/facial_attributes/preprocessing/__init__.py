"""Pipeline de preprocessing para reconocimiento de atributos faciales."""

from facial_attributes.preprocessing.inference import InferencePreprocessor
from facial_attributes.preprocessing.training import TrainingPreprocessor

__all__ = ["TrainingPreprocessor", "InferencePreprocessor"]
