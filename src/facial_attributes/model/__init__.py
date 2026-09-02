"""Módulo de modelo para reconocimiento de atributos faciales."""

from facial_attributes.model.classifier import FacialAttributeClassifier
from facial_attributes.model.losses import MultilabelLoss

__all__ = ["FacialAttributeClassifier", "MultilabelLoss"]
