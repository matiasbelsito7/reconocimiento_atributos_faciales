"""Pipeline de face processing para reconocimiento de atributos faciales."""

from facial_attributes.face_processing.extractor import FaceExtractor
from facial_attributes.face_processing.normalizer import FaceNormalizer
from facial_attributes.face_processing.pipeline import FaceProcessingPipeline

__all__ = ["FaceExtractor", "FaceNormalizer", "FaceProcessingPipeline"]
