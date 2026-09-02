"""Esquemas de registro de modelos."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ModelState(Enum):
    """Estado del modelo."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelMetrics:
    """Métricas de evaluación del modelo."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    hamming_loss: float = 0.0
    average_precision: float = 0.0
    per_attribute: dict[str, float] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuración utilizada para entrenar el modelo."""

    backbone: str = "resnet18"
    num_attributes: int = 40
    image_size: list[int] = field(default_factory=lambda: [224, 224])
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    dropout: float = 0.5
    optimizer: str = "adam"
    loss_function: str = "bce_with_logits"
    additional_config: dict[str, str] = field(default_factory=dict)


@dataclass
class DatasetInfo:
    """Información del dataset utilizado."""

    name: str = ""
    version: str = ""
    num_samples: int = 0
    num_attributes: int = 0
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    path: str = ""


@dataclass
class ArtifactInfo:
    """Información de un artefacto."""

    name: str = ""
    path: str = ""
    size_bytes: int = 0
    artifact_type: str = ""


@dataclass
class ModelMetadata:
    """Metadata completa de un modelo."""

    model_id: str = ""
    name: str = ""
    version: str = ""
    state: ModelState = ModelState.DEVELOPMENT
    created_at: str = ""
    updated_at: str = ""
    description: str = ""
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    config: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetInfo = field(default_factory=DatasetInfo)
    artifacts: list[ArtifactInfo] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    parent_model_id: str = ""

    def __post_init__(self) -> None:
        """Inicializar timestamps si no están definidos."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat()


@dataclass
class ModelVersion:
    """Versión de un modelo."""

    version_id: str = ""
    model_id: str = ""
    version: str = ""
    state: ModelState = ModelState.DEVELOPMENT
    created_at: str = ""
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    artifact_path: str = ""

    def __post_init__(self) -> None:
        """Inicializar timestamp si no está definido."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class ComparisonResult:
    """Resultado de comparación entre modelos."""

    model_a_id: str = ""
    model_b_id: str = ""
    metrics_comparison: dict[str, dict[str, float]] = field(default_factory=dict)
    winner: str = ""
    improvement_percent: dict[str, float] = field(default_factory=dict)
