"""Esquemas de configuración."""

from dataclasses import dataclass, field


@dataclass
class PathsConfig:
    """Configuración de rutas."""

    data_dir: str = "data/"
    raw_data_dir: str = "data/raw/"
    processed_data_dir: str = "data/processed/"
    models_dir: str = "models/"
    checkpoints_dir: str = "checkpoints/"
    logs_dir: str = "logs/"
    outputs_dir: str = "outputs/"


@dataclass
class LoggingConfig:
    """Configuración de logging."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/pipeline.log"


@dataclass
class PipelineConfig:
    """Configuración general del pipeline."""

    mode: str = "training"
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42
    device: str = "auto"


@dataclass
class ArchitectureConfig:
    """Configuración de arquitectura del modelo."""

    name: str = "facial_attribute_classifier"
    backbone: str = "resnet18"
    pretrained: bool = True
    num_attributes: int = 40


@dataclass
class InputConfig:
    """Configuración de entrada del modelo."""

    image_size: list[int] = field(default_factory=lambda: [224, 224])
    channels: int = 3
    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class OutputConfig:
    """Configuración de salida del modelo."""

    activation: str = "sigmoid"
    threshold: float = 0.5


@dataclass
class RegularizationConfig:
    """Configuración de regularización."""

    dropout: float = 0.5
    freeze_backbone: bool = False
    freeze_layers: int = 0


@dataclass
class ModelConfig:
    """Configuración del modelo."""

    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)


@dataclass
class HyperparametersConfig:
    """Hiperparámetros de entrenamiento."""

    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    weight_decay: float = 0.0001
    optimizer: str = "adam"
    scheduler: str = "reduce_on_plateau"


@dataclass
class EarlyStoppingConfig:
    """Configuración de early stopping."""

    enabled: bool = True
    patience: int = 10
    min_delta: float = 0.001
    monitor: str = "val_loss"
    mode: str = "min"


@dataclass
class CheckpointConfig:
    """Configuración de checkpoint."""

    monitor: str = "val_f1_score"
    mode: str = "max"
    save_best_only: bool = True
    save_last: bool = True


@dataclass
class DataConfig:
    """Configuración de datos."""

    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class AugmentationConfig:
    """Configuración de augmentación de datos."""

    enabled: bool = True
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_range: int = 15
    brightness_range: list[float] = field(default_factory=lambda: [0.8, 1.2])
    contrast_range: list[float] = field(default_factory=lambda: [0.8, 1.2])


@dataclass
class MLflowConfig:
    """Configuración de MLflow."""

    enabled: bool = True
    experiment_name: str = "facial_attributes"
    tracking_uri: str = "mlruns"


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento."""

    seed: int = 42
    hyperparameters: HyperparametersConfig = field(
        default_factory=HyperparametersConfig
    )
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)


@dataclass
class ThresholdsConfig:
    """Configuración de thresholds."""

    default: float = 0.5
    per_attribute: dict[str, float] = field(default_factory=dict)


@dataclass
class FaceDetectionConfig:
    """Configuración de detección de rostros."""

    model: str = "opencv_dnn"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_faces: int = 10
    model_path: str = "models/face_detection/"


@dataclass
class FaceExtractionConfig:
    """Configuración de extracción de rostros."""

    margin: float = 0.2
    largest_face_only: bool = True
    target_size: list[int] = field(default_factory=lambda: [224, 224])


@dataclass
class OptimizationConfig:
    """Configuración de optimización."""

    use_fp16: bool = False
    batch_inference: bool = False
    batch_size: int = 8


@dataclass
class InferenceOutputConfig:
    """Configuración de salida de inferencia."""

    format: str = "json"
    include_bbox: bool = True
    include_confidence: bool = True
    include_attributes: bool = True
    attribute_names: str = "auto"


@dataclass
class InferenceModelConfig:
    """Configuración de modelo para inferencia."""

    device: str = "auto"
    model_path: str = "checkpoints/best_model.pt"
    num_attributes: int = 40


@dataclass
class InferenceConfig:
    """Configuración de inferencia."""

    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    face_detection: FaceDetectionConfig = field(default_factory=FaceDetectionConfig)
    face_extraction: FaceExtractionConfig = field(default_factory=FaceExtractionConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    output: InferenceOutputConfig = field(default_factory=InferenceOutputConfig)
    inference: InferenceModelConfig = field(default_factory=InferenceModelConfig)


@dataclass
class ImageConfig:
    """Configuración de imágenes del dataset."""

    format: str = "jpg"
    target_size: list[int] = field(default_factory=lambda: [218, 178])
    channels: int = 3


@dataclass
class AttributesConfig:
    """Configuración de atributos del dataset."""

    num_attributes: int = 40
    type: str = "binary"
    columns: list[str] = field(default_factory=list)


@dataclass
class SplitsConfig:
    """Configuración de splits del dataset."""

    train: float = 0.7
    val: float = 0.15
    test: float = 0.15
    random_state: int = 42


@dataclass
class ValidationConfig:
    """Configuración de validación del dataset."""

    min_images: int = 1000
    check_attributes: bool = True
    check_corruption: bool = False


@dataclass
class DatasetPathsConfig:
    """Configuración de rutas del dataset."""

    root: str = "data/raw/"
    images: str = "data/raw/images/"
    attributes: str = "data/raw/annotations/celeba_attributes.csv"
    splits: str = "data/raw/annotations/"


@dataclass
class DatasetConfig:
    """Configuración de un dataset."""

    name: str = ""
    description: str = ""
    source: str = ""
    kaggle_dataset: str = ""
    version: str = "1.0"
    paths: DatasetPathsConfig = field(default_factory=DatasetPathsConfig)
    images: ImageConfig = field(default_factory=ImageConfig)
    attributes: AttributesConfig = field(default_factory=AttributesConfig)
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


@dataclass
class DatasetsConfig:
    """Configuración de datasets."""

    celeba: DatasetConfig = field(default_factory=DatasetConfig)
    sample: DatasetConfig = field(default_factory=DatasetConfig)
