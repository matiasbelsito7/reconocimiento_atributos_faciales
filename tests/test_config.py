"""Tests para el módulo de configuración."""

from pathlib import Path

import pytest
import yaml

from facial_attributes.config.loader import ConfigLoader
from facial_attributes.config.schemas import (
    ArchitectureConfig,
    DatasetsConfig,
    HyperparametersConfig,
    InferenceConfig,
    ModelConfig,
    PipelineConfig,
    ThresholdsConfig,
    TrainingConfig,
)


@pytest.fixture
def sample_pipeline_yaml(tmp_path: Path) -> Path:
    """Crear archivo pipeline.yaml de ejemplo."""
    config = {
        "mode": "training",
        "paths": {
            "data_dir": "data/",
            "raw_data_dir": "data/raw/",
        },
        "logging": {
            "level": "INFO",
        },
        "seed": 42,
        "device": "cpu",
    }
    config_path = tmp_path / "pipeline.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
def sample_model_yaml(tmp_path: Path) -> Path:
    """Crear archivo model.yaml de ejemplo."""
    config = {
        "architecture": {
            "name": "facial_attribute_classifier",
            "backbone": "resnet18",
            "pretrained": True,
            "num_attributes": 40,
        },
        "input": {
            "image_size": [224, 224],
            "channels": 3,
        },
        "output": {
            "activation": "sigmoid",
            "threshold": 0.5,
        },
        "regularization": {
            "dropout": 0.5,
        },
    }
    config_path = tmp_path / "model.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
def sample_training_yaml(tmp_path: Path) -> Path:
    """Crear archivo training.yaml de ejemplo."""
    config = {
        "seed": 42,
        "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
        },
        "early_stopping": {
            "enabled": True,
            "patience": 10,
        },
        "checkpoint": {
            "monitor": "val_f1_score",
        },
        "data": {
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
        },
        "augmentation": {
            "enabled": True,
        },
        "mlflow": {
            "enabled": True,
        },
    }
    config_path = tmp_path / "training.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
def sample_inference_yaml(tmp_path: Path) -> Path:
    """Crear archivo inference.yaml de ejemplo."""
    config = {
        "thresholds": {
            "default": 0.5,
        },
        "face_detection": {
            "model": "opencv_dnn",
            "confidence_threshold": 0.5,
        },
        "face_extraction": {
            "margin": 0.2,
            "largest_face_only": True,
        },
        "optimization": {
            "use_fp16": False,
        },
        "output": {
            "format": "json",
        },
        "inference": {
            "device": "cpu",
            "model_path": "checkpoints/best_model.pt",
        },
    }
    config_path = tmp_path / "inference.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
def sample_datasets_yaml(tmp_path: Path) -> Path:
    """Crear archivo datasets.yaml de ejemplo."""
    config = {
        "celeba": {
            "name": "CelebA",
            "source": "kaggle",
            "paths": {
                "root": "data/raw/",
            },
            "images": {
                "format": "jpg",
            },
            "attributes": {
                "num_attributes": 40,
            },
        },
        "sample": {
            "name": "Sample",
            "source": "local",
        },
    }
    config_path = tmp_path / "datasets.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


class TestPipelineConfig:
    """Tests para PipelineConfig."""

    def test_default_config(self) -> None:
        """Test de configuración por defecto."""
        config = PipelineConfig()

        assert config.mode == "training"
        assert config.seed == 42
        assert config.device == "auto"

    def test_custom_config(self) -> None:
        """Test de configuración personalizada."""
        config = PipelineConfig(mode="inference", seed=123, device="cpu")

        assert config.mode == "inference"
        assert config.seed == 123
        assert config.device == "cpu"


class TestModelConfig:
    """Tests para ModelConfig."""

    def test_default_config(self) -> None:
        """Test de configuración por defecto."""
        config = ModelConfig()

        assert config.architecture.backbone == "resnet18"
        assert config.architecture.num_attributes == 40
        assert config.input.image_size == [224, 224]

    def test_custom_config(self) -> None:
        """Test de configuración personalizada."""
        arch = ArchitectureConfig(backbone="resnet34", num_attributes=20)
        config = ModelConfig(architecture=arch)

        assert config.architecture.backbone == "resnet34"
        assert config.architecture.num_attributes == 20


class TestTrainingConfig:
    """Tests para TrainingConfig."""

    def test_default_config(self) -> None:
        """Test de configuración por defecto."""
        config = TrainingConfig()

        assert config.seed == 42
        assert config.hyperparameters.learning_rate == 0.001
        assert config.early_stopping.patience == 10

    def test_custom_config(self) -> None:
        """Test de configuración personalizada."""
        hp = HyperparametersConfig(learning_rate=0.01, batch_size=64)
        config = TrainingConfig(seed=123, hyperparameters=hp)

        assert config.seed == 123
        assert config.hyperparameters.learning_rate == 0.01
        assert config.hyperparameters.batch_size == 64


class TestInferenceConfig:
    """Tests para InferenceConfig."""

    def test_default_config(self) -> None:
        """Test de configuración por defecto."""
        config = InferenceConfig()

        assert config.thresholds.default == 0.5
        assert config.face_detection.model == "opencv_dnn"

    def test_custom_config(self) -> None:
        """Test de configuración personalizada."""
        thresholds = ThresholdsConfig(default=0.7)
        config = InferenceConfig(thresholds=thresholds)

        assert config.thresholds.default == 0.7


class TestDatasetsConfig:
    """Tests para DatasetsConfig."""

    def test_default_config(self) -> None:
        """Test de configuración por defecto."""
        config = DatasetsConfig()

        assert config.celeba.name == ""
        assert config.sample.name == ""


class TestConfigLoader:
    """Tests para ConfigLoader."""

    def test_loader_initialization(self, tmp_path: Path) -> None:
        """Test de inicialización del cargador."""
        loader = ConfigLoader(config_dir=tmp_path)

        assert loader.config_dir == tmp_path

    def test_load_pipeline(self, tmp_path: Path, sample_pipeline_yaml: Path) -> None:
        """Test de carga de configuración del pipeline."""
        loader = ConfigLoader(config_dir=tmp_path)

        config = loader.load_pipeline()

        assert isinstance(config, PipelineConfig)
        assert config.mode == "training"
        assert config.seed == 42

    def test_load_model(self, tmp_path: Path, sample_model_yaml: Path) -> None:
        """Test de carga de configuración del modelo."""
        loader = ConfigLoader(config_dir=tmp_path)

        config = loader.load_model()

        assert isinstance(config, ModelConfig)
        assert config.architecture.backbone == "resnet18"
        assert config.architecture.num_attributes == 40

    def test_load_training(self, tmp_path: Path, sample_training_yaml: Path) -> None:
        """Test de carga de configuración de entrenamiento."""
        loader = ConfigLoader(config_dir=tmp_path)

        config = loader.load_training()

        assert isinstance(config, TrainingConfig)
        assert config.seed == 42
        assert config.hyperparameters.learning_rate == 0.001

    def test_load_inference(self, tmp_path: Path, sample_inference_yaml: Path) -> None:
        """Test de carga de configuración de inferencia."""
        loader = ConfigLoader(config_dir=tmp_path)

        config = loader.load_inference()

        assert isinstance(config, InferenceConfig)
        assert config.thresholds.default == 0.5

    def test_load_datasets(self, tmp_path: Path, sample_datasets_yaml: Path) -> None:
        """Test de carga de configuración de datasets."""
        loader = ConfigLoader(config_dir=tmp_path)

        config = loader.load_datasets()

        assert isinstance(config, DatasetsConfig)
        assert config.celeba.name == "CelebA"

    def test_load_all(
        self,
        tmp_path: Path,
        sample_pipeline_yaml: Path,
        sample_model_yaml: Path,
        sample_training_yaml: Path,
        sample_inference_yaml: Path,
        sample_datasets_yaml: Path,
    ) -> None:
        """Test de carga de todas las configuraciones."""
        loader = ConfigLoader(config_dir=tmp_path)

        configs = loader.load_all()

        assert "pipeline" in configs
        assert "model" in configs
        assert "training" in configs
        assert "inference" in configs
        assert "datasets" in configs

        assert isinstance(configs["pipeline"], PipelineConfig)
        assert isinstance(configs["model"], ModelConfig)
        assert isinstance(configs["training"], TrainingConfig)
        assert isinstance(configs["inference"], InferenceConfig)
        assert isinstance(configs["datasets"], DatasetsConfig)
