"""Tests para el módulo de entrenamiento."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from facial_attributes.model.classifier import FacialAttributeClassifier, ModelConfig
from facial_attributes.training.checkpoint import CheckpointManager
from facial_attributes.training.config import TrainingConfig, set_seed
from facial_attributes.training.dataset import FacialAttributeDataset
from facial_attributes.training.metrics import MetricsCalculator, MetricsResult
from facial_attributes.training.trainer import EarlyStopping, Trainer


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Crear dataset de ejemplo para tests."""
    data_dir = tmp_path / "data"
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "image_id": range(1, 11),
            "Atr_smiling": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "Atr_eyeglasses": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    annotations_file = data_dir / "annotations.csv"
    df.to_csv(annotations_file, index=False)

    for img_id in range(1, 11):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        from PIL import Image

        Image.fromarray(img).save(images_dir / f"{img_id:06d}.jpg")

    return data_dir


@pytest.fixture
def sample_dataloader() -> DataLoader:
    """Crear DataLoader de ejemplo."""
    images = torch.randn(8, 3, 224, 224)
    labels = torch.randint(0, 2, (8, 2)).float()
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=4)


class TestTrainingConfig:
    """Tests para TrainingConfig."""

    def test_default_config(self) -> None:
        """Test de configuración por defecto."""
        config = TrainingConfig()

        assert config.seed == 42
        assert config.num_epochs == 50
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4

    def test_custom_config(self) -> None:
        """Test de configuración personalizada."""
        config = TrainingConfig(
            seed=123,
            num_epochs=100,
            batch_size=64,
            learning_rate=1e-3,
        )

        assert config.seed == 123
        assert config.num_epochs == 100
        assert config.batch_size == 64
        assert config.learning_rate == 1e-3

    def test_invalid_ratios(self) -> None:
        """Test de ratios inválidos."""
        with pytest.raises(ValueError):
            TrainingConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_get_device(self) -> None:
        """Test de obtención de dispositivo."""
        config = TrainingConfig(device="cpu")
        device = config.get_device()

        assert device == torch.device("cpu")


class TestSetSeed:
    """Tests para set_seed."""

    def test_set_seed_reproducibility(self) -> None:
        """Test de reproducibilidad con semilla."""
        set_seed(42)
        a = torch.randn(5)

        set_seed(42)
        b = torch.randn(5)

        torch.testing.assert_close(a, b)


class TestEarlyStopping:
    """Tests para EarlyStopping."""

    def test_no_stop_on_improvement(self) -> None:
        """Test de que no se detiene con mejora."""
        early_stopping = EarlyStopping(patience=3)

        assert early_stopping(1.0) is False
        assert early_stopping(0.9) is False
        assert early_stopping(0.8) is False

    def test_stop_on_no_improvement(self) -> None:
        """Test de que se detiene sin mejora."""
        early_stopping = EarlyStopping(patience=3)

        early_stopping(1.0)
        early_stopping(1.1)
        early_stopping(1.2)
        assert early_stopping(1.3) is True

    def test_reset_counter_on_improvement(self) -> None:
        """Test de reset del contador con mejora."""
        early_stopping = EarlyStopping(patience=3)

        early_stopping(1.0)
        early_stopping(1.1)
        early_stopping(1.2)
        early_stopping(0.9)
        assert early_stopping.counter == 0


class TestCheckpointManager:
    """Tests para CheckpointManager."""

    def test_save_checkpoint(self, tmp_path: Path) -> None:
        """Test de guardado de checkpoint."""
        manager = CheckpointManager(str(tmp_path / "checkpoints"))
        model = FacialAttributeClassifier(
            ModelConfig(pretrained=False, num_attributes=2)
        )
        optimizer = torch.optim.Adam(model.parameters())

        path = manager.save_checkpoint(
            model, optimizer, epoch=1, val_loss=0.5, config={}, is_best=True
        )

        assert path.exists()
        assert (tmp_path / "checkpoints" / "best_model.pt").exists()

    def test_load_checkpoint(self, tmp_path: Path) -> None:
        """Test de carga de checkpoint."""
        manager = CheckpointManager(str(tmp_path / "checkpoints"))
        model = FacialAttributeClassifier(
            ModelConfig(pretrained=False, num_attributes=2)
        )
        optimizer = torch.optim.Adam(model.parameters())

        manager.save_checkpoint(
            model, optimizer, epoch=1, val_loss=0.5, config={}, is_best=True
        )

        loaded_model = FacialAttributeClassifier(
            ModelConfig(pretrained=False, num_attributes=2)
        )
        loaded_optimizer = torch.optim.Adam(loaded_model.parameters())

        info = manager.load_checkpoint(loaded_model, loaded_optimizer)

        assert info["epoch"] == 1

    def test_list_checkpoints(self, tmp_path: Path) -> None:
        """Test de listado de checkpoints."""
        manager = CheckpointManager(str(tmp_path / "checkpoints"))
        model = FacialAttributeClassifier(
            ModelConfig(pretrained=False, num_attributes=2)
        )
        optimizer = torch.optim.Adam(model.parameters())

        manager.save_checkpoint(model, optimizer, epoch=1, val_loss=0.5, config={})
        manager.save_checkpoint(model, optimizer, epoch=2, val_loss=0.4, config={})

        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) == 2


class TestFacialAttributeDataset:
    """Tests para FacialAttributeDataset."""

    def test_dataset_creation(self, sample_dataset: Path) -> None:
        """Test de creación del dataset."""
        dataset = FacialAttributeDataset(
            annotations_file=sample_dataset / "annotations.csv",
            images_dir=sample_dataset / "images",
        )

        assert len(dataset) == 10

    def test_dataset_getitem(self, sample_dataset: Path) -> None:
        """Test de obtención de elemento."""
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        dataset = FacialAttributeDataset(
            annotations_file=sample_dataset / "annotations.csv",
            images_dir=sample_dataset / "images",
            transform=transform,
        )

        image, attributes = dataset[0]

        assert image.shape == (3, 224, 224)
        assert attributes.shape == (2,)

    def test_dataset_num_attributes(self, sample_dataset: Path) -> None:
        """Test de obtención de número de atributos."""
        dataset = FacialAttributeDataset(
            annotations_file=sample_dataset / "annotations.csv",
            images_dir=sample_dataset / "images",
        )

        assert dataset.get_num_attributes() == 2

    def test_dataset_class_weights(self, sample_dataset: Path) -> None:
        """Test de cálculo de pesos de clase."""
        dataset = FacialAttributeDataset(
            annotations_file=sample_dataset / "annotations.csv",
            images_dir=sample_dataset / "images",
        )

        weights = dataset.get_class_weights()

        assert weights.shape == (2,)
        assert all(w > 0 for w in weights)


class TestMetricsCalculator:
    """Tests para MetricsCalculator."""

    def test_calculate_metrics(self) -> None:
        """Test de cálculo de métricas."""
        calculator = MetricsCalculator()

        predictions = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8, 2)).float()

        result = calculator.calculate(predictions, targets)

        assert isinstance(result, MetricsResult)
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.f1 <= 1.0

    def test_calculate_per_attribute(self) -> None:
        """Test de métricas por atributo."""
        calculator = MetricsCalculator(attribute_names=["smiling", "glasses"])

        predictions = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8, 2)).float()

        result = calculator.calculate(predictions, targets)

        assert "smiling" in result.per_attribute
        assert "glasses" in result.per_attribute

    def test_calculate_loss(self) -> None:
        """Test de cálculo de pérdida."""
        calculator = MetricsCalculator()

        logits = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8, 2)).float()

        loss = calculator.calculate_loss(logits, targets)

        assert loss.shape == ()
        assert loss.item() >= 0.0


class TestTrainer:
    """Tests para Trainer."""

    def test_trainer_initialization(self) -> None:
        """Test de inicialización del entrenador."""
        config = TrainingConfig(device="cpu")
        trainer = Trainer(config)

        assert trainer is not None

    def test_trainer_setup_model(self) -> None:
        """Test de configuración del modelo."""
        config = TrainingConfig(device="cpu")
        trainer = Trainer(config)
        trainer.setup_model(num_attributes=2)

        assert trainer._model is not None
        assert trainer._loss_fn is not None

    def test_trainer_train_epoch(self, sample_dataloader: DataLoader) -> None:
        """Test de entrenamiento de una época."""
        config = TrainingConfig(device="cpu")
        trainer = Trainer(config)
        trainer.setup_model(num_attributes=2)

        metrics = trainer.train_epoch(sample_dataloader)

        assert "train_loss" in metrics
        assert metrics["train_loss"] >= 0.0

    def test_trainer_validate(self, sample_dataloader: DataLoader) -> None:
        """Test de validación."""
        config = TrainingConfig(device="cpu")
        trainer = Trainer(config)
        trainer.setup_model(num_attributes=2)

        metrics = trainer.validate(sample_dataloader)

        assert "val_loss" in metrics
        assert "val_accuracy" in metrics
        assert "val_f1" in metrics

    def test_trainer_predict(self, sample_dataloader: DataLoader) -> None:
        """Test de predicción."""
        config = TrainingConfig(device="cpu")
        trainer = Trainer(config)
        trainer.setup_model(num_attributes=2)

        predictions, targets = trainer.predict(sample_dataloader)

        assert predictions.shape[0] > 0
        assert targets.shape[0] > 0
