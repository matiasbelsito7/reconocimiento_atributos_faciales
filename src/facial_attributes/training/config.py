"""Configuración para entrenamiento reproducible."""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TrainingConfig:
    """Configuración completa de entrenamiento."""

    seed: int = 42
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 10
    min_delta: float = 1e-4

    num_attributes: int = 40
    backbone: str = "resnet18"
    pretrained: bool = True
    dropout_rate: float = 0.5

    auto_pos_weight: bool = True

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    experiment_name: str = "facial_attributes"

    device: str = "auto"

    def __post_init__(self) -> None:
        """Validar configuración."""
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("Los ratios de división deben sumar 1.0")

    def get_device(self) -> torch.device:
        """Obtener dispositivo disponible."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)


def set_seed(seed: int) -> None:
    """Establecer semillas para reproducibilidad.

    Args:
        seed: Semilla a utilizar.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_directories(config: TrainingConfig) -> None:
    """Crear directorios necesarios para entrenamiento.

    Args:
        config: Configuración de entrenamiento.
    """
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
