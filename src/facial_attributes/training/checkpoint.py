"""Sistema de checkpoints para entrenamiento."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class CheckpointData:
    """Datos de un checkpoint."""

    epoch: int
    model_state_dict: dict
    optimizer_state_dict: dict
    best_val_loss: float
    config: dict
    timestamp: str


class CheckpointManager:
    """Gestor de checkpoints para entrenamiento."""

    def __init__(self, checkpoint_dir: str = "checkpoints") -> None:
        """Inicializar gestor de checkpoints.

        Args:
            checkpoint_dir: Directorio para guardar checkpoints.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.best_checkpoint_path: Path | None = None

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        config: dict,
        is_best: bool = False,
    ) -> Path:
        """Guardar checkpoint.

        Args:
            model: Modelo a guardar.
            optimizer: Optimizador a guardar.
            epoch: Época actual.
            val_loss: Pérdida de validación.
            config: Configuración del entrenamiento.
            is_best: Si es el mejor modelo hasta ahora.

        Returns:
            Ruta del checkpoint guardado.
        """
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": config,
            "timestamp": datetime.now().isoformat(),
        }

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint_data, checkpoint_path)

        if is_best:
            self.best_val_loss = val_loss
            self.best_checkpoint_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint_data, self.best_checkpoint_path)

        metadata_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.json"
        metadata = {
            "epoch": epoch,
            "val_loss": val_loss,
            "is_best": is_best,
            "timestamp": checkpoint_data["timestamp"],
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return checkpoint_path

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        checkpoint_path: Path | None = None,
    ) -> dict:
        """Cargar checkpoint.

        Args:
            model: Modelo para cargar pesos.
            optimizer: Optimizador para cargar estado.
            checkpoint_path: Ruta del checkpoint (None = mejor modelo).

        Returns:
            Diccionario con información del checkpoint.
        """
        if checkpoint_path is None:
            checkpoint_path = self.best_checkpoint_path
            if checkpoint_path is None:
                raise FileNotFoundError("No hay checkpoints disponibles")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return {
            "epoch": checkpoint["epoch"],
            "best_val_loss": checkpoint["best_val_loss"],
            "timestamp": checkpoint["timestamp"],
        }

    def get_latest_checkpoint(self) -> Path | None:
        """Obtener el checkpoint más reciente.

        Returns:
            Ruta del checkpoint más reciente o None.
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            return None

        return max(checkpoints, key=lambda p: p.stat().st_mtime)

    def list_checkpoints(self) -> list[dict]:
        """Listar todos los checkpoints disponibles.

        Returns:
            Lista de diccionarios con información de checkpoints.
        """
        checkpoints = []
        for metadata_file in sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.json")
        ):
            with open(metadata_file) as f:
                metadata = json.load(f)
                metadata["path"] = str(metadata_file.with_suffix(".pt"))
                checkpoints.append(metadata)
        return checkpoints
