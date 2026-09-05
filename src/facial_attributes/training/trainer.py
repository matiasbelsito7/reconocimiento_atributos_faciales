"""Entrenador principal para modelos de atributos faciales."""

import time
from dataclasses import asdict
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from facial_attributes.model.classifier import FacialAttributeClassifier, ModelConfig
from facial_attributes.model.losses import MultilabelLoss
from facial_attributes.training.checkpoint import CheckpointManager
from facial_attributes.training.config import (
    TrainingConfig,
    create_directories,
    set_seed,
)
from facial_attributes.training.metrics import MetricsCalculator


class EarlyStopping:
    """Early stopping para detener entrenamiento si no hay mejora."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        """Inicializar early stopping.

        Args:
            patience: Épocas sin mejora antes de detener.
            min_delta: Mejora mínima para considerar mejora.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Evaluar si debe detener el entrenamiento.

        Args:
            val_loss: Pérdida de validación actual.

        Returns:
            True si debe detener, False si continúa.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """Entrenador para modelos de atributos faciales."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        """Inicializar entrenador.

        Args:
            config: Configuración de entrenamiento.
        """
        self.config = config or TrainingConfig()
        self.device = self.config.get_device()

        set_seed(self.config.seed)
        create_directories(self.config)

        self._model: FacialAttributeClassifier | None = None
        self._loss_fn: MultilabelLoss | None = None
        self._optimizer: torch.optim.Adam | None = None
        self._checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)
        self._metrics_calculator: MetricsCalculator | None = None
        self._early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
        )

    def setup_model(
        self,
        num_attributes: int,
        pos_weight: torch.Tensor | None = None,
        train_loader: DataLoader | None = None,
    ) -> None:
        """Configurar modelo, pérdida y optimizador.

        Args:
            num_attributes: Número de atributos a predecir.
            pos_weight: Pesos de clase positiva (None = auto si auto_pos_weight).
            train_loader: DataLoader de entrenamiento (necesario si auto_pos_weight).
        """
        model_config = ModelConfig(
            num_attributes=num_attributes,
            backbone=self.config.backbone,
            pretrained=self.config.pretrained,
            dropout_rate=self.config.dropout_rate,
        )
        self._model = FacialAttributeClassifier(model_config).to(self.device)

        self._loss_fn = MultilabelLoss()
        if pos_weight is not None:
            self._loss_fn.set_pos_weight(pos_weight.to(self.device))
        elif self.config.auto_pos_weight and train_loader is not None:
            auto_weights = self._calculate_pos_weight(train_loader)
            self._loss_fn.set_pos_weight(auto_weights.to(self.device))

        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self._metrics_calculator = MetricsCalculator()

    def _calculate_pos_weight(self, dataloader: DataLoader) -> torch.Tensor:
        """Calcular pos_weight desde el training set.

        Args:
            dataloader: DataLoader de entrenamiento.

        Returns:
            Tensor de pos_weight por atributo [num_attributes].
        """
        num_pos = torch.zeros(self.config.num_attributes)
        num_samples = 0
        for _, attributes in dataloader:
            num_pos += attributes.sum(dim=0)
            num_samples += attributes.shape[0]
        num_neg = num_samples - num_pos
        pos_weight = num_neg / (num_pos + 1e-6)
        return pos_weight

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Entrenar una época.

        Args:
            dataloader: DataLoader de entrenamiento.

        Returns:
            Diccionario con métricas de la época.
        """
        if self._model is None or self._loss_fn is None or self._optimizer is None:
            raise RuntimeError("Modelo no configurado. Llama a setup_model() primero.")

        self._model.train()
        total_loss = 0.0
        num_batches = 0

        for images, attributes in dataloader:
            images = images.to(self.device)
            attributes = attributes.to(self.device)

            self._optimizer.zero_grad()
            outputs = self._model(images)
            loss = self._loss_fn(outputs, attributes)
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {"train_loss": total_loss / num_batches}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Validar modelo.

        Args:
            dataloader: DataLoader de validación.

        Returns:
            Diccionario con métricas de validación.
        """
        if self._model is None or self._loss_fn is None:
            raise RuntimeError("Modelo no configurado. Llama a setup_model() primero.")

        self._model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        for images, attributes in dataloader:
            images = images.to(self.device)
            attributes = attributes.to(self.device)

            outputs = self._model(images)
            loss = self._loss_fn(outputs, attributes)

            total_loss += loss.item()
            all_predictions.append(outputs.cpu())
            all_targets.append(attributes.cpu())

        avg_loss = total_loss / len(dataloader)

        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)

        metrics = self._metrics_calculator.calculate(predictions, targets)

        return {
            "val_loss": avg_loss,
            "val_accuracy": metrics.accuracy,
            "val_precision": metrics.precision,
            "val_recall": metrics.recall,
            "val_f1": metrics.f1,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        experiment_name: str | None = None,
    ) -> dict[str, list[float]]:
        """Entrenar modelo completo.

        Args:
            train_loader: DataLoader de entrenamiento.
            val_loader: DataLoader de validación.
            experiment_name: Nombre del experimento para MLflow.

        Returns:
            Historial de entrenamiento.
        """
        if self._model is None:
            raise RuntimeError("Modelo no configurado. Llama a setup_model() primero.")

        experiment_name = experiment_name or self.config.experiment_name

        mlflow.set_experiment(experiment_name)

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
        }

        best_val_loss = float("inf")

        with mlflow.start_run():
            mlflow.log_params(asdict(self.config))

            start_time = time.time()

            for epoch in range(self.config.num_epochs):
                train_metrics = self.train_epoch(train_loader)
                val_metrics = self.validate(val_loader)

                history["train_loss"].append(train_metrics["train_loss"])
                history["val_loss"].append(val_metrics["val_loss"])
                history["val_accuracy"].append(val_metrics["val_accuracy"])
                history["val_f1"].append(val_metrics["val_f1"])

                mlflow.log_metrics(
                    {
                        "train_loss": train_metrics["train_loss"],
                        **val_metrics,
                    },
                    step=epoch,
                )

                is_best = val_metrics["val_loss"] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics["val_loss"]

                self._checkpoint_manager.save_checkpoint(
                    model=self._model,
                    optimizer=self._optimizer,
                    epoch=epoch,
                    val_loss=val_metrics["val_loss"],
                    config=asdict(self.config),
                    is_best=is_best,
                )

                if self._early_stopping(val_metrics["val_loss"]):
                    print(f"Early stopping en época {epoch}")
                    break

            duration = time.time() - start_time
            mlflow.log_metric("training_duration", duration)

        return history

    def load_model(self, checkpoint_path: Path | None = None) -> None:
        """Cargar modelo desde checkpoint.

        Args:
            checkpoint_path: Ruta del checkpoint (None = mejor modelo).
        """
        if self._model is None:
            raise RuntimeError("Modelo no configurado. Llama a setup_model() primero.")

        self._checkpoint_manager.load_checkpoint(
            self._model, self._optimizer, checkpoint_path
        )

    def predict(self, dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """Realizar predicciones.

        Args:
            dataloader: DataLoader con datos.

        Returns:
            Tupla de (predicciones, targets).
        """
        if self._model is None:
            raise RuntimeError("Modelo no configurado. Llama a setup_model() primero.")

        self._model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, attributes in dataloader:
                images = images.to(self.device)
                outputs = self._model(images)
                all_predictions.append(outputs.cpu())
                all_targets.append(attributes)

        return torch.cat(all_predictions), torch.cat(all_targets)
