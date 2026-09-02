"""Métricas de entrenamiento y evaluación."""

from dataclasses import dataclass, field

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class MetricsResult:
    """Resultado de métricas calculadas."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    per_attribute: dict[str, dict[str, float]] = field(default_factory=dict)


class MetricsCalculator:
    """Calculadora de métricas para clasificación multilabel."""

    def __init__(self, attribute_names: list[str] | None = None) -> None:
        """Inicializar calculadora.

        Args:
            attribute_names: Nombres de los atributos.
        """
        self.attribute_names = attribute_names or []

    def calculate(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
    ) -> MetricsResult:
        """Calcular métricas.

        Args:
            predictions: Predicciones del modelo [batch, num_attributes].
            targets: Valores reales [batch, num_attributes].
            threshold: Umbral para binarización.

        Returns:
            Resultado de métricas calculadas.
        """
        pred_proba = torch.sigmoid(predictions).cpu().numpy()
        target_np = targets.cpu().numpy()

        pred_binary = (pred_proba > threshold).astype(int)

        accuracy = accuracy_score(target_np, pred_binary)
        precision = precision_score(
            target_np, pred_binary, average="micro", zero_division=0
        )
        recall = recall_score(target_np, pred_binary, average="micro", zero_division=0)
        f1 = f1_score(target_np, pred_binary, average="micro", zero_division=0)

        per_attribute = {}
        num_attributes = target_np.shape[1]

        for i in range(num_attributes):
            attr_name = (
                self.attribute_names[i]
                if i < len(self.attribute_names)
                else f"attr_{i}"
            )
            per_attribute[attr_name] = {
                "accuracy": float(accuracy_score(target_np[:, i], pred_binary[:, i])),
                "precision": float(
                    precision_score(target_np[:, i], pred_binary[:, i], zero_division=0)
                ),
                "recall": float(
                    recall_score(target_np[:, i], pred_binary[:, i], zero_division=0)
                ),
                "f1": float(
                    f1_score(target_np[:, i], pred_binary[:, i], zero_division=0)
                ),
            }

        return MetricsResult(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            per_attribute=per_attribute,
        )

    def calculate_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pos_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calcular pérdida BCE.

        Args:
            logits: Salida del modelo.
            targets: Valores reales.
            pos_weight: Pesos de clase positiva.

        Returns:
            Pérdida calculada.
        """
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return loss_fn(logits, targets)
