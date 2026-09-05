"""Métricas de evaluación para clasificación multilabel."""

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class AttributeMetrics:
    """Métricas para un atributo individual."""

    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    support: int
    positive_rate: float
    prediction_rate: float


@dataclass
class EvaluationMetrics:
    """Métricas completas de evaluación."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    hamming: float
    average_precision: float
    macro_roc_auc: float
    per_attribute: list[AttributeMetrics] = field(default_factory=list)
    best_attributes: list[str] = field(default_factory=list)
    worst_attributes: list[str] = field(default_factory=list)


class MetricsCalculator:
    """Calculadora de métricas de evaluación."""

    def __init__(self, attribute_names: list[str] | None = None) -> None:
        """Inicializar calculadora.

        Args:
            attribute_names: Nombres de los atributos.
        """
        self.attribute_names = attribute_names or []

    def calculate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold: float = 0.5,
    ) -> EvaluationMetrics:
        """Calcular métricas completas.

        Args:
            predictions: Predicciones del modelo [num_samples, num_attributes].
            targets: Valores reales [num_samples, num_attributes].
            threshold: Umbral para binarización.

        Returns:
            Métricas de evaluación calculadas.
        """
        pred_binary = (predictions > threshold).astype(int)

        accuracy = accuracy_score(targets, pred_binary)
        precision = precision_score(
            targets, pred_binary, average="micro", zero_division=0
        )
        recall = recall_score(targets, pred_binary, average="micro", zero_division=0)
        f1 = f1_score(targets, pred_binary, average="micro", zero_division=0)
        hamming = hamming_loss(targets, pred_binary)

        try:
            avg_precision = average_precision_score(
                targets, predictions, average="micro"
            )
        except ValueError:
            avg_precision = 0.0

        per_attribute = self._calculate_per_attribute(predictions, targets, threshold)

        f1_per_attr = [attr.f1 for attr in per_attribute]
        macro_f1 = float(np.mean(f1_per_attr)) if f1_per_attr else 0.0

        roc_auc_per_attr = [attr.roc_auc for attr in per_attribute]
        macro_roc_auc = float(np.mean(roc_auc_per_attr)) if roc_auc_per_attr else 0.0

        sorted_attrs = sorted(per_attribute, key=lambda x: x.f1, reverse=True)
        best_attributes = [a.name for a in sorted_attrs[:5]]
        worst_attributes = [a.name for a in sorted_attrs[-5:]]

        return EvaluationMetrics(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            macro_f1=macro_f1,
            hamming=float(hamming),
            average_precision=float(avg_precision),
            macro_roc_auc=macro_roc_auc,
            per_attribute=per_attribute,
            best_attributes=best_attributes,
            worst_attributes=worst_attributes,
        )

    def _calculate_per_attribute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold: float,
    ) -> list[AttributeMetrics]:
        """Calcular métricas por atributo.

        Args:
            predictions: Predicciones del modelo.
            targets: Valores reales.
            threshold: Umbral para binarización.

        Returns:
            Lista de métricas por atributo.
        """
        pred_binary = (predictions > threshold).astype(int)
        num_attributes = targets.shape[1]

        per_attribute = []
        for i in range(num_attributes):
            attr_name = (
                self.attribute_names[i]
                if i < len(self.attribute_names)
                else f"attr_{i}"
            )

            accuracy = float(accuracy_score(targets[:, i], pred_binary[:, i]))
            precision = float(
                precision_score(targets[:, i], pred_binary[:, i], zero_division=0)
            )
            recall = float(
                recall_score(targets[:, i], pred_binary[:, i], zero_division=0)
            )
            f1 = float(f1_score(targets[:, i], pred_binary[:, i], zero_division=0))
            support = int(targets[:, i].sum())
            positive_rate = float(targets[:, i].mean())
            prediction_rate = float(pred_binary[:, i].mean())

            try:
                pr_auc = float(
                    average_precision_score(targets[:, i], predictions[:, i])
                )
            except ValueError:
                pr_auc = 0.0

            try:
                roc_auc = float(roc_auc_score(targets[:, i], predictions[:, i]))
            except ValueError:
                roc_auc = 0.0

            per_attribute.append(
                AttributeMetrics(
                    name=attr_name,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    pr_auc=pr_auc,
                    roc_auc=roc_auc,
                    support=support,
                    positive_rate=positive_rate,
                    prediction_rate=prediction_rate,
                )
            )

        return per_attribute
