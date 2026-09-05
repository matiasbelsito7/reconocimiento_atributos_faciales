"""Optimización de thresholds por atributo para clasificación multilabel."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score


@dataclass
class ThresholdResult:
    """Resultado de optimización de threshold para un atributo."""

    attribute: str
    threshold: float
    f1_score: float


class ThresholdOptimizer:
    """Optimizador de thresholds independiente por atributo."""

    def optimize(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        attribute_names: list[str],
        threshold_range: tuple[float, float] = (0.0, 1.0),
        step: float = 0.01,
    ) -> list[ThresholdResult]:
        """Optimizar threshold por atributo usando F1 como criterio.

        Args:
            predictions: Predicciones del modelo [num_samples, num_attributes].
            targets: Valores reales [num_samples, num_attributes].
            attribute_names: Nombres de los atributos.
            threshold_range: Rango de thresholds a evaluar.
            step: Paso entre thresholds.

        Returns:
            Lista de resultados con threshold óptimo por atributo.
        """
        num_attributes = targets.shape[1]
        thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)

        results = []
        for i in range(num_attributes):
            attr_name = attribute_names[i] if i < len(attribute_names) else f"attr_{i}"

            best_threshold = 0.5
            best_f1 = 0.0

            for threshold in thresholds:
                pred_binary = (predictions[:, i] > threshold).astype(int)
                f1 = f1_score(targets[:, i], pred_binary, zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = float(threshold)

            results.append(
                ThresholdResult(
                    attribute=attr_name,
                    threshold=best_threshold,
                    f1_score=best_f1,
                )
            )

        return results

    def apply_thresholds(
        self,
        predictions: np.ndarray,
        thresholds: list[ThresholdResult],
    ) -> np.ndarray:
        """Aplicar thresholds optimizados por atributo.

        Args:
            predictions: Predicciones del modelo [num_samples, num_attributes].
            thresholds: Lista de resultados de optimización.

        Returns:
            Predicciones binarizadas [num_samples, num_attributes].
        """
        pred_binary = np.zeros_like(predictions, dtype=int)
        for i, result in enumerate(thresholds):
            pred_binary[:, i] = (predictions[:, i] > result.threshold).astype(int)
        return pred_binary
