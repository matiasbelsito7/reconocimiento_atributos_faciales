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


@dataclass
class ThresholdWithMargin:
    """Threshold optimizado con margen de incerteza para un atributo."""

    attribute: str
    threshold: float
    margin: float


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
        pred_binary: np.ndarray = np.zeros_like(predictions, dtype=int)
        for i, result in enumerate(thresholds):
            pred_binary[:, i] = (predictions[:, i] > result.threshold).astype(int)
        return pred_binary

    def estimate_margins(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        thresholds: list[ThresholdResult],
        step: float = 0.01,
        reliability: float = 0.65,
        max_margin: float = 0.5,
    ) -> list[ThresholdWithMargin]:
        """Estimar margen de incerteza por atributo sobre datos de validación.

        El margen delimita la zona alrededor del umbral donde la predicción
        no es confiable: para la banda ``[threshold - margin, threshold + margin]``
        no se puede garantizar la clase, mientras que fuera de la banda la
        tasa empírica de la clase mayoritaria debe alcanzar ``reliability``.

        Los resultados deben estar alineados con las columnas de
        ``predictions`` (se asume el mismo orden que produce ``optimize``).

        Args:
            predictions: Predicciones del modelo [num_samples, num_attributes].
            targets: Valores reales [num_samples, num_attributes].
            thresholds: Thresholds optimizados, uno por columna.
            step: Paso de barrido del margen.
            reliability: Tasa empírica mínima de la clase mayoritaria fuera
                de la banda de incerteza.
            max_margin: Margen máximo a considerar.

        Returns:
            Lista de thresholds con margen estimado.
        """
        results = []
        for i, result in enumerate(thresholds):
            column = predictions[:, i]
            target_column = targets[:, i]
            margin = self._find_margin(
                column, target_column, result.threshold, step, reliability, max_margin
            )
            results.append(
                ThresholdWithMargin(
                    attribute=result.attribute,
                    threshold=result.threshold,
                    margin=margin,
                )
            )
        return results

    def apply_thresholds_with_margin(
        self,
        predictions: np.ndarray,
        thresholds: list[ThresholdWithMargin],
    ) -> np.ndarray:
        """Aplicar thresholds con margen de incerteza por atributo.

        La salida codifica la decisión ternaria por atributo:
        ``0`` = No (score por debajo de la banda de incerteza),
        ``1`` = Sí (score por encima de la banda de incerteza),
        ``2`` = Incierto (score dentro de la banda, no se puede garantizar).

        Args:
            predictions: Predicciones del modelo [num_samples, num_attributes].
            thresholds: Thresholds con margen, uno por columna.

        Returns:
            Decisiones ternarias [num_samples, num_attributes].
        """
        decisions: np.ndarray = np.zeros_like(predictions, dtype=int)
        for i, result in enumerate(thresholds):
            column = predictions[:, i]
            decisions[:, i] = np.where(
                column > result.threshold + result.margin,
                1,
                np.where(column < result.threshold - result.margin, 0, 2),
            )
        return decisions

    @staticmethod
    def _find_margin(
        column: np.ndarray,
        target_column: np.ndarray,
        threshold: float,
        step: float,
        reliability: float,
        max_margin: float,
    ) -> float:
        """Encontrar el menor margen con tasa empírica confiable fuera de la banda."""
        candidates = np.arange(0.0, max_margin + step, step)
        for margin in candidates:
            pos_mask = column > threshold + margin
            neg_mask = column < threshold - margin

            pos_rate = None
            if pos_mask.any():
                pos_rate = float(target_column[pos_mask].mean())
            neg_rate = None
            if neg_mask.any():
                neg_rate = 1.0 - float(target_column[neg_mask].mean())

            rates = [rate for rate in (pos_rate, neg_rate) if rate is not None]
            if rates and all(rate >= reliability for rate in rates):
                return float(margin)

        return float(max_margin)
