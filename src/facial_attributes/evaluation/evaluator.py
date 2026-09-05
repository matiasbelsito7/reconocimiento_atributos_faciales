"""Evaluador para modelos de atributos faciales."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from facial_attributes.evaluation.metrics import EvaluationMetrics, MetricsCalculator
from facial_attributes.evaluation.thresholds import ThresholdOptimizer, ThresholdResult


@dataclass
class ErrorSample:
    """Muestra de error para análisis."""

    index: int
    image_id: str | int
    predicted_attributes: dict[str, float]
    true_attributes: dict[str, int]
    mismatched_attributes: list[str]
    confidence: float


@dataclass
class EvaluationReport:
    """Reporte completo de evaluación."""

    metrics: EvaluationMetrics
    error_samples: list[ErrorSample]
    total_samples: int
    total_errors: int
    error_rate: float
    threshold: float


class Evaluator:
    """Evaluador de modelos de atributos faciales."""

    def __init__(self, attribute_names: list[str] | None = None) -> None:
        """Inicializar evaluador.

        Args:
            attribute_names: Nombres de los atributos.
        """
        self.attribute_names = attribute_names or []
        self._metrics_calculator = MetricsCalculator(attribute_names)

    def evaluate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        image_ids: list[str | int] | None = None,
        threshold: float = 0.5,
        num_error_samples: int = 10,
    ) -> EvaluationReport:
        """Evaluar predicciones.

        Args:
            predictions: Predicciones del modelo [num_samples, num_attributes].
            targets: Valores reales [num_samples, num_attributes].
            image_ids: IDs de las imágenes.
            threshold: Umbral para binarización.
            num_error_samples: Número de muestras de error a guardar.

        Returns:
            Reporte de evaluación completo.
        """
        metrics = self._metrics_calculator.calculate(predictions, targets, threshold)

        error_samples = self._analyze_errors(
            predictions, targets, image_ids, threshold, num_error_samples
        )

        total_errors = sum(1 for e in error_samples)
        error_rate = total_errors / len(predictions) if len(predictions) > 0 else 0.0

        return EvaluationReport(
            metrics=metrics,
            error_samples=error_samples,
            total_samples=len(predictions),
            total_errors=total_errors,
            error_rate=error_rate,
            threshold=threshold,
        )

    def evaluate_with_optimized_thresholds(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        image_ids: list[str | int] | None = None,
        num_error_samples: int = 10,
    ) -> tuple[EvaluationReport, list[ThresholdResult]]:
        """Evaluar predicciones con thresholds optimizados por atributo.

        Args:
            predictions: Predicciones del modelo [num_samples, num_attributes].
            targets: Valores reales [num_samples, num_attributes].
            image_ids: IDs de las imágenes.
            num_error_samples: Número de muestras de error a guardar.

        Returns:
            Tupla de (reporte de evaluación, thresholds optimizados).
        """
        optimizer = ThresholdOptimizer()
        optimized_thresholds = optimizer.optimize(
            predictions, targets, self.attribute_names
        )

        pred_binary = optimizer.apply_thresholds(predictions, optimized_thresholds)

        metrics = self._metrics_calculator.calculate(pred_binary, targets, 0.5)

        error_samples = self._analyze_errors_with_binary(
            pred_binary, targets, image_ids, num_error_samples
        )

        total_errors = sum(1 for e in error_samples)
        error_rate = total_errors / len(predictions) if len(predictions) > 0 else 0.0

        report = EvaluationReport(
            metrics=metrics,
            error_samples=error_samples,
            total_samples=len(predictions),
            total_errors=total_errors,
            error_rate=error_rate,
            threshold=0.5,
        )

        return report, optimized_thresholds

    def _analyze_errors_with_binary(
        self,
        pred_binary: np.ndarray,
        targets: np.ndarray,
        image_ids: list[str | int] | None,
        num_samples: int,
    ) -> list[ErrorSample]:
        """Analizar errores con predicciones ya binarizadas.

        Args:
            pred_binary: Predicciones binarizadas.
            targets: Valores reales.
            image_ids: IDs de las imágenes.
            num_samples: Número de muestras a analizar.

        Returns:
            Lista de muestras de error.
        """
        errors_per_sample = []
        for i in range(len(pred_binary)):
            mismatches = []
            for j in range(pred_binary.shape[1]):
                if pred_binary[i, j] != targets[i, j]:
                    attr_name = (
                        self.attribute_names[j]
                        if j < len(self.attribute_names)
                        else f"attr_{j}"
                    )
                    mismatches.append(attr_name)

            if mismatches:
                image_id = image_ids[i] if image_ids else i
                confidence = float(pred_binary[i].mean())

                predicted_attrs = {}
                true_attrs = {}
                for j in range(pred_binary.shape[1]):
                    attr_name = (
                        self.attribute_names[j]
                        if j < len(self.attribute_names)
                        else f"attr_{j}"
                    )
                    predicted_attrs[attr_name] = float(pred_binary[i, j])
                    true_attrs[attr_name] = int(targets[i, j])

                errors_per_sample.append(
                    ErrorSample(
                        index=i,
                        image_id=image_id,
                        predicted_attributes=predicted_attrs,
                        true_attributes=true_attrs,
                        mismatched_attributes=mismatches,
                        confidence=confidence,
                    )
                )

        errors_per_sample.sort(key=lambda x: x.confidence, reverse=True)

        return errors_per_sample[:num_samples]

    def _analyze_errors(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        image_ids: list[str | int] | None,
        threshold: float,
        num_samples: int,
    ) -> list[ErrorSample]:
        """Analizar errores de predicción.

        Args:
            predictions: Predicciones del modelo.
            targets: Valores reales.
            image_ids: IDs de las imágenes.
            threshold: Umbral para binarización.
            num_samples: Número de muestras a analizar.

        Returns:
            Lista de muestras de error.
        """
        pred_binary = (predictions > threshold).astype(int)

        errors_per_sample = []
        for i in range(len(predictions)):
            mismatches = []
            for j in range(predictions.shape[1]):
                if pred_binary[i, j] != targets[i, j]:
                    attr_name = (
                        self.attribute_names[j]
                        if j < len(self.attribute_names)
                        else f"attr_{j}"
                    )
                    mismatches.append(attr_name)

            if mismatches:
                image_id = image_ids[i] if image_ids else i
                confidence = float(predictions[i].mean())

                predicted_attrs = {}
                true_attrs = {}
                for j in range(predictions.shape[1]):
                    attr_name = (
                        self.attribute_names[j]
                        if j < len(self.attribute_names)
                        else f"attr_{j}"
                    )
                    predicted_attrs[attr_name] = float(predictions[i, j])
                    true_attrs[attr_name] = int(targets[i, j])

                errors_per_sample.append(
                    ErrorSample(
                        index=i,
                        image_id=image_id,
                        predicted_attributes=predicted_attrs,
                        true_attributes=true_attrs,
                        mismatched_attributes=mismatches,
                        confidence=confidence,
                    )
                )

        errors_per_sample.sort(key=lambda x: x.confidence, reverse=True)

        return errors_per_sample[:num_samples]

    def get_attribute_summary(self, report: EvaluationReport) -> pd.DataFrame:
        """Obtener resumen por atributo como DataFrame.

        Args:
            report: Reporte de evaluación.

        Returns:
            DataFrame con resumen por atributo.
        """
        data = []
        for attr in report.metrics.per_attribute:
            data.append(
                {
                    "attribute": attr.name,
                    "accuracy": attr.accuracy,
                    "precision": attr.precision,
                    "recall": attr.recall,
                    "f1": attr.f1,
                    "support": attr.support,
                    "positive_rate": attr.positive_rate,
                    "prediction_rate": attr.prediction_rate,
                }
            )

        return pd.DataFrame(data)

    def save_report(self, report: EvaluationReport, output_path: Path) -> None:
        """Guardar reporte de evaluación.

        Args:
            report: Reporte de evaluación.
            output_path: Ruta de salida.
        """
        output_path.mkdir(parents=True, exist_ok=True)

        metrics_data = {
            "accuracy": report.metrics.accuracy,
            "precision": report.metrics.precision,
            "recall": report.metrics.recall,
            "f1": report.metrics.f1,
            "hamming": report.metrics.hamming,
            "average_precision": report.metrics.average_precision,
            "best_attributes": report.metrics.best_attributes,
            "worst_attributes": report.metrics.worst_attributes,
        }

        with open(output_path / "metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2)

        summary_df = self.get_attribute_summary(report)
        summary_df.to_csv(output_path / "attribute_summary.csv", index=False)

        error_data = [
            {
                "image_id": e.image_id,
                "mismatched_attributes": e.mismatched_attributes,
                "confidence": e.confidence,
            }
            for e in report.error_samples
        ]

        with open(output_path / "error_analysis.json", "w") as f:
            json.dump(error_data, f, indent=2)
