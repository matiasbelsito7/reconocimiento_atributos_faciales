"""Tests para el módulo de evaluación."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from facial_attributes.evaluation.evaluator import EvaluationReport, Evaluator
from facial_attributes.evaluation.metrics import (
    AttributeMetrics,
    EvaluationMetrics,
    MetricsCalculator,
)
from facial_attributes.evaluation.thresholds import ThresholdOptimizer, ThresholdResult


@pytest.fixture
def sample_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Crear predicciones y targets de ejemplo."""
    np.random.seed(42)
    num_samples = 100
    num_attributes = 5

    targets = np.random.randint(0, 2, (num_samples, num_attributes)).astype(float)
    predictions = np.random.rand(num_samples, num_attributes)

    return predictions, targets


@pytest.fixture
def sample_attribute_names() -> list[str]:
    """Crear nombres de atributos de ejemplo."""
    return ["smiling", "glasses", "hat", "beard", "mustache"]


class TestAttributeMetrics:
    """Tests para AttributeMetrics."""

    def test_attribute_metrics_creation(self) -> None:
        """Test de creación de AttributeMetrics."""
        metrics = AttributeMetrics(
            name="smiling",
            accuracy=0.9,
            precision=0.85,
            recall=0.88,
            f1=0.86,
            pr_auc=0.92,
            roc_auc=0.91,
            support=50,
            positive_rate=0.5,
            prediction_rate=0.52,
        )

        assert metrics.name == "smiling"
        assert metrics.accuracy == 0.9
        assert metrics.f1 == 0.86
        assert metrics.pr_auc == 0.92
        assert metrics.roc_auc == 0.91


class TestEvaluationMetrics:
    """Tests para EvaluationMetrics."""

    def test_evaluation_metrics_creation(self) -> None:
        """Test de creación de EvaluationMetrics."""
        metrics = EvaluationMetrics(
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1=0.85,
            macro_f1=0.84,
            hamming=0.15,
            average_precision=0.88,
            macro_roc_auc=0.89,
        )

        assert metrics.accuracy == 0.85
        assert metrics.f1 == 0.85
        assert metrics.macro_f1 == 0.84
        assert metrics.hamming == 0.15
        assert metrics.macro_roc_auc == 0.89


class TestMetricsCalculator:
    """Tests para MetricsCalculator."""

    def test_calculator_initialization(self) -> None:
        """Test de inicialización del calculador."""
        calculator = MetricsCalculator()

        assert calculator is not None

    def test_calculator_with_attribute_names(
        self, sample_attribute_names: list[str]
    ) -> None:
        """Test de calculador con nombres de atributos."""
        calculator = MetricsCalculator(sample_attribute_names)

        assert calculator.attribute_names == sample_attribute_names

    def test_calculate_metrics(
        self,
        sample_predictions: tuple[np.ndarray, np.ndarray],
        sample_attribute_names: list[str],
    ) -> None:
        """Test de cálculo de métricas."""
        predictions, targets = sample_predictions
        calculator = MetricsCalculator(sample_attribute_names)

        metrics = calculator.calculate(predictions, targets)

        assert isinstance(metrics, EvaluationMetrics)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.f1 <= 1.0
        assert 0.0 <= metrics.hamming <= 1.0
        assert 0.0 <= metrics.macro_f1 <= 1.0
        assert 0.0 <= metrics.macro_roc_auc <= 1.0

    def test_calculate_per_attribute(
        self,
        sample_predictions: tuple[np.ndarray, np.ndarray],
        sample_attribute_names: list[str],
    ) -> None:
        """Test de métricas por atributo."""
        predictions, targets = sample_predictions
        calculator = MetricsCalculator(sample_attribute_names)

        metrics = calculator.calculate(predictions, targets)

        assert len(metrics.per_attribute) == len(sample_attribute_names)
        for attr in metrics.per_attribute:
            assert isinstance(attr, AttributeMetrics)
            assert 0.0 <= attr.f1 <= 1.0
            assert 0.0 <= attr.pr_auc <= 1.0
            assert 0.0 <= attr.roc_auc <= 1.0

    def test_best_worst_attributes(
        self,
        sample_predictions: tuple[np.ndarray, np.ndarray],
        sample_attribute_names: list[str],
    ) -> None:
        """Test de identificación de mejores y peores atributos."""
        predictions, targets = sample_predictions
        calculator = MetricsCalculator(sample_attribute_names)

        metrics = calculator.calculate(predictions, targets)

        assert len(metrics.best_attributes) > 0
        assert len(metrics.worst_attributes) > 0


class TestThresholdOptimizer:
    """Tests para ThresholdOptimizer."""

    def test_optimizer_initialization(self) -> None:
        """Test de inicialización del optimizador."""
        optimizer = ThresholdOptimizer()

        assert optimizer is not None

    def test_optimize_thresholds(
        self,
        sample_predictions: tuple[np.ndarray, np.ndarray],
        sample_attribute_names: list[str],
    ) -> None:
        """Test de optimización de thresholds."""
        predictions, targets = sample_predictions
        optimizer = ThresholdOptimizer()

        results = optimizer.optimize(predictions, targets, sample_attribute_names)

        assert len(results) == len(sample_attribute_names)
        for result in results:
            assert isinstance(result, ThresholdResult)
            assert 0.0 <= result.threshold <= 1.0
            assert 0.0 <= result.f1_score <= 1.0

    def test_apply_thresholds(
        self,
        sample_predictions: tuple[np.ndarray, np.ndarray],
        sample_attribute_names: list[str],
    ) -> None:
        """Test de aplicación de thresholds."""
        predictions, targets = sample_predictions
        optimizer = ThresholdOptimizer()

        results = optimizer.optimize(predictions, targets, sample_attribute_names)
        pred_binary = optimizer.apply_thresholds(predictions, results)

        assert pred_binary.shape == predictions.shape
        assert set(np.unique(pred_binary)).issubset({0, 1})


class TestEvaluator:
    """Tests para Evaluator."""

    def test_evaluator_initialization(self) -> None:
        """Test de inicialización del evaluador."""
        evaluator = Evaluator()

        assert evaluator is not None

    def test_evaluator_with_attribute_names(
        self, sample_attribute_names: list[str]
    ) -> None:
        """Test de evaluador con nombres de atributos."""
        evaluator = Evaluator(sample_attribute_names)

        assert evaluator.attribute_names == sample_attribute_names

    def test_evaluate(
        self,
        sample_predictions: tuple[np.ndarray, np.ndarray],
        sample_attribute_names: list[str],
    ) -> None:
        """Test de evaluación completa."""
        predictions, targets = sample_predictions
        evaluator = Evaluator(sample_attribute_names)

        report = evaluator.evaluate(predictions, targets)

        assert isinstance(report, EvaluationReport)
        assert report.total_samples == len(predictions)
        assert 0.0 <= report.error_rate <= 1.0

    def test_evaluate_with_image_ids(
        self,
        sample_predictions: tuple[np.ndarray, np.ndarray],
        sample_attribute_names: list[str],
    ) -> None:
        """Test de evaluación con IDs de imagen."""
        predictions, targets = sample_predictions
        evaluator = Evaluator(sample_attribute_names)
        image_ids = [f"img_{i:04d}" for i in range(len(predictions))]

        report = evaluator.evaluate(predictions, targets, image_ids=image_ids)

        assert report.total_samples == len(predictions)

    def test_error_analysis(
        self,
        sample_predictions: tuple[np.ndarray, np.ndarray],
        sample_attribute_names: list[str],
    ) -> None:
        """Test de análisis de errores."""
        predictions, targets = sample_predictions
        evaluator = Evaluator(sample_attribute_names)

        report = evaluator.evaluate(predictions, targets, num_error_samples=5)

        assert len(report.error_samples) <= 5

    def test_get_attribute_summary(
        self,
        sample_predictions: tuple[np.ndarray, np.ndarray],
        sample_attribute_names: list[str],
    ) -> None:
        """Test de obtención de resumen por atributo."""
        predictions, targets = sample_predictions
        evaluator = Evaluator(sample_attribute_names)

        report = evaluator.evaluate(predictions, targets)
        summary = evaluator.get_attribute_summary(report)

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == len(sample_attribute_names)

    def test_save_report(
        self,
        sample_predictions: tuple[np.ndarray, np.ndarray],
        sample_attribute_names: list[str],
        tmp_path: Path,
    ) -> None:
        """Test de guardado de reporte."""
        predictions, targets = sample_predictions
        evaluator = Evaluator(sample_attribute_names)

        report = evaluator.evaluate(predictions, targets)
        evaluator.save_report(report, tmp_path / "report")

        assert (tmp_path / "report" / "metrics.json").exists()
        assert (tmp_path / "report" / "attribute_summary.csv").exists()
        assert (tmp_path / "report" / "error_analysis.json").exists()

    def test_evaluate_with_optimized_thresholds(
        self,
        sample_predictions: tuple[np.ndarray, np.ndarray],
        sample_attribute_names: list[str],
    ) -> None:
        """Test de evaluación con thresholds optimizados."""
        predictions, targets = sample_predictions
        evaluator = Evaluator(sample_attribute_names)

        report, thresholds = evaluator.evaluate_with_optimized_thresholds(
            predictions, targets
        )

        assert isinstance(report, EvaluationReport)
        assert len(thresholds) == len(sample_attribute_names)
        for threshold in thresholds:
            assert isinstance(threshold, ThresholdResult)
            assert 0.0 <= threshold.threshold <= 1.0
