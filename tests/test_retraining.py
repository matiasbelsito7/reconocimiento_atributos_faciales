"""Tests para el módulo de pipeline de reentrenamiento."""

from pathlib import Path

import pandas as pd
import pytest

from facial_attributes.model_registry.schemas import ModelMetrics
from facial_attributes.retraining.criteria import AcceptanceCriteria, CriteriaResult
from facial_attributes.retraining.merger import DatasetMerger, MergeResult
from facial_attributes.retraining.pipeline import (
    RetrainingConfig,
    RetrainingPipeline,
    RetrainingResult,
    RetrainingStep,
)


@pytest.fixture
def sample_existing_annotations(tmp_path: Path) -> Path:
    """Crear anotaciones existentes de ejemplo."""
    data = {
        "image_id": [f"img_{i:04d}.jpg" for i in range(100)],
        "smiling": [1 if i % 2 == 0 else 0 for i in range(100)],
        "glasses": [1 if i % 3 == 0 else 0 for i in range(100)],
        "male": [1 if i % 4 == 0 else 0 for i in range(100)],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "existing_annotations.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_new_annotations(tmp_path: Path) -> Path:
    """Crear nuevas anotaciones de ejemplo."""
    data = {
        "image_id": [f"new_{i:04d}.jpg" for i in range(50)],
        "smiling": [1 if i % 2 == 0 else 0 for i in range(50)],
        "glasses": [1 if i % 3 == 0 else 0 for i in range(50)],
        "male": [1 if i % 4 == 0 else 0 for i in range(50)],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "new_annotations.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_images_dir(tmp_path: Path) -> Path:
    """Crear directorio de imágenes de ejemplo."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    for i in range(10):
        img_path = images_dir / f"img_{i:04d}.jpg"
        img_path.write_bytes(b"fake image data")

    return images_dir


class TestMergeResult:
    """Tests para MergeResult."""

    def test_merge_result_success(self) -> None:
        """Test de MergeResult exitoso."""
        result = MergeResult(
            success=True,
            total_samples=150,
            samples_from_existing=100,
            samples_from_new=50,
        )

        assert result.success is True
        assert result.total_samples == 150

    def test_merge_result_failure(self) -> None:
        """Test de MergeResult fallido."""
        result = MergeResult(
            success=False,
            error="Merge failed",
        )

        assert result.success is False
        assert result.error == "Merge failed"


class TestDatasetMerger:
    """Tests para DatasetMerger."""

    def test_merger_initialization(self, tmp_path: Path) -> None:
        """Test de inicialización del combinador."""
        merger = DatasetMerger(output_dir=tmp_path / "output")

        assert merger.output_dir.exists()

    def test_merge_datasets(
        self,
        tmp_path: Path,
        sample_existing_annotations: Path,
        sample_new_annotations: Path,
        sample_images_dir: Path,
    ) -> None:
        """Test de combinación de datasets."""
        merger = DatasetMerger(output_dir=tmp_path / "output")

        result = merger.merge_datasets(
            existing_annotations_path=sample_existing_annotations,
            new_annotations_path=sample_new_annotations,
            existing_images_dir=sample_images_dir,
            new_images_dir=sample_images_dir,
        )

        assert result.success is True
        assert result.total_samples > 0

    def test_validate_new_data(
        self,
        tmp_path: Path,
        sample_new_annotations: Path,
        sample_images_dir: Path,
    ) -> None:
        """Test de validación de nuevos datos."""
        merger = DatasetMerger(output_dir=tmp_path / "output")

        result = merger.validate_new_data(
            annotations_path=sample_new_annotations,
            images_dir=sample_images_dir,
        )

        assert result["valid"] is True
        assert result["num_samples"] == 50

    def test_validate_new_data_missing_file(self, tmp_path: Path) -> None:
        """Test de validación con archivo faltante."""
        merger = DatasetMerger(output_dir=tmp_path / "output")

        result = merger.validate_new_data(
            annotations_path=tmp_path / "nonexistent.csv",
            images_dir=tmp_path / "images",
        )

        assert result["valid"] is False
        assert len(result["issues"]) > 0


class TestCriteriaResult:
    """Tests para CriteriaResult."""

    def test_criteria_result_passed(self) -> None:
        """Test de CriteriaResult que pasó."""
        result = CriteriaResult(
            passed=True,
            summary="All criteria passed",
        )

        assert result.passed is True

    def test_criteria_result_failed(self) -> None:
        """Test de CriteriaResult que falló."""
        result = CriteriaResult(
            passed=False,
            summary="F1 score too low",
        )

        assert result.passed is False


class TestAcceptanceCriteria:
    """Tests para AcceptanceCriteria."""

    def test_criteria_initialization(self) -> None:
        """Test de inicialización de criterios."""
        criteria = AcceptanceCriteria(min_f1_score=0.8)

        assert criteria.min_f1_score == 0.8

    def test_check_acceptance_passed(self) -> None:
        """Test de verificación de aceptación que pasa."""
        criteria = AcceptanceCriteria(min_f1_score=0.9)

        new_metrics = ModelMetrics(accuracy=0.95, f1_score=0.93)
        previous_metrics = ModelMetrics(accuracy=0.90, f1_score=0.88)

        result = criteria.check_acceptance(new_metrics, previous_metrics)

        assert result.passed is True

    def test_check_acceptance_failed_regression(self) -> None:
        """Test de verificación de aceptación que falla por regresión."""
        criteria = AcceptanceCriteria(max_regression_percent=5.0)

        new_metrics = ModelMetrics(accuracy=0.80, f1_score=0.75)
        previous_metrics = ModelMetrics(accuracy=0.90, f1_score=0.88)

        result = criteria.check_acceptance(new_metrics, previous_metrics)

        assert result.passed is False

    def test_check_acceptance_below_minimum(self) -> None:
        """Test de verificación que falla por mínimo no alcanzado."""
        criteria = AcceptanceCriteria(min_f1_score=0.95)

        new_metrics = ModelMetrics(accuracy=0.95, f1_score=0.90)
        previous_metrics = ModelMetrics(accuracy=0.90, f1_score=0.88)

        result = criteria.check_acceptance(new_metrics, previous_metrics)

        assert result.passed is False

    def test_compare_models(self) -> None:
        """Test de comparación de modelos."""
        criteria = AcceptanceCriteria()

        new_metrics = ModelMetrics(accuracy=0.95, f1_score=0.93)
        previous_metrics = ModelMetrics(accuracy=0.90, f1_score=0.88)

        comparison = criteria.compare_models(new_metrics, previous_metrics)

        assert "accuracy" in comparison
        assert "f1_score" in comparison
        assert comparison["accuracy"]["improved"] is True


class TestRetrainingConfig:
    """Tests para RetrainingConfig."""

    def test_config_default(self) -> None:
        """Test de configuración por defecto."""
        config = RetrainingConfig()

        assert config.min_f1_score == 0.0
        assert config.max_regression_percent == 5.0

    def test_config_custom(self) -> None:
        """Test de configuración personalizada."""
        config = RetrainingConfig(
            min_f1_score=0.9,
            max_regression_percent=3.0,
        )

        assert config.min_f1_score == 0.9
        assert config.max_regression_percent == 3.0


class TestRetrainingStep:
    """Tests para RetrainingStep."""

    def test_step_default(self) -> None:
        """Test de paso por defecto."""
        step = RetrainingStep(name="test_step")

        assert step.name == "test_step"
        assert step.status == "pending"

    def test_step_completed(self) -> None:
        """Test de paso completado."""
        step = RetrainingStep(
            name="test_step",
            status="completed",
            duration_seconds=1.5,
        )

        assert step.status == "completed"
        assert step.duration_seconds == 1.5


class TestRetrainingResult:
    """Tests para RetrainingResult."""

    def test_result_success(self) -> None:
        """Test de resultado exitoso."""
        result = RetrainingResult(
            success=True,
            model_id="model_123",
        )

        assert result.success is True
        assert result.model_id == "model_123"

    def test_result_failure(self) -> None:
        """Test de resultado fallido."""
        result = RetrainingResult(
            success=False,
            error="Pipeline failed",
        )

        assert result.success is False
        assert result.error == "Pipeline failed"


class TestRetrainingPipeline:
    """Tests para RetrainingPipeline."""

    def test_pipeline_initialization(self, tmp_path: Path) -> None:
        """Test de inicialización del pipeline."""
        config = RetrainingConfig(
            output_dir=str(tmp_path / "output"),
            model_registry_dir=str(tmp_path / "registry"),
        )
        pipeline = RetrainingPipeline(config=config)

        assert pipeline is not None

    def test_pipeline_get_summary(self, tmp_path: Path) -> None:
        """Test de obtención de resumen del pipeline."""
        config = RetrainingConfig(
            output_dir=str(tmp_path / "output"),
            model_registry_dir=str(tmp_path / "registry"),
        )
        pipeline = RetrainingPipeline(config=config)

        summary = pipeline.get_pipeline_summary()

        assert "config" in summary
        assert "steps" in summary
        assert len(summary["steps"]) == 5

    def test_pipeline_validate_data(
        self,
        tmp_path: Path,
        sample_new_annotations: Path,
        sample_images_dir: Path,
    ) -> None:
        """Test de validación de datos en el pipeline."""
        config = RetrainingConfig(
            output_dir=str(tmp_path / "output"),
            model_registry_dir=str(tmp_path / "registry"),
        )
        pipeline = RetrainingPipeline(config=config)

        step = pipeline._step_validate_data(sample_new_annotations, sample_images_dir)

        assert step.status == "completed"

    def test_pipeline_merge_datasets(
        self,
        tmp_path: Path,
        sample_existing_annotations: Path,
        sample_new_annotations: Path,
        sample_images_dir: Path,
    ) -> None:
        """Test de combinación de datasets en el pipeline."""
        config = RetrainingConfig(
            existing_annotations_path=str(sample_existing_annotations),
            existing_images_dir=str(sample_images_dir),
            output_dir=str(tmp_path / "output"),
            model_registry_dir=str(tmp_path / "registry"),
        )
        pipeline = RetrainingPipeline(config=config)

        step = pipeline._step_merge_datasets(sample_new_annotations, sample_images_dir)

        assert step.status == "completed"
