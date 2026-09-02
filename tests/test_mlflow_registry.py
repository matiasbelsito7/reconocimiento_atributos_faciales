"""Tests para la integración de MLflow con Model Registry."""

from unittest.mock import MagicMock, patch

import pytest

from facial_attributes.model_registry.mlflow_registry import MLflowRegistry
from facial_attributes.model_registry.schemas import (
    DatasetInfo,
    ModelConfig,
    ModelMetrics,
)


@pytest.fixture
def mock_mlflow():
    """Mock de MLflow para tests."""
    with patch("facial_attributes.model_registry.mlflow_registry.mlflow") as mock:
        mock.set_tracking_uri = MagicMock()
        mock.set_experiment = MagicMock()
        mock.start_run = MagicMock()
        mock.log_metric = MagicMock()
        mock.log_param = MagicMock()
        mock.set_tag = MagicMock()
        mock.search_runs = MagicMock(return_value=MagicMock(iterrows=lambda: iter([])))
        yield mock


@pytest.fixture
def sample_metrics() -> ModelMetrics:
    """Métricas de ejemplo."""
    return ModelMetrics(
        accuracy=0.95,
        precision=0.93,
        recall=0.91,
        f1_score=0.92,
        hamming_loss=0.05,
        average_precision=0.88,
    )


@pytest.fixture
def sample_config() -> ModelConfig:
    """Configuración de ejemplo."""
    return ModelConfig(
        backbone="resnet18",
        num_attributes=40,
        image_size=[224, 224],
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
    )


@pytest.fixture
def sample_dataset() -> DatasetInfo:
    """Dataset de ejemplo."""
    return DatasetInfo(
        name="celeba",
        version="1.0",
        num_samples=202599,
        num_attributes=40,
    )


class TestMLflowRegistry:
    """Tests para MLflowRegistry."""

    def test_initialization(self, mock_mlflow) -> None:
        """Test de inicialización del registry MLflow."""
        registry = MLflowRegistry(experiment_name="test_experiment")

        assert registry.experiment_name == "test_experiment"
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")

    def test_initialization_with_tracking_uri(self, mock_mlflow) -> None:
        """Test de inicialización con tracking URI."""
        MLflowRegistry(
            experiment_name="test_experiment",
            tracking_uri="http://localhost:5000",
        )

        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")

    def test_log_training_run(
        self,
        mock_mlflow,
        sample_metrics,
        sample_config,
        sample_dataset,
    ) -> None:
        """Test de logging de entrenamiento."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        registry = MLflowRegistry()

        run_id = registry.log_training_run(
            run_name="test_run",
            model_name="test_model",
            version="1.0.0",
            metrics=sample_metrics,
            config=sample_config,
            dataset=sample_dataset,
        )

        assert run_id == "test_run_id"

    def test_log_training_run_with_model(
        self,
        mock_mlflow,
        sample_metrics,
    ) -> None:
        """Test de logging con modelo PyTorch."""
        mock_model = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        registry = MLflowRegistry()

        run_id = registry.log_training_run(
            run_name="test_run",
            model_name="test_model",
            version="1.0.0",
            metrics=sample_metrics,
            model=mock_model,
        )

        assert run_id == "test_run_id"
        mock_mlflow.pytorch.log_model.assert_called_once()

    def test_log_training_run_with_tags(
        self,
        mock_mlflow,
        sample_metrics,
    ) -> None:
        """Test de logging con tags."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        registry = MLflowRegistry()

        run_id = registry.log_training_run(
            run_name="test_run",
            model_name="test_model",
            version="1.0.0",
            metrics=sample_metrics,
            tags={"experiment": "test", "author": "test_user"},
        )

        assert run_id == "test_run_id"

    def test_log_training_run_logs_metrics(
        self,
        mock_mlflow,
        sample_metrics,
    ) -> None:
        """Test de que se loguean todas las métricas."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        registry = MLflowRegistry()

        registry.log_training_run(
            run_name="test_run",
            model_name="test_model",
            version="1.0.0",
            metrics=sample_metrics,
        )

        mock_mlflow.log_metric.assert_any_call("accuracy", 0.95)
        mock_mlflow.log_metric.assert_any_call("precision", 0.93)
        mock_mlflow.log_metric.assert_any_call("recall", 0.91)
        mock_mlflow.log_metric.assert_any_call("f1_score", 0.92)
        mock_mlflow.log_metric.assert_any_call("hamming_loss", 0.05)
        mock_mlflow.log_metric.assert_any_call("average_precision", 0.88)

    def test_log_training_run_logs_config(
        self,
        mock_mlflow,
        sample_metrics,
        sample_config,
    ) -> None:
        """Test de que se loguea la configuración."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        registry = MLflowRegistry()

        registry.log_training_run(
            run_name="test_run",
            model_name="test_model",
            version="1.0.0",
            metrics=sample_metrics,
            config=sample_config,
        )

        mock_mlflow.log_param.assert_any_call("backbone", "resnet18")
        mock_mlflow.log_param.assert_any_call("num_attributes", 40)
        mock_mlflow.log_param.assert_any_call("learning_rate", 0.001)
        mock_mlflow.log_param.assert_any_call("batch_size", 32)

    def test_log_training_run_logs_dataset(
        self,
        mock_mlflow,
        sample_metrics,
        sample_dataset,
    ) -> None:
        """Test de que se loguea la información del dataset."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        registry = MLflowRegistry()

        registry.log_training_run(
            run_name="test_run",
            model_name="test_model",
            version="1.0.0",
            metrics=sample_metrics,
            dataset=sample_dataset,
        )

        mock_mlflow.log_param.assert_any_call("dataset_name", "celeba")
        mock_mlflow.log_param.assert_any_call("dataset_version", "1.0")
        mock_mlflow.log_param.assert_any_call("num_samples", 202599)

    def test_get_run(self, mock_mlflow) -> None:
        """Test de obtención de run."""
        mock_run = MagicMock()
        mock_mlflow.get_run.return_value = mock_run

        registry = MLflowRegistry()
        run = registry.get_run("test_run_id")

        assert run == mock_run
        mock_mlflow.get_run.assert_called_once_with("test_run_id")

    def test_get_run_not_found(self, mock_mlflow) -> None:
        """Test de obtención de run no encontrado."""
        mock_mlflow.get_run.side_effect = Exception("Run not found")

        registry = MLflowRegistry()
        run = registry.get_run("nonexistent_run_id")

        assert run is None

    def test_get_model_versions(self, mock_mlflow) -> None:
        """Test de obtención de versiones de modelo."""
        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_version.run_id = "run_1"
        mock_version.status = "READY"
        mock_version.current_stage = "Production"
        mock_client.search_model_versions.return_value = [mock_version]
        mock_mlflow.MlflowClient.return_value = mock_client

        registry = MLflowRegistry()
        versions = registry.get_model_versions("test_model")

        assert len(versions) == 1
        assert versions[0]["version"] == "1"
        assert versions[0]["status"] == "READY"

    def test_transition_model_version(self, mock_mlflow) -> None:
        """Test de transición de versión de modelo."""
        mock_client = MagicMock()
        mock_mlflow.MlflowClient.return_value = mock_client

        registry = MLflowRegistry()
        result = registry.transition_model_version(
            model_name="test_model",
            version="1",
            stage="Production",
        )

        assert result is True
        mock_client.transition_model_version_stage.assert_called_once_with(
            name="test_model",
            version="1",
            stage="Production",
        )

    def test_transition_model_version_error(self, mock_mlflow) -> None:
        """Test de transición con error."""
        mock_client = MagicMock()
        mock_client.transition_model_version_stage.side_effect = Exception("Error")
        mock_mlflow.MlflowClient.return_value = mock_client

        registry = MLflowRegistry()
        result = registry.transition_model_version(
            model_name="test_model",
            version="1",
            stage="Production",
        )

        assert result is False

    def test_delete_model_version(self, mock_mlflow) -> None:
        """Test de eliminación de versión de modelo."""
        mock_client = MagicMock()
        mock_mlflow.MlflowClient.return_value = mock_client

        registry = MLflowRegistry()
        result = registry.delete_model_version("test_model", "1")

        assert result is True
        mock_client.delete_model_version.assert_called_once_with(
            name="test_model",
            version="1",
        )

    def test_delete_model_version_error(self, mock_mlflow) -> None:
        """Test de eliminación con error."""
        mock_client = MagicMock()
        mock_client.delete_model_version.side_effect = Exception("Error")
        mock_mlflow.MlflowClient.return_value = mock_client

        registry = MLflowRegistry()
        result = registry.delete_model_version("test_model", "1")

        assert result is False

    def test_search_runs(self, mock_mlflow) -> None:
        """Test de búsqueda de runs."""
        import pandas as pd

        mock_runs = pd.DataFrame(
            {
                "run_id": ["run_1", "run_2"],
                "run_name": ["run1", "run2"],
                "status": ["FINISHED", "FINISHED"],
                "tags.model_name": ["model_1", "model_2"],
                "tags.model_version": ["1.0", "2.0"],
                "metrics.f1_score": [0.9, 0.95],
                "metrics.accuracy": [0.85, 0.9],
            }
        )
        mock_mlflow.search_runs.return_value = mock_runs

        registry = MLflowRegistry()
        runs = registry.search_runs(model_name="model_1")

        assert len(runs) == 2
        assert runs[0]["run_id"] == "run_1"

    def test_get_experiment_summary(self, mock_mlflow) -> None:
        """Test de obtención de resumen del experimento."""
        import pandas as pd

        mock_runs = pd.DataFrame(
            {
                "status": ["FINISHED", "FINISHED", "FAILED"],
            }
        )
        mock_mlflow.search_runs.return_value = mock_runs

        registry = MLflowRegistry()
        summary = registry.get_experiment_summary()

        assert summary["experiment_name"] == "facial_attribute_recognition"
        assert summary["total_runs"] == 3
        assert summary["successful_runs"] == 2
        assert summary["failed_runs"] == 1

    def test_get_experiment_summary_error(self, mock_mlflow) -> None:
        """Test de obtención de resumen con error."""
        mock_mlflow.search_runs.side_effect = Exception("Error")

        registry = MLflowRegistry()
        summary = registry.get_experiment_summary()

        assert summary["experiment_name"] == "facial_attribute_recognition"
        assert summary["total_runs"] == 0
