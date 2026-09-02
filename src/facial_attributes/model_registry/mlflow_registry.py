"""Integración de MLflow con el Model Registry."""

import mlflow
import mlflow.pytorch
from mlflow.entities import Run

from facial_attributes.model_registry.schemas import (
    DatasetInfo,
    ModelConfig,
    ModelMetrics,
)


class MLflowRegistry:
    """Integración de MLflow como backend del Model Registry."""

    def __init__(
        self,
        experiment_name: str = "facial_attribute_recognition",
        tracking_uri: str | None = None,
    ) -> None:
        """Inicializar integración MLflow.

        Args:
            experiment_name: Nombre del experimento MLflow.
            tracking_uri: URI del tracking server MLflow.
        """
        self.experiment_name = experiment_name

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self._experiment = mlflow.set_experiment(experiment_name)

    def log_training_run(
        self,
        run_name: str,
        model_name: str,
        version: str,
        metrics: ModelMetrics,
        config: ModelConfig | None = None,
        dataset: DatasetInfo | None = None,
        model=None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Registrar un entrenamiento como run de MLflow.

        Args:
            run_name: Nombre del run.
            model_name: Nombre del modelo.
            version: Versión del modelo.
            metrics: Métricas de evaluación.
            config: Configuración del modelo.
            dataset: Información del dataset.
            model: Modelo PyTorch a guardar.
            tags: Tags adicionales.

        Returns:
            ID del run de MLflow.
        """
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("model_version", version)

            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)

            mlflow.log_metric("accuracy", metrics.accuracy)
            mlflow.log_metric("precision", metrics.precision)
            mlflow.log_metric("recall", metrics.recall)
            mlflow.log_metric("f1_score", metrics.f1_score)
            mlflow.log_metric("hamming_loss", metrics.hamming_loss)
            mlflow.log_metric("average_precision", metrics.average_precision)

            if metrics.per_attribute:
                for attr, value in metrics.per_attribute.items():
                    mlflow.log_metric(f"attr_{attr}", value)

            if config:
                mlflow.log_param("backbone", config.backbone)
                mlflow.log_param("num_attributes", config.num_attributes)
                mlflow.log_param("image_size", str(config.image_size))
                mlflow.log_param("learning_rate", config.learning_rate)
                mlflow.log_param("batch_size", config.batch_size)
                mlflow.log_param("epochs", config.epochs)
                mlflow.log_param("dropout", config.dropout)
                mlflow.log_param("optimizer", config.optimizer)
                mlflow.log_param("loss_function", config.loss_function)

            if dataset:
                mlflow.log_param("dataset_name", dataset.name)
                mlflow.log_param("dataset_version", dataset.version)
                mlflow.log_param("num_samples", dataset.num_samples)
                mlflow.log_param("num_attributes", dataset.num_attributes)

            if model is not None:
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="model",
                    registered_model_name=model_name,
                )

            return run.info.run_id

    def get_run(self, run_id: str) -> Run | None:
        """Obtener un run de MLflow.

        Args:
            run_id: ID del run.

        Returns:
            Run de MLflow o None si no existe.
        """
        try:
            return mlflow.get_run(run_id)
        except Exception:
            return None

    def get_model_versions(self, model_name: str) -> list[dict[str, str]]:
        """Obtener versiones de un modelo registrado en MLflow.

        Args:
            model_name: Nombre del modelo.

        Returns:
            Lista de versiones del modelo.
        """
        try:
            client = mlflow.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            return [
                {
                    "version": v.version,
                    "run_id": v.run_id,
                    "status": v.status,
                    "stage": v.current_stage,
                }
                for v in versions
            ]
        except Exception:
            return []

    def transition_model_version(
        self,
        model_name: str,
        version: str,
        stage: str,
    ) -> bool:
        """Transicionar versión de modelo a un stage.

        Args:
            model_name: Nombre del modelo.
            version: Versión del modelo.
            stage: Stage destino (Staging, Production, Archived).

        Returns:
            True si se transicionó correctamente.
        """
        try:
            client = mlflow.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
            )
            return True
        except Exception:
            return False

    def load_model(self, model_name: str, version: str | None = None):
        """Cargar modelo desde MLflow.

        Args:
            model_name: Nombre del modelo.
            version: Versión específica (None para Production).

        Returns:
            Modelo PyTorch cargado.
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/Production"

        return mlflow.pytorch.load_model(model_uri)

    def delete_model_version(self, model_name: str, version: str) -> bool:
        """Eliminar versión de modelo.

        Args:
            model_name: Nombre del modelo.
            version: Versión a eliminar.

        Returns:
            True si se eliminó correctamente.
        """
        try:
            client = mlflow.MlflowClient()
            client.delete_model_version(name=model_name, version=version)
            return True
        except Exception:
            return False

    def search_runs(
        self,
        model_name: str | None = None,
        max_results: int = 100,
    ) -> list[dict[str, str]]:
        """Buscar runs de MLflow.

        Args:
            model_name: Filtrar por nombre de modelo.
            max_results: Máximo de resultados.

        Returns:
            Lista de runs encontrados.
        """
        try:
            filter_str = ""
            if model_name:
                filter_str = f"tags.model_name = '{model_name}'"

            runs = mlflow.search_runs(
                experiment_ids=[self._experiment.experiment_id],
                filter_string=filter_str,
                max_results=max_results,
            )

            return [
                {
                    "run_id": row["run_id"],
                    "run_name": row.get("run_name", ""),
                    "status": row["status"],
                    "model_name": row.get("tags.model_name", ""),
                    "model_version": row.get("tags.model_version", ""),
                    "f1_score": str(row.get("metrics.f1_score", 0)),
                    "accuracy": str(row.get("metrics.accuracy", 0)),
                }
                for _, row in runs.iterrows()
            ]
        except Exception:
            return []

    def get_experiment_summary(self) -> dict[str, object]:
        """Obtener resumen del experimento MLflow.

        Returns:
            Diccionario con resumen del experimento.
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self._experiment.experiment_id],
                max_results=1000,
            )

            return {
                "experiment_name": self.experiment_name,
                "experiment_id": self._experiment.experiment_id,
                "total_runs": len(runs),
                "successful_runs": len(runs[runs["status"] == "FINISHED"]),
                "failed_runs": len(runs[runs["status"] == "FAILED"]),
            }
        except Exception:
            return {
                "experiment_name": self.experiment_name,
                "experiment_id": self._experiment.experiment_id,
                "total_runs": 0,
            }
