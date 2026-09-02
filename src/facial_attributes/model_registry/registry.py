"""Registro de modelos."""

import json
import uuid
from dataclasses import asdict
from pathlib import Path

from facial_attributes.model_registry.schemas import (
    ArtifactInfo,
    ComparisonResult,
    DatasetInfo,
    ModelConfig,
    ModelMetadata,
    ModelMetrics,
    ModelState,
    ModelVersion,
)


class ModelRegistry:
    """Registro de modelos para almacenamiento y versionado."""

    def __init__(self, registry_dir: str | Path = "models/registry") -> None:
        """Inicializar registro de modelos.

        Args:
            registry_dir: Directorio del registro de modelos.
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self.registry_dir / "registry.json"
        self._models: dict[str, ModelMetadata] = {}
        self._versions: dict[str, list[ModelVersion]] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Cargar registro desde disco."""
        if self._metadata_file.exists():
            with open(self._metadata_file) as f:
                data = json.load(f)
                for model_id, model_data in data.get("models", {}).items():
                    metrics = ModelMetrics(**model_data.get("metrics", {}))
                    config = ModelConfig(**model_data.get("config", {}))
                    dataset = DatasetInfo(**model_data.get("dataset", {}))
                    artifacts = [
                        ArtifactInfo(**a) for a in model_data.get("artifacts", [])
                    ]
                    self._models[model_id] = ModelMetadata(
                        model_id=model_id,
                        name=model_data.get("name", ""),
                        version=model_data.get("version", ""),
                        state=ModelState(model_data.get("state", "development")),
                        created_at=model_data.get("created_at", ""),
                        updated_at=model_data.get("updated_at", ""),
                        description=model_data.get("description", ""),
                        metrics=metrics,
                        config=config,
                        dataset=dataset,
                        artifacts=artifacts,
                        tags=model_data.get("tags", {}),
                        parent_model_id=model_data.get("parent_model_id", ""),
                    )

                for model_id, versions_data in data.get("versions", {}).items():
                    self._versions[model_id] = []
                    for v_data in versions_data:
                        metrics = ModelMetrics(**v_data.get("metrics", {}))
                        self._versions[model_id].append(
                            ModelVersion(
                                version_id=v_data.get("version_id", ""),
                                model_id=model_id,
                                version=v_data.get("version", ""),
                                state=ModelState(v_data.get("state", "development")),
                                created_at=v_data.get("created_at", ""),
                                metrics=metrics,
                                artifact_path=v_data.get("artifact_path", ""),
                            )
                        )

    def _save_registry(self) -> None:
        """Guardar registro en disco."""
        data = {"models": {}, "versions": {}}

        for model_id, model in self._models.items():
            model_dict = asdict(model)
            model_dict["state"] = model.state.value
            data["models"][model_id] = model_dict

        for model_id, versions in self._versions.items():
            data["versions"][model_id] = []
            for v in versions:
                v_dict = asdict(v)
                v_dict["state"] = v.state.value
                data["versions"][model_id].append(v_dict)

        with open(self._metadata_file, "w") as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        name: str,
        version: str,
        metrics: ModelMetrics | None = None,
        config: ModelConfig | None = None,
        dataset: DatasetInfo | None = None,
        description: str = "",
        tags: dict[str, str] | None = None,
    ) -> str:
        """Registrar un nuevo modelo.

        Args:
            name: Nombre del modelo.
            version: Versión del modelo.
            metrics: Métricas de evaluación.
            config: Configuración del modelo.
            dataset: Información del dataset.
            description: Descripción del modelo.
            tags: Tags adicionales.

        Returns:
            ID del modelo registrado.
        """
        model_id = str(uuid.uuid4())

        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            state=ModelState.DEVELOPMENT,
            description=description,
            metrics=metrics or ModelMetrics(),
            config=config or ModelConfig(),
            dataset=dataset or DatasetInfo(),
            tags=tags or {},
        )

        self._models[model_id] = metadata

        model_version = ModelVersion(
            version_id=str(uuid.uuid4()),
            model_id=model_id,
            version=version,
            state=ModelState.DEVELOPMENT,
            metrics=metrics or ModelMetrics(),
        )
        self._versions[model_id] = [model_version]

        self._save_registry()
        return model_id

    def get_model(self, model_id: str) -> ModelMetadata | None:
        """Obtener metadata de un modelo.

        Args:
            model_id: ID del modelo.

        Returns:
            Metadata del modelo o None si no existe.
        """
        return self._models.get(model_id)

    def get_model_versions(self, model_id: str) -> list[ModelVersion]:
        """Obtener versiones de un modelo.

        Args:
            model_id: ID del modelo.

        Returns:
            Lista de versiones del modelo.
        """
        return self._versions.get(model_id, [])

    def update_model_state(self, model_id: str, new_state: ModelState) -> bool:
        """Actualizar estado de un modelo.

        Args:
            model_id: ID del modelo.
            new_state: Nuevo estado.

        Returns:
            True si se actualizó correctamente, False otherwise.
        """
        if model_id not in self._models:
            return False

        model = self._models[model_id]
        model.state = new_state
        model.updated_at = __import__("datetime").datetime.now().isoformat()

        if model_id in self._versions and self._versions[model_id]:
            self._versions[model_id][-1].state = new_state

        self._save_registry()
        return True

    def update_model_metrics(self, model_id: str, metrics: ModelMetrics) -> bool:
        """Actualizar métricas de un modelo.

        Args:
            model_id: ID del modelo.
            metrics: Nuevas métricas.

        Returns:
            True si se actualizó correctamente, False otherwise.
        """
        if model_id not in self._models:
            return False

        self._models[model_id].metrics = metrics
        self._models[model_id].updated_at = (
            __import__("datetime").datetime.now().isoformat()
        )

        if model_id in self._versions and self._versions[model_id]:
            self._versions[model_id][-1].metrics = metrics

        self._save_registry()
        return True

    def add_artifact(
        self,
        model_id: str,
        name: str,
        path: str,
        artifact_type: str = "model",
    ) -> bool:
        """Agregar un artefacto a un modelo.

        Args:
            model_id: ID del modelo.
            name: Nombre del artefacto.
            path: Ruta del artefacto.
            artifact_type: Tipo de artefacto.

        Returns:
            True si se agregó correctamente, False otherwise.
        """
        if model_id not in self._models:
            return False

        artifact = ArtifactInfo(
            name=name,
            path=path,
            artifact_type=artifact_type,
        )
        self._models[model_id].artifacts.append(artifact)
        self._models[model_id].updated_at = (
            __import__("datetime").datetime.now().isoformat()
        )

        self._save_registry()
        return True

    def compare_models(
        self,
        model_a_id: str,
        model_b_id: str,
    ) -> ComparisonResult | None:
        """Comparar dos modelos.

        Args:
            model_a_id: ID del primer modelo.
            model_b_id: ID del segundo modelo.

        Returns:
            Resultado de comparación o None si algún modelo no existe.
        """
        model_a = self._models.get(model_a_id)
        model_b = self._models.get(model_b_id)

        if not model_a or not model_b:
            return None

        metrics_comparison = {
            "accuracy": {
                "model_a": model_a.metrics.accuracy,
                "model_b": model_b.metrics.accuracy,
            },
            "precision": {
                "model_a": model_a.metrics.precision,
                "model_b": model_b.metrics.precision,
            },
            "recall": {
                "model_a": model_a.metrics.recall,
                "model_b": model_b.metrics.recall,
            },
            "f1_score": {
                "model_a": model_a.metrics.f1_score,
                "model_b": model_b.metrics.f1_score,
            },
            "hamming_loss": {
                "model_a": model_a.metrics.hamming_loss,
                "model_b": model_b.metrics.hamming_loss,
            },
            "average_precision": {
                "model_a": model_a.metrics.average_precision,
                "model_b": model_b.metrics.average_precision,
            },
        }

        improvement_percent = {}
        for metric, values in metrics_comparison.items():
            if values["model_a"] != 0:
                improvement = (
                    (values["model_b"] - values["model_a"]) / abs(values["model_a"])
                ) * 100
                improvement_percent[metric] = improvement

        if model_b.metrics.f1_score > model_a.metrics.f1_score:
            winner = model_b_id
        elif model_b.metrics.f1_score < model_a.metrics.f1_score:
            winner = model_a_id
        else:
            winner = "tie"

        return ComparisonResult(
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            metrics_comparison=metrics_comparison,
            winner=winner,
            improvement_percent=improvement_percent,
        )

    def promote_to_production(self, model_id: str) -> bool:
        """Promocionar modelo a producción.

        Args:
            model_id: ID del modelo.

        Returns:
            True si se promocionó correctamente, False otherwise.
        """
        if model_id not in self._models:
            return False

        for _mid, model in self._models.items():
            if model.state == ModelState.PRODUCTION:
                model.state = ModelState.ARCHIVED
                model.updated_at = __import__("datetime").datetime.now().isoformat()

        self._models[model_id].state = ModelState.PRODUCTION
        self._models[model_id].updated_at = (
            __import__("datetime").datetime.now().isoformat()
        )

        if model_id in self._versions and self._versions[model_id]:
            self._versions[model_id][-1].state = ModelState.PRODUCTION

        self._save_registry()
        return True

    def get_production_model(self) -> ModelMetadata | None:
        """Obtener modelo en producción.

        Returns:
            Metadata del modelo en producción o None.
        """
        for model in self._models.values():
            if model.state == ModelState.PRODUCTION:
                return model
        return None

    def list_models(
        self,
        state: ModelState | None = None,
        name: str | None = None,
    ) -> list[ModelMetadata]:
        """Listar modelos filtrados.

        Args:
            state: Filtrar por estado.
            name: Filtrar por nombre.

        Returns:
            Lista de modelos que coinciden con los filtros.
        """
        models = list(self._models.values())

        if state:
            models = [m for m in models if m.state == state]

        if name:
            models = [m for m in models if m.name == name]

        return models

    def delete_model(self, model_id: str) -> bool:
        """Eliminar un modelo.

        Args:
            model_id: ID del modelo.

        Returns:
            True si se eliminó correctamente, False otherwise.
        """
        if model_id not in self._models:
            return False

        del self._models[model_id]
        if model_id in self._versions:
            del self._versions[model_id]

        self._save_registry()
        return True

    def get_registry_summary(self) -> dict[str, object]:
        """Obtener resumen del registro.

        Returns:
            Diccionario con resumen del registro.
        """
        state_counts = {}
        for model in self._models.values():
            state = model.state.value
            state_counts[state] = state_counts.get(state, 0) + 1

        return {
            "total_models": len(self._models),
            "models_by_state": state_counts,
            "total_versions": sum(len(v) for v in self._versions.values()),
        }
