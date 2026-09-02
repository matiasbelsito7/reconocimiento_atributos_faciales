"""Tests para el módulo de registro de modelos."""

from pathlib import Path

from facial_attributes.model_registry.registry import ModelRegistry
from facial_attributes.model_registry.schemas import (
    ComparisonResult,
    DatasetInfo,
    ModelConfig,
    ModelMetadata,
    ModelMetrics,
    ModelState,
    ModelVersion,
)


class TestModelState:
    """Tests para ModelState."""

    def test_model_states(self) -> None:
        """Test de estados del modelo."""
        assert ModelState.DEVELOPMENT.value == "development"
        assert ModelState.STAGING.value == "staging"
        assert ModelState.PRODUCTION.value == "production"
        assert ModelState.ARCHIVED.value == "archived"


class TestModelMetrics:
    """Tests para ModelMetrics."""

    def test_metrics_default(self) -> None:
        """Test de métricas por defecto."""
        metrics = ModelMetrics()

        assert metrics.accuracy == 0.0
        assert metrics.f1_score == 0.0

    def test_metrics_custom(self) -> None:
        """Test de métricas personalizadas."""
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.93,
            recall=0.97,
            f1_score=0.95,
        )

        assert metrics.accuracy == 0.95
        assert metrics.f1_score == 0.95


class TestModelConfig:
    """Tests para ModelConfig."""

    def test_config_default(self) -> None:
        """Test de configuración por defecto."""
        config = ModelConfig()

        assert config.backbone == "resnet18"
        assert config.num_attributes == 40

    def test_config_custom(self) -> None:
        """Test de configuración personalizada."""
        config = ModelConfig(
            backbone="resnet50",
            learning_rate=0.0001,
            batch_size=64,
        )

        assert config.backbone == "resnet50"
        assert config.learning_rate == 0.0001


class TestDatasetInfo:
    """Tests para DatasetInfo."""

    def test_dataset_info_default(self) -> None:
        """Test de información de dataset por defecto."""
        dataset = DatasetInfo()

        assert dataset.name == ""
        assert dataset.num_samples == 0

    def test_dataset_info_custom(self) -> None:
        """Test de información de dataset personalizada."""
        dataset = DatasetInfo(
            name="CelebA",
            num_samples=202599,
            num_attributes=40,
        )

        assert dataset.name == "CelebA"
        assert dataset.num_samples == 202599


class TestModelMetadata:
    """Tests para ModelMetadata."""

    def test_metadata_default(self) -> None:
        """Test de metadata por defecto."""
        metadata = ModelMetadata()

        assert metadata.model_id == ""
        assert metadata.state == ModelState.DEVELOPMENT
        assert metadata.created_at != ""

    def test_metadata_custom(self) -> None:
        """Test de metadata personalizada."""
        metrics = ModelMetrics(accuracy=0.95)
        metadata = ModelMetadata(
            model_id="model_123",
            name="test_model",
            version="1.0.0",
            metrics=metrics,
        )

        assert metadata.model_id == "model_123"
        assert metadata.metrics.accuracy == 0.95


class TestModelVersion:
    """Tests para ModelVersion."""

    def test_version_default(self) -> None:
        """Test de versión por defecto."""
        version = ModelVersion()

        assert version.version_id == ""
        assert version.state == ModelState.DEVELOPMENT

    def test_version_custom(self) -> None:
        """Test de versión personalizada."""
        version = ModelVersion(
            version_id="v1",
            model_id="model_123",
            version="1.0.0",
            state=ModelState.STAGING,
        )

        assert version.version_id == "v1"
        assert version.state == ModelState.STAGING


class TestComparisonResult:
    """Tests para ComparisonResult."""

    def test_comparison_result_default(self) -> None:
        """Test de resultado de comparación por defecto."""
        result = ComparisonResult()

        assert result.model_a_id == ""
        assert result.winner == ""


class TestModelRegistry:
    """Tests para ModelRegistry."""

    def test_registry_initialization(self, tmp_path: Path) -> None:
        """Test de inicialización del registro."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        assert registry.registry_dir.exists()

    def test_register_model(self, tmp_path: Path) -> None:
        """Test de registro de modelo."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        model_id = registry.register_model(
            name="test_model",
            version="1.0.0",
            description="Test model",
        )

        assert model_id is not None
        model = registry.get_model(model_id)
        assert model is not None
        assert model.name == "test_model"

    def test_get_model(self, tmp_path: Path) -> None:
        """Test de obtención de modelo."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        model_id = registry.register_model(name="test_model", version="1.0.0")

        model = registry.get_model(model_id)

        assert model is not None
        assert model.model_id == model_id

    def test_get_model_nonexistent(self, tmp_path: Path) -> None:
        """Test de obtención de modelo inexistente."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        model = registry.get_model("nonexistent")

        assert model is None

    def test_update_model_state(self, tmp_path: Path) -> None:
        """Test de actualización de estado."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        model_id = registry.register_model(name="test_model", version="1.0.0")

        result = registry.update_model_state(model_id, ModelState.STAGING)

        assert result is True
        model = registry.get_model(model_id)
        assert model.state == ModelState.STAGING

    def test_update_model_state_nonexistent(self, tmp_path: Path) -> None:
        """Test de actualización de estado de modelo inexistente."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        result = registry.update_model_state("nonexistent", ModelState.STAGING)

        assert result is False

    def test_update_model_metrics(self, tmp_path: Path) -> None:
        """Test de actualización de métricas."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        model_id = registry.register_model(name="test_model", version="1.0.0")

        metrics = ModelMetrics(accuracy=0.95, f1_score=0.93)
        result = registry.update_model_metrics(model_id, metrics)

        assert result is True
        model = registry.get_model(model_id)
        assert model.metrics.accuracy == 0.95

    def test_add_artifact(self, tmp_path: Path) -> None:
        """Test de agregado de artefacto."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        model_id = registry.register_model(name="test_model", version="1.0.0")

        result = registry.add_artifact(
            model_id,
            name="model.pt",
            path="models/test_model.pt",
            artifact_type="model",
        )

        assert result is True
        model = registry.get_model(model_id)
        assert len(model.artifacts) == 1

    def test_compare_models(self, tmp_path: Path) -> None:
        """Test de comparación de modelos."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        model_a_id = registry.register_model(name="model_a", version="1.0.0")
        model_b_id = registry.register_model(name="model_b", version="1.0.0")

        metrics_a = ModelMetrics(accuracy=0.90, f1_score=0.88)
        metrics_b = ModelMetrics(accuracy=0.95, f1_score=0.93)

        registry.update_model_metrics(model_a_id, metrics_a)
        registry.update_model_metrics(model_b_id, metrics_b)

        result = registry.compare_models(model_a_id, model_b_id)

        assert result is not None
        assert result.winner == model_b_id
        assert result.metrics_comparison["accuracy"]["model_a"] == 0.90
        assert result.metrics_comparison["accuracy"]["model_b"] == 0.95

    def test_compare_models_nonexistent(self, tmp_path: Path) -> None:
        """Test de comparación con modelo inexistente."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        result = registry.compare_models("nonexistent_a", "nonexistent_b")

        assert result is None

    def test_promote_to_production(self, tmp_path: Path) -> None:
        """Test de promoción a producción."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        model_id = registry.register_model(name="test_model", version="1.0.0")

        result = registry.promote_to_production(model_id)

        assert result is True
        model = registry.get_model(model_id)
        assert model.state == ModelState.PRODUCTION

    def test_promote_to_production_archives_previous(self, tmp_path: Path) -> None:
        """Test de que promoción archiva modelo anterior."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        model_a_id = registry.register_model(name="model_a", version="1.0.0")
        registry.promote_to_production(model_a_id)

        model_b_id = registry.register_model(name="model_b", version="2.0.0")
        registry.promote_to_production(model_b_id)

        model_a = registry.get_model(model_a_id)
        model_b = registry.get_model(model_b_id)

        assert model_a.state == ModelState.ARCHIVED
        assert model_b.state == ModelState.PRODUCTION

    def test_get_production_model(self, tmp_path: Path) -> None:
        """Test de obtención de modelo en producción."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        model_id = registry.register_model(name="test_model", version="1.0.0")
        registry.promote_to_production(model_id)

        prod_model = registry.get_production_model()

        assert prod_model is not None
        assert prod_model.state == ModelState.PRODUCTION

    def test_get_production_model_none(self, tmp_path: Path) -> None:
        """Test de obtención de modelo en producción cuando no hay."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        prod_model = registry.get_production_model()

        assert prod_model is None

    def test_list_models(self, tmp_path: Path) -> None:
        """Test de listado de modelos."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        registry.register_model(name="model_a", version="1.0.0")
        registry.register_model(name="model_b", version="1.0.0")

        models = registry.list_models()

        assert len(models) == 2

    def test_list_models_by_state(self, tmp_path: Path) -> None:
        """Test de listado de modelos por estado."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        model_a_id = registry.register_model(name="model_a", version="1.0.0")
        registry.register_model(name="model_b", version="1.0.0")

        registry.update_model_state(model_a_id, ModelState.STAGING)

        staging_models = registry.list_models(state=ModelState.STAGING)

        assert len(staging_models) == 1
        assert staging_models[0].name == "model_a"

    def test_delete_model(self, tmp_path: Path) -> None:
        """Test de eliminación de modelo."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        model_id = registry.register_model(name="test_model", version="1.0.0")

        result = registry.delete_model(model_id)

        assert result is True
        assert registry.get_model(model_id) is None

    def test_delete_model_nonexistent(self, tmp_path: Path) -> None:
        """Test de eliminación de modelo inexistente."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        result = registry.delete_model("nonexistent")

        assert result is False

    def test_get_registry_summary(self, tmp_path: Path) -> None:
        """Test de obtención de resumen."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        registry.register_model(name="model_a", version="1.0.0")
        registry.register_model(name="model_b", version="1.0.0")

        summary = registry.get_registry_summary()

        assert summary["total_models"] == 2
        assert "development" in summary["models_by_state"]

    def test_persistence(self, tmp_path: Path) -> None:
        """Test de persistencia del registro."""
        registry_dir = tmp_path / "registry"

        registry1 = ModelRegistry(registry_dir=registry_dir)
        model_id = registry1.register_model(name="test_model", version="1.0.0")

        registry2 = ModelRegistry(registry_dir=registry_dir)

        model = registry2.get_model(model_id)
        assert model is not None
        assert model.name == "test_model"
