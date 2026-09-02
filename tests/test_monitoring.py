"""Tests para el módulo de monitoreo."""

from pathlib import Path

import pytest

from facial_attributes.monitoring.alerts import (
    Alert,
    AlertManager,
    AlertSeverity,
    AlertThresholds,
    AlertType,
)
from facial_attributes.monitoring.logger import PredictionLogger, PredictionRecord
from facial_attributes.monitoring.metrics import (
    LatencyStats,
    MetricPoint,
    MetricsTracker,
)


class TestMetricPoint:
    """Tests para MetricPoint."""

    def test_metric_point_creation(self) -> None:
        """Test de creación de MetricPoint."""
        point = MetricPoint(timestamp=1000.0, value=0.95)

        assert point.timestamp == 1000.0
        assert point.value == 0.95
        assert point.metadata == {}

    def test_metric_point_with_metadata(self) -> None:
        """Test de creación de MetricPoint con metadata."""
        point = MetricPoint(
            timestamp=1000.0,
            value=0.95,
            metadata={"attribute": "smiling"},
        )

        assert point.metadata["attribute"] == "smiling"


class TestLatencyStats:
    """Tests para LatencyStats."""

    def test_latency_stats_default(self) -> None:
        """Test de LatencyStats por defecto."""
        stats = LatencyStats()

        assert stats.count == 0
        assert stats.total_ms == 0.0
        assert stats.avg_ms == 0.0

    def test_latency_stats_avg(self) -> None:
        """Test de cálculo de promedio de latencia."""
        stats = LatencyStats(count=3, total_ms=300.0)

        assert stats.avg_ms == 100.0


class TestMetricsTracker:
    """Tests para MetricsTracker."""

    def test_tracker_initialization(self) -> None:
        """Test de inicialización del rastreador."""
        tracker = MetricsTracker()

        assert tracker.get_total_predictions() == 0

    def test_record_prediction(self) -> None:
        """Test de registro de predicción."""
        tracker = MetricsTracker()

        tracker.record_prediction(
            scores={"smiling": 0.9, "glasses": 0.1},
            latency_ms=50.0,
        )

        assert tracker.get_total_predictions() == 1

    def test_get_prediction_distribution(self) -> None:
        """Test de obtención de distribución de predicciones."""
        tracker = MetricsTracker()

        tracker.record_prediction(scores={"smiling": 0.9}, latency_ms=50.0)
        tracker.record_prediction(scores={"smiling": 0.8}, latency_ms=50.0)

        dist = tracker.get_prediction_distribution("smiling")

        assert len(dist) == 2
        assert dist[0] == 0.9
        assert dist[1] == 0.8

    def test_get_attribute_stats(self) -> None:
        """Test de obtención de estadísticas de atributo."""
        tracker = MetricsTracker()

        tracker.record_prediction(scores={"smiling": 0.9}, latency_ms=50.0)
        tracker.record_prediction(scores={"smiling": 0.8}, latency_ms=50.0)

        stats = tracker.get_attribute_stats("smiling")

        assert stats["count"] == 2
        assert stats["mean"] == pytest.approx(0.85)
        assert stats["min"] == 0.8
        assert stats["max"] == 0.9

    def test_get_latency_stats(self) -> None:
        """Test de obtención de estadísticas de latencia."""
        tracker = MetricsTracker()

        tracker.record_prediction(scores={}, latency_ms=50.0)
        tracker.record_prediction(scores={}, latency_ms=100.0)

        stats = tracker.get_latency_stats()

        assert stats["count"] == 2
        assert stats["avg_ms"] == pytest.approx(75.0)
        assert stats["min_ms"] == 50.0
        assert stats["max_ms"] == 100.0

    def test_record_error(self) -> None:
        """Test de registro de error."""
        tracker = MetricsTracker()

        tracker.record_error("model_error")
        tracker.record_error("model_error")
        tracker.record_error("timeout")

        errors = tracker.get_error_stats()

        assert errors["model_error"] == 2
        assert errors["timeout"] == 1

    def test_reset(self) -> None:
        """Test de reinicio de métricas."""
        tracker = MetricsTracker()

        tracker.record_prediction(scores={"smiling": 0.9}, latency_ms=50.0)
        tracker.record_error("error")

        tracker.reset()

        assert tracker.get_total_predictions() == 0
        assert tracker.get_error_stats() == {}

    def test_get_summary(self) -> None:
        """Test de obtención de resumen."""
        tracker = MetricsTracker()

        tracker.record_prediction(scores={"smiling": 0.9}, latency_ms=50.0)

        summary = tracker.get_summary()

        assert "total_predictions" in summary
        assert "uptime_seconds" in summary
        assert "latency" in summary
        assert "errors" in summary
        assert "attributes_tracked" in summary


class TestPredictionRecord:
    """Tests para PredictionRecord."""

    def test_record_creation(self) -> None:
        """Test de creación de PredictionRecord."""
        record = PredictionRecord(
            prediction_id="pred_123",
            timestamp=1000.0,
            image_id="image_001",
            num_faces=1,
            faces=[{"bbox": {"x": 10, "y": 20, "w": 100, "h": 150}}],
            latency_ms=50.0,
            model_version="1.0.0",
        )

        assert record.prediction_id == "pred_123"
        assert record.num_faces == 1
        assert record.model_version == "1.0.0"


class TestPredictionLogger:
    """Tests para PredictionLogger."""

    def test_logger_initialization(self, tmp_path: Path) -> None:
        """Test de inicialización del logger."""
        logger = PredictionLogger(log_dir=tmp_path / "logs")

        assert logger.log_dir.exists()

    def test_log_prediction(self, tmp_path: Path) -> None:
        """Test de registro de predicción."""
        logger = PredictionLogger(log_dir=tmp_path / "logs")

        prediction_id = logger.log_prediction(
            image_id="image_001",
            faces=[{"bbox": {"x": 10, "y": 20, "w": 100, "h": 150}}],
            latency_ms=50.0,
        )

        assert prediction_id is not None
        assert logger.get_prediction_count() == 1

    def test_get_predictions(self, tmp_path: Path) -> None:
        """Test de obtención de predicciones."""
        logger = PredictionLogger(log_dir=tmp_path / "logs")

        logger.log_prediction(
            image_id="image_001",
            faces=[],
            latency_ms=50.0,
        )

        predictions = logger.get_predictions()

        assert len(predictions) == 1
        assert predictions[0].image_id == "image_001"

    def test_clear_logs(self, tmp_path: Path) -> None:
        """Test de limpieza de logs."""
        logger = PredictionLogger(log_dir=tmp_path / "logs")

        logger.log_prediction(image_id="image_001", faces=[], latency_ms=50.0)

        count = logger.clear_logs()

        assert count == 1
        assert logger.get_prediction_count() == 0


class TestAlert:
    """Tests para Alert."""

    def test_alert_creation(self) -> None:
        """Test de creación de Alert."""
        alert = Alert(
            alert_id="alert_1",
            alert_type=AlertType.DISTRIBUTION_SHIFT,
            severity=AlertSeverity.WARNING,
            message="Test alert",
            timestamp=1000.0,
        )

        assert alert.alert_id == "alert_1"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.acknowledged is False


class TestAlertThresholds:
    """Tests para AlertThresholds."""

    def test_default_thresholds(self) -> None:
        """Test de umbrales por defecto."""
        thresholds = AlertThresholds()

        assert thresholds.distribution_shift_std == 0.2
        assert thresholds.error_rate_threshold == 0.1
        assert thresholds.latency_threshold_ms == 1000.0
        assert thresholds.volume_drop_percent == 50.0


class TestAlertManager:
    """Tests para AlertManager."""

    def test_manager_initialization(self) -> None:
        """Test de inicialización del gestor de alertas."""
        manager = AlertManager()

        assert manager.get_unacknowledged_alerts() == []

    def test_check_distribution_shift(self) -> None:
        """Test de verificación de cambio de distribución."""
        manager = AlertManager()

        alert = manager.check_distribution_shift(
            current_mean=0.9,
            baseline_mean=0.5,
            current_std=0.1,
            attribute="smiling",
        )

        assert alert is not None
        assert alert.alert_type == AlertType.DISTRIBUTION_SHIFT

    def test_check_distribution_shift_no_alert(self) -> None:
        """Test de verificación de distribución sin alerta."""
        manager = AlertManager()

        alert = manager.check_distribution_shift(
            current_mean=0.51,
            baseline_mean=0.5,
            current_std=0.1,
            attribute="smiling",
        )

        assert alert is None

    def test_check_error_rate(self) -> None:
        """Test de verificación de tasa de errores."""
        manager = AlertManager()

        alert = manager.check_error_rate(
            error_count=15,
            total_predictions=100,
        )

        assert alert is not None
        assert alert.alert_type == AlertType.ERROR_RATE

    def test_check_error_rate_no_alert(self) -> None:
        """Test de verificación de errores sin alerta."""
        manager = AlertManager()

        alert = manager.check_error_rate(
            error_count=5,
            total_predictions=100,
        )

        assert alert is None

    def test_check_latency(self) -> None:
        """Test de verificación de latencia."""
        manager = AlertManager()

        alert = manager.check_latency(
            current_latency_ms=1500.0,
            baseline_latency_ms=500.0,
        )

        assert alert is not None
        assert alert.alert_type == AlertType.LATENCY_DEGRADATION

    def test_check_volume_drop(self) -> None:
        """Test de verificación de caída de volumen."""
        manager = AlertManager()

        alert = manager.check_volume_drop(
            current_volume=30,
            baseline_volume=100,
        )

        assert alert is not None
        assert alert.alert_type == AlertType.VOLUME_DROP

    def test_suppress_alert_type(self) -> None:
        """Test de supresión de tipo de alerta."""
        manager = AlertManager()
        manager.suppress_alert_type(AlertType.DISTRIBUTION_SHIFT)

        alert = manager.check_distribution_shift(
            current_mean=0.9,
            baseline_mean=0.5,
            current_std=0.1,
            attribute="smiling",
        )

        assert alert is None

    def test_acknowledge_alert(self) -> None:
        """Test de reconocimiento de alerta."""
        manager = AlertManager()

        alert = manager.check_error_rate(error_count=15, total_predictions=100)

        assert manager.acknowledge_alert(alert.alert_id) is True
        assert alert.acknowledged is True

    def test_get_alerts_filtered(self) -> None:
        """Test de obtención de alertas filtradas."""
        manager = AlertManager()

        manager.check_error_rate(error_count=15, total_predictions=100)
        manager.check_latency(current_latency_ms=1500.0, baseline_latency_ms=500.0)

        error_alerts = manager.get_alerts(alert_type=AlertType.ERROR_RATE)

        assert len(error_alerts) == 1

    def test_clear_alerts(self) -> None:
        """Test de limpieza de alertas."""
        manager = AlertManager()

        manager.check_error_rate(error_count=15, total_predictions=100)

        count = manager.clear_alerts()

        assert count == 1
        assert manager.get_unacknowledged_alerts() == []
