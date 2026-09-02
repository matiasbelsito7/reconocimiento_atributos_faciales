"""Gestionador de alertas de monitoreo."""

import time
from dataclasses import dataclass, field
from enum import Enum


class AlertSeverity(Enum):
    """Severidad de alerta."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Tipo de alerta."""

    DISTRIBUTION_SHIFT = "distribution_shift"
    ERROR_RATE = "error_rate"
    LATENCY_DEGRADATION = "latency_degradation"
    VOLUME_DROP = "volume_drop"


@dataclass
class Alert:
    """Alerta de monitoreo."""

    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: float
    metadata: dict[str, str] = field(default_factory=dict)
    acknowledged: bool = False


@dataclass
class AlertThresholds:
    """Umbrales para alertas."""

    distribution_shift_std: float = 0.2
    error_rate_threshold: float = 0.1
    latency_threshold_ms: float = 1000.0
    volume_drop_percent: float = 50.0


class AlertManager:
    """Gestionador de alertas de monitoreo."""

    def __init__(self, thresholds: AlertThresholds | None = None) -> None:
        """Inicializar gestionador de alertas.

        Args:
            thresholds: Umbrales para alertas.
        """
        self.thresholds = thresholds or AlertThresholds()
        self._alerts: list[Alert] = []
        self._alert_counter: int = 0
        self._suppressed_types: set[AlertType] = set()

    def check_distribution_shift(
        self,
        current_mean: float,
        baseline_mean: float,
        current_std: float,
        attribute: str,
    ) -> Alert | None:
        """Verificar cambio en distribución de predicciones.

        Args:
            current_mean: Media actual.
            baseline_mean: Media de referencia.
            current_std: Desviación estándar actual.
            attribute: Nombre del atributo.

        Returns:
            Alerta si se detecta cambio significativo, None otherwise.
        """
        if AlertType.DISTRIBUTION_SHIFT in self._suppressed_types:
            return None

        if current_std == 0:
            return None

        z_score = abs(current_mean - baseline_mean) / current_std

        if z_score > self.thresholds.distribution_shift_std * 3:
            alert = self._create_alert(
                alert_type=AlertType.DISTRIBUTION_SHIFT,
                severity=AlertSeverity.WARNING,
                message=f"Significant distribution shift detected for attribute '{attribute}'",
                metadata={
                    "attribute": attribute,
                    "current_mean": str(current_mean),
                    "baseline_mean": str(baseline_mean),
                    "z_score": str(z_score),
                },
            )
            return alert

        return None

    def check_error_rate(
        self,
        error_count: int,
        total_predictions: int,
    ) -> Alert | None:
        """Verificar tasa de errores.

        Args:
            error_count: Número de errores.
            total_predictions: Total de predicciones.

        Returns:
            Alerta si la tasa de errores es alta, None otherwise.
        """
        if AlertType.ERROR_RATE in self._suppressed_types:
            return None

        if total_predictions == 0:
            return None

        error_rate = error_count / total_predictions

        if error_rate > self.thresholds.error_rate_threshold:
            alert = self._create_alert(
                alert_type=AlertType.ERROR_RATE,
                severity=AlertSeverity.CRITICAL,
                message=f"High error rate detected: {error_rate:.2%}",
                metadata={
                    "error_count": str(error_count),
                    "total_predictions": str(total_predictions),
                    "error_rate": str(error_rate),
                },
            )
            return alert

        return None

    def check_latency(
        self,
        current_latency_ms: float,
        baseline_latency_ms: float,
    ) -> Alert | None:
        """Verificar degradación de latencia.

        Args:
            current_latency_ms: Latencia actual en ms.
            baseline_latency_ms: Latencia de referencia en ms.

        Returns:
            Alerta si la latencia es alta, None otherwise.
        """
        if AlertType.LATENCY_DEGRADATION in self._suppressed_types:
            return None

        if current_latency_ms > self.thresholds.latency_threshold_ms:
            alert = self._create_alert(
                alert_type=AlertType.LATENCY_DEGRADATION,
                severity=AlertSeverity.WARNING,
                message=f"High latency detected: {current_latency_ms:.2f}ms",
                metadata={
                    "current_latency_ms": str(current_latency_ms),
                    "baseline_latency_ms": str(baseline_latency_ms),
                },
            )
            return alert

        if baseline_latency_ms > 0:
            latency_increase = (
                current_latency_ms - baseline_latency_ms
            ) / baseline_latency_ms
            if latency_increase > 0.5:
                alert = self._create_alert(
                    alert_type=AlertType.LATENCY_DEGRADATION,
                    severity=AlertSeverity.WARNING,
                    message=f"Latency increased by {latency_increase:.2%}",
                    metadata={
                        "current_latency_ms": str(current_latency_ms),
                        "baseline_latency_ms": str(baseline_latency_ms),
                        "increase_percent": str(latency_increase),
                    },
                )
                return alert

        return None

    def check_volume_drop(
        self,
        current_volume: int,
        baseline_volume: int,
    ) -> Alert | None:
        """Verificar caída en volumen de predicciones.

        Args:
            current_volume: Volumen actual.
            baseline_volume: Volumen de referencia.

        Returns:
            Alerta si hay caída significativa, None otherwise.
        """
        if AlertType.VOLUME_DROP in self._suppressed_types:
            return None

        if baseline_volume == 0:
            return None

        drop_percent = ((baseline_volume - current_volume) / baseline_volume) * 100

        if drop_percent > self.thresholds.volume_drop_percent:
            alert = self._create_alert(
                alert_type=AlertType.VOLUME_DROP,
                severity=AlertSeverity.WARNING,
                message=f"Volume drop detected: {drop_percent:.1f}%",
                metadata={
                    "current_volume": str(current_volume),
                    "baseline_volume": str(baseline_volume),
                    "drop_percent": str(drop_percent),
                },
            )
            return alert

        return None

    def _create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        metadata: dict[str, str],
    ) -> Alert:
        """Crear una alerta.

        Args:
            alert_type: Tipo de alerta.
            severity: Severidad de la alerta.
            message: Mensaje de la alerta.
            metadata: Metadata adicional.

        Returns:
            Alerta creada.
        """
        self._alert_counter += 1
        alert = Alert(
            alert_id=f"alert_{self._alert_counter}",
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=time.time(),
            metadata=metadata,
        )
        self._alerts.append(alert)
        return alert

    def suppress_alert_type(self, alert_type: AlertType) -> None:
        """Suprimir un tipo de alerta.

        Args:
            alert_type: Tipo de alerta a suprimir.
        """
        self._suppressed_types.add(alert_type)

    def unsuppress_alert_type(self, alert_type: AlertType) -> None:
        """Remover supresión de un tipo de alerta.

        Args:
            alert_type: Tipo de alerta a des-superimir.
        """
        self._suppressed_types.discard(alert_type)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Reconocer una alerta.

        Args:
            alert_id: ID de la alerta.

        Returns:
            True si la alerta fue reconocida, False otherwise.
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_alerts(
        self,
        severity: AlertSeverity | None = None,
        alert_type: AlertType | None = None,
        acknowledged: bool | None = None,
    ) -> list[Alert]:
        """Obtener alertas filtradas.

        Args:
            severity: Filtrar por severidad.
            alert_type: Filtrar por tipo.
            acknowledged: Filtrar por estado de reconocimiento.

        Returns:
            Lista de alertas que coinciden con los filtros.
        """
        filtered = self._alerts

        if severity:
            filtered = [a for a in filtered if a.severity == severity]

        if alert_type:
            filtered = [a for a in filtered if a.alert_type == alert_type]

        if acknowledged is not None:
            filtered = [a for a in filtered if a.acknowledged == acknowledged]

        return filtered

    def get_unacknowledged_alerts(self) -> list[Alert]:
        """Obtener alertas no reconocidas.

        Returns:
            Lista de alertas no reconocidas.
        """
        return self.get_alerts(acknowledged=False)

    def clear_alerts(self) -> int:
        """Limpiar todas las alertas.

        Returns:
            Número de alertas eliminadas.
        """
        count = len(self._alerts)
        self._alerts.clear()
        return count
