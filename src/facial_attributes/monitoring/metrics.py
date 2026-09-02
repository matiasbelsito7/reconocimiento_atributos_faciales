"""Rastreador de métricas de monitoreo."""

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class MetricPoint:
    """Punto de métrica individual."""

    timestamp: float
    value: float
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Estadísticas de latencia."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    last_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        """Promedio de latencia."""
        if self.count == 0:
            return 0.0
        return self.total_ms / self.count


class MetricsTracker:
    """Rastreador de métricas de monitoreo."""

    def __init__(self, window_size: int = 1000) -> None:
        """Inicializar rastreador de métricas.

        Args:
            window_size: Tamaño de ventana para métricas recientes.
        """
        self.window_size = window_size
        self._prediction_scores: dict[str, list[MetricPoint]] = defaultdict(list)
        self._prediction_volume: list[MetricPoint] = []
        self._latency_stats = LatencyStats()
        self._error_counts: dict[str, int] = defaultdict(int)
        self._total_predictions: int = 0
        self._start_time: float = time.time()

    def record_prediction(
        self,
        scores: dict[str, float],
        latency_ms: float,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Registrar una predicción.

        Args:
            scores: Scores de predicción por atributo.
            latency_ms: Latencia de la predicción en milisegundos.
            metadata: Metadata adicional de la predicción.
        """
        timestamp = time.time()
        meta = metadata or {}

        for attribute, score in scores.items():
            point = MetricPoint(timestamp=timestamp, value=score, metadata=meta)
            self._prediction_scores[attribute].append(point)

            if len(self._prediction_scores[attribute]) > self.window_size:
                self._prediction_scores[attribute].pop(0)

        volume_point = MetricPoint(timestamp=timestamp, value=1.0, metadata=meta)
        self._prediction_volume.append(volume_point)

        if len(self._prediction_volume) > self.window_size:
            self._prediction_volume.pop(0)

        self._latency_stats.count += 1
        self._latency_stats.total_ms += latency_ms
        self._latency_stats.min_ms = min(self._latency_stats.min_ms, latency_ms)
        self._latency_stats.max_ms = max(self._latency_stats.max_ms, latency_ms)
        self._latency_stats.last_ms = latency_ms

        self._total_predictions += 1

    def record_error(
        self, error_type: str, metadata: dict[str, str] | None = None
    ) -> None:
        """Registrar un error.

        Args:
            error_type: Tipo de error.
            metadata: Metadata adicional del error.
        """
        self._error_counts[error_type] += 1

    def get_prediction_distribution(
        self, attribute: str, last_n: int | None = None
    ) -> list[float]:
        """Obtener distribución de predicciones para un atributo.

        Args:
            attribute: Nombre del atributo.
            last_n: Últimos N puntos a retornar.

        Returns:
            Lista de scores.
        """
        points = self._prediction_scores.get(attribute, [])
        if last_n:
            points = points[-last_n:]
        return [p.value for p in points]

    def get_attribute_stats(self, attribute: str) -> dict[str, float]:
        """Obtener estadísticas para un atributo.

        Args:
            attribute: Nombre del atributo.

        Returns:
            Diccionario con estadísticas.
        """
        scores = self.get_prediction_distribution(attribute)
        if not scores:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = variance**0.5

        return {
            "mean": mean,
            "std": std,
            "min": min(scores),
            "max": max(scores),
            "count": len(scores),
        }

    def get_latency_stats(self) -> dict[str, float]:
        """Obtener estadísticas de latencia.

        Returns:
            Diccionario con estadísticas de latencia.
        """
        return {
            "avg_ms": self._latency_stats.avg_ms,
            "min_ms": (
                self._latency_stats.min_ms
                if self._latency_stats.min_ms != float("inf")
                else 0.0
            ),
            "max_ms": self._latency_stats.max_ms,
            "last_ms": self._latency_stats.last_ms,
            "count": self._latency_stats.count,
        }

    def get_error_stats(self) -> dict[str, int]:
        """Obtener estadísticas de errores.

        Returns:
            Diccionario con conteo de errores por tipo.
        """
        return dict(self._error_counts)

    def get_total_predictions(self) -> int:
        """Obtener total de predicciones registradas.

        Returns:
            Total de predicciones.
        """
        return self._total_predictions

    def get_uptime_seconds(self) -> float:
        """Obtener tiempo de actividad en segundos.

        Returns:
            Tiempo de actividad.
        """
        return time.time() - self._start_time

    def reset(self) -> None:
        """Reiniciar todas las métricas."""
        self._prediction_scores.clear()
        self._prediction_volume.clear()
        self._latency_stats = LatencyStats()
        self._error_counts.clear()
        self._total_predictions = 0
        self._start_time = time.time()

    def get_summary(self) -> dict[str, object]:
        """Obtener resumen de todas las métricas.

        Returns:
            Diccionario con resumen de métricas.
        """
        return {
            "total_predictions": self._total_predictions,
            "uptime_seconds": self.get_uptime_seconds(),
            "latency": self.get_latency_stats(),
            "errors": self.get_error_stats(),
            "attributes_tracked": list(self._prediction_scores.keys()),
        }
