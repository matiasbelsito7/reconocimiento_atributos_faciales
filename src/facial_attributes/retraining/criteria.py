"""Criterios de aceptación para reentrenamiento."""

from dataclasses import dataclass, field

from facial_attributes.model_registry.schemas import ModelMetrics


@dataclass
class CriteriaResult:
    """Resultado de verificación de criterios."""

    passed: bool
    details: dict[str, dict[str, object]] = field(default_factory=dict)
    summary: str = ""


class AcceptanceCriteria:
    """Criterios de aceptación para reentrenamiento."""

    def __init__(
        self,
        min_f1_score: float = 0.0,
        max_regression_percent: float = 5.0,
        required_metrics: list[str] | None = None,
    ) -> None:
        """Inicializar criterios de aceptación.

        Args:
            min_f1_score: F1 score mínimo aceptable.
            max_regression_percent: Máximo porcentaje de regresión permitido.
            required_metrics: Métricas requeridas para comparación.
        """
        self.min_f1_score = min_f1_score
        self.max_regression_percent = max_regression_percent
        self.required_metrics = required_metrics or [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "hamming_loss",
        ]

    def check_acceptance(
        self,
        new_metrics: ModelMetrics,
        previous_metrics: ModelMetrics,
    ) -> CriteriaResult:
        """Verificar si el nuevo modelo cumple los criterios de aceptación.

        Args:
            new_metrics: Métricas del nuevo modelo.
            previous_metrics: Métricas del modelo anterior.

        Returns:
            Resultado de la verificación.
        """
        details = {}
        all_passed = True

        if new_metrics.f1_score < self.min_f1_score:
            all_passed = False
            details["min_f1_score"] = {
                "passed": False,
                "required": self.min_f1_score,
                "actual": new_metrics.f1_score,
                "message": f"F1 score {new_metrics.f1_score:.4f} below minimum {self.min_f1_score:.4f}",
            }
        else:
            details["min_f1_score"] = {
                "passed": True,
                "required": self.min_f1_score,
                "actual": new_metrics.f1_score,
                "message": "F1 score meets minimum requirement",
            }

        for metric_name in self.required_metrics:
            new_value = getattr(new_metrics, metric_name, 0)
            prev_value = getattr(previous_metrics, metric_name, 0)

            is_lower_better = metric_name == "hamming_loss"

            if prev_value != 0:
                change_percent = ((new_value - prev_value) / abs(prev_value)) * 100
            else:
                change_percent = 0.0 if new_value == 0 else 100.0

            if is_lower_better:
                passed = (
                    change_percent <= self.max_regression_percent
                    or new_value <= prev_value
                )
            else:
                passed = (
                    change_percent >= -self.max_regression_percent
                    or new_value >= prev_value
                )

            if not passed:
                all_passed = False

            details[metric_name] = {
                "passed": passed,
                "previous": prev_value,
                "current": new_value,
                "change_percent": change_percent,
                "is_lower_better": is_lower_better,
                "message": f"{'OK' if passed else 'REGRESSION'}: {metric_name} changed {change_percent:+.2f}%",
            }

        summary = self._generate_summary(details, all_passed)

        return CriteriaResult(
            passed=all_passed,
            details=details,
            summary=summary,
        )

    def _generate_summary(
        self, details: dict[str, dict[str, object]], all_passed: bool
    ) -> str:
        """Generar resumen de la verificación.

        Args:
            details: Detalles de las métricas.
            all_passed: Si todas las métricas pasaron.

        Returns:
            Resumen de la verificación.
        """
        passed_count = sum(1 for d in details.values() if d["passed"])
        total_count = len(details)

        if all_passed:
            return (
                f"All {total_count} criteria passed. Model is ready for registration."
            )
        else:
            failed = [name for name, d in details.items() if not d["passed"]]
            return f"Failed {total_count - passed_count}/{total_count} criteria: {', '.join(failed)}"

    def compare_models(
        self,
        new_metrics: ModelMetrics,
        previous_metrics: ModelMetrics,
    ) -> dict[str, object]:
        """Comparar métricas de dos modelos.

        Args:
            new_metrics: Métricas del nuevo modelo.
            previous_metrics: Métricas del modelo anterior.

        Returns:
            Diccionario con comparación detallada.
        """
        comparison = {}

        for metric_name in self.required_metrics:
            new_value = getattr(new_metrics, metric_name, 0)
            prev_value = getattr(previous_metrics, metric_name, 0)

            if prev_value != 0:
                change_percent = ((new_value - prev_value) / abs(prev_value)) * 100
            else:
                change_percent = 0.0 if new_value == 0 else 100.0

            comparison[metric_name] = {
                "previous": prev_value,
                "current": new_value,
                "change": new_value - prev_value,
                "change_percent": change_percent,
                "improved": (
                    new_value > prev_value
                    if metric_name != "hamming_loss"
                    else new_value < prev_value
                ),
            }

        return comparison
