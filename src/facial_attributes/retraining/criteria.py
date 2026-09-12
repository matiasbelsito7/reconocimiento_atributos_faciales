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
        comparison: dict[str, object] = {}

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


@dataclass
class PromotionDecision:
    """Decisión de promoción de un candidato a champion.

    Attributes:
        should_promote: Si el candidato debe reemplazar al champion actual.
        wins: Métricas en las que el candidato supera al champion.
        losses: Métricas en las que el candidato es superado por el champion.
        ties: Métricas con valor idéntico.
        total_metrics: Métricas totales consideradas.
        per_metric: Detalle por métrica.
        summary: Resumen legible de la decisión.
    """

    should_promote: bool
    wins: int = 0
    losses: int = 0
    ties: int = 0
    total_metrics: int = 0
    per_metric: dict[str, dict[str, object]] = field(default_factory=dict)
    summary: str = ""


class MajorityPromotionPolicy:
    """Política que promueve un candidato solo si mejora la mayoría de métricas.

    El champion solo se reemplaza cuando el candidato gana en más de la mitad
    de las métricas comparadas, evitando promociones por una sola métrica
    (ej: ganar F1 pero empeorar accuracy, precision y recall).
    """

    HIGHER_BETTER = ("accuracy", "precision", "recall", "f1_score", "average_precision")
    LOWER_BETTER = ("hamming_loss",)

    def __init__(self, compared_metrics: list[str] | None = None) -> None:
        """Inicializar política.

        Args:
            compared_metrics: Métricas a comparar. Por defecto todas las de
                ``ModelMetrics`` con dirección conocida.
        """
        self.compared_metrics = compared_metrics or [
            *self.HIGHER_BETTER,
            *self.LOWER_BETTER,
        ]

    def evaluate(
        self,
        candidate: ModelMetrics,
        champion: ModelMetrics,
    ) -> PromotionDecision:
        """Evaluar si el candidato debe reemplazar al champion.

        Args:
            candidate: Métricas del modelo candidato.
            champion: Métricas del modelo champion vigente.

        Returns:
            Decisión de promoción con detalle por métrica.
        """
        per_metric: dict[str, dict[str, object]] = {}
        wins = losses = ties = 0
        active = 0

        for metric_name in self.compared_metrics:
            cand_value = getattr(candidate, metric_name, 0.0)
            champ_value = getattr(champion, metric_name, 0.0)

            if cand_value != 0.0 or champ_value != 0.0:
                active += 1

            is_lower_better = metric_name in self.LOWER_BETTER
            if cand_value == champ_value:
                result = "tie"
                ties += 1
            elif (is_lower_better and cand_value < champ_value) or (
                not is_lower_better and cand_value > champ_value
            ):
                result = "win"
                wins += 1
            else:
                result = "loss"
                losses += 1

            per_metric[metric_name] = {
                "candidate": cand_value,
                "champion": champ_value,
                "result": result,
            }

        total = active
        should_promote = wins > total / 2

        if should_promote:
            summary = (
                f"Candidate wins {wins}/{total} metrics, promotes as new champion."
            )
        else:
            summary = f"Candidate wins {wins}/{total} metrics, keeps champion."

        return PromotionDecision(
            should_promote=should_promote,
            wins=wins,
            losses=losses,
            ties=ties,
            total_metrics=total,
            per_metric=per_metric,
            summary=summary,
        )
