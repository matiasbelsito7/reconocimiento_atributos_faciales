"""Funciones de pérdida para clasificación multilabel."""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LossConfig:
    """Configuración de la función de pérdida."""

    pos_weight: torch.Tensor | None = None
    reduction: str = "mean"


class MultilabelLoss(nn.Module):
    """Pérdida BCE para clasificación multilabel.

    Wrapper sobre BCEWithLogitsLoss con soporte para pesos de clase.
    """

    def __init__(self, config: LossConfig | None = None) -> None:
        super().__init__()
        self.config = config or LossConfig()
        self._loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=self.config.pos_weight,
            reduction=self.config.reduction,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calcular pérdida.

        Args:
            logits: Salida del modelo [batch_size, num_attributes].
            targets: Etiquetas binarias [batch_size, num_attributes].

        Returns:
            Pérdida escalar.
        """
        return self._loss_fn(logits, targets)

    def set_pos_weight(self, pos_weight: torch.Tensor) -> None:
        """Actualizar pesos de clase positiva."""
        self.config.pos_weight = pos_weight
        self._loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction=self.config.reduction,
        )
