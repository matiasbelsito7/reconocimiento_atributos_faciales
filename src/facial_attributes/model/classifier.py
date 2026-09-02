"""Clasificador de atributos faciales usando ResNet."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.models as models


@dataclass
class ModelConfig:
    """Configuración del modelo."""

    num_attributes: int = 40
    backbone: str = "resnet18"
    pretrained: bool = True
    dropout_rate: float = 0.5
    freeze_backbone: bool = False


class FacialAttributeClassifier(nn.Module):
    """Clasificador multilabel de atributos faciales.

    Usa un backbone ResNet pre-entrenado con una capa de clasificación
    personalizada para predicción de atributos binarios independientes.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self._backbone = self._create_backbone()
        self._classifier = self._create_classifier()

        if self.config.freeze_backbone:
            self._freeze_backbone()

    def _create_backbone(self) -> nn.Module:
        """Crear backbone ResNet."""
        if self.config.backbone == "resnet18":
            weights = (
                models.ResNet18_Weights.IMAGENET1K_V1
                if self.config.pretrained
                else None
            )
            backbone = models.resnet18(weights=weights)
        elif self.config.backbone == "resnet34":
            weights = (
                models.ResNet34_Weights.IMAGENET1K_V1
                if self.config.pretrained
                else None
            )
            backbone = models.resnet34(weights=weights)
        elif self.config.backbone == "resnet50":
            weights = (
                models.ResNet50_Weights.IMAGENET1K_V1
                if self.config.pretrained
                else None
            )
            backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Backbone no soportado: {self.config.backbone}")

        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self._num_features = num_features
        return backbone

    def _create_classifier(self) -> nn.Module:
        """Crear capa de clasificación."""
        return nn.Sequential(
            nn.Dropout(p=self.config.dropout_rate),
            nn.Linear(self._num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=self.config.dropout_rate),
            nn.Linear(512, self.config.num_attributes),
        )

    def _freeze_backbone(self) -> None:
        """Congelar pesos del backbone."""
        for param in self._backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor de entrada [batch_size, 3, 224, 224].

        Returns:
            Tensor de scores [batch_size, num_attributes].
        """
        features = self._backbone(x)
        scores = self._classifier(features)
        return scores

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predecir probabilidades.

        Args:
            x: Tensor de entrada.

        Returns:
            Probabilidades sigmoid [0, 1].
        """
        scores = self.forward(x)
        return torch.sigmoid(scores)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predecir atributos binarios.

        Args:
            x: Tensor de entrada.
            threshold: Umbral para binarización.

        Returns:
            Predicciones binarias [0, 1].
        """
        probas = self.predict_proba(x)
        return (probas > threshold).float()

    def get_num_parameters(self) -> dict[str, int]:
        """Obtener conteo de parámetros."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
