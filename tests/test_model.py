"""Tests para el módulo de modelo."""

import pytest
import torch

from facial_attributes.model.classifier import FacialAttributeClassifier, ModelConfig
from facial_attributes.model.losses import LossConfig, MultilabelLoss


class TestModelConfig:
    """Tests para ModelConfig."""

    def test_default_config(self) -> None:
        """Test de configuración por defecto."""
        config = ModelConfig()

        assert config.num_attributes == 40
        assert config.backbone == "resnet18"
        assert config.pretrained is True
        assert config.dropout_rate == 0.5

    def test_custom_config(self) -> None:
        """Test de configuración personalizada."""
        config = ModelConfig(
            num_attributes=20,
            backbone="resnet34",
            pretrained=False,
            dropout_rate=0.3,
        )

        assert config.num_attributes == 20
        assert config.backbone == "resnet34"
        assert config.pretrained is False
        assert config.dropout_rate == 0.3


class TestFacialAttributeClassifier:
    """Tests para FacialAttributeClassifier."""

    def test_model_creation(self) -> None:
        """Test de creación del modelo."""
        config = ModelConfig(pretrained=False)
        model = FacialAttributeClassifier(config)

        assert model is not None

    def test_model_forward(self) -> None:
        """Test de forward pass."""
        config = ModelConfig(pretrained=False, num_attributes=40)
        model = FacialAttributeClassifier(config)
        model.eval()

        x = torch.randn(2, 3, 224, 224)
        output = model(x)

        assert output.shape == (2, 40)

    def test_model_predict_proba(self) -> None:
        """Test de predicción de probabilidades."""
        config = ModelConfig(pretrained=False, num_attributes=40)
        model = FacialAttributeClassifier(config)
        model.eval()

        x = torch.randn(2, 3, 224, 224)
        probas = model.predict_proba(x)

        assert probas.shape == (2, 40)
        assert probas.min() >= 0.0
        assert probas.max() <= 1.0

    def test_model_predict(self) -> None:
        """Test de predicción binaria."""
        config = ModelConfig(pretrained=False, num_attributes=40)
        model = FacialAttributeClassifier(config)
        model.eval()

        x = torch.randn(2, 3, 224, 224)
        predictions = model.predict(x, threshold=0.5)

        assert predictions.shape == (2, 40)
        assert set(predictions.unique().tolist()).issubset({0.0, 1.0})

    def test_model_get_num_parameters(self) -> None:
        """Test de conteo de parámetros."""
        config = ModelConfig(pretrained=False, num_attributes=40)
        model = FacialAttributeClassifier(config)

        params = model.get_num_parameters()

        assert "total" in params
        assert "trainable" in params
        assert params["total"] > 0

    def test_model_different_backbones(self) -> None:
        """Test de diferentes backbones."""
        for backbone in ["resnet18", "resnet34", "resnet50"]:
            config = ModelConfig(backbone=backbone, pretrained=False)
            model = FacialAttributeClassifier(config)
            model.eval()

            x = torch.randn(1, 3, 224, 224)
            output = model(x)

            assert output.shape == (1, 40)

    def test_model_invalid_backbone(self) -> None:
        """Test de backbone inválido."""
        config = ModelConfig(backbone="invalid", pretrained=False)

        with pytest.raises(ValueError, match="Backbone no soportado"):
            FacialAttributeClassifier(config)

    def test_model_freeze_backbone(self) -> None:
        """Test de congelamiento de backbone."""
        config = ModelConfig(pretrained=False, freeze_backbone=True)
        model = FacialAttributeClassifier(config)

        for param in model._backbone.parameters():
            assert param.requires_grad is False


class TestLossConfig:
    """Tests para LossConfig."""

    def test_default_config(self) -> None:
        """Test de configuración por defecto."""
        config = LossConfig()

        assert config.pos_weight is None
        assert config.reduction == "mean"


class TestMultilabelLoss:
    """Tests para MultilabelLoss."""

    def test_loss_creation(self) -> None:
        """Test de creación de pérdida."""
        loss_fn = MultilabelLoss()

        assert loss_fn is not None

    def test_loss_computation(self) -> None:
        """Test de cálculo de pérdida."""
        loss_fn = MultilabelLoss()

        logits = torch.randn(4, 40)
        targets = torch.randint(0, 2, (4, 40)).float()

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_loss_with_pos_weight(self) -> None:
        """Test de pérdida con pesos de clase."""
        pos_weight = torch.ones(40) * 2.0
        config = LossConfig(pos_weight=pos_weight)
        loss_fn = MultilabelLoss(config)

        logits = torch.randn(4, 40)
        targets = torch.randint(0, 2, (4, 40)).float()

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_loss_set_pos_weight(self) -> None:
        """Test de actualización de pesos."""
        loss_fn = MultilabelLoss()

        new_pos_weight = torch.ones(40) * 3.0
        loss_fn.set_pos_weight(new_pos_weight)

        assert loss_fn.config.pos_weight is not None
