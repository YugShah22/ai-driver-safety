import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))

from ml.cnn.config import CNNConfig
from ml.transfer_learning.model import TransferCNN


def _cfg(num_classes: int = 6, dropout: float = 0.3) -> CNNConfig:
    return CNNConfig(num_classes=num_classes, dropout=dropout)


def _model(
    num_classes: int = 6,
    backbone: str = "resnet18",
    pretrained: bool = False,
    freeze: bool = True,
) -> TransferCNN:
    return TransferCNN(
        config=_cfg(num_classes=num_classes),
        backbone_name=backbone,
        pretrained=pretrained,
        freeze=freeze,
    )


class TestTransferCNNConstruction:
    def test_default_construction_succeeds(self):
        model = _model()
        assert isinstance(model, TransferCNN)

    def test_is_nn_module(self):
        model = _model()
        assert isinstance(model, torch.nn.Module)

    def test_config_stored(self):
        cfg = _cfg(num_classes=6)
        model = TransferCNN(cfg, pretrained=False)
        assert model.config is cfg


class TestTransferCNNOutputShape:
    def test_output_shape_default(self):
        model = _model(num_classes=6)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 6)

    def test_output_shape_batch_1(self):
        model = _model(num_classes=6)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 6)

    def test_output_shape_batch_4(self):
        model = _model(num_classes=6)
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 6)


class TestTransferCNNNumClasses:
    @pytest.mark.parametrize("num_classes", [2, 4, 6, 10])
    def test_num_classes_from_config(self, num_classes: int):
        model = _model(num_classes=num_classes)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape[1] == num_classes

    def test_six_classes_indian_road(self):
        model = _model(num_classes=6)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 6)


class TestTransferCNNFrozenBackbone:
    def test_backbone_frozen_by_default(self):
        model = _model(freeze=True)
        for name, param in model.backbone.named_parameters():
            assert not param.requires_grad

    def test_head_trainable_when_backbone_frozen(self):
        model = _model(freeze=True)
        for name, param in model.head.named_parameters():
            assert param.requires_grad

    def test_trainable_params_reduced_when_frozen(self):
        model_frozen = _model(freeze=True)
        model_unfrozen = _model(freeze=False)
        assert model_frozen.count_trainable_parameters() < model_unfrozen.count_trainable_parameters()

    def test_total_params_unchanged_by_freeze(self):
        model = _model(freeze=True)
        total_frozen = model.count_total_parameters()
        model.unfreeze_backbone()
        total_unfrozen = model.count_total_parameters()
        assert total_frozen == total_unfrozen


class TestTransferCNNUnfreeze:
    def test_unfreeze_makes_all_backbone_params_trainable(self):
        model = _model(freeze=True)
        model.unfreeze_backbone()
        for name, param in model.backbone.named_parameters():
            assert param.requires_grad

    def test_unfreeze_does_not_affect_head(self):
        model = _model(freeze=True)
        model.unfreeze_backbone()
        for name, param in model.head.named_parameters():
            assert param.requires_grad


class TestTransferCNNFreezeCycle:
    def test_freeze_unfreeze_refreeze(self):
        model = _model(freeze=False)

        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad

        model.unfreeze_backbone()
        for param in model.backbone.parameters():
            assert param.requires_grad

        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad

    def test_count_trainable_reflects_stage(self):
        model = _model(freeze=False)
        full_trainable = model.count_trainable_parameters()

        model.freeze_backbone()
        head_only_trainable = model.count_trainable_parameters()

        assert head_only_trainable < full_trainable

        model.unfreeze_backbone()
        assert model.count_trainable_parameters() == full_trainable


class TestTransferCNNBackbones:
    @pytest.mark.parametrize("backbone", [
        "resnet18",
        "resnet50",
        "efficientnet_b0",
        "mobilenet_v3_small",
    ])
    def test_backbone_forward(self, backbone: str):
        model = TransferCNN(
            config=_cfg(num_classes=6),
            backbone_name=backbone,
            pretrained=False,
            freeze=True,
        )
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 6)

    def test_invalid_backbone_raises(self):
        with pytest.raises(ValueError):
            TransferCNN(
                config=_cfg(),
                backbone_name="invalid",
                pretrained=False,
            )
