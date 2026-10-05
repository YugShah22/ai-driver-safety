from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.ann.config import ANNConfig
from ml.ann.model import RiskANN
from ml.ann.train import ANNTrainer, ANNTrainingResult
from ml.traditional_ml.schema import N_FEATURES


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _config(**kwargs) -> ANNConfig:
    defaults = dict(
        input_dim=N_FEATURES,
        hidden_dims=(32, 16),
        num_classes=2,
        epochs=2,
        batch_size=16,
        device="cpu",
        checkpoint_dir="",
    )
    defaults.update(kwargs)
    return ANNConfig(**defaults)


def _random_data(n=80, input_dim=N_FEATURES, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, input_dim)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n).astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# ANNConfig
# ---------------------------------------------------------------------------

class TestANNConfig:
    def test_defaults(self):
        cfg = ANNConfig()
        assert cfg.input_dim == 18
        assert cfg.num_classes == 2
        assert cfg.epochs == 20

    def test_resolve_device_cpu(self):
        cfg = ANNConfig(device="cpu")
        assert cfg.resolve_device() == "cpu"

    def test_resolve_device_auto(self):
        cfg = ANNConfig(device="auto")
        result = cfg.resolve_device()
        assert result in ("cpu", "cuda")

    def test_custom_hidden_dims(self):
        cfg = ANNConfig(hidden_dims=(128, 64, 32))
        assert cfg.hidden_dims == (128, 64, 32)

    def test_input_dim_matches_n_features(self):
        # ANNConfig default should match Phase 11 FEATURE_NAMES length
        assert ANNConfig().input_dim == N_FEATURES


# ---------------------------------------------------------------------------
# RiskANN — construction
# ---------------------------------------------------------------------------

class TestRiskANNConstruction:
    def test_init_default(self):
        model = RiskANN(_config())
        assert isinstance(model, torch.nn.Module)

    def test_single_hidden_layer(self):
        model = RiskANN(_config(hidden_dims=(64,)))
        assert model is not None

    def test_no_hidden_layers(self):
        model = RiskANN(_config(hidden_dims=()))
        assert model is not None

    def test_many_hidden_layers(self):
        model = RiskANN(_config(hidden_dims=(128, 64, 32, 16)))
        assert model is not None

    def test_custom_num_classes(self):
        model = RiskANN(_config(num_classes=5))
        x = torch.randn(4, N_FEATURES)
        out = model(x)
        assert out.shape == (4, 5)


# ---------------------------------------------------------------------------
# RiskANN — forward pass
# ---------------------------------------------------------------------------

class TestRiskANNForward:
    def test_output_shape_batch(self):
        model = RiskANN(_config())
        x = torch.randn(8, N_FEATURES)
        out = model(x)
        assert out.shape == (8, 2)

    def test_output_shape_single(self):
        model = RiskANN(_config())
        model.eval()
        x = torch.randn(1, N_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_output_is_logits_not_proba(self):
        # Logits can be any real number; probabilities would sum to 1
        model = RiskANN(_config())
        x = torch.randn(10, N_FEATURES)
        out = model(x)
        # They should NOT all sum to 1 per row (that would be softmax output)
        row_sums = out.sum(dim=1)
        assert not torch.allclose(row_sums, torch.ones(10), atol=0.01)

    def test_configurable_input_dim(self):
        model = RiskANN(_config(input_dim=32, hidden_dims=(16,), num_classes=3))
        x = torch.randn(4, 32)
        out = model(x)
        assert out.shape == (4, 3)

    def test_wrong_input_dim_raises(self):
        model = RiskANN(_config(input_dim=18))
        x = torch.randn(4, 10)   # wrong dim
        with pytest.raises(Exception):
            model(x)

    def test_gradient_flows(self):
        model = RiskANN(_config())
        x = torch.randn(4, N_FEATURES)
        y = torch.randint(0, 2, (4,))
        loss = torch.nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_predict_returns_integers(self):
        model = RiskANN(_config())
        x = torch.randn(8, N_FEATURES)
        preds = model.predict(x)
        assert preds.dtype == torch.int64
        assert preds.shape == (8,)

    def test_predict_values_in_class_range(self):
        model = RiskANN(_config(num_classes=3))
        x = torch.randn(20, N_FEATURES)
        preds = model.predict(x)
        assert preds.min() >= 0
        assert preds.max() < 3

    def test_eval_mode_no_batchnorm_error(self):
        # BN with batch_size=1 fails in train mode, not in eval
        model = RiskANN(_config())
        model.eval()
        x = torch.randn(1, N_FEATURES)
        out = model(x)
        assert out.shape == (1, 2)


# ---------------------------------------------------------------------------
# RiskANN — save / load
# ---------------------------------------------------------------------------

class TestRiskANNSerialization:
    def test_save_and_load(self, tmp_path):
        cfg = _config(hidden_dims=(32, 16), num_classes=2)
        model = RiskANN(cfg)
        model.eval()
        path = tmp_path / "ann.pt"
        model.save(path)
        loaded = RiskANN.load(path)  # already eval from load()
        x = torch.randn(4, N_FEATURES)
        with torch.no_grad():
            torch.testing.assert_close(model(x), loaded(x))

    def test_load_without_config(self, tmp_path):
        cfg = _config(input_dim=10, hidden_dims=(8,), num_classes=3)
        model = RiskANN(cfg)
        path = tmp_path / "ann.pt"
        model.save(path)
        loaded = RiskANN.load(path)   # config reconstructed from checkpoint
        assert loaded.config.input_dim == 10
        assert loaded.config.num_classes == 3

    def test_load_sets_eval_mode(self, tmp_path):
        model = RiskANN(_config())
        path = tmp_path / "ann.pt"
        model.save(path)
        loaded = RiskANN.load(path)
        assert not loaded.training


# ---------------------------------------------------------------------------
# ANNTrainer — smoke test
# ---------------------------------------------------------------------------

class TestANNTrainerSmoke:
    def _trainer(self, tmp_path, **kwargs):
        cfg = _config(epochs=2, batch_size=16, checkpoint_dir=str(tmp_path), **kwargs)
        model = RiskANN(cfg)
        X, y = _random_data(n=80)
        X_val, y_val = _random_data(n=20, seed=1)
        return ANNTrainer(model, cfg, X, y, X_val, y_val), model, X

    def test_train_returns_result(self, tmp_path):
        trainer, _, _ = self._trainer(tmp_path)
        result = trainer.train()
        assert isinstance(result, ANNTrainingResult)

    def test_result_has_correct_epoch_count(self, tmp_path):
        trainer, _, _ = self._trainer(tmp_path)
        result = trainer.train()
        assert len(result.train_loss) == 2
        assert len(result.val_loss) == 2

    def test_losses_are_finite(self, tmp_path):
        trainer, _, _ = self._trainer(tmp_path)
        result = trainer.train()
        for loss in result.train_loss + result.val_loss:
            assert np.isfinite(loss)

    def test_best_val_acc_in_range(self, tmp_path):
        trainer, _, _ = self._trainer(tmp_path)
        result = trainer.train()
        assert 0.0 <= result.best_val_acc() <= 1.0

    def test_checkpoint_saved(self, tmp_path):
        trainer, _, _ = self._trainer(tmp_path)
        trainer.train()
        saved = list(tmp_path.glob("*.pt"))
        assert len(saved) > 0

    def test_best_checkpoint_loadable(self, tmp_path):
        cfg = _config(epochs=2, batch_size=16, checkpoint_dir=str(tmp_path))
        model = RiskANN(cfg)
        X, y = _random_data(n=80)
        X_val, y_val = _random_data(n=20, seed=1)
        trainer = ANNTrainer(model, cfg, X, y, X_val, y_val)
        trainer.train()
        loaded = ANNTrainer.load_checkpoint(tmp_path / cfg.best_ckpt_name, cfg)
        assert isinstance(loaded, RiskANN)

    def test_model_produces_predictions_after_training(self, tmp_path):
        trainer, model, X = self._trainer(tmp_path)
        trainer.train()
        model.eval()
        x = torch.from_numpy(X[:5]).float()
        preds = model.predict(x)
        assert preds.shape == (5,)

    def test_three_class_smoke(self, tmp_path):
        cfg = _config(epochs=1, batch_size=16, num_classes=3,
                      checkpoint_dir=str(tmp_path))
        model = RiskANN(cfg)
        X, y = _random_data(n=60, n_classes=3)
        X_val, y_val = _random_data(n=20, n_classes=3, seed=2)
        result = ANNTrainer(model, cfg, X, y, X_val, y_val).train()
        assert len(result.train_loss) == 1

    def test_sgd_optimizer(self, tmp_path):
        cfg = _config(epochs=1, batch_size=16, optimizer="sgd",
                      checkpoint_dir=str(tmp_path))
        model = RiskANN(cfg)
        X, y = _random_data(n=60)
        X_val, y_val = _random_data(n=20, seed=3)
        result = ANNTrainer(model, cfg, X, y, X_val, y_val).train()
        assert np.isfinite(result.train_loss[0])


# ---------------------------------------------------------------------------
# ANNTrainingResult
# ---------------------------------------------------------------------------

class TestANNTrainingResult:
    def test_best_val_acc_empty(self):
        r = ANNTrainingResult()
        assert r.best_val_acc() == 0.0

    def test_best_epoch_empty(self):
        r = ANNTrainingResult()
        assert r.best_epoch() == -1

    def test_best_epoch_correct(self):
        r = ANNTrainingResult()
        r.val_acc = [0.5, 0.8, 0.7]
        assert r.best_epoch() == 1
