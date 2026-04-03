"""Tests for model_config.py — data pipeline, model, early stopping, scalers."""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytest
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler

from model_config import (
    set_seed, load_and_preprocess_data, make_loaders,
    PermeabilityMLP, EarlyStopping, train_one_epoch, validate,
    save_scaler, load_scaler, CONFIG,
)
N_FEATURES = 11
N_TARGETS = 3


# =====================================================================
# set_seed
# =====================================================================

class TestSetSeed:
    def test_reproducibility(self):
        """Same seed produces identical random tensors."""
        set_seed(0)
        a = torch.randn(5)
        set_seed(0)
        b = torch.randn(5)
        assert torch.equal(a, b)


# =====================================================================
# PermeabilityMLP
# =====================================================================

class TestPermeabilityMLP:
    def test_output_shape(self, small_model):
        x = torch.randn(8, N_FEATURES)
        out = small_model(x)
        assert out.shape == (8, N_TARGETS)

    def test_single_sample(self, small_model):
        x = torch.randn(1, N_FEATURES)
        out = small_model(x)
        assert out.shape == (1, N_TARGETS)

    @pytest.mark.parametrize("layers", [[16], [32, 32], [64, 64, 64]])
    def test_different_architectures(self, layers):
        model = PermeabilityMLP(N_FEATURES, N_TARGETS, layers, dropout_rate=0.0)
        out = model(torch.randn(4, N_FEATURES))
        assert out.shape == (4, N_TARGETS)

    def test_no_nan_output(self, small_model):
        out = small_model(torch.randn(8, N_FEATURES))
        assert torch.isfinite(out).all()

    def test_parameter_count(self):
        model = PermeabilityMLP(N_FEATURES, N_TARGETS, [16, 16], dropout_rate=0.0)
        # Linear(11,16)=192, Linear(16,16)=272, Linear(16,3)=51 → 515
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == 515

    def test_deterministic_with_no_dropout(self, small_model):
        small_model.eval()
        x = torch.randn(4, N_FEATURES)
        out1 = small_model(x)
        out2 = small_model(x)
        assert torch.equal(out1, out2)


# =====================================================================
# EarlyStopping
# =====================================================================

class TestEarlyStopping:
    def test_no_trigger_on_improvement(self, small_model):
        es = EarlyStopping(patience=5)
        for loss in [1.0, 0.9, 0.8, 0.7, 0.6]:
            assert es(loss, small_model) is False

    def test_triggers_after_patience(self, small_model):
        es = EarlyStopping(patience=3)
        es(1.0, small_model)  # best
        es(1.0, small_model)  # no improvement, counter=1
        es(1.0, small_model)  # counter=2
        assert es(1.0, small_model) is True  # counter=3 >= patience

    def test_resets_on_improvement(self, small_model):
        es = EarlyStopping(patience=3)
        es(1.0, small_model)
        es(1.0, small_model)  # counter=1
        es(1.0, small_model)  # counter=2
        es(0.5, small_model)  # big improvement → reset
        assert es.counter == 0
        # Should need 3 more flat calls to trigger
        es(0.5, small_model)
        es(0.5, small_model)
        assert es(0.5, small_model) is True

    def test_min_delta_respected(self, small_model):
        es = EarlyStopping(patience=3, min_delta=0.1)
        es(1.0, small_model)  # best = 1.0
        # Improvement of 0.05 is less than min_delta=0.1, so doesn't count
        es(0.95, small_model)
        assert es.counter == 1

    def test_restore_best(self, small_model):
        es = EarlyStopping(patience=5)
        # Record weights at best loss
        es(1.0, small_model)
        best_param = small_model.network[0].weight.data.clone()

        # Modify model weights manually (simulating further training)
        with torch.no_grad():
            small_model.network[0].weight.fill_(999.0)

        # Worsening losses
        es(2.0, small_model)
        es(3.0, small_model)

        # Restore should bring back the original weights
        es.restore_best(small_model)
        assert torch.equal(small_model.network[0].weight.data, best_param)


# =====================================================================
# train_one_epoch / validate
# =====================================================================

class TestTraining:
    def test_train_one_epoch_returns_finite_loss(self, small_model, trained_loaders, device):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
        loss = train_one_epoch(small_model, trained_loaders["train_loader"],
                               criterion, optimizer, device)
        assert np.isfinite(loss) and loss >= 0

    def test_train_one_epoch_updates_weights(self, small_model, trained_loaders, device):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
        before = small_model.network[0].weight.data.clone()
        train_one_epoch(small_model, trained_loaders["train_loader"],
                        criterion, optimizer, device)
        after = small_model.network[0].weight.data
        assert not torch.equal(before, after)

    def test_validate_returns_finite_loss(self, small_model, trained_loaders, device):
        criterion = nn.MSELoss()
        loss = validate(small_model, trained_loaders["val_loader"], criterion, device)
        assert np.isfinite(loss) and loss >= 0

    def test_validate_does_not_update_weights(self, small_model, trained_loaders, device):
        criterion = nn.MSELoss()
        before = small_model.network[0].weight.data.clone()
        validate(small_model, trained_loaders["val_loader"], criterion, device)
        after = small_model.network[0].weight.data
        assert torch.equal(before, after)

    def test_loss_decreases_over_epochs(self, small_model, trained_loaders, device):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
        loss_1 = train_one_epoch(small_model, trained_loaders["train_loader"],
                                 criterion, optimizer, device)
        for _ in range(9):
            train_one_epoch(small_model, trained_loaders["train_loader"],
                            criterion, optimizer, device)
        loss_10 = train_one_epoch(small_model, trained_loaders["train_loader"],
                                  criterion, optimizer, device)
        assert loss_10 < loss_1


# =====================================================================
# make_loaders
# =====================================================================

class TestMakeLoaders:
    def test_keys(self, trained_loaders):
        assert set(trained_loaders.keys()) == {"train_loader", "val_loader"}

    def test_batch_shape(self, trained_loaders):
        xb, yb = next(iter(trained_loaders["train_loader"]))
        assert xb.shape[1] == N_FEATURES
        assert yb.shape[1] == N_TARGETS
        assert xb.shape[0] <= 8  # batch_size

    def test_total_samples(self, trained_loaders):
        total = sum(xb.shape[0] for xb, _ in trained_loaders["train_loader"])
        assert total == 44  # n_train from synthetic_data


# =====================================================================
# save_scaler / load_scaler
# =====================================================================

class TestScalerSerialization:
    def test_roundtrip(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = np.random.default_rng(0).uniform(0, 10, (20, 3)).astype(np.float64)
        scaler.fit(data)

        d = save_scaler(scaler)
        restored = load_scaler(d)

        test = np.array([[1.0, 5.0, 9.0]])
        np.testing.assert_allclose(
            scaler.transform(test), restored.transform(test)
        )

    def test_dict_keys(self):
        scaler = MinMaxScaler()
        scaler.fit(np.random.default_rng(0).uniform(0, 1, (10, 2)))
        d = save_scaler(scaler)
        expected = {"data_min_", "data_max_", "data_range_", "scale_",
                    "min_", "feature_range", "n_features_in_", "n_samples_seen_"}
        assert set(d.keys()) == expected

    def test_json_serializable(self):
        scaler = MinMaxScaler()
        scaler.fit(np.random.default_rng(0).uniform(0, 1, (10, 3)))
        d = save_scaler(scaler)
        # Should not raise
        json.dumps(d)


# =====================================================================
# load_and_preprocess_data (integration with synthetic CSV)
# =====================================================================

class TestLoadAndPreprocess:
    def _make_config(self, csv_path, tmp_path):
        cfg = deepcopy(CONFIG)
        cfg["data_path"] = csv_path
        cfg["output_dir"] = str(tmp_path / "output")
        return cfg

    def test_output_keys(self, synthetic_csv, tmp_path):
        cfg = self._make_config(synthetic_csv, tmp_path)
        data = load_and_preprocess_data(cfg)
        expected_keys = {
            "scaler_X", "scaler_Y",
            "X_train_scaled", "Y_train_scaled",
            "X_val_scaled", "Y_val_scaled",
            "X_test_scaled", "Y_test_scaled",
            "X_train_t", "Y_train_t", "X_val_t", "Y_val_t",
            "Y_val_raw", "Y_test_raw", "Y_val_model", "Y_test_model",
            "names_val", "names_test", "n_features", "n_targets",
        }
        assert expected_keys.issubset(set(data.keys()))

    def test_split_sizes(self, synthetic_csv, tmp_path):
        cfg = self._make_config(synthetic_csv, tmp_path)
        data = load_and_preprocess_data(cfg)
        n_total = 50
        n_train = data["X_train_scaled"].shape[0]
        n_val = data["X_val_scaled"].shape[0]
        n_test = data["X_test_scaled"].shape[0]
        assert n_train + n_val + n_test == n_total

    def test_scaler_fit_on_train_only(self, synthetic_csv, tmp_path):
        cfg = self._make_config(synthetic_csv, tmp_path)
        data = load_and_preprocess_data(cfg)
        n_train = data["X_train_scaled"].shape[0]
        assert data["scaler_X"].n_samples_seen_ == n_train

    def test_bad_fractions_raises(self, synthetic_csv, tmp_path):
        cfg = self._make_config(synthetic_csv, tmp_path)
        cfg["train_frac"] = 0.8
        cfg["val_frac"] = 0.2
        cfg["test_frac"] = 0.1  # sum = 1.1
        with pytest.raises(ValueError, match="sum to 1.0"):
            load_and_preprocess_data(cfg)

    def test_missing_column_raises(self, tmp_path):
        # CSV missing K_xx
        rng = np.random.default_rng(0)
        n = 20
        data = {"sample_name": [f"s{i}" for i in range(n)]}
        for col in CONFIG["feature_cols"]:
            data[col] = rng.uniform(0, 1, n)
        data["K_yy"] = rng.uniform(1e-10, 1e-2, n)
        data["K_zz"] = rng.uniform(1e-10, 1e-2, n)
        # K_xx intentionally missing

        csv_path = str(tmp_path / "bad.csv")
        pd.DataFrame(data).to_csv(csv_path, index=False)

        cfg = deepcopy(CONFIG)
        cfg["data_path"] = csv_path
        cfg["output_dir"] = str(tmp_path / "out")
        with pytest.raises(ValueError, match="Missing columns"):
            load_and_preprocess_data(cfg)

    def test_nan_in_features_raises(self, tmp_path):
        rng = np.random.default_rng(0)
        n = 20
        data = {"sample_name": [f"s{i}" for i in range(n)]}
        for col in CONFIG["feature_cols"]:
            data[col] = rng.uniform(0, 1, n)
        for col in CONFIG["target_cols"]:
            data[col] = rng.uniform(1e-10, 1e-2, n)
        # Inject NaN
        data["porosity"][0] = float("nan")

        csv_path = str(tmp_path / "nan.csv")
        pd.DataFrame(data).to_csv(csv_path, index=False)

        cfg = deepcopy(CONFIG)
        cfg["data_path"] = csv_path
        cfg["output_dir"] = str(tmp_path / "out")
        with pytest.raises(ValueError, match="Non-finite"):
            load_and_preprocess_data(cfg)
