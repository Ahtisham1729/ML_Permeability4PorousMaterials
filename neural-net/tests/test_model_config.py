"""Tests for model_config.py — scaler serialization, early stopping, data pipeline."""

import json
import numpy as np
import pandas as pd
import torch
import pytest
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler

from model_config import (
    load_and_preprocess_data, EarlyStopping,
    save_scaler, load_scaler, CONFIG,
)


# =====================================================================
# Scaler Serialization — if this breaks, all inference results are wrong
# =====================================================================

class TestScalerSerialization:
    def test_roundtrip_produces_identical_transforms(self):
        """Save → load a scaler and verify it transforms data identically."""
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(np.random.default_rng(0).uniform(0, 10, (20, 3)))

        restored = load_scaler(save_scaler(scaler))

        test = np.array([[1.0, 5.0, 9.0]])
        np.testing.assert_allclose(
            scaler.transform(test), restored.transform(test)
        )

    def test_dict_is_json_serializable(self):
        """Scaler dict must serialize cleanly for torch.save with weights_only=True."""
        scaler = MinMaxScaler()
        scaler.fit(np.random.default_rng(0).uniform(0, 1, (10, 3)))
        json.dumps(save_scaler(scaler))  # should not raise

    def test_dict_has_all_required_keys(self):
        scaler = MinMaxScaler()
        scaler.fit(np.random.default_rng(0).uniform(0, 1, (10, 2)))
        expected = {"data_min_", "data_max_", "data_range_", "scale_",
                    "min_", "feature_range", "n_features_in_", "n_samples_seen_"}
        assert set(save_scaler(scaler).keys()) == expected


# =====================================================================
# EarlyStopping — stateful logic that's easy to break
# =====================================================================

class TestEarlyStopping:
    def test_does_not_trigger_while_improving(self, small_model):
        es = EarlyStopping(patience=5)
        for loss in [1.0, 0.9, 0.8, 0.7, 0.6]:
            assert es(loss, small_model) is False

    def test_triggers_after_patience_flat_losses(self, small_model):
        es = EarlyStopping(patience=3)
        es(1.0, small_model)   # best
        es(1.0, small_model)   # counter=1
        es(1.0, small_model)   # counter=2
        assert es(1.0, small_model) is True  # counter=3 >= patience

    def test_resets_counter_on_improvement(self, small_model):
        es = EarlyStopping(patience=3)
        es(1.0, small_model)
        es(1.0, small_model)   # counter=1
        es(1.0, small_model)   # counter=2
        es(0.5, small_model)   # big improvement → reset
        assert es.counter == 0

    def test_restore_best_recovers_weights(self, small_model):
        es = EarlyStopping(patience=5)
        es(1.0, small_model)  # saves these weights as best
        best_param = small_model.network[0].weight.data.clone()

        # Corrupt the weights
        with torch.no_grad():
            small_model.network[0].weight.fill_(999.0)
        es(2.0, small_model)

        es.restore_best(small_model)
        assert torch.equal(small_model.network[0].weight.data, best_param)


# =====================================================================
# load_and_preprocess_data — validates the full data pipeline
# =====================================================================

class TestLoadAndPreprocess:
    def _make_config(self, csv_path, tmp_path):
        cfg = deepcopy(CONFIG)
        cfg["data_path"] = csv_path
        cfg["output_dir"] = str(tmp_path / "output")
        return cfg

    def test_split_sizes_sum_to_total(self, synthetic_csv, tmp_path):
        cfg = self._make_config(synthetic_csv, tmp_path)
        data = load_and_preprocess_data(cfg)
        n_train = data["X_train_scaled"].shape[0]
        n_val = data["X_val_scaled"].shape[0]
        n_test = data["X_test_scaled"].shape[0]
        assert n_train + n_val + n_test == 50

    def test_scaler_fit_on_train_only(self, synthetic_csv, tmp_path):
        """Scalers must be fit on training data only to prevent data leakage."""
        cfg = self._make_config(synthetic_csv, tmp_path)
        data = load_and_preprocess_data(cfg)
        assert data["scaler_X"].n_samples_seen_ == data["X_train_scaled"].shape[0]

    def test_bad_fractions_raises(self, synthetic_csv, tmp_path):
        cfg = self._make_config(synthetic_csv, tmp_path)
        cfg["train_frac"] = 0.8
        cfg["val_frac"] = 0.2
        cfg["test_frac"] = 0.1  # sum = 1.1
        with pytest.raises(ValueError, match="sum to 1.0"):
            load_and_preprocess_data(cfg)

    def test_missing_column_raises(self, tmp_path):
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
        data["porosity"][0] = float("nan")

        csv_path = str(tmp_path / "nan.csv")
        pd.DataFrame(data).to_csv(csv_path, index=False)

        cfg = deepcopy(CONFIG)
        cfg["data_path"] = csv_path
        cfg["output_dir"] = str(tmp_path / "out")
        with pytest.raises(ValueError, match="Non-finite"):
            load_and_preprocess_data(cfg)
