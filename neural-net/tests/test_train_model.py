"""Tests for train_model.py — metrics, predict, build_model, param loading."""

import json
import numpy as np
import torch
import pytest
from copy import deepcopy

from model_config import CONFIG, PermeabilityMLP
from train_model import mdape, nrmse, compute_metrics, predict, build_model, load_params_from_json
N_FEATURES = 11
N_TARGETS = 3


# =====================================================================
# mdape
# =====================================================================

class TestMdape:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mdape(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 220.0])
        # Both have 10% error → median = 0.10
        assert abs(mdape(y_true, y_pred) - 0.10) < 1e-10

    def test_handles_zero_true(self):
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.1, 1.1])
        result = mdape(y_true, y_pred)
        assert np.isfinite(result)


# =====================================================================
# nrmse
# =====================================================================

class TestNrmse:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert nrmse(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([0.0, 10.0])
        y_pred = np.array([1.0, 9.0])
        # RMSE = 1.0, range = 10 → NRMSE = 0.1
        assert abs(nrmse(y_true, y_pred) - 0.1) < 1e-10

    def test_constant_true(self):
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([4.0, 5.0, 6.0])
        # Range = 0, eps prevents division by zero
        result = nrmse(y_true, y_pred)
        assert np.isfinite(result)


# =====================================================================
# compute_metrics
# =====================================================================

class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = np.random.default_rng(0).uniform(0, 1, (20, 3)).astype(np.float32)
        m = compute_metrics(y, y, physical=True)
        for comp in m["per_component"]:
            assert abs(comp["R2"] - 1.0) < 1e-6
            assert comp["MAE"] == 0.0
            assert comp["RMSE"] == 0.0
        assert abs(m["overall"]["R2"] - 1.0) < 1e-6

    def test_structure(self):
        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 1, (20, 3))
        y_pred = rng.uniform(0, 1, (20, 3))
        m = compute_metrics(y_true, y_pred, physical=True)
        assert len(m["per_component"]) == 3
        assert isinstance(m["overall"], dict)
        for comp in m["per_component"]:
            assert {"R2", "MAE", "RMSE", "NRMSE", "MdAPE"} == set(comp.keys())

    def test_physical_false_omits_extra_keys(self):
        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 1, (20, 3))
        y_pred = rng.uniform(0, 1, (20, 3))
        m = compute_metrics(y_true, y_pred, physical=False)
        for comp in m["per_component"]:
            assert "NRMSE" not in comp
            assert "MdAPE" not in comp


# =====================================================================
# predict
# =====================================================================

class TestPredict:
    def test_output_shapes(self, small_model, synthetic_data, device):
        config = deepcopy(CONFIG)
        config["use_log_targets"] = True
        y_model, y_phys = predict(
            small_model, synthetic_data["X_val_scaled"],
            synthetic_data["scaler_Y"], config, device,
        )
        n = synthetic_data["X_val_scaled"].shape[0]
        assert y_model.shape == (n, N_TARGETS)
        assert y_phys.shape == (n, N_TARGETS)

    def test_log_inverse(self, small_model, synthetic_data, device):
        config = deepcopy(CONFIG)
        config["use_log_targets"] = True
        y_model, y_phys = predict(
            small_model, synthetic_data["X_val_scaled"],
            synthetic_data["scaler_Y"], config, device,
        )
        # y_phys should be 10^(y_model)
        np.testing.assert_allclose(y_phys, np.power(10.0, y_model), rtol=1e-5)

    def test_no_log(self, small_model, synthetic_data, device):
        config = deepcopy(CONFIG)
        config["use_log_targets"] = False
        y_model, y_phys = predict(
            small_model, synthetic_data["X_val_scaled"],
            synthetic_data["scaler_Y"], config, device,
        )
        np.testing.assert_allclose(y_phys, y_model, rtol=1e-5)

    def test_deterministic(self, small_model, synthetic_data, device):
        config = deepcopy(CONFIG)
        config["use_log_targets"] = True
        y1, _ = predict(small_model, synthetic_data["X_val_scaled"],
                         synthetic_data["scaler_Y"], config, device)
        y2, _ = predict(small_model, synthetic_data["X_val_scaled"],
                         synthetic_data["scaler_Y"], config, device)
        np.testing.assert_array_equal(y1, y2)


# =====================================================================
# build_model
# =====================================================================

class TestBuildModel:
    def test_returns_mlp(self, mini_config):
        model = build_model(mini_config, N_FEATURES, N_TARGETS)
        assert isinstance(model, PermeabilityMLP)

    def test_respects_config(self):
        config = deepcopy(CONFIG)
        config["hidden_layers"] = [32, 64]
        config["dropout_rate"] = 0.0
        model = build_model(config, N_FEATURES, N_TARGETS)
        # 2 hidden + 1 output = 3 Linear layers
        linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        assert len(linears) == 3


# =====================================================================
# load_params_from_json
# =====================================================================

class TestLoadParamsFromJson:
    def test_merges_correctly(self, tmp_path):
        params = {"best_params": {"hidden_layers": [64], "dropout_rate": 0.1}}
        path = str(tmp_path / "params.json")
        with open(path, "w") as f:
            json.dump(params, f)

        result = load_params_from_json(path, CONFIG)
        assert result["hidden_layers"] == [64]
        assert result["dropout_rate"] == 0.1
        # Unmentioned keys should keep defaults
        assert result["max_epochs"] == CONFIG["max_epochs"]

    def test_does_not_mutate_base(self, tmp_path):
        original = deepcopy(CONFIG)
        params = {"best_params": {"hidden_layers": [64]}}
        path = str(tmp_path / "params.json")
        with open(path, "w") as f:
            json.dump(params, f)

        load_params_from_json(path, CONFIG)
        assert CONFIG["hidden_layers"] == original["hidden_layers"]

    def test_flat_json(self, tmp_path):
        # JSON without best_params wrapper should also work
        params = {"hidden_layers": [128, 128], "dropout_rate": 0.05}
        path = str(tmp_path / "flat.json")
        with open(path, "w") as f:
            json.dump(params, f)

        result = load_params_from_json(path, CONFIG)
        assert result["hidden_layers"] == [128, 128]
