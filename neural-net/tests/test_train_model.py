"""Tests for train_model.py — metrics, predict, param loading."""

import json
import numpy as np
import pytest
from copy import deepcopy

from model_config import CONFIG
from train_model import mdape, nrmse, compute_metrics, predict, load_params_from_json

N_FEATURES = 11
N_TARGETS = 3


# =====================================================================
# Metrics — pure math, easy to get wrong
# =====================================================================

class TestMdape:
    def test_perfect_prediction_is_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mdape(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 220.0])
        assert abs(mdape(y_true, y_pred) - 0.10) < 1e-10

    def test_handles_zero_in_true(self):
        """eps floor prevents division by zero."""
        result = mdape(np.array([0.0, 1.0]), np.array([0.1, 1.1]))
        assert np.isfinite(result)


class TestNrmse:
    def test_perfect_prediction_is_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert nrmse(y, y) == 0.0

    def test_known_value(self):
        # RMSE = 1.0, range = 10 → NRMSE = 0.1
        assert abs(nrmse(np.array([0.0, 10.0]), np.array([1.0, 9.0])) - 0.1) < 1e-10

    def test_constant_true_does_not_crash(self):
        """Zero range uses eps to avoid division by zero."""
        result = nrmse(np.array([5.0, 5.0, 5.0]), np.array([4.0, 5.0, 6.0]))
        assert np.isfinite(result)


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = np.random.default_rng(0).uniform(0, 1, (20, 3)).astype(np.float32)
        m = compute_metrics(y, y, physical=True)
        for comp in m["per_component"]:
            assert abs(comp["R2"] - 1.0) < 1e-6
        assert abs(m["overall"]["R2"] - 1.0) < 1e-6

    def test_physical_false_omits_nrmse_and_mdape(self):
        rng = np.random.default_rng(0)
        m = compute_metrics(rng.uniform(0, 1, (20, 3)), rng.uniform(0, 1, (20, 3)), physical=False)
        for comp in m["per_component"]:
            assert "NRMSE" not in comp
            assert "MdAPE" not in comp


# =====================================================================
# predict — catches the log10 ↔ raw K mix-up
# =====================================================================

class TestPredict:
    def test_log_inverse_correctness(self, small_model, synthetic_data, device):
        """With log targets, physical K should be 10^(model-space prediction)."""
        config = deepcopy(CONFIG)
        config["use_log_targets"] = True
        y_model, y_phys = predict(
            small_model, synthetic_data["X_val_scaled"],
            synthetic_data["scaler_Y"], config, device,
        )
        np.testing.assert_allclose(y_phys, np.power(10.0, y_model), rtol=1e-5)

    def test_no_log_returns_same_values(self, small_model, synthetic_data, device):
        config = deepcopy(CONFIG)
        config["use_log_targets"] = False
        y_model, y_phys = predict(
            small_model, synthetic_data["X_val_scaled"],
            synthetic_data["scaler_Y"], config, device,
        )
        np.testing.assert_allclose(y_phys, y_model, rtol=1e-5)


# =====================================================================
# load_params_from_json — ensures tuned params merge correctly
# =====================================================================

class TestLoadParamsFromJson:
    def test_merges_and_keeps_defaults(self, tmp_path):
        params = {"best_params": {"hidden_layers": [64], "dropout_rate": 0.1}}
        path = str(tmp_path / "params.json")
        with open(path, "w") as f:
            json.dump(params, f)

        result = load_params_from_json(path, CONFIG)
        assert result["hidden_layers"] == [64]
        assert result["dropout_rate"] == 0.1
        assert result["max_epochs"] == CONFIG["max_epochs"]  # default preserved

    def test_does_not_mutate_base_config(self, tmp_path):
        original = deepcopy(CONFIG)
        path = str(tmp_path / "params.json")
        with open(path, "w") as f:
            json.dump({"best_params": {"hidden_layers": [64]}}, f)

        load_params_from_json(path, CONFIG)
        assert CONFIG["hidden_layers"] == original["hidden_layers"]
