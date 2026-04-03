"""Shared fixtures for neural-net tests."""

import json
import numpy as np
import pandas as pd
import torch
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler

import pytest

# Add parent directory to path so bare imports work
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_config import CONFIG, PermeabilityMLP, make_loaders


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = CONFIG["feature_cols"]   # 11 features
TARGET_COLS = CONFIG["target_cols"]     # 3 targets (K_xx, K_yy, K_zz)
N_FEATURES = len(FEATURE_COLS)
N_TARGETS = len(TARGET_COLS)
N_SAMPLES = 64


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    """CPU-only device for all tests."""
    return torch.device("cpu")


@pytest.fixture
def mini_config(tmp_path):
    """Minimal CONFIG for fast CPU tests."""
    cfg = deepcopy(CONFIG)
    cfg.update({
        "hidden_layers": [16, 16],
        "batch_size": 8,
        "max_epochs": 5,
        "dropout_rate": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "early_stop_patience": 3,
        "early_stop_min_delta": 1e-8,
        "scheduler_patience": 2,
        "max_grad_norm": 1.0,
        "output_dir": str(tmp_path / "output"),
    })
    return cfg


@pytest.fixture
def synthetic_data():
    """
    Build a dataset dict matching the output of load_and_preprocess_data(),
    using random data (64 samples, split 44/13/7).
    """
    rng = np.random.default_rng(42)

    # Random features in [0, 1] and targets (raw K) in [1e-10, 1e-2]
    X = rng.uniform(0, 1, size=(N_SAMPLES, N_FEATURES)).astype(np.float32)
    Y_raw = rng.uniform(1e-10, 1e-2, size=(N_SAMPLES, N_TARGETS)).astype(np.float32)
    Y_model = np.log10(np.maximum(Y_raw, 1e-30)).astype(np.float32)
    names = np.array([f"sample_{i}" for i in range(N_SAMPLES)])

    # Split: 70/20/10 → 44/13/7
    n_train, n_val = 44, 13
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, N_SAMPLES)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    Y_train_m, Y_val_m, Y_test_m = Y_model[train_idx], Y_model[val_idx], Y_model[test_idx]
    Y_val_raw, Y_test_raw = Y_raw[val_idx], Y_raw[test_idx]

    # Fit scalers on training data only
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_Y = MinMaxScaler(feature_range=(0, 1))

    X_train_s = scaler_X.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler_X.transform(X_val).astype(np.float32)
    X_test_s = scaler_X.transform(X_test).astype(np.float32)

    Y_train_s = scaler_Y.fit_transform(Y_train_m).astype(np.float32)
    Y_val_s = scaler_Y.transform(Y_val_m).astype(np.float32)
    Y_test_s = scaler_Y.transform(Y_test_m).astype(np.float32)

    return {
        "scaler_X": scaler_X,
        "scaler_Y": scaler_Y,
        "X_train_scaled": X_train_s,
        "Y_train_scaled": Y_train_s,
        "X_val_scaled": X_val_s,
        "Y_val_scaled": Y_val_s,
        "X_test_scaled": X_test_s,
        "Y_test_scaled": Y_test_s,
        "X_train_t": torch.from_numpy(X_train_s),
        "Y_train_t": torch.from_numpy(Y_train_s),
        "X_val_t": torch.from_numpy(X_val_s),
        "Y_val_t": torch.from_numpy(Y_val_s),
        "Y_val_raw": Y_val_raw,
        "Y_test_raw": Y_test_raw,
        "Y_val_model": Y_val_m,
        "Y_test_model": Y_test_m,
        "names_val": names[val_idx],
        "names_test": names[test_idx],
        "n_features": N_FEATURES,
        "n_targets": N_TARGETS,
    }


@pytest.fixture
def small_model():
    """Tiny MLP for fast tests (no dropout for determinism)."""
    return PermeabilityMLP(
        n_inputs=N_FEATURES,
        n_outputs=N_TARGETS,
        hidden_layers=[16, 16],
        dropout_rate=0.0,
    )


@pytest.fixture
def trained_loaders(synthetic_data):
    """Train and val DataLoaders from synthetic data."""
    return make_loaders(synthetic_data, batch_size=8)


@pytest.fixture
def synthetic_csv(tmp_path):
    """Write a small CSV with all required columns and return the path."""
    rng = np.random.default_rng(99)
    n = 50
    data = {}
    data["sample_name"] = [f"s_{i}" for i in range(n)]
    for col in FEATURE_COLS:
        data[col] = rng.uniform(0, 1, size=n)
    for col in TARGET_COLS:
        data[col] = rng.uniform(1e-10, 1e-2, size=n)

    csv_path = tmp_path / "test_data.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return str(csv_path)
