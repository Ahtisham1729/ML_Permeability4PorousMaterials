#!/usr/bin/env python3
"""
Shared Configuration and Utilities
==================================
Contains CONFIG, data loading, model definition, and utilities shared between
tune_hyperparams.py and train_model.py.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Data
    "data_path": "256-final.csv",
    "feature_cols": [
        "a20", "a11", "a02", "a10", "a01", "a00",
        "m", "porosity", "tortuosity_geometric_x", "tortuosity_geometric_y", "tortuosity_geometric_z"
    ],
    "target_cols": ["K_xx", "K_yy", "K_zz"],
    "metadata_col": "sample_name",

    # Split: 70/20/10
    "train_frac": 0.70,
    "val_frac": 0.20,
    "test_frac": 0.10,

    # Targets
    "use_log_targets": True,
    "k_floor": 1e-30,

    # Preprocessing
    "random_state": 42,
    "batch_size": 128,

    # Model (defaults - can be overridden by tuned params)
    "hidden_layers": [512, 512, 512, 512],
    "dropout_rate": 0.005,

    # Training
    "max_epochs": 2500,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "scheduler_factor": 0.5,
    "scheduler_patience": 15,
    "early_stop_patience": 100,
    "early_stop_min_delta": 1e-8,

    # Optuna HPO settings (used by tune_hyperparams.py)
    "optuna_trials": 50,
    "optuna_timeout_sec": None,
    "optuna_seed": 42,
    "optuna_pruner": "median",
    "optuna_startup_trials": 10,
    "optuna_warmup_steps": 10,
    "optuna_study_name": "permeability_mlp_hpo",
    "optuna_db": None,

    # Tuning epoch budget (per trial)
    "tune_max_epochs": 400,
    "tune_early_stop_patience": 60,
    "tune_early_stop_min_delta": 1e-7,

    # Output paths
    "output_dir": "forward_model_output",
    "optuna_results_csv": "optuna_trials.csv",
    "best_params_json": "optuna_best_params.json",
    "model_path": "best_model.pt",
    "val_output_csv": "val_results.csv",
    "test_output_csv": "test_results.csv",
    "plot_path_val": "parity_val.png",
    "plot_path_test": "parity_test.png",
    "training_history_path": "training_history.png",

    # -----------------------------------------------------------------
    # Inverse Model — Optuna HPO settings
    # -----------------------------------------------------------------
    "inverse_optuna_trials": 50,
    "inverse_optuna_timeout_sec": None,
    "inverse_optuna_seed": 42,
    "inverse_optuna_pruner": "median",
    "inverse_optuna_startup_trials": 10,
    "inverse_optuna_warmup_steps": 10,
    "inverse_optuna_study_name": "inverse_mlp_hpo",
    "inverse_optuna_db": None,

    # Tuning epoch budget (per trial)
    "inverse_tune_max_epochs": 400,
    "inverse_tune_early_stop_patience": 60,
    "inverse_tune_early_stop_min_delta": 1e-7,

    # Output paths for inverse HPO
    "inverse_optuna_results_csv": "inverse_optuna_trials.csv",
    "inverse_best_params_json": "inverse_optuna_best_params.json",
}

# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# Data Loading & Preprocessing
# =============================================================================

def load_and_preprocess_data(config: dict) -> dict:
    """
    Load CSV, optionally log-transform targets, split into train/val/test (70/20/10),
    scale features and targets (fit scalers on TRAIN only), and return arrays + metadata.
    """
    print("=" * 60)
    print("DATA LOADING & PREPROCESSING")
    print("=" * 60)

    train_frac = float(config["train_frac"])
    val_frac = float(config["val_frac"])
    test_frac = float(config["test_frac"])
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0; got {train_frac+val_frac+test_frac:.6f}")

    df = pd.read_csv(config["data_path"])
    print(f"Loaded {len(df)} samples from {config['data_path']}")

    required_cols = config["feature_cols"] + config["target_cols"] + [config["metadata_col"]]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    X = df[config["feature_cols"]].values.astype(np.float32)
    Y = df[config["target_cols"]].values.astype(np.float32)
    sample_names = df[config["metadata_col"]].values

    if not np.isfinite(X).all():
        raise ValueError("Non-finite values in features (NaN/Inf). Clean before training.")
    if not np.isfinite(Y).all():
        raise ValueError("Non-finite values in targets (NaN/Inf). Clean before training.")

    # Model-space targets
    if config["use_log_targets"]:
        k_floor = float(config.get("k_floor", 0.0))
        if (Y <= 0).any():
            print("[WARN] Non-positive K detected; applying floor before log10.")
        Y_model = np.log10(np.maximum(Y, k_floor)).astype(np.float32)
        print("Targets: log10(K)")
        print(f"  Raw K range:    [{Y.min():.3e}, {Y.max():.3e}]")
        print(f"  log10(K) range: [{Y_model.min():.3f}, {Y_model.max():.3f}]")
    else:
        Y_model = Y
        print("Targets: raw K")
        print(f"  Raw K range: [{Y.min():.3e}, {Y.max():.3e}]")

    # 70/20/10 split via two-stage splitting
    indices = np.arange(len(X))
    temp_frac = val_frac + test_frac

    train_idx, temp_idx = train_test_split(
        indices, test_size=temp_frac, random_state=config["random_state"]
    )

    val_within_temp = val_frac / temp_frac
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1.0 - val_within_temp), random_state=config["random_state"]
    )

    print("\nSplit sizes:")
    print(f"  Train: {len(train_idx)} ({len(train_idx)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(val_idx)} ({len(val_idx)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(test_idx)} ({len(test_idx)/len(X)*100:.1f}%)")

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    Y_train_m, Y_val_m, Y_test_m = Y_model[train_idx], Y_model[val_idx], Y_model[test_idx]
    Y_val_raw, Y_test_raw = Y[val_idx], Y[test_idx]

    names_val, names_test = sample_names[val_idx], sample_names[test_idx]

    # Fit scalers ONLY on train
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_Y = MinMaxScaler(feature_range=(0, 1))

    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)

    Y_train_s = scaler_Y.fit_transform(Y_train_m)
    Y_val_s = scaler_Y.transform(Y_val_m)
    Y_test_s = scaler_Y.transform(Y_test_m)

    # Store tensors for fast loader rebuilding
    X_train_t = torch.from_numpy(X_train_s.astype(np.float32))
    Y_train_t = torch.from_numpy(Y_train_s.astype(np.float32))
    X_val_t = torch.from_numpy(X_val_s.astype(np.float32))
    Y_val_t = torch.from_numpy(Y_val_s.astype(np.float32))

    return {
        "scaler_X": scaler_X,
        "scaler_Y": scaler_Y,

        "X_train_scaled": X_train_s,
        "Y_train_scaled": Y_train_s,
        "X_val_scaled": X_val_s,
        "Y_val_scaled": Y_val_s,
        "X_test_scaled": X_test_s,
        "Y_test_scaled": Y_test_s,

        "X_train_t": X_train_t,
        "Y_train_t": Y_train_t,
        "X_val_t": X_val_t,
        "Y_val_t": Y_val_t,

        "Y_val_raw": Y_val_raw,
        "Y_test_raw": Y_test_raw,
        "Y_val_model": Y_val_m,
        "Y_test_model": Y_test_m,
        "names_val": names_val,
        "names_test": names_test,

        "n_features": X.shape[1],
        "n_targets": Y.shape[1],
    }


def make_loaders(data: dict, batch_size: int) -> dict:
    """Create train/val loaders from cached tensors."""
    train_loader = DataLoader(
        TensorDataset(data["X_train_t"], data["Y_train_t"]),
        batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(data["X_val_t"], data["Y_val_t"]),
        batch_size=batch_size, shuffle=False, drop_last=False
    )
    return {"train_loader": train_loader, "val_loader": val_loader}

# =============================================================================
# Model Definition
# =============================================================================

class PermeabilityMLP(nn.Module):
    """MLP regressor: [Linear→GELU→Dropout]×N → Linear."""
    def __init__(self, n_inputs: int, n_outputs: int, hidden_layers: list[int], dropout_rate: float = 0.05):
        super().__init__()
        layers = []
        in_features = n_inputs
        for h in hidden_layers:
            layers += [nn.Linear(in_features, h), nn.GELU(), nn.Dropout(dropout_rate)]
            in_features = h
        layers.append(nn.Linear(in_features, n_outputs))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Kaiming init."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
