#!/usr/bin/env python3
"""
Shared Configuration and Utilities
==================================
Contains CONFIG, data loading, model definition, and utilities shared between
tune_hyperparams.py and train_model.py.
"""

import logging
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger("permeability")


def setup_logging(level: int = logging.INFO):
    """Configure the 'permeability' logger with a console handler."""
    root = logging.getLogger("permeability")
    if root.handlers:
        return  # already configured
    root.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Data (override via PERMEABILITY_DATA_PATH env var or --data CLI arg)
    "data_path": os.environ.get("PERMEABILITY_DATA_PATH", "256-final.csv"),
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
    "max_grad_norm": 1.0,

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
    logger.info("=" * 60)
    logger.info("DATA LOADING & PREPROCESSING")
    logger.info("=" * 60)

    train_frac = float(config["train_frac"])
    val_frac = float(config["val_frac"])
    test_frac = float(config["test_frac"])
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0; got {train_frac+val_frac+test_frac:.6f}")

    df = pd.read_csv(config["data_path"])
    logger.info("Loaded %d samples from %s", len(df), config["data_path"])

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
            logger.warning("Non-positive K detected; applying floor before log10.")
        Y_model = np.log10(np.maximum(Y, k_floor)).astype(np.float32)
        logger.info("Targets: log10(K)")
        logger.info("  Raw K range:    [%s, %s]", f"{Y.min():.3e}", f"{Y.max():.3e}")
        logger.info("  log10(K) range: [%.3f, %.3f]", Y_model.min(), Y_model.max())
    else:
        Y_model = Y
        logger.info("Targets: raw K")
        logger.info("  Raw K range: [%s, %s]", f"{Y.min():.3e}", f"{Y.max():.3e}")

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

    logger.info("Split sizes:")
    logger.info("  Train: %d (%.1f%%)", len(train_idx), len(train_idx)/len(X)*100)
    logger.info("  Val:   %d (%.1f%%)", len(val_idx), len(val_idx)/len(X)*100)
    logger.info("  Test:  %d (%.1f%%)", len(test_idx), len(test_idx)/len(X)*100)

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


def make_loaders(data: dict, batch_size: int, seed: int = 42) -> dict:
    """Create train/val loaders from cached tensors with a seeded generator."""
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(data["X_train_t"], data["Y_train_t"]),
        batch_size=batch_size, shuffle=True, drop_last=False, generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(data["X_val_t"], data["Y_val_t"]),
        batch_size=batch_size, shuffle=False, drop_last=False,
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


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """Early stopping on validation loss; stores best weights."""
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_weights = None
        self.counter = 0

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_weights = deepcopy(model.state_dict())
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


# =============================================================================
# Training Utilities
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device,
                    max_grad_norm: float = 0.0) -> float:
    """Single training epoch. Applies gradient clipping if max_grad_norm > 0."""
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


# =============================================================================
# Scaler Serialization
# =============================================================================

def save_scaler(scaler: MinMaxScaler) -> dict:
    """Serialize a MinMaxScaler to a plain dict of numpy arrays."""
    return {
        "data_min_": scaler.data_min_.tolist(),
        "data_max_": scaler.data_max_.tolist(),
        "data_range_": scaler.data_range_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "min_": scaler.min_.tolist(),
        "feature_range": list(scaler.feature_range),
        "n_features_in_": int(scaler.n_features_in_),
        "n_samples_seen_": int(scaler.n_samples_seen_),
    }


def load_scaler(d: dict) -> MinMaxScaler:
    """Reconstruct a MinMaxScaler from a plain dict."""
    scaler = MinMaxScaler(feature_range=tuple(d["feature_range"]))
    scaler.data_min_ = np.array(d["data_min_"], dtype=np.float64)
    scaler.data_max_ = np.array(d["data_max_"], dtype=np.float64)
    scaler.data_range_ = np.array(d["data_range_"], dtype=np.float64)
    scaler.scale_ = np.array(d["scale_"], dtype=np.float64)
    scaler.min_ = np.array(d["min_"], dtype=np.float64)
    scaler.n_features_in_ = int(d["n_features_in_"])
    scaler.n_samples_seen_ = int(d["n_samples_seen_"])
    return scaler
