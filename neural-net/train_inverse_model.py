#!/usr/bin/env python3
"""
Inverse Design Training Script
===============================
Trains an inverse MLP with forward consistency loss.

Forward model (frozen):  features (11) → K (3)
Inverse model (trained): K (3)         → features (11)

Three loss modes (selectable via --loss_mode):

  1. "geometry"  (default):
     L = α × L_fwd  +  β × L_geo
     L_fwd = ||FNN(Inv(K)) - K||²
     L_geo = ||Inv(K) - θ_true||²
     → Eq. 23 from reference: forward consistency + geometry reconstruction

  2. "bounding":
     L = L_fwd  +  λ × L_bound
     L_bound penalises predictions outside [θ_min, θ_max] with zero penalty inside
     → Replaces geometry loss to avoid averaging artefact

  3. "fwd_only":
     L = L_fwd
     → Pure forward consistency, no parameter regularisation

Usage:
    python train_inverse_model.py -f diagonal_model.pt
    python train_inverse_model.py -f diagonal_model.pt --loss_mode bounding
    python train_inverse_model.py -f diagonal_model.pt --loss_mode geometry --w_param 0.05
    python train_inverse_model.py -f diagonal_model.pt --resume inverse_best.pt

    # Optuna hyperparameter tuning (tune + train in one run):
    python train_inverse_model.py -f diagonal_model.pt --tune

    # Tune only (save best params, skip training):
    python train_inverse_model.py -f diagonal_model.pt --tune_only

    # Train with previously tuned parameters:
    python train_inverse_model.py -f diagonal_model.pt --params inverse_optuna_best_params.json
"""

import argparse
import gc
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict
from sklearn.metrics import r2_score

try:
    import optuna
except ImportError:
    optuna = None  # Only required when --tune is used

# Import from your existing forward model infrastructure
from model_config import (
    CONFIG, set_seed, setup_logging, load_and_preprocess_data, PermeabilityMLP,
    EarlyStopping, save_scaler, load_scaler,
)

logger = logging.getLogger("permeability")


# =============================================================================
# Inverse Model Architecture
# =============================================================================

class InverseMLP(nn.Module):
    """
    Inverse MLP: K (3) → all forward model input features (11)

    Input:  [K_xx, K_yy, K_zz]  (3, scaled)
    Output: [a20, a11, a02, a10, a01, a00, m, porosity,
             tortuosity_geometric_x, tortuosity_geometric_y,
             tortuosity_geometric_z]  (11, scaled)

    Note: The model predicts all 11 features internally (needed for forward
    consistency through the FNN), but only the 8 design parameters
    (a20..a00, m, porosity) are reported and exported — tortuosity is
    an auxiliary prediction not needed for microstructure generation.
    """

    def __init__(self, n_inputs: int = 3, n_outputs: int = 11,
                 hidden_layers: list = None, dropout_rate: float = 0.1):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [256, 512, 512, 512, 256]

        layers = []
        prev = n_inputs
        for h in hidden_layers:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            ])
            prev = h
        layers.append(nn.Linear(prev, n_outputs))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small init for final layer
        final = [m for m in self.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.xavier_uniform_(final.weight, gain=0.1)
        nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# =============================================================================
# Loss Functions
# =============================================================================

def bounding_loss(theta_pred: torch.Tensor,
                  theta_min: torch.Tensor,
                  theta_max: torch.Tensor) -> torch.Tensor:
    """
    Bounding penalty loss — zero inside [θ_min, θ_max], quadratic outside.

    From the reference paper: replaces direct geometry loss to avoid
    forcing predictions into a tight band (averaging artefact).

    Args:
        theta_pred: predicted parameters, shape (batch, n_features)
        theta_min:  per-feature lower bounds, shape (n_features,)
        theta_max:  per-feature upper bounds, shape (n_features,)

    Returns:
        scalar loss
    """
    below = torch.clamp(theta_min - theta_pred, min=0) ** 2
    above = torch.clamp(theta_pred - theta_max, min=0) ** 2
    return torch.mean(below + above)


# =============================================================================
# Configuration
# =============================================================================

INVERSE_CONFIG = {
    # Architecture
    "hidden_layers": [256, 512, 512, 512, 256],
    "dropout_rate": 0.1,

    # Design parameters (reported/exported) vs auxiliary (predicted internally only)
    "design_param_cols": ["a20", "a11", "a02", "a10", "a01", "a00", "m", "porosity"],
    "auxiliary_cols": ["tortuosity_geometric_x", "tortuosity_geometric_y", "tortuosity_geometric_z"],

    # Training
    "batch_size": 128,
    "max_epochs": 1000,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,

    # Scheduler
    "scheduler_factor": 0.5,
    "scheduler_patience": 30,

    # Early stopping
    "early_stop_patience": 100,
    "early_stop_min_delta": 1e-8,
    "max_grad_norm": 1.0,

    # Loss mode: "geometry", "bounding", or "fwd_only"
    "loss_mode": "geometry",

    # Weights for geometry mode:  L = w_fwd * L_fwd + w_param * L_geo
    "w_forward_consistency": 1.0,
    "w_parameter_recon": 0.1,

    # Weight for bounding mode:  L = L_fwd + w_bounding * L_bound
    "w_bounding": 1.0,
    "bound_margin": 0.0,  # expand bounds by this fraction (0 = exact data range)

    # Saving
    "save_every": 20,
    "output_dir": "inverse_model_output",
}


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_inverse_data(forward_checkpoint_path: str, device: torch.device):
    """
    Load data using the same pipeline as forward training, then reorganise
    for inverse training:  K (targets) become inputs, features become outputs.
    """
    logger.info("=" * 60)
    logger.info("DATA PREPARATION (reusing forward model pipeline)")
    logger.info("=" * 60)

    # Load forward checkpoint
    checkpoint = torch.load(forward_checkpoint_path, map_location=device, weights_only=True)
    fwd_config = checkpoint["config"]
    scaler_X = load_scaler(checkpoint["scaler_X"])
    scaler_Y = load_scaler(checkpoint["scaler_Y"])

    logger.info("Forward config loaded from checkpoint")
    logger.info("  Features (%d): %s", len(fwd_config["feature_cols"]), fwd_config["feature_cols"])
    logger.info("  Targets  (%d): %s", len(fwd_config["target_cols"]), fwd_config["target_cols"])

    # Reload data using the SAME config
    data = load_and_preprocess_data(fwd_config)

    n_features = data["n_features"]
    n_targets = data["n_targets"]

    # Build and load forward model (frozen)
    forward_model = PermeabilityMLP(
        n_inputs=n_features,
        n_outputs=n_targets,
        hidden_layers=fwd_config["hidden_layers"],
        dropout_rate=fwd_config["dropout_rate"],
    )
    forward_model.load_state_dict(checkpoint["model_state_dict"])
    forward_model.to(device)
    forward_model.eval()
    for p in forward_model.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in forward_model.parameters())
    logger.info("Forward model loaded: %s parameters (frozen)", f"{n_params:,}")

    # Prepare tensors
    dtype = torch.float32

    k_train = torch.tensor(data["Y_train_scaled"], dtype=dtype, device=device)
    theta_train = torch.tensor(data["X_train_scaled"], dtype=dtype, device=device)

    k_val = torch.tensor(data["Y_val_scaled"], dtype=dtype, device=device)
    theta_val = torch.tensor(data["X_val_scaled"], dtype=dtype, device=device)

    k_test = torch.tensor(data["Y_test_scaled"], dtype=dtype, device=device)
    theta_test = torch.tensor(data["X_test_scaled"], dtype=dtype, device=device)

    # Compute parameter bounds from training data (for bounding loss)
    theta_min = theta_train.min(dim=0).values
    theta_max = theta_train.max(dim=0).values

    logger.info("Inverse training data:")
    logger.info("  Train: K %s → θ %s", k_train.shape, theta_train.shape)
    logger.info("  Val:   K %s → θ %s", k_val.shape, theta_val.shape)
    logger.info("  Test:  K %s → θ %s", k_test.shape, theta_test.shape)

    return {
        "k_train": k_train, "theta_train": theta_train,
        "k_val": k_val, "theta_val": theta_val,
        "k_test": k_test, "theta_test": theta_test,
        "theta_min": theta_min, "theta_max": theta_max,
        "forward_model": forward_model,
        "fwd_config": fwd_config,
        "scaler_X": scaler_X,
        "scaler_Y": scaler_Y,
        "n_features": n_features,
        "n_targets": n_targets,
        "data": data,
    }


# =============================================================================
# Optuna HPO for Inverse Model
# =============================================================================

def suggest_inverse_hidden_layers(trial) -> list[int]:
    """
    Sample a symmetric (encoder-decoder) width schedule for the inverse MLP.

    The inverse model maps a low-dimensional input (K, 3 dims) to a
    higher-dimensional output (features, 11 dims), so the architecture
    expands then contracts (bottleneck style).
    """
    n_layers = trial.suggest_int("n_layers", 3, 7)
    hidden_base = trial.suggest_int("hidden_base", 128, 1024, step=64)
    width_decay = trial.suggest_float("width_decay", 0.5, 1.0)

    # Build expanding half
    half = n_layers // 2
    dims_up = []
    w = float(hidden_base) * (width_decay ** half)  # start narrow
    for _ in range(half):
        dims_up.append(int(max(32, round(w / 8) * 8)))
        w /= width_decay  # expand towards base

    # Middle layer at full width
    dims_mid = [int(max(32, round(float(hidden_base) / 8) * 8))]

    # Build contracting half
    dims_down = []
    w = float(hidden_base)
    for _ in range(n_layers - half - 1):
        w *= width_decay
        dims_down.append(int(max(32, round(w / 8) * 8)))

    return dims_up + dims_mid + dims_down


def run_inverse_optuna_hpo(prep: dict, inv_config: dict, config: dict,
                           device: torch.device) -> dict:
    """
    Run Optuna HPO to find optimal hyperparameters for the inverse MLP.

    Optimises validation forward-consistency MSE (scaled space) as the
    primary objective — this is the metric that matters most for inverse
    design quality.
    """
    if optuna is None:
        raise ImportError("Optuna is required for --tune. Install with: pip install optuna")

    logger.info("=" * 60)
    logger.info("OPTUNA HYPERPARAMETER OPTIMIZATION (Inverse Model)")
    logger.info("=" * 60)

    loss_mode = inv_config["loss_mode"]
    logger.info("  Loss mode: %s", loss_mode)

    sampler = optuna.samplers.TPESampler(seed=int(config["inverse_optuna_seed"]))

    pruner_name = str(config.get("inverse_optuna_pruner", "median")).lower()
    if pruner_name == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(config["inverse_optuna_startup_trials"]),
            n_warmup_steps=int(config["inverse_optuna_warmup_steps"]),
        )
    else:
        pruner = optuna.pruners.NopPruner()

    storage = config.get("inverse_optuna_db", None)
    study = optuna.create_study(
        study_name=str(config["inverse_optuna_study_name"]),
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=bool(storage),
    )

    # Cached tensors
    k_train, theta_train = prep["k_train"], prep["theta_train"]
    k_val, theta_val = prep["k_val"], prep["theta_val"]
    forward_model = prep["forward_model"]
    n_features = prep["n_features"]
    n_targets = prep["n_targets"]

    # Bounding loss bounds
    theta_min = prep["theta_min"]
    theta_max = prep["theta_max"]

    tune_max_epochs = int(config["inverse_tune_max_epochs"])
    tune_es_patience = int(config["inverse_tune_early_stop_patience"])
    tune_es_min_delta = float(config["inverse_tune_early_stop_min_delta"])

    def objective(trial: optuna.Trial) -> float:
        base_seed = int(config["random_state"])
        set_seed(base_seed + int(trial.number))

        # --- Architecture ---
        hidden_layers = suggest_inverse_hidden_layers(trial)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3)

        # --- Optimiser ---
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

        # --- Scheduler ---
        scheduler_factor = trial.suggest_float("scheduler_factor", 0.3, 0.8)
        scheduler_patience = trial.suggest_int("scheduler_patience", 8, 30)

        # --- Loss weights (mode-dependent) ---
        if loss_mode == "geometry":
            w_fwd = 1.0
            w_param = trial.suggest_float("w_parameter_recon", 0.01, 1.0, log=True)
        elif loss_mode == "bounding":
            w_fwd = 1.0
            w_bound = trial.suggest_float("w_bounding", 0.1, 10.0, log=True)
            bound_margin = trial.suggest_float("bound_margin", 0.0, 0.2)
            theta_range = theta_max - theta_min
            theta_min_b = theta_min - bound_margin * theta_range
            theta_max_b = theta_max + bound_margin * theta_range
        else:  # fwd_only
            w_fwd = 1.0

        # --- DataLoader ---
        dataset = TensorDataset(k_train, theta_train)
        loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)

        # --- Build model ---
        model = InverseMLP(
            n_inputs=n_targets,
            n_outputs=n_features,
            hidden_layers=hidden_layers,
            dropout_rate=float(dropout_rate),
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=float(scheduler_factor),
            patience=int(scheduler_patience),
        )

        early_stop = EarlyStopping(
            patience=tune_es_patience,
            min_delta=tune_es_min_delta,
        )

        best_metric = float("inf")
        hpo_max_grad_norm = float(inv_config.get("max_grad_norm", 0.0))

        for epoch in range(1, tune_max_epochs + 1):
            # --- Train ---
            model.train()
            for k_batch, theta_batch in loader:
                optimizer.zero_grad(set_to_none=True)
                theta_pred = model(k_batch)
                k_recon = forward_model(theta_pred)
                loss_fwd = torch.mean((k_recon - k_batch) ** 2)

                if loss_mode == "geometry":
                    loss_reg = torch.mean((theta_pred - theta_batch) ** 2)
                    loss = w_fwd * loss_fwd + w_param * loss_reg
                elif loss_mode == "bounding":
                    loss_reg = bounding_loss(theta_pred, theta_min_b, theta_max_b)
                    loss = w_fwd * loss_fwd + w_bound * loss_reg
                else:
                    loss = w_fwd * loss_fwd

                loss.backward()
                if hpo_max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), hpo_max_grad_norm)
                optimizer.step()

            # --- Validate (forward consistency MSE on val set) ---
            model.eval()
            with torch.no_grad():
                tp_val = model(k_val)
                kr_val = forward_model(tp_val)
                val_fwd_mse = torch.mean((kr_val - k_val) ** 2).item()

                if loss_mode == "geometry":
                    val_reg = torch.mean((tp_val - theta_val) ** 2).item()
                    val_total = w_fwd * val_fwd_mse + w_param * val_reg
                elif loss_mode == "bounding":
                    val_reg = bounding_loss(tp_val, theta_min_b, theta_max_b).item()
                    val_total = w_fwd * val_fwd_mse + w_bound * val_reg
                else:
                    val_total = w_fwd * val_fwd_mse

            scheduler.step(val_total)

            # Report forward consistency MSE as the optimisation target
            trial.report(val_fwd_mse, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_fwd_mse < best_metric - 1e-12:
                best_metric = val_fwd_mse

            if early_stop(val_total, model):
                break

        del model, optimizer, scheduler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return float(best_metric)

    study.optimize(
        objective,
        n_trials=int(config["inverse_optuna_trials"]),
        timeout=config["inverse_optuna_timeout_sec"],
        show_progress_bar=True,
    )

    logger.info("Best trial:")
    logger.info("  value (val forward-consistency MSE): %s", f"{study.best_value:.6e}")
    logger.info("  params: %s", study.best_params)

    # Save trials table
    output_dir = Path(inv_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    trials_csv = str(output_dir / config["inverse_optuna_results_csv"])
    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_csv, index=False)
    logger.info("Saved Optuna trials: %s", trials_csv)

    # Save best params with reconstructed hidden_layers
    best_params = study.best_params.copy()

    if "n_layers" in best_params and "hidden_base" in best_params and "width_decay" in best_params:
        n_layers = int(best_params["n_layers"])
        hidden_base = int(best_params["hidden_base"])
        width_decay = float(best_params["width_decay"])

        half = n_layers // 2
        dims_up = []
        w = float(hidden_base) * (width_decay ** half)
        for _ in range(half):
            dims_up.append(int(max(32, round(w / 8) * 8)))
            w /= width_decay
        dims_mid = [int(max(32, round(float(hidden_base) / 8) * 8))]
        dims_down = []
        w = float(hidden_base)
        for _ in range(n_layers - half - 1):
            w *= width_decay
            dims_down.append(int(max(32, round(w / 8) * 8)))
        best_params["hidden_layers"] = dims_up + dims_mid + dims_down

    output_json = {
        "best_value": float(study.best_value),
        "loss_mode": loss_mode,
        "best_params": best_params,
    }

    params_path = str(output_dir / config["inverse_best_params_json"])
    with open(params_path, "w") as f:
        json.dump(output_json, f, indent=2)
    logger.info("Saved best params: %s", params_path)

    return output_json


def load_inverse_params_from_json(path: str, base_config: dict) -> dict:
    """Load tuned inverse parameters from JSON and merge with base config."""
    with open(path, "r") as f:
        data = json.load(f)

    params = data.get("best_params", data)
    config = deepcopy(base_config)

    # Direct mappings
    param_keys = [
        "hidden_layers", "dropout_rate", "learning_rate", "weight_decay",
        "batch_size", "scheduler_factor", "scheduler_patience",
        "w_parameter_recon", "w_bounding", "bound_margin",
    ]
    for key in param_keys:
        if key in params:
            config[key] = params[key]

    # Override loss_mode if saved in the JSON
    if "loss_mode" in data:
        config["loss_mode"] = data["loss_mode"]

    return config


# =============================================================================
# Training
# =============================================================================

def train_inverse(inv_config: dict, prep: dict, device: torch.device,
                  resume_path: str = None):
    """Train the inverse model."""

    logger.info("=" * 60)
    logger.info("INVERSE MODEL TRAINING")
    logger.info("=" * 60)

    output_dir = Path(inv_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    # Unpack
    k_train, theta_train = prep["k_train"], prep["theta_train"]
    k_val, theta_val = prep["k_val"], prep["theta_val"]
    forward_model = prep["forward_model"]
    n_features = prep["n_features"]
    n_targets = prep["n_targets"]

    # Bounding loss bounds (with optional margin)
    margin = inv_config.get("bound_margin", 0.0)
    theta_range = prep["theta_max"] - prep["theta_min"]
    theta_min_b = prep["theta_min"] - margin * theta_range
    theta_max_b = prep["theta_max"] + margin * theta_range

    # DataLoader (seeded for reproducibility)
    g = torch.Generator()
    g.manual_seed(int(CONFIG.get("random_state", 42)))
    dataset = TensorDataset(k_train, theta_train)
    loader = DataLoader(dataset, batch_size=inv_config["batch_size"], shuffle=True,
                        generator=g)

    # Build inverse model
    inverse_model = InverseMLP(
        n_inputs=n_targets,
        n_outputs=n_features,
        hidden_layers=inv_config["hidden_layers"],
        dropout_rate=inv_config["dropout_rate"],
    ).to(device)

    if resume_path:
        state = torch.load(resume_path, map_location=device, weights_only=True)
        sd = state if isinstance(state, dict) and "model_state_dict" not in state else state.get("model_state_dict", state)
        clean = OrderedDict()
        for k, v in sd.items():
            clean[k.replace("module.", "")] = v
        inverse_model.load_state_dict(clean)
        logger.info("Resumed from: %s", resume_path)

    n_params = sum(p.numel() for p in inverse_model.parameters() if p.requires_grad)
    logger.info("Inverse model: K (%d) → features (%d)", n_targets, n_features)
    logger.info("  Architecture: %d → %s → %d", n_targets, inv_config["hidden_layers"], n_features)
    logger.info("  Trainable parameters: %s", f"{n_params:,}")

    # Loss configuration
    loss_mode = inv_config["loss_mode"]
    w_fwd = inv_config["w_forward_consistency"]

    if loss_mode == "geometry":
        w_param = inv_config["w_parameter_recon"]
        logger.info("Loss mode: GEOMETRY")
        logger.info("  L = %s×||FNN(Inv(K)) - K||²  +  %s×||Inv(K) - θ_true||²", w_fwd, w_param)
    elif loss_mode == "bounding":
        w_bound = inv_config["w_bounding"]
        logger.info("Loss mode: BOUNDING")
        logger.info("  L = %s×||FNN(Inv(K)) - K||²  +  %s×L_bound", w_fwd, w_bound)
        logger.info("  L_bound = 0 inside bounds, quadratic outside")
        if margin > 0:
            logger.info("  Bounds expanded by %.0f%% margin", margin*100)
    elif loss_mode == "fwd_only":
        logger.info("Loss mode: FORWARD ONLY")
        logger.info("  L = %s×||FNN(Inv(K)) - K||²", w_fwd)
    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}. Use 'geometry', 'bounding', or 'fwd_only'.")

    # Optimizer, scheduler, early stopping
    optimizer = optim.AdamW(
        inverse_model.parameters(),
        lr=inv_config["learning_rate"],
        weight_decay=inv_config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=inv_config["scheduler_factor"],
        patience=inv_config["scheduler_patience"],
    )
    early_stop = EarlyStopping(
        patience=inv_config["early_stop_patience"],
        min_delta=inv_config["early_stop_min_delta"],
    )

    max_grad_norm = float(inv_config.get("max_grad_norm", 0.0))
    history = {
        "train_fwd": [], "val_fwd": [],
        "train_reg": [], "val_reg": [],
        "train_total": [], "val_total": [],
        "lr": [],
    }

    logger.info("Starting training for %d epochs...", inv_config["max_epochs"])

    for epoch in range(1, inv_config["max_epochs"] + 1):

        # --- Train ---
        inverse_model.train()
        for k_batch, theta_batch in loader:
            optimizer.zero_grad(set_to_none=True)

            theta_pred = inverse_model(k_batch)
            k_recon = forward_model(theta_pred)

            loss_fwd = torch.mean((k_recon - k_batch) ** 2)

            if loss_mode == "geometry":
                loss_reg = torch.mean((theta_pred - theta_batch) ** 2)
                loss = w_fwd * loss_fwd + w_param * loss_reg
            elif loss_mode == "bounding":
                loss_reg = bounding_loss(theta_pred, theta_min_b, theta_max_b)
                loss = w_fwd * loss_fwd + w_bound * loss_reg
            else:  # fwd_only
                loss_reg = torch.tensor(0.0, device=device)
                loss = w_fwd * loss_fwd

            loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(inverse_model.parameters(), max_grad_norm)
            optimizer.step()

        # --- Evaluate (train + val only; test is held out until after training) ---
        inverse_model.eval()
        with torch.no_grad():
            # Train
            tp_train = inverse_model(k_train)
            kr_train = forward_model(tp_train)
            tr_fwd = torch.mean((kr_train - k_train) ** 2).item()
            if loss_mode == "geometry":
                tr_reg = torch.mean((tp_train - theta_train) ** 2).item()
            elif loss_mode == "bounding":
                tr_reg = bounding_loss(tp_train, theta_min_b, theta_max_b).item()
            else:
                tr_reg = 0.0

            # Val
            tp_val = inverse_model(k_val)
            kr_val = forward_model(tp_val)
            va_fwd = torch.mean((kr_val - k_val) ** 2).item()
            if loss_mode == "geometry":
                va_reg = torch.mean((tp_val - theta_val) ** 2).item()
            elif loss_mode == "bounding":
                va_reg = bounding_loss(tp_val, theta_min_b, theta_max_b).item()
            else:
                va_reg = 0.0

        if loss_mode == "geometry":
            tr_total = w_fwd * tr_fwd + w_param * tr_reg
            va_total = w_fwd * va_fwd + w_param * va_reg
        elif loss_mode == "bounding":
            tr_total = w_fwd * tr_fwd + w_bound * tr_reg
            va_total = w_fwd * va_fwd + w_bound * va_reg
        else:
            tr_total = w_fwd * tr_fwd
            va_total = w_fwd * va_fwd

        lr = optimizer.param_groups[0]["lr"]

        history["train_fwd"].append(tr_fwd)
        history["val_fwd"].append(va_fwd)
        history["train_reg"].append(tr_reg)
        history["val_reg"].append(va_reg)
        history["train_total"].append(tr_total)
        history["val_total"].append(va_total)
        history["lr"].append(lr)

        scheduler.step(va_total)

        reg_label = {"geometry": "geo", "bounding": "bound", "fwd_only": "—"}[loss_mode]
        if epoch == 1 or epoch % 20 == 0:
            logger.info("Epoch %4d | train: fwd=%s %s=%s | val: fwd=%s %s=%s | lr=%s",
                        epoch, f"{tr_fwd:.3e}", reg_label, f"{tr_reg:.3e}",
                        f"{va_fwd:.3e}", reg_label, f"{va_reg:.3e}", f"{lr:.1e}")

        if early_stop(va_total, inverse_model):
            logger.info("Early stop at epoch %d | best val=%s", epoch, f"{early_stop.best_loss:.6e}")
            break

        # Periodic checkpoint
        if epoch % inv_config["save_every"] == 0:
            torch.save({
                "model_state_dict": inverse_model.state_dict(),
                "inv_config": inv_config,
                "epoch": epoch,
            }, output_dir / "checkpoints" / f"inverse_epoch_{epoch}.pt")

    # Restore best
    early_stop.restore_best(inverse_model)
    logger.info("Restored best weights | best val=%s", f"{early_stop.best_loss:.6e}")

    # Save final (scalers as plain dicts for weights_only=True compatibility)
    torch.save({
        "model_state_dict": inverse_model.state_dict(),
        "inv_config": inv_config,
        "fwd_config": prep["fwd_config"],
        "scaler_X": save_scaler(prep["scaler_X"]),
        "scaler_Y": save_scaler(prep["scaler_Y"]),
    }, output_dir / "inverse_best.pt")
    logger.info("Saved: %s", output_dir / "inverse_best.pt")

    return inverse_model, history


# =============================================================================
# Evaluation
# =============================================================================

def to_physical_K(k_scaled: np.ndarray, scaler_Y, use_log: bool) -> np.ndarray:
    """Convert scaled K back to physical units."""
    k_model = scaler_Y.inverse_transform(k_scaled)
    if use_log:
        return np.power(10.0, k_model)
    return k_model


def evaluate_inverse(inverse_model: nn.Module, prep: dict, inv_config: dict,
                     device: torch.device):
    """Full evaluation with forward consistency, parameter metrics, and plots."""

    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    output_dir = Path(inv_config["output_dir"])
    fwd_config = prep["fwd_config"]
    forward_model = prep["forward_model"]
    scaler_X = prep["scaler_X"]
    scaler_Y = prep["scaler_Y"]
    use_log = fwd_config["use_log_targets"]
    feature_names = fwd_config["feature_cols"]
    target_names = fwd_config["target_cols"]

    # Identify design parameter indices (exclude tortuosity)
    design_cols = inv_config.get("design_param_cols",
                                 ["a20", "a11", "a02", "a10", "a01", "a00", "m", "porosity"])
    design_idx = [i for i, name in enumerate(feature_names) if name in design_cols]
    design_names = [feature_names[i] for i in design_idx]

    inverse_model.eval()

    for split_name, k_tensor, theta_tensor in [
        ("Validation", prep["k_val"], prep["theta_val"]),
        ("Test", prep["k_test"], prep["theta_test"]),
    ]:
        with torch.no_grad():
            theta_pred = inverse_model(k_tensor)
            k_achieved = forward_model(theta_pred)

        k_true_sc = k_tensor.cpu().numpy()
        k_ach_sc = k_achieved.cpu().numpy()
        theta_pred_sc = theta_pred.cpu().numpy()
        theta_true_sc = theta_tensor.cpu().numpy()

        # Physical K
        k_true_phys = to_physical_K(k_true_sc, scaler_Y, use_log)
        k_ach_phys = to_physical_K(k_ach_sc, scaler_Y, use_log)

        # Relative error
        rel_err = np.abs(k_ach_phys - k_true_phys) / (np.abs(k_true_phys) + 1e-20)

        logger.info("--- %s (%d samples) ---", split_name, len(k_true_sc))

        fwd_mse = np.mean((k_ach_sc - k_true_sc) ** 2)
        logger.info("  Forward consistency MSE (scaled): %s", f"{fwd_mse:.6e}")

        logger.info("  Permeability reconstruction (physical):")
        logger.info("    Mean relative error:   %.2f%%", np.mean(rel_err)*100)
        logger.info("    Median relative error: %.2f%%", np.median(rel_err)*100)
        logger.info("    Samples < 5%% error:    %.1f%%", np.mean(rel_err < 0.05)*100)
        logger.info("    Samples < 10%% error:   %.1f%%", np.mean(rel_err < 0.10)*100)

        for i, name in enumerate(target_names):
            r2 = r2_score(
                np.log10(np.maximum(k_true_phys[:, i], 1e-30)),
                np.log10(np.maximum(k_ach_phys[:, i], 1e-30)),
            )
            logger.info("    %s: MRE=%.2f%%  R²(log)=%.4f", name, np.mean(rel_err[:, i])*100, r2)

        # Parameter reconstruction — design parameters only
        theta_pred_phys = scaler_X.inverse_transform(theta_pred_sc)
        theta_true_phys = scaler_X.inverse_transform(theta_true_sc)

        logger.info("  Design parameter reconstruction:")
        for i in design_idx:
            name = feature_names[i]
            mae = np.mean(np.abs(theta_pred_phys[:, i] - theta_true_phys[:, i]))
            rng = np.max(theta_true_phys[:, i]) - np.min(theta_true_phys[:, i])
            nrmse_val = mae / (rng + 1e-20)
            logger.info("    %35s: MAE=%.4f  NRMSE=%.4f", name, mae, nrmse_val)

        # --- K parity plot ---
        _plot_k_parity(
            k_true_phys, k_ach_phys, target_names,
            str(output_dir / f"inverse_parity_K_{split_name.lower()}.png"),
            f"Inverse Design — {split_name} (Forward Consistency)",
        )

        # --- Parameter parity plots (design params only, no tortuosity) ---
        _plot_parameter_parity(
            theta_true_phys[:, design_idx],
            theta_pred_phys[:, design_idx],
            design_names,
            str(output_dir / f"inverse_parity_params_{split_name.lower()}.png"),
            f"Design Parameter Predictions — {split_name} ({inv_config['loss_mode']} loss)",
        )

    # Example inverse designs (design params only)
    _print_examples(inverse_model, forward_model, prep, scaler_X, scaler_Y,
                    feature_names, target_names, design_idx, design_names,
                    use_log, device)


# =============================================================================
# Plotting
# =============================================================================

def _plot_k_parity(k_true, k_pred, names, path, title):
    """Parity plot for permeability in log10 space."""
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5))
    if len(names) == 1:
        axes = [axes]

    for ax, name, i in zip(axes, names, range(len(names))):
        yt = np.log10(np.maximum(k_true[:, i], 1e-30))
        yp = np.log10(np.maximum(k_pred[:, i], 1e-30))
        ax.scatter(yt, yp, alpha=0.4, s=12, edgecolors="none")
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        pad = 0.05 * (hi - lo + 1e-12)
        lims = [lo - pad, hi + pad]
        ax.plot(lims, lims, "r--", lw=1.5, alpha=0.8)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f"Target {name}")
        ax.set_ylabel(f"Achieved {name}")
        ax.set_title(f"{name}", fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("  Saved: %s", path)


def _plot_parameter_parity(theta_true, theta_pred, names, path, title):
    """
    Parameter parity plots — one subplot per design parameter,
    matching the style of Figure 26 from the reference paper.

    Layout adapts to number of parameters (e.g. 3×3 for 8 params).
    Each subplot shows actual vs predicted with 1:1 reference line.
    """
    n = len(names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes_flat = axes.flatten()

    for i in range(n):
        ax = axes_flat[i]
        name = names[i]
        yt = theta_true[:, i]
        yp = theta_pred[:, i]

        ax.scatter(yt, yp, alpha=0.3, s=10, edgecolors="none", c="steelblue")

        # 1:1 reference line
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        pad = 0.05 * (hi - lo + 1e-12)
        lims = [lo - pad, hi + pad]
        ax.plot(lims, lims, "r--", lw=1.5, alpha=0.8)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name} prediction", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)

    # Hide unused subplots
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle(title, y=1.01, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("  Saved: %s", path)


def plot_inverse_history(history: dict, inv_config: dict, output_path: str):
    """Plot training curves."""
    loss_mode = inv_config["loss_mode"]
    reg_label = {
        "geometry": "||Inv(K) − θ_true||²",
        "bounding": "L_bound",
        "fwd_only": "N/A",
    }[loss_mode]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    epochs = range(1, len(history["train_fwd"]) + 1)

    # Forward consistency
    axes[0].plot(epochs, history["train_fwd"], label="Train", lw=1.2)
    axes[0].plot(epochs, history["val_fwd"], label="Val", lw=1.2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE")
    axes[0].set_title("Forward Consistency ||FNN(Inv(K)) − K||²")
    axes[0].set_yscale("log"); axes[0].grid(True, alpha=0.3); axes[0].legend()

    # Regularisation term
    if loss_mode != "fwd_only":
        axes[1].plot(epochs, history["train_reg"], label="Train", lw=1.2)
        axes[1].plot(epochs, history["val_reg"], label="Val", lw=1.2)
        axes[1].set_title(f"Regularisation: {reg_label}")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No regularisation\n(fwd_only mode)",
                     ha="center", va="center", fontsize=12, transform=axes[1].transAxes)
        axes[1].set_title("Regularisation")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log"); axes[1].grid(True, alpha=0.3)

    # LR
    axes[2].plot(epochs, history["lr"], lw=1.2)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR")
    axes[2].set_title("Learning Rate")
    axes[2].set_yscale("log"); axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"Training History — {loss_mode} loss", y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved: %s", output_path)


# =============================================================================
# Example Inverse Designs
# =============================================================================

def _print_examples(inverse_model, forward_model, prep, scaler_X, scaler_Y,
                    feature_names, target_names, design_idx, design_names,
                    use_log, device, n=5):
    """Print example inverse designs from test set (design parameters only)."""
    logger.info("=" * 60)
    logger.info("EXAMPLE INVERSE DESIGNS (from test set)")
    logger.info("=" * 60)

    k_test = prep["k_test"]
    theta_test = prep["theta_test"]

    with torch.no_grad():
        theta_pred = inverse_model(k_test)
        k_achieved = forward_model(theta_pred)

    k_test_np = k_test.cpu().numpy()
    k_ach_np = k_achieved.cpu().numpy()
    theta_pred_np = theta_pred.cpu().numpy()
    theta_true_np = theta_test.cpu().numpy()

    k_test_phys = to_physical_K(k_test_np, scaler_Y, use_log)
    k_ach_phys = to_physical_K(k_ach_np, scaler_Y, use_log)
    theta_pred_phys = scaler_X.inverse_transform(theta_pred_np)
    theta_true_phys = scaler_X.inverse_transform(theta_true_np)

    for idx in range(min(n, len(k_test))):
        rel = np.abs(k_ach_phys[idx] - k_test_phys[idx]) / (np.abs(k_test_phys[idx]) + 1e-20)
        logger.info("--- Sample %d ---", idx+1)
        for i, name in enumerate(target_names):
            logger.info("  %s: target=%s  achieved=%s  error=%.2f%%",
                        name, f"{k_test_phys[idx, i]:.4e}",
                        f"{k_ach_phys[idx, i]:.4e}", rel[i]*100)
        logger.info("  Predicted design parameters vs true:")
        for i, name in zip(design_idx, design_names):
            logger.info("    %12s: pred=%12.4f  true=%12.4f",
                        name, theta_pred_phys[idx, i], theta_true_phys[idx, i])


# =============================================================================
# Inference
# =============================================================================

def inverse_design_from_checkpoint(checkpoint_path: str, k_target_physical: np.ndarray):
    """
    Standalone inference: given target K in physical units, predict design parameters.

    Args:
        checkpoint_path: path to inverse_best.pt
        k_target_physical: array shape (3,) or (N, 3) — target [K_xx, K_yy, K_zz]

    Returns:
        list of dicts with predicted features and achieved K
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    inv_config = ckpt["inv_config"]
    fwd_config = ckpt["fwd_config"]
    scaler_X = load_scaler(ckpt["scaler_X"])
    scaler_Y = load_scaler(ckpt["scaler_Y"])
    use_log = fwd_config["use_log_targets"]

    n_features = len(fwd_config["feature_cols"])
    n_targets = len(fwd_config["target_cols"])

    # Load inverse model
    inverse_model = InverseMLP(
        n_inputs=n_targets, n_outputs=n_features,
        hidden_layers=inv_config["hidden_layers"],
        dropout_rate=inv_config["dropout_rate"],
    )
    inverse_model.load_state_dict(ckpt["model_state_dict"])
    inverse_model.to(device).eval()

    # Load forward model for verification
    fwd_ckpt = torch.load(fwd_config.get("model_path", "diagonal_model.pt"),
                           map_location=device, weights_only=True)
    forward_model = PermeabilityMLP(
        n_inputs=n_features, n_outputs=n_targets,
        hidden_layers=fwd_config["hidden_layers"],
        dropout_rate=fwd_config["dropout_rate"],
    )
    forward_model.load_state_dict(fwd_ckpt["model_state_dict"])
    forward_model.to(device).eval()

    # Scale target K
    k_arr = np.atleast_2d(k_target_physical).astype(np.float32)
    if (k_arr <= 0).any() and use_log:
        raise ValueError("All target K values must be positive when using log targets.")
    k_log = np.log10(k_arr + 1e-20) if use_log else k_arr
    k_scaled = scaler_Y.transform(k_log)

    # OOD check: warn if any scaled values fall outside [0, 1] (training range)
    logger = logging.getLogger("permeability")
    ood_low = k_scaled < -0.05
    ood_high = k_scaled > 1.05
    if ood_low.any() or ood_high.any():
        target_names_ood = fwd_config["target_cols"]
        msgs = []
        for j, name in enumerate(target_names_ood):
            col = k_scaled[:, j]
            if (col < -0.05).any() or (col > 1.05).any():
                k_min_train = float(np.power(10.0, scaler_Y.data_min_[j]) if use_log
                                    else scaler_Y.data_min_[j])
                k_max_train = float(np.power(10.0, scaler_Y.data_max_[j]) if use_log
                                    else scaler_Y.data_max_[j])
                msgs.append(f"  {name}: training range [{k_min_train:.3e}, {k_max_train:.3e}]")
        logger.warning(
            "Target K values are outside the training distribution. "
            "Predictions may be unreliable.\n" + "\n".join(msgs)
        )

    k_tensor = torch.tensor(k_scaled, dtype=torch.float32, device=device)

    with torch.no_grad():
        theta_pred_sc = inverse_model(k_tensor)
        k_ach_sc = forward_model(theta_pred_sc)

    theta_pred_phys = scaler_X.inverse_transform(theta_pred_sc.cpu().numpy())
    k_ach_phys = to_physical_K(k_ach_sc.cpu().numpy(), scaler_Y, use_log)

    feature_names = fwd_config["feature_cols"]
    target_names = fwd_config["target_cols"]
    design_cols = inv_config.get("design_param_cols",
                                 ["a20", "a11", "a02", "a10", "a01", "a00", "m", "porosity"])
    design_idx = [i for i, name in enumerate(feature_names) if name in design_cols]
    design_names = [feature_names[i] for i in design_idx]

    results = []
    for i in range(len(k_arr)):
        rel = np.abs(k_ach_phys[i] - k_arr[i]) / (np.abs(k_arr[i]) + 1e-20)
        results.append({
            "target_K": {name: float(k_arr[i, j]) for j, name in enumerate(target_names)},
            "achieved_K": {name: float(k_ach_phys[i, j]) for j, name in enumerate(target_names)},
            "relative_error": {name: float(rel[j]) for j, name in enumerate(target_names)},
            "design_parameters": {name: float(theta_pred_phys[i, idx])
                                  for idx, name in zip(design_idx, design_names)},
        })

    return results


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train inverse design model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Loss modes:
  geometry   L = α×L_fwd + β×L_geo       (Eq. 23 from reference)
  bounding   L = L_fwd + λ×L_bound       (bounding penalty, avoids averaging)
  fwd_only   L = L_fwd                   (pure forward consistency)

Examples:
  python train_inverse_model.py -f diagonal_model.pt
  python train_inverse_model.py -f diagonal_model.pt --loss_mode bounding
  python train_inverse_model.py -f diagonal_model.pt --loss_mode geometry --w_param 0.05
  python train_inverse_model.py -f diagonal_model.pt --tune
  python train_inverse_model.py -f diagonal_model.pt --params inverse_optuna_best_params.json
        """,
    )
    parser.add_argument(
        "--forward_checkpoint", "-f", type=str, required=True,
        help="Path to trained forward model checkpoint (.pt)",
    )
    parser.add_argument(
        "--loss_mode", "-l", type=str, default=None,
        choices=["geometry", "bounding", "fwd_only"],
        help="Loss function mode (default: geometry)",
    )
    parser.add_argument(
        "--w_param", type=float, default=None,
        help="Weight for geometry loss term (geometry mode only)",
    )
    parser.add_argument(
        "--w_bounding", type=float, default=None,
        help="Weight for bounding loss term (bounding mode only)",
    )
    parser.add_argument(
        "--bound_margin", type=float, default=None,
        help="Fraction to expand bounds (bounding mode, 0=exact data range)",
    )
    parser.add_argument(
        "--resume", "-r", type=str, default=None,
        help="Path to inverse model checkpoint to resume from",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run Optuna HPO to find best hyperparameters before training",
    )
    parser.add_argument(
        "--params", "-p", type=str, default=None,
        help="Path to JSON file with tuned hyperparameters (from --tune or manual)",
    )
    parser.add_argument(
        "--tune_only", action="store_true",
        help="Run Optuna HPO only (skip final training)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    logger.info("=" * 60)
    logger.info("INVERSE DESIGN MODEL TRAINING")
    logger.info("  Forward: features (11) → K (3)")
    logger.info("  Inverse: K (3) → features (11)")
    logger.info("=" * 60)

    set_seed(CONFIG["random_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("  GPU: %s", torch.cuda.get_device_name(0))

    inv_config = deepcopy(INVERSE_CONFIG)

    # Apply CLI overrides
    if args.loss_mode:
        inv_config["loss_mode"] = args.loss_mode
    if args.w_param is not None:
        inv_config["w_parameter_recon"] = args.w_param
    if args.w_bounding is not None:
        inv_config["w_bounding"] = args.w_bounding
    if args.bound_margin is not None:
        inv_config["bound_margin"] = args.bound_margin
    if args.output_dir:
        inv_config["output_dir"] = args.output_dir

    # Prepare data & load forward model
    prep = prepare_inverse_data(args.forward_checkpoint, device)

    # --- Load tuned params from JSON (if provided) ---
    if args.params:
        logger.info("Loading tuned parameters from: %s", args.params)
        inv_config = load_inverse_params_from_json(args.params, inv_config)

    # --- Optuna HPO (if requested) ---
    if args.tune or args.tune_only:
        hpo_result = run_inverse_optuna_hpo(prep, inv_config, CONFIG, device)

        # Apply best params to inv_config for subsequent training
        inv_config = load_inverse_params_from_json(
            str(Path(inv_config["output_dir"]) / CONFIG["inverse_best_params_json"]),
            inv_config,
        )

        if args.tune_only:
            output_dir = Path(inv_config["output_dir"])
            logger.info("=" * 60)
            logger.info("TUNING COMPLETE (--tune_only)")
            logger.info("=" * 60)
            logger.info("Best parameters saved to: %s", output_dir / CONFIG["inverse_best_params_json"])
            logger.info("To train with these parameters, run:")
            logger.info("  python train_inverse_model.py -f %s --params %s",
                        args.forward_checkpoint, output_dir / CONFIG["inverse_best_params_json"])
            return

    # Print final configuration
    logger.info("Configuration:")
    logger.info("  hidden_layers      = %s", inv_config["hidden_layers"])
    logger.info("  dropout_rate       = %s", inv_config["dropout_rate"])
    logger.info("  learning_rate      = %s", inv_config["learning_rate"])
    logger.info("  weight_decay       = %s", inv_config["weight_decay"])
    logger.info("  batch_size         = %s", inv_config["batch_size"])
    logger.info("  scheduler_factor   = %s", inv_config["scheduler_factor"])
    logger.info("  scheduler_patience = %s", inv_config["scheduler_patience"])
    logger.info("  loss_mode          = %s", inv_config["loss_mode"])

    # Train
    inverse_model, history = train_inverse(
        inv_config, prep, device, resume_path=args.resume
    )

    # Plot history
    output_dir = Path(inv_config["output_dir"])
    plot_inverse_history(history, inv_config,
                         str(output_dir / "inverse_training_history.png"))

    # Evaluate (includes both K and parameter parity plots)
    evaluate_inverse(inverse_model, prep, inv_config, device)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Loss mode: %s", inv_config["loss_mode"])
    logger.info("Outputs in: %s", output_dir)
    logger.info("  Best model:       %s", output_dir / "inverse_best.pt")
    logger.info("  History plot:      %s", output_dir / "inverse_training_history.png")
    logger.info("  K parity plots:   %s", output_dir / "inverse_parity_K_*.png")
    logger.info("  Param parity:     %s", output_dir / "inverse_parity_params_*.png")
    logger.info("For inference:")
    logger.info("  from train_inverse_model import inverse_design_from_checkpoint")
    logger.info("  results = inverse_design_from_checkpoint(")
    logger.info("      '%s',", output_dir / "inverse_best.pt")
    logger.info("      k_target_physical=np.array([1.0, 1.5, 0.8])")
    logger.info("  )")


if __name__ == "__main__":
    main()
