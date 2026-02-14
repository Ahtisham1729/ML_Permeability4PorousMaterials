#!/usr/bin/env python3
"""
Hyperparameter Tuning Script (Optuna)
=====================================
Runs Optuna HPO to find optimal hyperparameters for the permeability MLP.
Saves best parameters to JSON for use by train_model.py.

Usage:
    python tune_hyperparams.py
"""

import gc
import json
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
from copy import deepcopy
from pathlib import Path

try:
    import optuna
except ImportError:
    raise ImportError("Optuna is required. Install with: pip install optuna")

# Import shared components
from model_config import (
    CONFIG, set_seed, load_and_preprocess_data, make_loaders,
    PermeabilityMLP, EarlyStopping, train_one_epoch, validate,
)


@torch.no_grad()
def val_mae_modelspace(model: nn.Module, X_val_scaled: np.ndarray, Y_val_model: np.ndarray,
                       scaler_Y, device: torch.device) -> float:
    """Compute MAE in model-space on the VAL set."""
    model.eval()
    X_t = torch.from_numpy(X_val_scaled.astype(np.float32)).to(device)
    y_scaled = model(X_t).cpu().numpy()
    y_model_pred = scaler_Y.inverse_transform(y_scaled)
    return float(mean_absolute_error(Y_val_model.reshape(-1), y_model_pred.reshape(-1)))


def suggest_hidden_layers(trial) -> list[int]:
    """Sample a monotone (optionally decaying) width schedule."""
    n_layers = trial.suggest_int("n_layers", 2, 6)
    hidden_base = trial.suggest_int("hidden_base", 64, 1024, step=64)
    width_decay = trial.suggest_float("width_decay", 0.6, 1.0)
    dims = []
    w = float(hidden_base)
    for _ in range(n_layers):
        dims.append(int(max(16, round(w / 8) * 8)))
        w *= width_decay
    return dims


def run_optuna_hpo(data: dict, config: dict, device: torch.device) -> dict:
    print("\n" + "=" * 60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    sampler = optuna.samplers.TPESampler(seed=int(config["optuna_seed"]))

    pruner_name = str(config.get("optuna_pruner", "median")).lower()
    if pruner_name == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(config["optuna_startup_trials"]),
            n_warmup_steps=int(config["optuna_warmup_steps"]),
        )
    else:
        pruner = optuna.pruners.NopPruner()

    storage = config.get("optuna_db", None)
    study = optuna.create_study(
        study_name=str(config["optuna_study_name"]),
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=bool(storage),
    )

    # Cached arrays for fast objective eval
    X_val_scaled = data["X_val_scaled"]
    Y_val_model = data["Y_val_model"]
    scaler_Y = data["scaler_Y"]

    n_features = data["n_features"]
    n_targets = data["n_targets"]

    def objective(trial: optuna.Trial) -> float:
        base_seed = int(config["random_state"])
        set_seed(base_seed + int(trial.number))

        # Hyperparameters
        hidden_layers = suggest_hidden_layers(trial)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.2)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

        scheduler_factor = trial.suggest_float("scheduler_factor", 0.3, 0.8)
        scheduler_patience = trial.suggest_int("scheduler_patience", 8, 25)

        loaders = make_loaders(data, int(batch_size))

        model = PermeabilityMLP(
            n_inputs=n_features,
            n_outputs=n_targets,
            hidden_layers=hidden_layers,
            dropout_rate=float(dropout_rate),
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=float(scheduler_factor),
            patience=int(scheduler_patience),
        )

        early_stop = EarlyStopping(
            patience=int(config["tune_early_stop_patience"]),
            min_delta=float(config["tune_early_stop_min_delta"]),
        )

        best_metric = float("inf")
        max_epochs = int(config["tune_max_epochs"])

        max_grad_norm = float(config.get("max_grad_norm", 0.0))

        for epoch in range(1, max_epochs + 1):
            _ = train_one_epoch(model, loaders["train_loader"], criterion, optimizer, device,
                                max_grad_norm=max_grad_norm)
            val_loss_scaled = validate(model, loaders["val_loader"], criterion, device)
            scheduler.step(val_loss_scaled)

            metric = val_mae_modelspace(model, X_val_scaled, Y_val_model, scaler_Y, device)

            trial.report(metric, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if metric < best_metric - 1e-12:
                best_metric = metric

            if early_stop(metric, model):
                break

        del model, optimizer, scheduler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return float(best_metric)

    study.optimize(
        objective,
        n_trials=int(config["optuna_trials"]),
        timeout=config["optuna_timeout_sec"],
        show_progress_bar=True,
    )

    print("\nBest trial:")
    print(f"  value (val MAE model-space): {study.best_value:.6e}")
    print(f"  params: {study.best_params}")

    # Save trials table
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    trials_csv = str(output_dir / config["optuna_results_csv"])
    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_csv, index=False)
    print(f"Saved Optuna trials: {trials_csv}")

    # Save best params with reconstructed hidden_layers for easy loading
    best_params = study.best_params.copy()

    # Reconstruct hidden_layers from the params
    if "n_layers" in best_params and "hidden_base" in best_params and "width_decay" in best_params:
        n_layers = int(best_params["n_layers"])
        hidden_base = int(best_params["hidden_base"])
        width_decay = float(best_params["width_decay"])
        dims = []
        w = float(hidden_base)
        for _ in range(n_layers):
            dims.append(int(max(16, round(w / 8) * 8)))
            w *= width_decay
        best_params["hidden_layers"] = dims

    output = {
        "best_value": float(study.best_value),
        "best_params": best_params,
    }

    params_path = str(output_dir / config["best_params_json"])
    with open(params_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved best params: {params_path}")

    return output


def main():
    print("\n" + "=" * 60)
    print("PERMEABILITY MLP - HYPERPARAMETER TUNING")
    print("=" * 60 + "\n")

    set_seed(CONFIG["random_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    data = load_and_preprocess_data(CONFIG)

    hpo_result = run_optuna_hpo(data, CONFIG, device)

    output_dir = Path(CONFIG["output_dir"])
    params_path = output_dir / CONFIG["best_params_json"]

    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"\nBest parameters saved to: {params_path}")
    print("\nTo train with these parameters, run:")
    print(f"  python train_model.py --params {params_path}")


if __name__ == "__main__":
    main()
