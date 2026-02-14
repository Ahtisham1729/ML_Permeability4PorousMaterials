#!/usr/bin/env python3
"""
Model Training Script
=====================
Trains the permeability MLP using either:
  1. Parameters from a JSON file (output of tune_hyperparams.py)
  2. Default parameters from model_config.py

Usage:
    python train_model.py                              # Use default params
    python train_model.py --params best_params.json   # Use tuned params
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from copy import deepcopy
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Import shared components
from model_config import CONFIG, set_seed, load_and_preprocess_data, make_loaders, PermeabilityMLP

# =============================================================================
# Training
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


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
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


def build_model(config: dict, n_features: int, n_targets: int) -> nn.Module:
    model = PermeabilityMLP(
        n_inputs=n_features,
        n_outputs=n_targets,
        hidden_layers=config["hidden_layers"],
        dropout_rate=config["dropout_rate"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel:")
    print(f"  Input: {n_features} → Hidden: {config['hidden_layers']} → Output: {n_targets}")
    print(f"  Trainable parameters: {n_params:,}")
    return model


def train_model(model: nn.Module, loaders: dict, config: dict, device: torch.device) -> dict:
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    model = model.to(device)
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
    )

    early_stop = EarlyStopping(
        patience=config["early_stop_patience"],
        min_delta=config["early_stop_min_delta"],
    )

    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, config["max_epochs"] + 1):
        tr = train_one_epoch(model, loaders["train_loader"], criterion, optimizer, device)
        va = validate(model, loaders["val_loader"], criterion, device)

        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        history["lr"].append(lr)

        scheduler.step(va)

        if epoch == 1 or epoch % 20 == 0:
            print(f"Epoch {epoch:4d} | train={tr:.6e} | val={va:.6e} | lr={lr:.2e}")

        if early_stop(va, model):
            print(f"\nEarly stop at epoch {epoch} | best val={early_stop.best_loss:.6e}")
            break

    early_stop.restore_best(model)
    print(f"Restored best weights | best val={early_stop.best_loss:.6e}")
    return history

# =============================================================================
# Metrics
# =============================================================================

def mdape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-30) -> float:
    """Median absolute percentage error."""
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.median(np.abs((y_pred - y_true) / denom)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-30) -> float:
    """Normalized RMSE by (max-min)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    scale = float(np.max(y_true) - np.min(y_true))
    return float(rmse / max(scale, eps))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, physical: bool = True) -> dict:
    """Compute regression metrics per component and overall."""
    out = {"per_component": [], "overall": {}}

    for i in range(y_true.shape[1]):
        yt, yp = y_true[:, i], y_pred[:, i]
        m = {
            "R2": float(r2_score(yt, yp)),
            "MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        }
        if physical:
            m["NRMSE"] = float(nrmse(yt, yp))
            m["MdAPE"] = float(mdape(yt, yp))
        out["per_component"].append(m)

    yt_all, yp_all = y_true.flatten(), y_pred.flatten()
    out["overall"] = {
        "R2": float(r2_score(yt_all, yp_all)),
        "MAE": float(mean_absolute_error(yt_all, yp_all)),
        "RMSE": float(np.sqrt(mean_squared_error(yt_all, yp_all))),
    }
    if physical:
        out["overall"]["NRMSE"] = float(nrmse(yt_all, yp_all))
        out["overall"]["MdAPE"] = float(mdape(yt_all, yp_all))

    return out


def print_metrics_block(title: str, target_names: list[str], m: dict, physical: bool):
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)
    for name, mi in zip(target_names, m["per_component"]):
        if physical:
            print(f"{name}: R2={mi['R2']:.4f} | MAE={mi['MAE']:.3e} | RMSE={mi['RMSE']:.3e} | "
                  f"NRMSE={mi['NRMSE']:.3e} | MdAPE={mi['MdAPE']*100:.2f}%")
        else:
            print(f"{name}: R2={mi['R2']:.4f} | MAE={mi['MAE']:.3e} | RMSE={mi['RMSE']:.3e}")

    o = m["overall"]
    if physical:
        print(f"Overall: R2={o['R2']:.4f} | MAE={o['MAE']:.3e} | RMSE={o['RMSE']:.3e} | "
              f"NRMSE={o['NRMSE']:.3e} | MdAPE={o['MdAPE']*100:.2f}%")
    else:
        print(f"Overall: R2={o['R2']:.4f} | MAE={o['MAE']:.3e} | RMSE={o['RMSE']:.3e}")

# =============================================================================
# Prediction / Export / Plots
# =============================================================================

@torch.no_grad()
def predict(model: nn.Module, X_scaled: np.ndarray, scaler_Y, config: dict, device: torch.device):
    """Predict and convert to physical K."""
    model.eval()
    X_t = torch.from_numpy(X_scaled.astype(np.float32)).to(device)
    y_scaled = model(X_t).cpu().numpy()
    y_model = scaler_Y.inverse_transform(y_scaled)

    if config["use_log_targets"]:
        y_phys = np.power(10.0, y_model)
    else:
        y_phys = y_model

    return y_model, y_phys


def export_results_csv(path: str, sample_names, y_true_phys: np.ndarray, y_pred_phys: np.ndarray):
    df = pd.DataFrame({
        "sample_name": sample_names,
        "True_Kxx": y_true_phys[:, 0],
        "Pred_Kxx": y_pred_phys[:, 0],
        "True_Kyy": y_true_phys[:, 1],
        "Pred_Kyy": y_pred_phys[:, 1],
        "True_Kzz": y_true_phys[:, 2],
        "Pred_Kzz": y_pred_phys[:, 2],
    })
    df.to_csv(path, index=False)
    print(f"Exported: {path}")


def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str],
                plot_path: str, title: str):
    """Parity plot with log10 scale."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    floor = 1e-300

    for ax, name, i in zip(axes, target_names, range(3)):
        yt = np.log10(np.maximum(y_true[:, i], floor))
        yp = np.log10(np.maximum(y_pred[:, i], floor))

        ax.scatter(yt, yp, alpha=0.5, s=15, edgecolors="none")

        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        pad = 0.05 * (hi - lo + 1e-12)
        lims = [lo - pad, hi + pad]

        ax.plot(lims, lims, "k--", lw=1.5, label="1:1")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"log10(True {name} [m²])")
        ax.set_ylabel(f"log10(Pred {name} [m²])")
        ax.set_title(f"{name} (R²={r2_score(yt, yp):.4f})")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    plt.suptitle(title, y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved parity plot: {plot_path}")


def plot_training_history(history: dict, output_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train", lw=1.5)
    ax1.plot(epochs, history["val_loss"], label="Val", lw=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE (scaled target)")
    ax1.set_title("Loss Curves")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["lr"], lw=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("LR")
    ax2.set_title("LR Schedule")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved training history: {output_path}")

# =============================================================================
# Main
# =============================================================================

def load_params_from_json(path: str, base_config: dict) -> dict:
    """Load tuned parameters from JSON and merge with base config."""
    with open(path, "r") as f:
        data = json.load(f)

    params = data.get("best_params", data)
    config = deepcopy(base_config)

    # Direct mappings
    param_keys = [
        "hidden_layers", "dropout_rate", "learning_rate", "weight_decay",
        "batch_size", "scheduler_factor", "scheduler_patience"
    ]
    for key in param_keys:
        if key in params:
            config[key] = params[key]

    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Train permeability MLP model")
    parser.add_argument(
        "--params", "-p",
        type=str,
        default=None,
        help="Path to JSON file with tuned hyperparameters (from tune_hyperparams.py)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("PERMEABILITY MLP - MODEL TRAINING")
    print("=" * 60 + "\n")

    set_seed(CONFIG["random_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load config (with tuned params if provided)
    if args.params:
        print(f"\nLoading parameters from: {args.params}")
        config = load_params_from_json(args.params, CONFIG)
    else:
        print("\nUsing default parameters from model_config.py")
        config = deepcopy(CONFIG)

    print(f"\nConfiguration:")
    print(f"  hidden_layers      = {config['hidden_layers']}")
    print(f"  dropout_rate       = {config['dropout_rate']}")
    print(f"  learning_rate      = {config['learning_rate']}")
    print(f"  weight_decay       = {config['weight_decay']}")
    print(f"  batch_size         = {config['batch_size']}")
    print(f"  scheduler_factor   = {config['scheduler_factor']}")
    print(f"  scheduler_patience = {config['scheduler_patience']}")

    # Load and preprocess data
    data = load_and_preprocess_data(config)
    loaders = make_loaders(data, int(config["batch_size"]))

    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve output paths inside output directory
    model_path = str(output_dir / config["model_path"])
    val_csv = str(output_dir / config["val_output_csv"])
    test_csv = str(output_dir / config["test_output_csv"])
    plot_val = str(output_dir / config["plot_path_val"])
    plot_test = str(output_dir / config["plot_path_test"])
    history_plot = str(output_dir / config["training_history_path"])

    # Build and train model
    model = build_model(config, data["n_features"], data["n_targets"])
    history = train_model(model, loaders, config, device)

    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "scaler_X": data["scaler_X"],
        "scaler_Y": data["scaler_Y"],
    }, model_path)
    print(f"\nSaved checkpoint: {model_path}")

    # Predictions
    y_val_pred_m, y_val_pred_k = predict(model, data["X_val_scaled"], data["scaler_Y"], config, device)
    y_test_pred_m, y_test_pred_k = predict(model, data["X_test_scaled"], data["scaler_Y"], config, device)

    y_val_true_k = data["Y_val_raw"]
    y_test_true_k = data["Y_test_raw"]
    y_val_true_m = data["Y_val_model"]
    y_test_true_m = data["Y_test_model"]

    # Metrics
    targets = config["target_cols"]

    m_val_phys = compute_metrics(y_val_true_k, y_val_pred_k, physical=True)
    m_test_phys = compute_metrics(y_test_true_k, y_test_pred_k, physical=True)
    m_val_model = compute_metrics(y_val_true_m, y_val_pred_m, physical=False)
    m_test_model = compute_metrics(y_test_true_m, y_test_pred_m, physical=False)

    print_metrics_block("VALIDATION METRICS (Physical K)", targets, m_val_phys, physical=True)
    print_metrics_block("TEST METRICS (Physical K) [HELD-OUT]", targets, m_test_phys, physical=True)

    label = "log10(K)" if config["use_log_targets"] else "K"
    print_metrics_block(f"VALIDATION METRICS (Model space = {label})", targets, m_val_model, physical=False)
    print_metrics_block(f"TEST METRICS (Model space = {label}) [HELD-OUT]", targets, m_test_model, physical=False)

    # Export
    export_results_csv(val_csv, data["names_val"], y_val_true_k, y_val_pred_k)
    export_results_csv(test_csv, data["names_test"], y_test_true_k, y_test_pred_k)

    # Plots
    plot_parity(y_val_true_k, y_val_pred_k, targets, plot_val, "Parity (Validation) - log10(K)")
    plot_parity(y_test_true_k, y_test_pred_k, targets, plot_test, "Parity (Test) - log10(K) [Held-out]")
    plot_training_history(history, history_plot)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nOutputs in: {output_dir}")
    print(f"  • Model:            {model_path}")
    print(f"  • Val CSV:          {val_csv}")
    print(f"  • Test CSV:         {test_csv}")
    print(f"  • Val parity plot:  {plot_val}")
    print(f"  • Test parity plot: {plot_test}")
    print(f"  • History plot:     {history_plot}")


if __name__ == "__main__":
    main()
