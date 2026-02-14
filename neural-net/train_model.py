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
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from copy import deepcopy
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Import shared components
from model_config import (
    CONFIG, set_seed, setup_logging, load_and_preprocess_data, make_loaders,
    PermeabilityMLP, EarlyStopping, train_one_epoch, validate,
    save_scaler, load_scaler,
)

logger = logging.getLogger("permeability")


def build_model(config: dict, n_features: int, n_targets: int) -> nn.Module:
    model = PermeabilityMLP(
        n_inputs=n_features,
        n_outputs=n_targets,
        hidden_layers=config["hidden_layers"],
        dropout_rate=config["dropout_rate"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model:")
    logger.info("  Input: %d → Hidden: %s → Output: %d", n_features, config["hidden_layers"], n_targets)
    logger.info("  Trainable parameters: %s", f"{n_params:,}")
    return model


def train_model(model: nn.Module, loaders: dict, config: dict, device: torch.device) -> dict:
    logger.info("=" * 60)
    logger.info("TRAINING")
    logger.info("=" * 60)

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

    max_grad_norm = float(config.get("max_grad_norm", 0.0))
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, config["max_epochs"] + 1):
        tr = train_one_epoch(model, loaders["train_loader"], criterion, optimizer, device,
                             max_grad_norm=max_grad_norm)
        va = validate(model, loaders["val_loader"], criterion, device)

        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        history["lr"].append(lr)

        scheduler.step(va)

        if epoch == 1 or epoch % 20 == 0:
            logger.info("Epoch %4d | train=%s | val=%s | lr=%s",
                        epoch, f"{tr:.6e}", f"{va:.6e}", f"{lr:.2e}")

        if early_stop(va, model):
            logger.info("Early stop at epoch %d | best val=%s", epoch, f"{early_stop.best_loss:.6e}")
            break

    early_stop.restore_best(model)
    logger.info("Restored best weights | best val=%s", f"{early_stop.best_loss:.6e}")
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
    logger.info("-" * 60)
    logger.info(title)
    logger.info("-" * 60)
    for name, mi in zip(target_names, m["per_component"]):
        if physical:
            logger.info("%s: R2=%.4f | MAE=%s | RMSE=%s | NRMSE=%s | MdAPE=%.2f%%",
                        name, mi["R2"], f"{mi['MAE']:.3e}", f"{mi['RMSE']:.3e}",
                        f"{mi['NRMSE']:.3e}", mi["MdAPE"]*100)
        else:
            logger.info("%s: R2=%.4f | MAE=%s | RMSE=%s",
                        name, mi["R2"], f"{mi['MAE']:.3e}", f"{mi['RMSE']:.3e}")

    o = m["overall"]
    if physical:
        logger.info("Overall: R2=%.4f | MAE=%s | RMSE=%s | NRMSE=%s | MdAPE=%.2f%%",
                     o["R2"], f"{o['MAE']:.3e}", f"{o['RMSE']:.3e}",
                     f"{o['NRMSE']:.3e}", o["MdAPE"]*100)
    else:
        logger.info("Overall: R2=%.4f | MAE=%s | RMSE=%s",
                     o["R2"], f"{o['MAE']:.3e}", f"{o['RMSE']:.3e}")

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
    logger.info("Exported: %s", path)


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
    logger.info("Saved parity plot: %s", plot_path)


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
    logger.info("Saved training history: %s", output_path)

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
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="Path to CSV data file (overrides config and PERMEABILITY_DATA_PATH env var)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    logger.info("=" * 60)
    logger.info("PERMEABILITY MLP - MODEL TRAINING")
    logger.info("=" * 60)

    set_seed(CONFIG["random_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("  GPU: %s", torch.cuda.get_device_name(0))

    # Load config (with tuned params if provided)
    if args.params:
        logger.info("Loading parameters from: %s", args.params)
        config = load_params_from_json(args.params, CONFIG)
    else:
        logger.info("Using default parameters from model_config.py")
        config = deepcopy(CONFIG)

    if args.data:
        config["data_path"] = args.data

    logger.info("Configuration:")
    logger.info("  hidden_layers      = %s", config["hidden_layers"])
    logger.info("  dropout_rate       = %s", config["dropout_rate"])
    logger.info("  learning_rate      = %s", config["learning_rate"])
    logger.info("  weight_decay       = %s", config["weight_decay"])
    logger.info("  batch_size         = %s", config["batch_size"])
    logger.info("  scheduler_factor   = %s", config["scheduler_factor"])
    logger.info("  scheduler_patience = %s", config["scheduler_patience"])

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

    # Save checkpoint (scalers as plain dicts for weights_only=True compatibility)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "scaler_X": save_scaler(data["scaler_X"]),
        "scaler_Y": save_scaler(data["scaler_Y"]),
    }, model_path)
    logger.info("Saved checkpoint: %s", model_path)

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

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Outputs in: %s", output_dir)
    logger.info("  Model:            %s", model_path)
    logger.info("  Val CSV:          %s", val_csv)
    logger.info("  Test CSV:         %s", test_csv)
    logger.info("  Val parity plot:  %s", plot_val)
    logger.info("  Test parity plot: %s", plot_test)
    logger.info("  History plot:     %s", history_plot)


if __name__ == "__main__":
    main()
