from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def scores(actual, predicted):
    if not actual.index.equals(predicted.index):
        raise ValueError("Forecast dates do not match the evaluation dates")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Evaluation values must be finite")
    error = actual.to_numpy() - predicted.to_numpy()
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))), "days": len(actual)}


def prediction_table(actual, predictions):
    frame = pd.DataFrame({"actual": actual, **{name: forecast.prediction for name, forecast in predictions.items()}})
    frame.index.name = "Date"
    return frame


def plot_forecasts(train, actual, predictions, selected, path: Path, synthetic=False):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    prefix = "Synthetic example: " if synthetic else ""
    train.iloc[-730:].plot(ax=axes[0], color="0.65", linewidth=0.8, label="Observed before forecast origin")
    actual.plot(ax=axes[0], color="black", linewidth=1, label="Held-out observations")
    predictions[selected].prediction.plot(ax=axes[0], linewidth=1.6, label=f"Selected: {selected}")
    axes[0].axvline(actual.index[0], linestyle=":", color="black")
    axes[0].set_title(f"{prefix}Fixed-origin forecast ({len(actual)} days)")
    actual.plot(ax=axes[1], color="black", linewidth=1.5, label="Held-out observations")
    for name, forecast in predictions.items():
        forecast.prediction.plot(ax=axes[1], linewidth=1.5 if name == selected else 0.9, alpha=0.9 if name == selected else 0.65, label=name)
    for axis in axes:
        axis.set_xlabel("Date")
        axis.set_ylabel("Hydrometry (source units)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="best")
    fig.savefig(path, dpi=150)
    plt.close(fig)
