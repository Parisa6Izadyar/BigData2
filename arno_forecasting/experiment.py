import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from . import __version__
from .data import TARGET, chronological_split
from .evaluate import plot_forecasts, prediction_table, scores
from .models import CANDIDATES, forecast_candidate


def run_experiment(frame, output: Path, quality, horizon=365, validation_days=365,
                   weather_mode="climatology", maxiter=200, synthetic=False):
    split = chronological_split(frame, horizon, validation_days)
    validation, validation_metrics, failures = {}, {}, {}
    for name in CANDIDATES:
        print(f"Validation: {name}", flush=True)
        try:
            forecast = forecast_candidate(name, split.train, split.validation, weather_mode, maxiter)
            validation[name] = forecast
            validation_metrics[name] = {**scores(split.validation[TARGET], forecast.prediction), "warnings": forecast.warnings}
        except (ValueError, np.linalg.LinAlgError) as error:
            failures[name] = str(error)
            print(f"Unavailable: {name}: {error}", flush=True)
    if not validation:
        raise ValueError("No candidate produced a valid validation forecast")
    selected = min(validation, key=lambda name: (validation_metrics[name]["mae"], name))
    development = pd.concat([split.train, split.validation])
    test, test_metrics = {}, {}
    for name in validation:
        print(f"Test: {name}", flush=True)
        try:
            forecast = forecast_candidate(name, development, split.test, weather_mode, maxiter)
            test[name] = forecast
            test_metrics[name] = {**scores(split.test[TARGET], forecast.prediction), "warnings": forecast.warnings}
        except (ValueError, np.linalg.LinAlgError) as error:
            failures[name + "_test_fit"] = str(error)
            if name == selected:
                raise ValueError(f"The validation-selected model could not produce its test forecast: {error}") from error
    output.mkdir(parents=True, exist_ok=True)
    prediction_table(split.validation[TARGET], validation).to_csv(output / "validation_predictions.csv", lineterminator="\n")
    prediction_table(split.test[TARGET], test).to_csv(output / "test_predictions.csv", lineterminator="\n")
    plot_forecasts(development[TARGET], split.test[TARGET], test, selected, output / "forecast_comparison.png", synthetic)
    partitions = {}
    for name, data in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        partitions[name] = {"days": len(data), "start": str(data.index[0].date()), "end": str(data.index[-1].date())}
        pd.DataFrame({"Date": data.index}).to_csv(output / f"{name}_dates.csv", index=False, lineterminator="\n")
    manifest = {
        "pipeline_version": __version__, "synthetic": synthetic,
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "statsmodels", "matplotlib")},
        "data_quality": quality, "partitions": partitions,
        "protocol": "Fixed-origin multistep forecast; no target updates within the horizon",
        "weather_mode": weather_mode,
        "weather_assumption": "Future weather is training-only calendar-day climatology" if weather_mode == "climatology" else "Conditional hindcast using observed future weather",
        "model_settings": {"order": [1, 0, 1], "seasonal_order": [0, 0, 0, 0], "annual_fourier_harmonics": 2, "annual_period_days": 365.25, "maxiter": maxiter},
        "selection_metric": "validation MAE", "selected_model": selected,
        "validation_metrics": validation_metrics, "test_metrics": test_metrics,
        "unavailable_models": failures,
    }
    (output / "metrics.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Selected by validation MAE: {selected}")
    print(f"Finished. Results: {output.resolve()}")
    return manifest
