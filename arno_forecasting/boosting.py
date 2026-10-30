"""Lagged gradient boosting with a fixed forecast origin."""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .data import TARGET, WEATHER
from .features import WeatherFeatures

XGB_SETTINGS = {
    "n_estimators": 1000,
    "max_depth": 5,
    "learning_rate": 0.03,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": 1,
    "random_state": 42,
}


def lag_row(date, target_history, weather_history):
    row = {
        "year": date.year,
        "quarter": date.quarter,
        "month": date.month,
        "dayofyear": date.dayofyear,
        "dayofmonth": date.day,
        "dayofweek": date.dayofweek,
    }
    for lag in range(1, 8):
        row[f"target_lag_{lag}"] = target_history[-lag]
    for column in WEATHER:
        for lag in range(1, 8):
            row[f"{column}_lag_{lag}"] = weather_history[column][-lag]
    return row


def recursive_xgboost(train, future, weather_mode="climatology"):
    weather = WeatherFeatures().fit(train)
    past_weather = weather.observed(train)
    if weather_mode == "observed":
        if future[WEATHER].isna().any().any():
            raise ValueError("Observed-weather evaluation requires all future weather values")
        future_weather = weather.observed(future)
    elif weather_mode == "climatology":
        future_weather = weather.climatology(future.index)
    else:
        raise ValueError("Weather mode must be climatology or observed")

    targets = train[TARGET].to_numpy(dtype=float).tolist()
    history = {column: past_weather[column].tolist() for column in WEATHER}
    rows = [lag_row(date, targets[:i], {c: history[c][:i] for c in WEATHER})
            for i, date in enumerate(train.index) if i >= 7]
    if not rows:
        raise ValueError("XGBoost needs more than seven training days")
    model = XGBRegressor(**XGB_SETTINGS)
    model.fit(pd.DataFrame(rows), np.asarray(targets[7:]))

    predictions = []
    for date in future.index:
        row = pd.DataFrame([lag_row(date, targets, history)])
        value = float(model.predict(row)[0])
        if not np.isfinite(value):
            raise ValueError("xgboost_recursive produced a nonfinite forecast")
        predictions.append(value)
        # Later target lags contain predictions from this forecast, not held-out observations.
        targets.append(value)
        for column in WEATHER:
            history[column].append(float(future_weather.loc[date, column]))
    return pd.Series(predictions, index=future.index, dtype=float)
