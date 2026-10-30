from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .data import TARGET
from .features import WeatherFeatures, annual_features
from .boosting import recursive_xgboost

CANDIDATES = ("persistence", "seasonal_naive_7", "seasonal_naive_365", "arma_fourier", "sarimax_weather", "xgboost_recursive")


@dataclass
class Forecast:
    prediction: pd.Series
    warnings: list[str]


def baseline(train_y, index, period):
    if len(train_y) < period:
        raise ValueError(f"At least {period} training days are required for this baseline")
    values = np.resize(train_y.iloc[-period:].to_numpy(), len(index))
    return pd.Series(values, index=index, dtype=float)


def forecast_candidate(name, train, future, weather_mode="climatology", maxiter=200):
    if weather_mode not in ("climatology", "observed"):
        raise ValueError("Weather mode must be climatology or observed")
    index = future.index
    if name == "xgboost_recursive":
        return Forecast(recursive_xgboost(train, future, weather_mode), [])
    if name == "persistence":
        return Forecast(baseline(train[TARGET], index, 1), [])
    if name.startswith("seasonal_naive_"):
        period = int(name.rsplit("_", 1)[1])
        return Forecast(baseline(train[TARGET], index, period), [])
    if name not in ("arma_fourier", "sarimax_weather"):
        raise ValueError(f"Unknown model: {name}")
    train_x, future_x = annual_features(train.index), annual_features(index)
    if name == "sarimax_weather":
        weather = WeatherFeatures().fit(train)
        train_x = train_x.join(weather.observed(train))
        if weather_mode == "observed":
            if future[["Mean_Temp", "Mean_Rainfall"]].isna().any().any():
                raise ValueError("Observed-weather evaluation requires all future weather values")
            future_weather = weather.observed(future)
        else:
            future_weather = weather.climatology(index)
        future_x = future_x.join(future_weather)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SARIMAX(train[TARGET], exog=train_x, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), trend="c")
        fitted = model.fit(disp=False, maxiter=maxiter)
        if not fitted.mle_retvals.get("converged", False):
            raise ValueError(f"{name} optimizer did not converge within {maxiter} iterations")
        predicted = fitted.get_forecast(steps=len(index), exog=future_x).predicted_mean
    prediction = pd.Series(np.asarray(predicted, dtype=float), index=index)
    if not np.isfinite(prediction).all():
        raise ValueError(f"{name} produced nonfinite forecasts")
    return Forecast(prediction, sorted({str(w.message) for w in caught}))
