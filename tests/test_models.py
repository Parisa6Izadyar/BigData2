import numpy as np
import pandas as pd
import pytest

from arno_forecasting.data import TARGET, WEATHER, synthetic_data
from arno_forecasting.models import baseline, forecast_candidate


def test_multistep_baseline_reuses_training_history():
    target = pd.Series(range(10), index=pd.date_range("2020-01-01", periods=10))
    future = pd.date_range("2020-01-11", periods=8)
    assert baseline(target, future, 3).tolist() == [7, 8, 9, 7, 8, 9, 7, 8]


def test_forecast_does_not_use_future_weather_or_targets():
    frame = synthetic_data(900)
    train, future = frame.iloc[:800], frame.iloc[800:]
    expected = forecast_candidate("sarimax_weather", train, future).prediction
    changed = future.copy(); changed[[TARGET, *WEATHER]] = 1e6
    actual = forecast_candidate("sarimax_weather", train, changed).prediction
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_observed_weather_is_explicitly_conditional():
    frame = synthetic_data(900)
    train, future = frame.iloc[:800], frame.iloc[800:].copy()
    expected = forecast_candidate("sarimax_weather", train, future, weather_mode="observed").prediction
    future["Mean_Temp"] += 10
    changed = forecast_candidate("sarimax_weather", train, future, weather_mode="observed").prediction
    assert not np.allclose(expected, changed)


def test_nonconverged_model_is_reported(monkeypatch):
    class FailedFit:
        mle_retvals = {"converged": False}
    class Model:
        def __init__(self, *args, **kwargs): pass
        def fit(self, **kwargs): return FailedFit()
    monkeypatch.setattr("arno_forecasting.models.SARIMAX", Model)
    frame = synthetic_data(500)
    with pytest.raises(ValueError, match="did not converge"):
        forecast_candidate("arma_fourier", frame.iloc[:400], frame.iloc[400:])
