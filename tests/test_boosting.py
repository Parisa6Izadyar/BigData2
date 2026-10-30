import numpy as np

from arno_forecasting.boosting import recursive_xgboost
from arno_forecasting.data import TARGET, WEATHER, synthetic_data


def test_recursive_booster_uses_its_own_predictions(monkeypatch):
    class IncrementModel:
        def __init__(self, **kwargs):
            pass

        def fit(self, features, targets):
            assert "target_lag_1" in features

        def predict(self, features):
            return features["target_lag_1"].to_numpy() + 1

    monkeypatch.setattr("arno_forecasting.boosting.XGBRegressor", IncrementModel)
    frame = synthetic_data(400)
    train, future = frame.iloc[:-3], frame.iloc[-3:]
    actual = recursive_xgboost(train, future)
    np.testing.assert_allclose(actual, train[TARGET].iloc[-1] + np.arange(1, 4))


def test_recursive_booster_ignores_held_out_targets_and_weather():
    frame = synthetic_data(450)
    train, future = frame.iloc[:400], frame.iloc[400:]
    expected = recursive_xgboost(train, future)
    altered = future.copy()
    altered[[TARGET, *WEATHER]] = 1e6
    actual = recursive_xgboost(train, altered)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_booster_fits_without_test_or_validation_labels(monkeypatch):
    frame = synthetic_data(400)
    train, future = frame.iloc[:380], frame.iloc[380:]

    class FitRecorder:
        def __init__(self, **kwargs):
            assert "early_stopping_rounds" not in kwargs

        def fit(self, features, targets, **kwargs):
            assert kwargs == {}
            assert len(features) == len(train) - 7
            np.testing.assert_allclose(targets, train[TARGET].iloc[7:])

        def predict(self, features):
            return np.zeros(len(features))

    monkeypatch.setattr("arno_forecasting.boosting.XGBRegressor", FitRecorder)
    recursive_xgboost(train, future)
