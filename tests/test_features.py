import numpy as np
import pandas as pd

from arno_forecasting.data import WEATHER, synthetic_data
from arno_forecasting.features import WeatherFeatures, annual_features


def test_weather_statistics_fit_training_values_only():
    frame = synthetic_data(1460)
    train = frame.iloc[:1000].copy()
    train.loc[train.index[0], "Mean_Temp"] = np.nan
    features = WeatherFeatures().fit(train)
    assert features.medians["Mean_Temp"] == train["Mean_Temp"].median()
    future = frame.iloc[1000:].copy()
    expected = features.climatology(future.index)
    future[WEATHER] = 1e6
    pd.testing.assert_frame_equal(features.climatology(future.index), expected)
    assert np.isfinite(features.observed(train)).all().all()


def test_unseen_leap_day_uses_adjacent_training_days():
    frame = synthetic_data(365)
    features = WeatherFeatures().fit(frame)
    index = pd.DatetimeIndex(["2016-02-29"])
    actual = features.climatology(index)
    expected = (features.calendar.loc[["02-28", "03-01"]].mean() - features.means) / features.scales
    np.testing.assert_allclose(actual.iloc[0], expected)


def test_annual_features_use_day_frequency():
    index = pd.DatetimeIndex(["2000-01-01", "2000-07-01", "2001-01-01"])
    x = annual_features(index)
    assert list(x.columns) == ["annual_sin_1", "annual_cos_1", "annual_sin_2", "annual_cos_2"]
    assert x.iloc[0]["annual_cos_1"] == 1
    assert x.iloc[1]["annual_cos_1"] < -0.99
    assert x.iloc[2]["annual_cos_1"] > 0.99
