import numpy as np
import pandas as pd
import pytest

from arno_forecasting.data import TARGET, synthetic_data
from arno_forecasting.evaluate import scores
from arno_forecasting.experiment import run_experiment
from arno_forecasting.models import Forecast


def test_model_selection_uses_validation_not_test(tmp_path, monkeypatch):
    frame = synthetic_data(1200)
    validation_index = frame.index[-200:-100]
    calls = []
    def predict(name, train, future, *args):
        calls.append((name, train.index.max(), future.index.min()))
        if future.index.equals(validation_index):
            values = frame.loc[future.index, TARGET] + (0 if name == "persistence" else 1)
        else:
            values = frame.loc[future.index, TARGET] + (5 if name == "persistence" else 0)
        return Forecast(pd.Series(values, index=future.index), [])
    monkeypatch.setattr("arno_forecasting.experiment.forecast_candidate", predict)
    result = run_experiment(frame, tmp_path, {}, horizon=100, validation_days=100, synthetic=True)
    assert result["selected_model"] == "persistence"
    assert result["test_metrics"]["persistence"]["mae"] > result["test_metrics"]["arma_fourier"]["mae"]
    assert all(train_end < future_start for _, train_end, future_start in calls)


def test_metric_values_and_date_alignment():
    index = pd.date_range("2020-01-01", periods=2)
    actual = pd.Series([1.0, 3.0], index=index)
    prediction = pd.Series([2.0, 1.0], index=index)
    result = scores(actual, prediction)
    assert result["mae"] == 1.5
    assert result["rmse"] == np.sqrt(2.5)
    with pytest.raises(ValueError, match="dates"):
        scores(actual, prediction.shift(freq="D"))
