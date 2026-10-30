import numpy as np
import pandas as pd
import pytest

from arno_forecasting.data import TARGET, chronological_split, synthetic_data, validate_frame


def table():
    return synthetic_data(1100).reset_index()


def test_chronological_partitions_cover_each_day_once():
    frame = synthetic_data(1460)
    split = chronological_split(frame)
    assert [len(x) for x in (split.train, split.validation, split.test)] == [730, 365, 365]
    assert split.train.index.max() + pd.Timedelta(days=1) == split.validation.index.min()
    assert split.validation.index.max() + pd.Timedelta(days=1) == split.test.index.min()
    assert pd.concat([split.train, split.validation, split.test]).index.equals(frame.index)


def test_unsorted_dates_are_sorted_without_moving_values():
    data = table().sample(frac=1, random_state=3)
    actual, quality = validate_frame(data)
    expected = synthetic_data(1100)
    pd.testing.assert_frame_equal(actual, expected)
    assert quality["inserted_days"] == 0


def test_duplicate_dates_are_rejected():
    data = table()
    with pytest.raises(ValueError, match="Duplicate dates"):
        validate_frame(pd.concat([data, data.iloc[:1]]))


@pytest.mark.parametrize("change", ["missing_day", "missing_target"])
def test_targets_are_not_imputed(change):
    data = table()
    if change == "missing_day":
        data = data.drop(index=500)
    else:
        data.loc[500, TARGET] = np.nan
    with pytest.raises(ValueError, match="Target values are not imputed"):
        validate_frame(data)


@pytest.mark.parametrize("format", ["%Y-%m-%d", "%d/%m/%Y"])
def test_supported_date_strings_preserve_dates(format):
    data = table(); data["Date"] = data["Date"].dt.strftime(format)
    actual, _ = validate_frame(data)
    assert actual.index.equals(synthetic_data(1100).index)


def test_insufficient_history_is_rejected():
    with pytest.raises(ValueError, match="At least 1095"):
        chronological_split(synthetic_data(1000))


def test_infinite_values_are_rejected():
    data = table(); data.loc[10, "Mean_Temp"] = np.inf
    with pytest.raises(ValueError, match="Infinite values"):
        validate_frame(data)
