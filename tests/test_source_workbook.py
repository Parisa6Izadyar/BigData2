from pathlib import Path

import pytest

from arno_forecasting.data import TARGET, chronological_split, load_data


def test_source_workbook_profile_and_zero_values_are_preserved():
    path = Path(__file__).resolve().parents[1] / "data/RIVER_TRAIN_DATASET_A1.xlsx"
    if not path.is_file():
        pytest.skip("The source workbook is not distributed; supply an authorized local copy")
    frame, quality = load_data(path)
    assert quality["source_sha256"] == "6e3562fb761ddcb834c0f3d75567dcfb8e6e1ffea760ba96228352d928509dc4"
    assert quality["input_rows"] == 6249
    assert quality["inserted_days"] == 0
    assert quality["target_zero_values"] == 177
    assert quality["date_start"] == "2000-06-25"
    assert quality["date_end"] == "2017-08-03"
    split = chronological_split(frame)
    assert (len(split.train), len(split.validation), len(split.test)) == (5519, 365, 365)
    assert (split.validation[TARGET].eq(0).sum(), split.test[TARGET].eq(0).sum()) == (4, 2)
