from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Actual_Hydrometry"
WEATHER = ["Mean_Temp", "Mean_Rainfall"]


def validate_frame(frame: pd.DataFrame):
    required = ["Date", TARGET, *WEATHER]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    frame = frame[required].copy()
    raw_dates = frame.pop("Date")
    if pd.api.types.is_datetime64_any_dtype(raw_dates):
        dates = pd.to_datetime(raw_dates, errors="raise")
    else:
        raw_dates = raw_dates.astype(str).str.strip()
        if raw_dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            dates = pd.to_datetime(raw_dates, format="%Y-%m-%d", errors="raise")
        elif raw_dates.str.fullmatch(r"\d{2}/\d{2}/\d{4}").all():
            dates = pd.to_datetime(raw_dates, format="%d/%m/%Y", errors="raise")
        else:
            raise ValueError("Dates must be Excel dates, YYYY-MM-DD or DD/MM/YYYY")
    if dates.isna().any() or dates.dt.tz is not None or (dates != dates.dt.normalize()).any():
        raise ValueError("Dates must be nonmissing calendar days without a time or timezone")
    frame.index = pd.DatetimeIndex(dates, name="Date")
    if frame.index.duplicated().any():
        raise ValueError("Duplicate dates must be resolved before forecasting")
    frame = frame.sort_index()
    if frame.empty:
        raise ValueError("The input series is empty")
    for column in (TARGET, *WEATHER):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if np.isinf(frame[column].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite values in {column}")
    input_rows = len(frame)
    frame = frame.asfreq("D")
    quality = {
        "input_rows": input_rows,
        "calendar_days": len(frame),
        "inserted_days": len(frame) - input_rows,
        "date_start": str(frame.index.min().date()),
        "date_end": str(frame.index.max().date()),
        "missing_values": {c: int(frame[c].isna().sum()) for c in frame},
        "target_zero_values": int(frame[TARGET].eq(0).sum()),
    }
    if frame[TARGET].isna().any():
        raise ValueError(f"Hydrometry is missing on {quality['missing_values'][TARGET]} days; provide a complete daily target series. Target values are not imputed")
    return frame, quality


def load_data(path: Path):
    if not path.is_file():
        raise ValueError(f"Input file not found: {path}. Place the original dataset in data/ or pass --input. Use 'demo' only for a synthetic installation check")
    if path.suffix.lower() == ".xlsx":
        source = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        source = pd.read_csv(path)
    else:
        raise ValueError("Input must be an XLSX or CSV file")
    frame, quality = validate_frame(source)
    quality["source_file"] = path.name
    quality["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return frame, quality


@dataclass
class Splits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(frame, horizon=365, validation_days=365, min_train=365):
    if min(horizon, validation_days, min_train) < 1:
        raise ValueError("Split lengths must be positive")
    if len(frame) < horizon + validation_days + min_train:
        raise ValueError(f"At least {horizon + validation_days + min_train} daily rows are required")
    split = Splits(frame.iloc[:-horizon - validation_days].copy(),
                   frame.iloc[-horizon - validation_days:-horizon].copy(), frame.iloc[-horizon:].copy())
    assert split.train.index.max() < split.validation.index.min() < split.test.index.min()
    return split


def synthetic_data(days=1460, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=days, freq="D")
    phase = 2 * np.pi * np.arange(days) / 365.25
    temperature = 15 + 8 * np.sin(phase) + rng.normal(0, 0.8, days)
    rainfall = rng.gamma(1.2, 1.5, days)
    noise = rng.normal(0, 0.05, days)
    for i in range(1, days):
        noise[i] += 0.65 * noise[i - 1]
    target = 1.5 + 0.3 * np.cos(phase) + 0.02 * rainfall + noise
    return validate_frame(pd.DataFrame({"Date": dates, TARGET: target, "Mean_Temp": temperature, "Mean_Rainfall": rainfall}))[0]
