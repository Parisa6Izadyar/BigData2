import numpy as np
import pandas as pd

from .data import WEATHER


def annual_features(index):
    elapsed = (index - pd.Timestamp("2000-01-01")).days.to_numpy()
    values = {}
    for harmonic in (1, 2):
        phase = 2 * np.pi * harmonic * elapsed / 365.25
        values[f"annual_sin_{harmonic}"] = np.sin(phase)
        values[f"annual_cos_{harmonic}"] = np.cos(phase)
    return pd.DataFrame(values, index=index)


class WeatherFeatures:
    def fit(self, train):
        self.medians = train[WEATHER].median()
        if self.medians.isna().any():
            raise ValueError("A weather column has no observed training values")
        observed = train[WEATHER].fillna(self.medians)
        self.means = observed.mean()
        self.scales = observed.std(ddof=0).replace(0, 1)
        self.calendar = observed.groupby(observed.index.strftime("%m-%d")).mean()
        return self

    def observed(self, frame):
        weather = frame[WEATHER].fillna(self.medians)
        return (weather - self.means) / self.scales

    def climatology(self, index):
        rows = []
        for date in index:
            key = date.strftime("%m-%d")
            if key in self.calendar.index:
                values = self.calendar.loc[key]
            elif key == "02-29" and {"02-28", "03-01"}.issubset(self.calendar.index):
                values = self.calendar.loc[["02-28", "03-01"]].mean()
            else:
                values = self.medians
            rows.append(values.to_numpy())
        weather = pd.DataFrame(rows, index=index, columns=WEATHER)
        return (weather - self.means) / self.scales
