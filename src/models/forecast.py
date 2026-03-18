from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


@dataclass
class ForecastModel:
    results: SARIMAXResults
    freq: str

    def forecast(self, *, horizon: int) -> pd.DataFrame:
        pred = self.results.get_forecast(steps=horizon)
        mean = pred.predicted_mean
        conf = pred.conf_int()

        # Build future index based on last date.
        last_ds = pd.to_datetime(self.results.data.dates[-1])
        future_ds = pd.date_range(start=last_ds + pd.tseries.frequencies.to_offset(self.freq), periods=horizon, freq=self.freq)

        out = pd.DataFrame(
            {
                "ds": future_ds,
                "yhat": np.asarray(mean, dtype=float),
                "yhat_lower": np.asarray(conf.iloc[:, 0], dtype=float),
                "yhat_upper": np.asarray(conf.iloc[:, 1], dtype=float),
            }
        )
        return out

    def save(self, path: str) -> None:
        joblib.dump({"results": self.results, "freq": self.freq}, path)

    @staticmethod
    def load(path: str) -> "ForecastModel":
        obj = joblib.load(path)
        return ForecastModel(results=obj["results"], freq=obj["freq"])


def train_sarimax(df: pd.DataFrame, *, freq: str = "D") -> ForecastModel:
    """Train a simple SARIMAX model on a univariate series."""

    series = df.set_index("ds")["y"].asfreq(freq)
    series = series.interpolate(method="time")

    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)
    return ForecastModel(results=results, freq=freq)
