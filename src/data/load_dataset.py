from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class TimeSeriesDataset:
    df: pd.DataFrame  # columns: ds (datetime64[ns]), y (float)


def load_co2_daily() -> TimeSeriesDataset:
    """Load the Mauna Loa CO2 dataset (real-world) and return a daily series.

    Uses statsmodels' built-in dataset to avoid external credentials.
    """

    data = sm.datasets.co2.load_pandas().data
    df = data.copy()

    # Dataset index is weekly; make it explicit.
    df = df.reset_index().rename(columns={"index": "ds", "co2": "y"})
    df["ds"] = pd.to_datetime(df["ds"], utc=False)

    # Resample to daily frequency and interpolate missing days.
    df = (
        df.set_index("ds")["y"]
        .resample("D")
        .mean()
        .interpolate(method="time")
        .to_frame()
        .reset_index()
    )

    df["y"] = df["y"].astype(float)
    return TimeSeriesDataset(df=df)
