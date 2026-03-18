from __future__ import annotations

import pandas as pd


def make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dayofweek"] = out["ds"].dt.dayofweek.astype(int)
    out["month"] = out["ds"].dt.month.astype(int)
    out["day"] = out["ds"].dt.day.astype(int)
    return out


def make_lag_rolling_features(df: pd.DataFrame, *, lags: list[int], rolling_windows: list[int]) -> pd.DataFrame:
    out = df.copy()

    for lag in lags:
        out[f"lag_{lag}"] = out["y"].shift(lag)

    for w in rolling_windows:
        out[f"roll_mean_{w}"] = out["y"].shift(1).rolling(window=w, min_periods=max(2, w // 2)).mean()
        out[f"roll_std_{w}"] = out["y"].shift(1).rolling(window=w, min_periods=max(2, w // 2)).std()

    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with ds, y, and model features; rows with NaNs dropped."""

    out = df.copy()
    out = make_time_features(out)
    out = make_lag_rolling_features(out, lags=[1, 2, 7, 14], rolling_windows=[7, 14])
    out = out.dropna().reset_index(drop=True)
    return out
