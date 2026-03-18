from __future__ import annotations

import os

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.features.feature_engineering import build_feature_frame
from src.models.anomaly import train_isolation_forest
from src.models.forecast import train_sarimax
from src.services.pipeline import load_training_data

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Anomaly & Forecast Dashboard", layout="wide")

st.title("Anomaly Detection and Forecasting Dashboard")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", API_URL)
    horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=120, value=30, step=1)
    limit = st.slider("Anomaly points to display", min_value=200, max_value=8000, value=2000, step=100)


@st.cache_data(show_spinner=False)
def _load_df() -> pd.DataFrame:
    return load_training_data()


@st.cache_resource(show_spinner=False)
def _train_models(contamination: float = 0.02):
    df = _load_df()
    feature_df = build_feature_frame(df)
    anomaly_model = train_isolation_forest(feature_df, contamination=contamination)
    forecast_model = train_sarimax(df, freq="D")
    return anomaly_model, forecast_model


def _fetch_anomalies_from_api(*, api_url: str, limit: int) -> pd.DataFrame:
    r = requests.get(f"{api_url}/anomalies", params={"limit": limit}, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json()["rows"])


def _fetch_forecast_from_api(*, api_url: str, horizon: int) -> pd.DataFrame:
    r = requests.get(f"{api_url}/forecast", params={"horizon": horizon}, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json()["rows"])


def _compute_anomalies_locally(*, limit: int) -> pd.DataFrame:
    df = _load_df()
    feature_df = build_feature_frame(df)
    anomaly_model, _ = _train_models()
    scored = anomaly_model.score(feature_df)
    if len(scored) > limit:
        scored = scored.iloc[-limit:].copy()
    scored["ds"] = pd.to_datetime(scored["ds"])
    return scored


def _compute_forecast_locally(*, horizon: int) -> pd.DataFrame:
    _, forecast_model = _train_models()
    pred = forecast_model.forecast(horizon=horizon)
    pred["ds"] = pd.to_datetime(pred["ds"])
    return pred

col1, col2 = st.columns(2)

with col1:
    st.subheader("Anomalies")
    try:
        anom = _fetch_anomalies_from_api(api_url=api_url, limit=limit)
        anom["ds"] = pd.to_datetime(anom["ds"])
    except Exception as e:
        st.info("API not reachable; running in local mode inside Streamlit.")
        try:
            anom = _compute_anomalies_locally(limit=limit)
        except Exception as e2:
            st.error(f"Failed to compute anomalies locally: {e2}")
            anom = pd.DataFrame(columns=["ds", "y", "anomaly_score", "is_anomaly"])

    if not anom.empty:

        normal = anom[~anom["is_anomaly"]]
        anomalies = anom[anom["is_anomaly"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=normal["ds"], y=normal["y"], mode="lines", name="y"))
        fig.add_trace(
            go.Scatter(
                x=anomalies["ds"],
                y=anomalies["y"],
                mode="markers",
                name="anomaly",
            )
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Forecast")
    try:
        fc = _fetch_forecast_from_api(api_url=api_url, horizon=horizon)
        fc["ds"] = pd.to_datetime(fc["ds"])
    except Exception as e:
        st.info("API not reachable; running forecast locally inside Streamlit.")
        try:
            fc = _compute_forecast_locally(horizon=horizon)
        except Exception as e2:
            st.error(f"Failed to compute forecast locally: {e2}")
            fc = pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])

    if not fc.empty:

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines", name="yhat"))
        fig.add_trace(
            go.Scatter(
                x=fc["ds"],
                y=fc["yhat_upper"],
                mode="lines",
                name="upper",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc["ds"],
                y=fc["yhat_lower"],
                mode="lines",
                name="lower",
                line=dict(width=0),
                fill="tonexty",
                showlegend=False,
            )
        )
        st.plotly_chart(fig, use_container_width=True)
