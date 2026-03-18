from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Anomaly & Forecast Dashboard", layout="wide")

st.title("Anomaly Detection and Forecasting Dashboard")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", API_URL)
    horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=120, value=30, step=1)
    limit = st.slider("Anomaly points to display", min_value=200, max_value=8000, value=2000, step=100)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Anomalies")
    try:
        r = requests.get(f"{api_url}/anomalies", params={"limit": limit}, timeout=30)
        r.raise_for_status()
        rows = r.json()["rows"]
        anom = pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Failed to fetch anomalies: {e}")
        anom = pd.DataFrame(columns=["ds", "y", "anomaly_score", "is_anomaly"])

    if not anom.empty:
        anom["ds"] = pd.to_datetime(anom["ds"])

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
        r = requests.get(f"{api_url}/forecast", params={"horizon": horizon}, timeout=30)
        r.raise_for_status()
        frows = r.json()["rows"]
        fc = pd.DataFrame(frows)
    except Exception as e:
        st.error(f"Failed to fetch forecast: {e}")
        fc = pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])

    if not fc.empty:
        fc["ds"] = pd.to_datetime(fc["ds"])

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
