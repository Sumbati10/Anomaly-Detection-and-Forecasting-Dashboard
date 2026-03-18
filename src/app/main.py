from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src.features.feature_engineering import build_feature_frame
from src.services.pipeline import load_models, load_training_data, train_and_save

app = FastAPI(title="Anomaly Detection & Forecasting API", version="0.1.0")

_ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
_anomaly_model = None
_forecast_model = None


def _ensure_models_loaded() -> None:
    global _anomaly_model, _forecast_model

    if _anomaly_model is not None and _forecast_model is not None:
        return

    anomaly_path = _ARTIFACTS_DIR / "anomaly_model.joblib"
    forecast_path = _ARTIFACTS_DIR / "forecast_model.joblib"

    if not anomaly_path.exists() or not forecast_path.exists():
        train_and_save(artifacts_dir=_ARTIFACTS_DIR)

    _anomaly_model, _forecast_model = load_models(artifacts_dir=_ARTIFACTS_DIR)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/train")
def train(contamination: float = Query(default=0.02, ge=0.0001, le=0.5)) -> dict:
    global _anomaly_model, _forecast_model
    train_and_save(artifacts_dir=_ARTIFACTS_DIR, contamination=contamination)
    _anomaly_model = None
    _forecast_model = None
    _ensure_models_loaded()
    return {"status": "trained", "artifacts_dir": str(_ARTIFACTS_DIR)}


@app.get("/anomalies")
def anomalies(limit: int = Query(default=5000, ge=10, le=20000)) -> dict:
    _ensure_models_loaded()

    df = load_training_data()
    feature_df = build_feature_frame(df)
    scored = _anomaly_model.score(feature_df)

    if len(scored) > limit:
        scored = scored.iloc[-limit:].copy()

    scored = scored.copy()
    scored["ds"] = pd.to_datetime(scored["ds"]).dt.strftime("%Y-%m-%d")

    return {
        "rows": scored.to_dict(orient="records"),
        "count": int(len(scored)),
    }


@app.get("/forecast")
def forecast(horizon: int = Query(default=30, ge=1, le=365)) -> dict:
    _ensure_models_loaded()

    try:
        pred = _forecast_model.forecast(horizon=horizon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    pred = pred.copy()
    pred["ds"] = pd.to_datetime(pred["ds"]).dt.strftime("%Y-%m-%d")

    return {
        "rows": pred.to_dict(orient="records"),
        "horizon": horizon,
    }
