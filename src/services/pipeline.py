from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.load_dataset import load_co2_daily
from src.features.feature_engineering import build_feature_frame
from src.models.anomaly import AnomalyModel, train_isolation_forest
from src.models.forecast import ForecastModel, train_sarimax


@dataclass(frozen=True)
class ArtifactPaths:
    anomaly_model_path: Path
    forecast_model_path: Path


def default_artifact_paths(artifacts_dir: str | Path) -> ArtifactPaths:
    d = Path(artifacts_dir)
    return ArtifactPaths(
        anomaly_model_path=d / "anomaly_model.joblib",
        forecast_model_path=d / "forecast_model.joblib",
    )


def load_training_data() -> pd.DataFrame:
    dataset = load_co2_daily()
    return dataset.df.copy()


def train_and_save(*, artifacts_dir: str | Path, contamination: float = 0.02) -> ArtifactPaths:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    paths = default_artifact_paths(artifacts_dir)

    df = load_training_data()
    feature_df = build_feature_frame(df)

    anomaly_model = train_isolation_forest(feature_df, contamination=contamination)
    anomaly_model.save(str(paths.anomaly_model_path))

    forecast_model = train_sarimax(df, freq="D")
    forecast_model.save(str(paths.forecast_model_path))

    return paths


def load_models(*, artifacts_dir: str | Path) -> tuple[AnomalyModel, ForecastModel]:
    paths = default_artifact_paths(artifacts_dir)
    anomaly_model = AnomalyModel.load(str(paths.anomaly_model_path))
    forecast_model = ForecastModel.load(str(paths.forecast_model_path))
    return anomaly_model, forecast_model
