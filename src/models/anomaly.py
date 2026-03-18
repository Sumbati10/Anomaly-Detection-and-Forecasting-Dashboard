from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class AnomalyModel:
    scaler: StandardScaler
    model: IsolationForest
    feature_columns: list[str]

    def score(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        X = feature_df[self.feature_columns].to_numpy(dtype=float)
        Xs = self.scaler.transform(X)

        # IsolationForest: higher is more normal; we'll convert to anomaly_score where higher is more anomalous.
        normal_score = self.model.score_samples(Xs)
        anomaly_score = -normal_score
        is_anomaly = self.model.predict(Xs) == -1

        out = feature_df[["ds", "y"]].copy()
        out["anomaly_score"] = anomaly_score
        out["is_anomaly"] = is_anomaly
        return out

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "scaler": self.scaler,
                "model": self.model,
                "feature_columns": self.feature_columns,
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "AnomalyModel":
        obj = joblib.load(path)
        return AnomalyModel(
            scaler=obj["scaler"],
            model=obj["model"],
            feature_columns=list(obj["feature_columns"]),
        )


def train_isolation_forest(feature_df: pd.DataFrame, *, contamination: float = 0.02, random_state: int = 42) -> AnomalyModel:
    feature_cols = [c for c in feature_df.columns if c not in {"ds", "y"}]

    X = feature_df[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=400,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(Xs)

    return AnomalyModel(scaler=scaler, model=model, feature_columns=feature_cols)
