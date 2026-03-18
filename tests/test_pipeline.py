from __future__ import annotations

from pathlib import Path

from src.services.pipeline import load_models, train_and_save


def test_train_and_load_models(tmp_path: Path) -> None:
    train_and_save(artifacts_dir=tmp_path)
    anomaly_model, forecast_model = load_models(artifacts_dir=tmp_path)

    assert anomaly_model.feature_columns
    assert forecast_model.freq == "D"
