# Anomaly Detection and Forecasting Dashboard



![Dashboard Screenshot](Screenshot%20From%202026-03-18%2016-15-02.png)

## Problem statement
Build a small production-style system that:

- Detects anomalies in a univariate time series.
- Produces short-term forecasts.
- Exposes both capabilities via a web API.
- Provides a simple dashboard to visualize anomalies and forecasts.

## Dataset
This project uses the Mauna Loa CO6 daily CO2 dataset via `statsmodels` (no external API keys required).

- The dataset is loaded in `src/data/load_dataset.py`.
- It is resampled to daily frequency and interpolated.

## Approach
### Feature engineering (anomaly detection)
Anomaly detection uses engineered features built from the time series:

- Calendar features: day of week, month, day
- Lag features: 1, 2, 7, 14
- Rolling statistics: mean/std over 7 and 14 day windows

Feature engineering is implemented in `src/features/feature_engineering.py`.

### Anomaly detection model
- Model: `IsolationForest`
- Output:
  - `anomaly_score` (higher means more anomalous)
  - `is_anomaly` (boolean)

Implementation: `src/models/anomaly.py`.

### Forecasting model
- Model: `SARIMAX` (statsmodels)
- Seasonal weekly component (period 7)
- Returns prediction mean with confidence interval

Implementation: `src/models/forecast.py`.

## Project layout
- `src/app/main.py`: FastAPI app
- `src/services/pipeline.py`: training + artifact IO
- `train.py`: CLI training entrypoint
- `dashboard/app.py`: Streamlit dashboard
- `artifacts/`: saved model artifacts (joblib)

## Local setup (recommended: virtualenv)
From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Run the API
```bash
python -m uvicorn src.app.main:app --reload --port 8000
```

API endpoints:
- `GET /health`
- `POST /train?contamination=0.02`
- `GET /anomalies?limit=2000`
- `GET /forecast?horizon=30`
- Swagger docs: `GET /docs`

## Run the dashboard
In another terminal (same env):

```bash
streamlit run dashboard/app.py
```

The dashboard calls the API URL from the sidebar (default: `http://127.0.0.1:8000`).

## Run tests
```bash
pytest -q
```

## Docker (optional)
If you use Docker Compose:

```bash
docker compose up --build
```

Then open:
- API: `http://127.0.0.1:8000/docs`
- Dashboard: `http://127.0.0.1:8501`
