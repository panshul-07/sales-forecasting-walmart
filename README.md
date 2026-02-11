# Walmart Sales Forecasting (FastAPI + React + Plotly)

This project was refactored into a clean full-stack architecture with:
- `backend/` for model training, diagnostics, and forecasting APIs
- `frontend/` for an interactive React + Plotly dashboard
- `notebooks/` for report notebooks with outputs (accuracy, heatmaps, model visuals)
- `data/` for dataset files (`walmart_sales.csv`)
- `artifacts/` for trained model bundles
- `legacy/` for previous notebook + Streamlit code

## Live Deployment

- Frontend URL (after deploy): `https://walmart-sales-frontend.onrender.com`
- Backend API URL (after deploy): `https://walmart-sales-api.onrender.com`
- API Docs (after deploy): `https://walmart-sales-api.onrender.com/docs`

If a link returns `404`, trigger a Render redeploy and wait for the first successful build before sharing the URL publicly.

## Models Used

- Forecast model: `VotingRegressor` (ensemble of `RandomForestRegressor` + `GradientBoostingRegressor`)
- Interpretable model: `OLS` from `statsmodels`
- OLS target: `log1p(Weekly_Sales)`
- OLS robust covariance: `HC3`
- Feature set: store/economic variables + engineered seasonality + lag and rolling features
- Full model + preprocessing + results report:
  - `/Users/panshulaj/Documents/sales-forecasting-walmart/docs/MODEL_REPORT.md`

## Why this fixes the OLS issues

The previous implementation had weak statistical validation and unstable normality behavior (high kurtosis / poor JB results). The new pipeline improves this through:
- feature engineering with lags + rolling windows + seasonality terms
- outlier clipping (IQR)
- log-transformed OLS target (`log1p(Weekly_Sales)`)
- robust covariance (`HC3`) for OLS
- expanded parametric diagnostics

### Added validation tests
- Jarque-Bera
- Omnibus
- Shapiro-Wilk
- D'Agostino K2
- Breusch-Pagan
- Durbin-Watson
- Breusch-Godfrey
- Ljung-Box
- Ramsey RESET
- one-sample t-test on residual mean
- VIF per feature

## Folder Structure

```text
.
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── services/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   ├── package.json
│   └── Dockerfile
├── notebooks/
│   ├── Walmart_Model_Report.ipynb
│   ├── Walmart_Model_Report.py
│   └── outputs/
├── data/
├── artifacts/
├── scripts/train_model.py
├── docker-compose.yml
├── render.yaml
└── legacy/
```

## Dataset

If `data/walmart_sales.csv` exists, it is used automatically.
If not, the backend generates a realistic synthetic Walmart-like dataset so the project still runs end-to-end.

## Local Run

### 1. Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python scripts/train_model.py
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend: `http://localhost:5173`

Set API base if needed:

```bash
export VITE_API_BASE_URL="http://localhost:8000"
```

## Notebook Outputs (pybooks)

Generate and refresh all report outputs:

```bash
source .venv/bin/activate
python notebooks/Walmart_Model_Report.py
```

Open this notebook:
- `notebooks/Walmart_Model_Report.ipynb`

Generated report artifacts are saved in:
- `notebooks/outputs/model_summary.json`
- `notebooks/outputs/correlation_heatmap.html`
- `notebooks/outputs/diagnostic_heatmap.html`
- `notebooks/outputs/actual_vs_predicted.html`
- `notebooks/outputs/feature_importance_rf.html`
- `notebooks/outputs/residual_distribution.html`

## Docker Run

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Web UI: `http://localhost:5173`

## Deploy on Render

A Blueprint file is included: `render.yaml`

Services provisioned:
- `walmart-sales-api` (Python web service)
- `walmart-sales-frontend` (Static React site)

Before deploy, set env vars in Render dashboard:
- `FRONTEND_URL` on API service
- `VITE_API_BASE_URL` on static frontend service to your API URL

Then deploy using Render Blueprint from repo root.

If your frontend URL still shows `Not Found`, check these in Render:
- static service name exists (`walmart-sales-frontend`)
- static rewrite route is active (`/* -> /index.html`)
- latest deploy status is `Live`
