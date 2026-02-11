from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    BootstrapResponse,
    PredictionRequest,
    PredictionResponse,
    SensitivityPoint,
    SensitivityResponse,
    StoreHistoryPoint,
    StoreHistoryResponse,
    TrainResponse,
)
from .services.data import load_walmart_data
from .services.features import make_features
from .services.forecast import build_feature_row, sensitivity_points
from .services.modeling import load_artifacts, train_pipeline

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts"

app = FastAPI(title="Walmart Sales Forecasting API", version="2.0.0")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    os.getenv("FRONTEND_URL", ""),
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x for x in origins if x],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_context():
    raw = load_walmart_data(ROOT)
    feat = make_features(raw)
    return raw, feat


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/train", response_model=TrainResponse)
def train_model():
    _, feat = _load_context()
    artifacts = train_pipeline(feat, ARTIFACT_DIR)
    return TrainResponse(
        message="Model trained successfully",
        metrics=artifacts.metrics,
        diagnostics=artifacts.diagnostics,
    )


@app.get("/api/v1/bootstrap", response_model=BootstrapResponse)
def bootstrap():
    _, feat = _load_context()
    bundle = load_artifacts(ARTIFACT_DIR)
    if bundle is None:
        bundle = train_pipeline(feat, ARTIFACT_DIR).__dict__
    stores = sorted(feat["Store"].unique().tolist())
    return BootstrapResponse(stores=stores, default_store=stores[0], metrics=bundle["metrics"])


@app.get("/api/v1/metrics")
def get_metrics():
    bundle = load_artifacts(ARTIFACT_DIR)
    if bundle is None:
        _, feat = _load_context()
        bundle = train_pipeline(feat, ARTIFACT_DIR).__dict__
    return {"metrics": bundle["metrics"], "diagnostics": bundle["diagnostics"]}


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_sales(payload: PredictionRequest):
    _, feat = _load_context()
    bundle = load_artifacts(ARTIFACT_DIR)
    if bundle is None:
        bundle = train_pipeline(feat, ARTIFACT_DIR).__dict__
    row = build_feature_row(feat, payload.model_dump())
    pred = float(bundle["model"].predict(row)[0])
    return PredictionResponse(predicted_sales=pred, model_name="VotingRegressor")


@app.get("/api/v1/store/{store_id}/history", response_model=StoreHistoryResponse)
def store_history(store_id: int, limit: int = Query(default=40, ge=10, le=200)):
    raw, feat = _load_context()
    bundle = load_artifacts(ARTIFACT_DIR)
    if bundle is None:
        bundle = train_pipeline(feat, ARTIFACT_DIR).__dict__

    data = feat[feat["Store"] == store_id].sort_values("Date").tail(limit)
    if data.empty:
        raise HTTPException(status_code=404, detail="Store not found")

    preds = bundle["model"].predict(data[bundle["feature_columns"]])
    points = [
        StoreHistoryPoint(
            date=row.Date.date(),
            actual_sales=float(row.Weekly_Sales),
            predicted_sales=float(pred),
        )
        for row, pred in zip(data.itertuples(index=False), preds)
    ]
    return StoreHistoryResponse(store=store_id, points=points)


@app.get("/api/v1/store/{store_id}/sensitivity", response_model=SensitivityResponse)
def sensitivity(store_id: int, feature: str = Query(default="Temperature")):
    _, feat = _load_context()
    bundle = load_artifacts(ARTIFACT_DIR)
    if bundle is None:
        bundle = train_pipeline(feat, ARTIFACT_DIR).__dict__

    points = sensitivity_points(feat, bundle["model"], store_id, feature)
    return SensitivityResponse(feature=feature, points=[SensitivityPoint(**p) for p in points])


@app.get("/")
def root():
    return {"service": "walmart-sales-forecasting-api", "docs": "/docs"}
