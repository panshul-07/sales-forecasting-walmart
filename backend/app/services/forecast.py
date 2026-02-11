from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS


def build_feature_row(df_feat: pd.DataFrame, payload: dict) -> pd.DataFrame:
    store = payload["store"]
    date = pd.to_datetime(payload["date"])

    store_hist = df_feat[df_feat["Store"] == store].sort_values("Date")
    if store_hist.empty:
        store_hist = df_feat.sort_values("Date").tail(8).copy()

    lag_1 = float(store_hist["Weekly_Sales"].iloc[-1])
    lag_4 = float(store_hist["Weekly_Sales"].iloc[-4] if len(store_hist) >= 4 else lag_1)
    rolling_4 = float(store_hist["Weekly_Sales"].tail(4).mean())
    rolling_8 = float(store_hist["Weekly_Sales"].tail(8).mean())

    week = int(date.isocalendar().week)

    row = {
        "Store": int(store),
        "Holiday_Flag": int(payload["holiday_flag"]),
        "Temperature": float(payload["temperature"]),
        "Fuel_Price": float(payload["fuel_price"]),
        "CPI": float(payload["cpi"]),
        "Unemployment": float(payload["unemployment"]),
        "Month": int(date.month),
        "Week": week,
        "Year": int(date.year),
        "Week_Sin": float(np.sin(2 * np.pi * week / 52)),
        "Week_Cos": float(np.cos(2 * np.pi * week / 52)),
        "Lag_1": lag_1,
        "Lag_4": lag_4,
        "Rolling_4_Mean": rolling_4,
        "Rolling_8_Mean": rolling_8,
    }

    return pd.DataFrame([row])[FEATURE_COLUMNS]


def sensitivity_points(df_feat: pd.DataFrame, model, store: int, feature: str):
    feature_ranges = {
        "Temperature": np.linspace(20, 110, 70),
        "Fuel_Price": np.linspace(2.0, 5.5, 70),
        "CPI": np.linspace(180, 320, 70),
        "Unemployment": np.linspace(3.0, 14.5, 70),
    }
    if feature not in feature_ranges:
        raise ValueError(f"Unsupported feature: {feature}")

    base = {
        "store": store,
        "date": str(df_feat[df_feat["Store"] == store]["Date"].max().date()),
        "holiday_flag": 0,
        "temperature": float(df_feat["Temperature"].median()),
        "fuel_price": float(df_feat["Fuel_Price"].median()),
        "cpi": float(df_feat["CPI"].median()),
        "unemployment": float(df_feat["Unemployment"].median()),
    }

    feature_map = {
        "Temperature": "temperature",
        "Fuel_Price": "fuel_price",
        "CPI": "cpi",
        "Unemployment": "unemployment",
    }

    points = []
    for value in feature_ranges[feature]:
        payload = base.copy()
        payload[feature_map[feature]] = float(value)
        row = build_feature_row(df_feat, payload)
        pred = float(model.predict(row)[0])
        points.append({"x": float(value), "y": pred})

    return points
