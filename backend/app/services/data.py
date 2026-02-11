from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "Store",
    "Date",
    "Weekly_Sales",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
]


def _generate_synthetic_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    stores = np.arange(1, 46)
    dates = pd.date_range("2010-02-05", periods=143, freq="W-FRI")
    rows = []

    for store in stores:
        store_level = rng.normal(170000, 22000)
        trend = rng.uniform(-130, 150)
        for i, dt in enumerate(dates):
            holiday = int((dt.month == 11 and dt.day >= 20) or (dt.month == 12 and dt.day <= 31))
            temp = 62 + 18 * np.sin(2 * np.pi * dt.dayofyear / 365.25) + rng.normal(0, 7)
            fuel = 2.8 + 0.012 * i + rng.normal(0, 0.08)
            cpi = 210 + 0.18 * i + rng.normal(0, 1.5)
            unemp = 8.5 - 0.012 * i + rng.normal(0, 0.25)

            seasonality = 14000 * np.sin(2 * np.pi * dt.isocalendar().week / 52)
            noise = rng.normal(0, 7000)
            sales = (
                store_level
                + trend * i
                + seasonality
                + 16500 * holiday
                - 600 * (temp - 70) ** 2 / 100
                - 2400 * fuel
                - 90 * cpi
                - 3100 * unemp
                + noise
            )
            rows.append(
                {
                    "Store": store,
                    "Date": dt,
                    "Weekly_Sales": max(5000, sales),
                    "Holiday_Flag": holiday,
                    "Temperature": temp,
                    "Fuel_Price": fuel,
                    "CPI": cpi,
                    "Unemployment": unemp,
                }
            )

    return pd.DataFrame(rows)


def _resolve_data_path(root: Path) -> Path | None:
    candidates = [
        os.getenv("WALMART_DATA_PATH"),
        str(root / "data" / "walmart_sales.csv"),
        str(root / "data" / "Walmart Data Analysis and Forcasting.csv"),
        str(root / "walmart_sales.csv"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def load_walmart_data(root: Path) -> pd.DataFrame:
    data_path = _resolve_data_path(root)
    if data_path is not None:
        df = pd.read_csv(data_path)
    else:
        df = _generate_synthetic_data()
        out_path = root / "data" / "walmart_sales.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values(["Store", "Date"]).reset_index(drop=True)

    for col in ["Store", "Holiday_Flag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    numeric_cols = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna().reset_index(drop=True)
