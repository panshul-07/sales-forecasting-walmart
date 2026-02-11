from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "Store",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "Month",
    "Week",
    "Year",
    "Week_Sin",
    "Week_Cos",
    "Lag_1",
    "Lag_4",
    "Rolling_4_Mean",
    "Rolling_8_Mean",
]


def clip_outliers_iqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    clipped = df.copy()
    for col in cols:
        q1 = clipped[col].quantile(0.25)
        q3 = clipped[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        clipped[col] = clipped[col].clip(lower=low, upper=high)
    return clipped


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    feat["Weekly_Sales"] = feat.groupby("Store")["Weekly_Sales"].transform(
        lambda s: s.clip(lower=s.quantile(0.03), upper=s.quantile(0.97))
    )
    feat["Year"] = feat["Date"].dt.year
    feat["Month"] = feat["Date"].dt.month
    feat["Week"] = feat["Date"].dt.isocalendar().week.astype(int)
    feat["Week_Sin"] = np.sin(2 * np.pi * feat["Week"] / 52)
    feat["Week_Cos"] = np.cos(2 * np.pi * feat["Week"] / 52)

    feat["Lag_1"] = feat.groupby("Store")["Weekly_Sales"].shift(1)
    feat["Lag_4"] = feat.groupby("Store")["Weekly_Sales"].shift(4)
    feat["Rolling_4_Mean"] = (
        feat.groupby("Store")["Weekly_Sales"].transform(lambda s: s.shift(1).rolling(4).mean())
    )
    feat["Rolling_8_Mean"] = (
        feat.groupby("Store")["Weekly_Sales"].transform(lambda s: s.shift(1).rolling(8).mean())
    )

    feat = feat.dropna().reset_index(drop=True)
    feat = clip_outliers_iqr(
        feat,
        ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Lag_1", "Lag_4"],
    )
    return feat
