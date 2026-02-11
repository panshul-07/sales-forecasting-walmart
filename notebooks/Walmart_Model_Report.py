from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.data import load_walmart_data
from backend.app.services.features import FEATURE_COLUMNS, make_features
from backend.app.services.modeling import train_pipeline

OUT = ROOT / "notebooks" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    raw = load_walmart_data(ROOT)
    feat = make_features(raw)
    artifacts = train_pipeline(feat, ROOT / "artifacts")

    bundle = {
        "model_used": {
            "forecast_model": "VotingRegressor(RandomForestRegressor + GradientBoostingRegressor)",
            "interpretable_model": "OLS (log1p target, HC3 robust errors)",
            "features": FEATURE_COLUMNS,
        },
        "metrics": artifacts.metrics,
        "diagnostics": artifacts.diagnostics,
    }
    (OUT / "model_summary.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    corr_cols = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Holiday_Flag"]
    corr = raw[corr_cols].corr().round(3)
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="Tealgrn", title="Feature Correlation Heatmap")
    fig_corr.write_html(OUT / "correlation_heatmap.html", include_plotlyjs="cdn")

    sample = feat.sample(n=min(1400, len(feat)), random_state=42)
    preds = artifacts.model.predict(sample[FEATURE_COLUMNS])
    fig_scatter = px.scatter(
        x=sample["Weekly_Sales"],
        y=preds,
        labels={"x": "Actual Weekly Sales", "y": "Predicted Weekly Sales"},
        title="Actual vs Predicted Sales",
        opacity=0.45,
    )
    min_v = float(min(sample["Weekly_Sales"].min(), preds.min()))
    max_v = float(max(sample["Weekly_Sales"].max(), preds.max()))
    fig_scatter.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Ideal", line=dict(dash="dash")))
    fig_scatter.write_html(OUT / "actual_vs_predicted.html", include_plotlyjs="cdn")

    y_log = np.log1p(feat["Weekly_Sales"])
    X_ols = sm.add_constant(feat[FEATURE_COLUMNS], has_constant="add")
    residuals = y_log - artifacts.ols_result.predict(X_ols)
    fig_res = px.histogram(pd.DataFrame({"residuals": residuals}), x="residuals", nbins=60, title="OLS Residual Distribution")
    fig_res.write_html(OUT / "residual_distribution.html", include_plotlyjs="cdn")

    rf_model = artifacts.model.named_estimators_["rf"]
    importance = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": rf_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    fig_imp = px.bar(importance.head(12), x="importance", y="feature", orientation="h", title="RandomForest Feature Importance")
    fig_imp.write_html(OUT / "feature_importance_rf.html", include_plotlyjs="cdn")

    diag_vals = {
        "JB p": artifacts.diagnostics["normality"]["jarque_bera_p"],
        "Shapiro p": artifacts.diagnostics["normality"]["shapiro_p"],
        "Breusch-Pagan p": artifacts.diagnostics["heteroskedasticity"]["breusch_pagan_p"],
        "Breusch-Godfrey p": artifacts.diagnostics["autocorrelation"]["breusch_godfrey_p"],
        "Ljung-Box p": artifacts.diagnostics["autocorrelation"]["ljung_box_p_lag10"],
        "RESET p": artifacts.diagnostics["specification"]["ramsey_reset_p"],
    }
    diag_df = pd.DataFrame(
        {"test": list(diag_vals.keys()), "neg_log10_p": [-np.log10(max(v, 1e-12)) for v in diag_vals.values()]}
    )
    fig_diag = px.imshow(
        [diag_df["neg_log10_p"].tolist()],
        x=diag_df["test"].tolist(),
        y=["-log10(p)"],
        text_auto=True,
        color_continuous_scale="OrRd",
        title="Diagnostic Test Heatmap",
    )
    fig_diag.write_html(OUT / "diagnostic_heatmap.html", include_plotlyjs="cdn")

    print("Notebook outputs generated at", OUT)
    for key, val in artifacts.metrics.items():
        print(f"{key}: {val:.6f}")


if __name__ == "__main__":
    main()
