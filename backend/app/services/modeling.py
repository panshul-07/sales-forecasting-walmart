from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox, het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera, omni_normtest

from .features import FEATURE_COLUMNS


@dataclass
class TrainingArtifacts:
    model: Any
    ols_result: Any
    feature_columns: list[str]
    metrics: Dict[str, float]
    diagnostics: Dict[str, Dict[str, float]]


def _split_by_time(df_feat: pd.DataFrame):
    cutoff = df_feat["Date"].quantile(0.8)
    train = df_feat[df_feat["Date"] <= cutoff].copy()
    test = df_feat[df_feat["Date"] > cutoff].copy()
    return train, test


def _vif_table(X: pd.DataFrame) -> Dict[str, float]:
    X_const = sm.add_constant(X)
    values = {}
    for i, col in enumerate(X_const.columns):
        if col == "const":
            continue
        values[col] = float(variance_inflation_factor(X_const.values, i))
    return values


def _diagnostics(ols_result, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
    residuals = ols_result.resid
    jb_stat, jb_p, skew, kurt = jarque_bera(residuals)
    omni_stat, omni_p = omni_normtest(residuals)
    shapiro_stat, shapiro_p = stats.shapiro(residuals.sample(min(5000, len(residuals)), random_state=42))
    k2_stat, k2_p = stats.normaltest(residuals)
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, sm.add_constant(X))
    dw = durbin_watson(residuals)
    bg_stat, bg_p, _, _ = acorr_breusch_godfrey(ols_result, nlags=2)
    lb_df = acorr_ljungbox(residuals, lags=[10], return_df=True)
    reset_result = linear_reset(ols_result, power=2, use_f=True)
    t_stat, t_p = stats.ttest_1samp(residuals, popmean=0.0)

    return {
        "normality": {
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_p": float(jb_p),
            "omnibus_stat": float(omni_stat),
            "omnibus_p": float(omni_p),
            "shapiro_stat": float(shapiro_stat),
            "shapiro_p": float(shapiro_p),
            "dagostino_k2_stat": float(k2_stat),
            "dagostino_k2_p": float(k2_p),
            "residual_skew": float(skew),
            "residual_kurtosis": float(kurt),
        },
        "heteroskedasticity": {
            "breusch_pagan_stat": float(bp_stat),
            "breusch_pagan_p": float(bp_p),
        },
        "autocorrelation": {
            "durbin_watson": float(dw),
            "breusch_godfrey_stat": float(bg_stat),
            "breusch_godfrey_p": float(bg_p),
            "ljung_box_stat_lag10": float(lb_df["lb_stat"].iloc[0]),
            "ljung_box_p_lag10": float(lb_df["lb_pvalue"].iloc[0]),
        },
        "specification": {
            "ramsey_reset_f": float(reset_result.fvalue),
            "ramsey_reset_p": float(reset_result.pvalue),
            "residual_mean_t_stat": float(t_stat),
            "residual_mean_t_p": float(t_p),
        },
        "multicollinearity": _vif_table(X),
        "fit": {
            "r_squared": float(ols_result.rsquared),
            "adj_r_squared": float(ols_result.rsquared_adj),
            "aic": float(ols_result.aic),
            "bic": float(ols_result.bic),
        },
    }


def train_pipeline(df_feat: pd.DataFrame, artifact_dir: Path) -> TrainingArtifacts:
    train_df, test_df = _split_by_time(df_feat)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["Weekly_Sales"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["Weekly_Sales"]

    model = VotingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=260, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1)),
            ("gbr", GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    y_train_log = np.log1p(y_train)
    X_ols_train = sm.add_constant(X_train, has_constant="add")
    ols = sm.OLS(y_train_log, X_ols_train).fit(cov_type="HC3")

    X_ols_test = sm.add_constant(X_test, has_constant="add").reindex(columns=X_ols_train.columns, fill_value=0.0)
    ols_test_pred = np.expm1(ols.predict(X_ols_test))

    metrics = {
        "ensemble_train_r2": float(r2_score(y_train, pred_train)),
        "ensemble_test_r2": float(r2_score(y_test, pred_test)),
        "ensemble_test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "ensemble_test_mae": float(mean_absolute_error(y_test, pred_test)),
        "ols_test_r2": float(r2_score(y_test, ols_test_pred)),
        "ols_test_rmse": float(np.sqrt(mean_squared_error(y_test, ols_test_pred))),
        "ols_test_mae": float(mean_absolute_error(y_test, ols_test_pred)),
    }

    diagnostics = _diagnostics(ols, X_train, y_train_log)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "ols": ols,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": metrics,
            "diagnostics": diagnostics,
        },
        artifact_dir / "walmart_forecasting_bundle.joblib",
    )

    return TrainingArtifacts(
        model=model,
        ols_result=ols,
        feature_columns=FEATURE_COLUMNS,
        metrics=metrics,
        diagnostics=diagnostics,
    )


def load_artifacts(artifact_dir: Path):
    bundle_path = artifact_dir / "walmart_forecasting_bundle.joblib"
    if not bundle_path.exists():
        return None
    return joblib.load(bundle_path)
