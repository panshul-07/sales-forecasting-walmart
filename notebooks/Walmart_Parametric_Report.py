from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox, het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.data import load_walmart_data
from backend.app.services.features import FEATURE_COLUMNS, make_features

OUT = ROOT / "notebooks" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


@dataclass
class SpecResult:
    name: str
    model: any
    transform: str
    test_metrics: dict
    diagnostics: dict
    pass_count: int
    total_tests: int


def winsorize_series(s: pd.Series, low: float = 0.02, high: float = 0.98) -> pd.Series:
    lo, hi = s.quantile(low), s.quantile(high)
    return s.clip(lower=lo, upper=hi)


def calc_vif(X: pd.DataFrame) -> dict:
    Xc = sm.add_constant(X, has_constant="add")
    out = {}
    for i, col in enumerate(Xc.columns):
        if col == "const":
            continue
        out[col] = float(variance_inflation_factor(Xc.values, i))
    return out


def diagnostics(model, X: pd.DataFrame, resid: pd.Series) -> dict:
    jb_s, jb_p, skew, kurt = jarque_bera(resid)
    sh_s, sh_p = stats.shapiro(resid.sample(min(5000, len(resid)), random_state=42))
    bp_s, bp_p, _, _ = het_breuschpagan(resid, sm.add_constant(X, has_constant="add"))
    bg_s, bg_p, _, _ = acorr_breusch_godfrey(model, nlags=2)
    lb = acorr_ljungbox(resid, lags=[10], return_df=True)
    reset = linear_reset(model, power=2, use_f=True)
    dw = durbin_watson(resid)
    k2_s, k2_p = stats.normaltest(resid)
    t_s, t_p = stats.ttest_1samp(resid, 0.0)
    return {
        "jarque_bera_p": float(jb_p),
        "shapiro_p": float(sh_p),
        "dagostino_k2_p": float(k2_p),
        "breusch_pagan_p": float(bp_p),
        "breusch_godfrey_p": float(bg_p),
        "ljung_box_p": float(lb["lb_pvalue"].iloc[0]),
        "ramsey_reset_p": float(reset.pvalue),
        "durbin_watson": float(dw),
        "residual_kurtosis": float(kurt),
        "residual_skew": float(skew),
        "residual_mean_t_p": float(t_p),
        "max_vif": float(max(calc_vif(X).values())),
    }


def pass_score(d: dict) -> tuple[int, int]:
    checks = {
        "JB": d["jarque_bera_p"] > 0.05,
        "Shapiro": d["shapiro_p"] > 0.05,
        "K2": d["dagostino_k2_p"] > 0.05,
        "BP": d["breusch_pagan_p"] > 0.05,
        "BG": d["breusch_godfrey_p"] > 0.05,
        "LjungBox": d["ljung_box_p"] > 0.05,
        "RESET": d["ramsey_reset_p"] > 0.05,
        "MeanZero": d["residual_mean_t_p"] > 0.05,
        "DW": 1.5 <= d["durbin_watson"] <= 2.5,
        "Kurtosis": 2.0 <= d["residual_kurtosis"] <= 4.0,
        "VIF": d["max_vif"] < 10,
    }
    return int(sum(checks.values())), len(checks)


def build_specs(train: pd.DataFrame, test: pd.DataFrame) -> list[SpecResult]:
    X_train = train[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]
    y_train_raw = train["Weekly_Sales"]
    y_test = test["Weekly_Sales"]

    specs = []

    # Spec 1: Raw OLS
    m1 = sm.OLS(y_train_raw, sm.add_constant(X_train, has_constant="add")).fit()
    pred1 = m1.predict(sm.add_constant(X_test, has_constant="add"))
    d1 = diagnostics(m1, X_train, m1.resid)
    p1, t1 = pass_score(d1)
    specs.append(
        SpecResult(
            name="raw_ols",
            model=m1,
            transform="none",
            test_metrics={
                "r2": float(r2_score(y_test, pred1)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred1))),
                "mae": float(mean_absolute_error(y_test, pred1)),
            },
            diagnostics=d1,
            pass_count=p1,
            total_tests=t1,
        )
    )

    # Spec 2: Log OLS + HC3
    y_train_log = np.log1p(y_train_raw)
    m2 = sm.OLS(y_train_log, sm.add_constant(X_train, has_constant="add")).fit(cov_type="HC3")
    pred2 = np.expm1(m2.predict(sm.add_constant(X_test, has_constant="add")))
    d2 = diagnostics(m2, X_train, m2.resid)
    p2, t2 = pass_score(d2)
    specs.append(
        SpecResult(
            name="log_ols_hc3",
            model=m2,
            transform="log1p",
            test_metrics={
                "r2": float(r2_score(y_test, pred2)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred2))),
                "mae": float(mean_absolute_error(y_test, pred2)),
            },
            diagnostics=d2,
            pass_count=p2,
            total_tests=t2,
        )
    )

    # Spec 3: Winsor + BoxCox + HC3
    y_w = winsorize_series(y_train_raw, 0.03, 0.97)
    y_box, lmbda = stats.boxcox(np.maximum(y_w, 1.0))
    m3 = sm.OLS(y_box, sm.add_constant(X_train, has_constant="add")).fit(cov_type="HC3")
    pred3_t = m3.predict(sm.add_constant(X_test, has_constant="add"))
    pred3 = np.power(np.maximum(pred3_t * lmbda + 1, 1e-9), 1 / lmbda) if abs(lmbda) > 1e-9 else np.exp(pred3_t)
    d3 = diagnostics(m3, X_train, m3.resid)
    p3, t3 = pass_score(d3)
    specs.append(
        SpecResult(
            name="winsor_boxcox_ols_hc3",
            model=m3,
            transform=f"boxcox(lambda={lmbda:.4f})",
            test_metrics={
                "r2": float(r2_score(y_test, pred3)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred3))),
                "mae": float(mean_absolute_error(y_test, pred3)),
            },
            diagnostics=d3,
            pass_count=p3,
            total_tests=t3,
        )
    )

    # Spec 4: Robustly trimmed log OLS
    m4_base = sm.OLS(y_train_log, sm.add_constant(X_train, has_constant="add")).fit(cov_type="HC3")
    infl = OLSInfluence(m4_base)
    cooks = infl.cooks_distance[0]
    thr = 4 / len(X_train)
    keep = cooks < thr
    Xt = X_train.loc[keep]
    yt = y_train_log.loc[keep]
    m4 = sm.OLS(yt, sm.add_constant(Xt, has_constant="add")).fit(cov_type="HC3")
    pred4 = np.expm1(m4.predict(sm.add_constant(X_test, has_constant="add")))
    d4 = diagnostics(m4, Xt, m4.resid)
    p4, t4 = pass_score(d4)
    specs.append(
        SpecResult(
            name="trimmed_log_ols_hc3",
            model=m4,
            transform="log1p + cooks_trim",
            test_metrics={
                "r2": float(r2_score(y_test, pred4)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred4))),
                "mae": float(mean_absolute_error(y_test, pred4)),
            },
            diagnostics=d4,
            pass_count=p4,
            total_tests=t4,
        )
    )

    # Spec 5: Aggregated-date OLS with nonlinear terms for better parametric fit
    agg_cols = [
        "Weekly_Sales",
        "Holiday_Flag",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Week_Sin",
        "Week_Cos",
        "Lag_1",
        "Rolling_4_Mean",
    ]
    train_agg = train.groupby("Date", as_index=False)[agg_cols].mean()
    test_agg = test.groupby("Date", as_index=False)[agg_cols].mean()
    for df in (train_agg, test_agg):
        df["Temp2"] = df["Temperature"] ** 2
        df["Fuel2"] = df["Fuel_Price"] ** 2
        df["Unemp2"] = df["Unemployment"] ** 2
        df["Interaction"] = df["Holiday_Flag"] * df["Week_Cos"]

    agg_features = [
        "Holiday_Flag",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Week_Sin",
        "Week_Cos",
        "Lag_1",
        "Rolling_4_Mean",
        "Temp2",
        "Fuel2",
        "Unemp2",
        "Interaction",
    ]
    X5_train = train_agg[agg_features]
    X5_test = test_agg[agg_features]
    y5_train = np.log1p(train_agg["Weekly_Sales"])
    y5_test = test_agg["Weekly_Sales"]
    m5 = sm.OLS(y5_train, sm.add_constant(X5_train, has_constant="add")).fit(cov_type="HC3")
    pred5 = np.expm1(m5.predict(sm.add_constant(X5_test, has_constant="add")))
    d5 = diagnostics(m5, X5_train, m5.resid)
    p5, t5 = pass_score(d5)
    specs.append(
        SpecResult(
            name="aggregated_nonlinear_log_ols_hc3",
            model=m5,
            transform="log1p + date aggregation + nonlinear terms",
            test_metrics={
                "r2": float(r2_score(y5_test, pred5)),
                "rmse": float(np.sqrt(mean_squared_error(y5_test, pred5))),
                "mae": float(mean_absolute_error(y5_test, pred5)),
            },
            diagnostics=d5,
            pass_count=p5,
            total_tests=t5,
        )
    )

    return specs


def main() -> None:
    raw = load_walmart_data(ROOT)
    feat = make_features(raw)
    cutoff = feat["Date"].quantile(0.8)
    train = feat[feat["Date"] <= cutoff].copy()
    test = feat[feat["Date"] > cutoff].copy()

    specs = build_specs(train, test)
    best = sorted(specs, key=lambda s: (s.pass_count, s.test_metrics["r2"]), reverse=True)[0]

    summary_rows = []
    for s in specs:
        summary_rows.append(
            {
                "spec": s.name,
                "transform": s.transform,
                "pass_count": s.pass_count,
                "total_tests": s.total_tests,
                "pass_ratio": s.pass_count / s.total_tests,
                **s.test_metrics,
                **s.diagnostics,
            }
        )

    df = pd.DataFrame(summary_rows).sort_values(["pass_count", "r2"], ascending=[False, False])
    df.to_csv(OUT / "parametric_spec_comparison.csv", index=False)

    report = {
        "chosen_spec": best.name,
        "chosen_transform": best.transform,
        "chosen_pass_count": best.pass_count,
        "chosen_total_tests": best.total_tests,
        "chosen_primary_parametric_pass_count": int(
            sum(
                [
                    best.diagnostics["jarque_bera_p"] > 0.05,
                    best.diagnostics["shapiro_p"] > 0.05,
                    best.diagnostics["dagostino_k2_p"] > 0.05,
                    best.diagnostics["breusch_pagan_p"] > 0.05,
                    best.diagnostics["breusch_godfrey_p"] > 0.05,
                    best.diagnostics["ljung_box_p"] > 0.05,
                    best.diagnostics["ramsey_reset_p"] > 0.05,
                    best.diagnostics["residual_mean_t_p"] > 0.05,
                    1.5 <= best.diagnostics["durbin_watson"] <= 2.5,
                    2.0 <= best.diagnostics["residual_kurtosis"] <= 4.0,
                ]
            )
        ),
        "chosen_primary_parametric_total_tests": 10,
        "metrics": best.test_metrics,
        "diagnostics": best.diagnostics,
        "all_specs": summary_rows,
        "note": "Real retail data may not satisfy all strict parametric assumptions simultaneously; this report picks the best-validating specification.",
    }
    (OUT / "parametric_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Chosen spec:", best.name)
    print("Pass count:", f"{best.pass_count}/{best.total_tests}")
    print("R2:", round(best.test_metrics["r2"], 6))
    print("RMSE:", round(best.test_metrics["rmse"], 6))


if __name__ == "__main__":
    main()
