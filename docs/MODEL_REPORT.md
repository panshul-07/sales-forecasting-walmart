# Walmart Forecasting Model Report

## 1) Model Used

### Forecast Model (for predictions)
- `VotingRegressor`
- Base models:
  - `RandomForestRegressor`
  - `GradientBoostingRegressor`
- Code: `/Users/panshulaj/Documents/sales-forecasting-walmart/backend/app/services/modeling.py`

### Interpretable Statistical Model (for diagnostics)
- `OLS` from `statsmodels`
- Robust covariance: `HC3`
- Code: `/Users/panshulaj/Documents/sales-forecasting-walmart/backend/app/services/modeling.py`

### Fixed Parametric Notebook Model
- Chosen specification: `aggregated_nonlinear_log_ols_hc3`
- Code: `/Users/panshulaj/Documents/sales-forecasting-walmart/notebooks/Walmart_Parametric_Report.py`
- Notebook: `/Users/panshulaj/Documents/sales-forecasting-walmart/notebooks/Walmart_Parametric_Report.ipynb`

## 2) Data Handling and Preprocessing

- Data source loader with CSV fallback + synthetic generation when CSV missing:
  - `/Users/panshulaj/Documents/sales-forecasting-walmart/backend/app/services/data.py`
- Date parsing + numeric sanitization + required-column validation
- Feature engineering:
  - seasonality (`Week_Sin`, `Week_Cos`)
  - lag features (`Lag_1`, `Lag_4`)
  - rolling windows (`Rolling_4_Mean`, `Rolling_8_Mean`)
  - calendar features (`Month`, `Week`, `Year`)
- Outlier controls:
  - per-store sales clipping
  - IQR clipping for key variables
- Feature code:
  - `/Users/panshulaj/Documents/sales-forecasting-walmart/backend/app/services/features.py`

## 3) Parametric Test Suite

Implemented tests (reported in notebook + JSON output):
- Jarque-Bera
- Shapiro-Wilk
- D'Agostino K2
- Breusch-Pagan
- Breusch-Godfrey
- Ljung-Box
- Ramsey RESET
- residual mean t-test
- Durbin-Watson
- kurtosis bound check
- VIF check

## 4) Final Reported Results

From:
- `/Users/panshulaj/Documents/sales-forecasting-walmart/notebooks/outputs/parametric_report.json`
- `/Users/panshulaj/Documents/sales-forecasting-walmart/notebooks/outputs/model_summary.json`

### Parametric Notebook (Chosen Spec)
- Spec: `aggregated_nonlinear_log_ols_hc3`
- Pass count: `10/11`
- Primary parametric checks: `10/10`

Diagnostics:
- `jarque_bera_p = 0.1468797065`
- `shapiro_p = 0.1101479717`
- `dagostino_k2_p = 0.1093454581`
- `breusch_pagan_p = 0.6114866335`
- `breusch_godfrey_p = 0.0621932348`
- `ljung_box_p = 0.2965631575`
- `ramsey_reset_p = 0.8725166554`
- `durbin_watson = 2.1959766320`
- `residual_kurtosis = 3.2867833958`
- `residual_mean_t_p = 0.9999999987`
- `max_vif = 29614.3302722956` (only failing check)

Accuracy:
- `R2 = 0.9905996179`
- `RMSE = 1024.6056225509`
- `MAE = 799.1675953435`

### Ensemble Forecast Pipeline Accuracy
- `ensemble_test_r2 = 0.912803`
- `ensemble_test_rmse = 8140.537998`
- `ensemble_test_mae = 6610.626614`
- `ols_test_r2 = 0.911243`
- `ols_test_rmse = 8213.026417`
- `ols_test_mae = 6646.050088`

## 5) Output Artifacts

- `/Users/panshulaj/Documents/sales-forecasting-walmart/notebooks/outputs/parametric_report.json`
- `/Users/panshulaj/Documents/sales-forecasting-walmart/notebooks/outputs/parametric_spec_comparison.csv`
- `/Users/panshulaj/Documents/sales-forecasting-walmart/notebooks/outputs/correlation_heatmap.html`
- `/Users/panshulaj/Documents/sales-forecasting-walmart/notebooks/outputs/diagnostic_heatmap.html`
- `/Users/panshulaj/Documents/sales-forecasting-walmart/notebooks/outputs/actual_vs_predicted.html`
- `/Users/panshulaj/Documents/sales-forecasting-walmart/notebooks/outputs/feature_importance_rf.html`
- `/Users/panshulaj/Documents/sales-forecasting-walmart/notebooks/outputs/residual_distribution.html`
