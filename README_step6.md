# Step 6 - Modeling & Forecasting (S1 2026)

This step reproduces the main ARTES report outputs using monthly forecasts on `IM_RI=10` (new vehicles).

## Outputs covered

1. Total monthly market forecast (VP + VU)
2. VP/VU decomposition forecast
3. ARTES brands combined forecast (Renault + Dacia + Nissan)
4. City watch table (S1 2025 vs S1 2024)

## Main script

- `step6_modeling.py`

## Generated files

- `step6_forecast_s1_2026.csv`
- `step6_metrics_summary.csv`
- `step6_backtest_2025_total.csv`
- `step6_city_watch_s1_2025_vs_2024.csv`
- `step6_artes_monthly_history.csv`
- `12_Step6_Forecasts_S1_2026.png`

## Run

```powershell
python step6_modeling.py
```

## Notes on dependencies

- The script uses optional models if available:
  - `statsmodels` for SARIMAX
  - `prophet` for Prophet
  - `xgboost` for XGBoost
- If one of these is missing, the script falls back automatically (without crashing).
- Required baseline libs: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `openpyxl`.
