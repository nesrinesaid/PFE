# Step 6 - Modeling & Forecasting (S1 2026)

This step reproduces the main ARTES report outputs using monthly forecasts on `IM_RI=10` (new vehicles). The ensemble model combines SARIMAX, Prophet, and XGBoost for robust predictions.

## Outputs covered

1. **Total monthly market forecast** (VP + VU) — SARIMAX + Prophet ensemble
2. **VP/VU decomposition forecast** — XGBoost share model
3. **ARTES brands combined forecast** (Renault + Dacia + Nissan) — XGBoost volume model
4. **City watch table** (S1 2025 vs S1 2024) — Transaction-level monitoring
5. **Scenario analysis** — Multiple macro scenarios (baseline/optimiste/prudent)

## Main script

- `step6_modeling.py` — Core modeling logic
- `add_macro_scenarios.py` — Populate Excel with example scenarios
- `produce_artes_report.py` — Post-process step6 outputs into final report

## Generated files

### Base outputs (always generated)

- `step6_forecast_s1_2026.csv` — Monthly totals + VP/VU + ARTES
- `step6_metrics_summary.csv` — Backtest metrics (MAE, RMSE, sMAPE)
- `step6_backtest_2025_total.csv` — Actual vs predicted for 2025
- `step6_city_watch_s1_2025_vs_2024.csv` — Top cities, growth %
- `step6_artes_monthly_history.csv` — Historical ARTES volumes
- `12_Step6_Forecasts_S1_2026.png` — 4-panel visualization

### Scenario outputs (if scenario sheets detected)

- `step6_forecast_s1_2026_baseline.csv`, `_optimiste.csv`, `_prudent.csv`
- `step6_metrics_summary_baseline.csv`, `_optimiste.csv`, `_prudent.csv`
- `rapport_ARTES_s1_2026_by_scenario.csv` — Consolidated summary
- `rapport_ARTES_monthly_by_scenario.csv` — Monthly detail
- `rapport_ARTES_s1_2026_by_scenario.png` — Scenario comparison chart
- `rapport_ARTES_s1_2026_summary.txt` — Auto-generated text report

## Run

```powershell
# Standard forecast (no scenarios)
python step6_modeling.py

# After populating scenarios in Excel
python add_macro_scenarios.py
python step6_modeling.py
python produce_artes_report.py
```

## Understanding the Outputs

### Forecast CSV columns

| Column | Meaning |
|--------|---------|
| Date | Month (YYYY-01-01 format) |
| PREV_TOTAL_MARCHE | Forecast total volume (VP + VU) |
| PREV_VP | Forecast passenger vehicles |
| PREV_VU | Forecast commercial vehicles |
| PART_VP | Forecast VP share % |
| PART_VU | Forecast VU share % |
| PREV_PART_ARTES | ARTES market share % |
| PREV_VOL_ARTES | ARTES volume (Renault + Dacia + Nissan) |
| RAMADAN_DAYS | Days affected by Ramadan in that month |

### Metrics CSV columns

| Column | Meaning |
|--------|---------|
| name | Model name (SARIMAX_TOTAL, PROPHET_TOTAL, etc.) |
| mae | Mean Absolute Error (avg error in units) |
| rmse | Root Mean Squared Error (penalizes outliers) |
| smape | Symmetric MAPE (% error, 0-100%) |

## Interpreting Metrics

**sMAPE (Symmetric Mean Absolute Percentage Error)**

- **sMAPE < 10%**: Excellent accuracy
- **sMAPE 10–20%**: Good accuracy, production-ready
- **sMAPE 20–30%**: Fair accuracy, acceptable for automotive market
- **sMAPE 30–50%**: Moderate issues, review external factors
- **sMAPE > 50%**: Poor, requires investigation or model retuning

**Example**:
- Backtest ENSEMBLE_TOTAL sMAPE = 21.6% → Good forecast reliability
- If sMAPE > 40%, check: Ramadan dates, external events (lockdowns, fuel prices), missing macro data

## Scenario Setup

The script automatically detects scenario sheets in `data/donnees_externes_tunisie.xlsx` using these patterns:

| Sheet Name | Pattern Recognition |
|------------|---------------------|
| `baseline` | Contains "baseline" (case-insensitive) |
| `optimiste` | Contains "optimiste" |
| `prudent` | Contains "prudent" or "conserv" |
| Custom | Any sheet with "scenario" in name |

### Expected columns in scenario sheets

- **ANNEE** (year, required if DATE not present)
- **MOIS** (month, required if DATE not present) or
- **DATE** (single date column, optional)
- **Numeric columns**: GDP_GROWTH, INFLATION, etc. (up to 6 used)

### Example scenario sheet structure

```csv
ANNEE,MOIS,PIB_INDEX,INFLATION
2019,1,100.0,6.0
2019,2,100.5,6.1
...
2026,6,110.5,5.2
```

## Dependencies

### Required

- `pandas` — Data manipulation
- `numpy` — Numerics
- `matplotlib` — Visualization
- `scikit-learn` — Preprocessing & ensemble models
- `openpyxl` — Excel I/O

### Optional (auto-detected fallback)

- `statsmodels>=0.14` — SARIMAX forecasting
- `prophet>=1.1.5` — Prophet timeseries (Facebook)
- `xgboost>=2.0` — XGBoost models for shares/ARTES

If optional packages are missing, the script prints warnings but continues using available models.

## Expected Runtime

- **First run**: ~2–3 minutes (SARIMAX estimation takes time)
- **Scenario runs**: +30 seconds per scenario (Prophet is faster after initial setup)
- **Report generation**: <30 seconds

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `KeyError: 'TOTAL'` | Missing column in input data | Verify step5 ran successfully |
| SARIMAX convergence warning | Volatile data, hard to fit | Expected; check sMAPE still reasonable |
| Prophet import fails | Not installed | `pip install prophet` |
| Scenario CSVs not generated | No scenario sheets in Excel | Run `add_macro_scenarios.py` first |
| Very high sMAPE (>50%) | External shock (pandemic, fuel crisis) | Check Ramadan dates; consider external regressors |
| No output files | Step 6 crashed silently | Check terminal output for errors |

## Example Output

### Forecast snippet

```
Date,PREV_TOTAL_MARCHE,PREV_VP,PREV_VU,PREV_PART_ARTES,PREV_VOL_ARTES
2026-01-01,7286,5670,1616,0.071,516
2026-02-01,6555,5043,1512,0.068,446
2026-03-01,7456,5772,1684,0.073,543
2026-04-01,6834,5203,1631,0.069,471
2026-05-01,7123,5412,1711,0.072,513
2026-06-01,6945,5267,1678,0.070,486
```

### Metrics snippet

```
name,mae,rmse,smape
SARIMAX_TOTAL,1559.81,1920.84,21.63
PROPHET_TOTAL,1687.34,2045.21,23.18
ENSEMBLE_TOTAL,1589.45,1959.33,21.89
XGB_SHARE_VP,0.034,0.046,4.29
ARTES_VOL,2134.67,2687.92,32.70
```

## Integration with Pipeline

Step 6 depends on outputs from **Step 5** (`data_prepared_final.csv`, daily aggregates + lag features). It does NOT use the transaction-level `data_prepared_final_full.csv`.

```
data_cleaned_enriched.csv (from step 2.5)
    ↓
step6_modeling.py → loads monthly base + external features
    ↓
Scenarios + forecasts → rapport_ARTES_*.csv/png/txt
```
