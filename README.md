# PFE: Forecasting Tunisian Vehicle Sales (ARTES Report)

**Projet de Fin d'Études** — Time-series forecasting pipeline for the Tunisian automotive market (ARTES brands).

## Project Overview

This project builds an end-to-end data pipeline to clean, enrich, and model Tunisian vehicle sales data, producing monthly forecasts for S1 2026 (January–June). The deliverables include:

- **Total market forecast** (VP/VU split)
- **ARTES brand volumes** (Renault + Dacia + Nissan)
- **City-level monitoring** (S1 2025 vs S1 2024)
- **Multi-scenario analysis** (baseline/optimistic/pessimistic macro assumptions)
- **Backtest metrics** (SARIMAX, Prophet, XGBoost ensemble)

**Key techniques**: Data cleaning, feature engineering, SARIMAX + Prophet ensemble, XGBoost for shares, Ramadan seasonality handling, scenario analysis.

## Project Structure

```
PFE/
├── step3_cleaning.py              # Load & clean 8x Excel files → data_intermediate.csv
├── step2_5_enrich_data.py         # Enrich with segments, countries → data_cleaned_enriched.csv
├── step4_eda.py                   # Exploratory plots + insights
├── step5_preparation.py           # Daily aggregation, features, normalization → modeling CSVs
├── step6_modeling.py              # SARIMAX + Prophet + XGBoost ensemble → forecasts
├── add_macro_scenarios.py         # Populate Excel with baseline/optimiste/prudent scenarios
├── produce_artes_report.py        # Consolidate scenario outputs → final report
│
├── valider_pipeline.py            # Validation utilities (column checks, completeness)
├── verifier_pret.py               # Pre-step6 readiness checks
│
├── data/                          # Input Excel files
│   ├── Cumule2019PFE.xlsx
│   ├── ... (6 more Excel files)
│   └── donnees_externes_tunisie.xlsx  (macro scenarios)
│
├── README.md                      # This file (project overview)
├── README_step6.md                # Step 6 detailed documentation
├── SCENARIO_SETUP.md              # Scenario configuration guide
├── requirements.txt               # Python dependencies
│
└── outputs/                       # Generated data & reports
    ├── data_intermediate.csv
    ├── data_cleaned_enriched.csv
    ├── data_prepared_final.csv
    ├── data_prepared_final_full.csv
    ├── step6_forecast_s1_2026.csv
    ├── rapport_ARTES_s1_2026_by_scenario.csv
    ├── rapport_ARTES_s1_2026_by_scenario.png
    └── ... (metrics, backtests, city watch, etc.)
```

## Quick Start

### 1. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

See **Installation** section below for details.

### 2. Run the Full Pipeline

```powershell
# Step 3: Clean & combine 8 Excel files
python step3_cleaning.py

# Step 2.5: Enrich with segment/country/market-type data
python step2_5_enrich_data.py

# Step 4: Exploratory Data Analysis (plots)
python step4_eda.py

# Step 5: Prepare data for modeling (daily aggregation, features)
python step5_preparation.py

# Verify readiness before step 6
python verifier_pret.py

# Step 6: Forecast with SARIMAX + Prophet + XGBoost
python step6_modeling.py
```

**Expected runtime**: ~5 minutes (mostly step 6 SARIMAX fitting)

### 3. Add Scenarios & Generate Report

```powershell
# Populate Excel with example scenarios (baseline/optimiste/prudent)
python add_macro_scenarios.py

# Re-run step 6 with scenarios detected automatically
python step6_modeling.py

# Consolidate scenario outputs into final report
python produce_artes_report.py
```

**Output files ready for mémoire**:
- `rapport_ARTES_s1_2026_by_scenario.csv` — Summary table
- `rapport_ARTES_s1_2026_by_scenario.png` — Visualization
- `rapport_ARTES_s1_2026_summary.txt` — Auto-generated text

## Installation

### Prerequisites

- **Python 3.10+** (tested with 3.10–3.13)
- **pip** or **conda** package manager
- **Excel files** in `data/` folder

### Option A: Full Installation (All Models)

```bash
pip install -r requirements.txt
```

This installs:
- Core: pandas, numpy, matplotlib, scikit-learn, openpyxl
- Models: statsmodels (SARIMAX), prophet (Prophet), xgboost (XGBoost)

**Note**: Prophet installation may take 2–3 minutes on first run.

### Option B: Minimal Installation (No Prophet)

If Prophet installation is slow or problematic:

```bash
pip install pandas>=2.0 numpy>=1.24 matplotlib>=3.7 scikit-learn>=1.3 openpyxl>=3.1 statsmodels>=0.14 xgboost>=2.0
```

The scripts will auto-detect missing models and fall back gracefully.

### Option C: Using Conda

```bash
conda create -n pfe python=3.11
conda activate pfe
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import pandas, numpy, matplotlib, sklearn, openpyxl, statsmodels, prophet, xgboost; print('✅ All dependencies installed')"
```

If any package is missing, you'll see an import error—just run `pip install <package>` again.

## Data Flow

```
Data Inputs (Excel files in data/)
    ↓
Step 3: Clean & combine (data_intermediate.csv)
    ↓
Step 2.5: Enrich with segments (data_cleaned_enriched.csv)
    ↓
Step 4: EDA visualizations + insights
    ↓
Step 5: Prepare for modeling
    ├── Daily aggregation by market type
    ├── Feature engineering (lags, moving averages, temporal encoding)
    ├── Normalization (StandardScaler fit on train)
    └── Outputs: data_prepared_final.csv (daily) + data_prepared_final_full.csv (transaction-level)
    ↓
Step 6: Forecasting (with optional scenarios)
    ├── Monthly base + external regressors merge
    ├── Models: SARIMAX (baseline) + Prophet + XGBoost (shares/ARTES)
    ├── Backtesting on 2025 data
    ├── S1 2026 forecast (6 months)
    ├── Scenario runs (if baseline/optimiste/prudent detected in Excel)
    └── Outputs: step6_*.csv, PNG, metrics, city watch
    ↓
Report generation (produce_artes_report.py)
    └── Outputs: rapport_ARTES_*.csv/png/txt (ready for mémoire)
```

## Documentation

- **README_step6.md** — Detailed Step 6 documentation: outputs, metrics interpretation, troubleshooting
- **SCENARIO_SETUP.md** — How to set up & customize macroeconomic scenarios
- **valider_pipeline.py** — Validation utilities (check column presence, data completeness)
- **verifier_pret.py** — Pre-step6 readiness checker

## Key Files

| File | Purpose |
|------|---------|
| `step3_cleaning.py` | Clean Excel files, standardize columns, remove duplicates |
| `step2_5_enrich_data.py` | Add SEGMENT, SOUS_SEGMENT, GROUPE, PAYS_DORIGINE, CONTINENT, TYPE_MARCHE |
| `step4_eda.py` | Generate 12 EDA plots (distributions, seasonality, top brands/cities) |
| `step5_preparation.py` | Daily aggregation, lag/MA features, normalization, Ramadan flag, 2026 placeholder |
| `step6_modeling.py` | SARIMAX + Prophet + XGBoost ensemble, backtesting, scenario support |
| `add_macro_scenarios.py` | Populate Excel scenarios (run before step6 for multi-scenario analysis) |
| `produce_artes_report.py` | Consolidate scenario CSVs → summary table, PNG, text for mémoire |

## Models & Methodology

### Forecasting Approach (Step 6)

1. **SARIMAX** — Seasonal ARIMA(1,1,1)×(1,1,1)₁₂ on monthly totals + external regressors
2. **Prophet** — Facebook Prophet with yearly seasonality + regressors (if installed)
3. **XGBoost** — Gradient boosting for VP share & ARTES share (lag/MA features, Ramadan flag)
4. **Ensemble** — Weighted average (60% SARIMAX + 40% Prophet) or fallback to best available model

### Features Used

- **Temporal**: Month (sin/cos encoding), Ramadan days, day-of-week one-hot
- **Lags**: Lag 1, 3, 6, 12 months (for shares & ARTES)
- **Moving Averages**: 3, 6, 12-month rolling averages
- **External**: PIB_INDEX, INFLATION (if provided in scenarios)
- **Target variables**: TOTAL, VP, VU, PART_VP, ARTES_VOL, ARTES_SHARE

### Train/Val/Test Split

- **Train**: 2019–2023 (used to fit SARIMAX, Prophet, XGBoost)
- **Backtest**: 2025 (evaluate model accuracy)
- **Forecast**: S1 2026 (Jan–Jun, 6 months ahead)

## Interpretation

### Forecast Metrics (from step6_metrics_summary.csv)

| Metric | Meaning |
|--------|---------|
| **MAE** | Mean Absolute Error — avg forecast error in units |
| **RMSE** | Root Mean Squared Error — penalizes large errors |
| **sMAPE** | Symmetric MAPE (%) — <10% excellent, 10–20% good, 20–30% fair |

**Example**: sMAPE=21.6% on ENSEMBLE_TOTAL means the forecast is typically off by ~22%, which is good for automotive market.

### Output CSV Format (step6_forecast_s1_2026.csv)

```
Date,PREV_TOTAL_MARCHE,PREV_VP,PREV_VU,PART_VP,PREV_PART_ARTES,PREV_VOL_ARTES,RAMADAN_DAYS
2026-01-01,7286,5670,1616,0.778,0.071,516,0
2026-02-01,6555,5043,1512,0.770,0.068,446,0
2026-03-01,7456,5772,1684,0.774,0.073,543,0
...
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'prophet'` | Run `pip install prophet` (or skip Prophet — script auto-detects) |
| Step 5 hangs or crashes | Check input Excel files exist in `data/` folder |
| Step 6 produces very high sMAPE (>50%) | Check Ramadan dates; external shocks (lockdown, fuel crisis) may impact accuracy |
| No scenario CSVs generated | Verify scenario sheet names contain "baseline", "optimiste", or "prudent" |
| Forecast values look unrealistic | Check external regressors in Excel; ensure dates align monthly |

## Contributing & Modifications

To customize the pipeline:

1. **Change gap-fill strategy** → Edit `STRATEGIE_REMPLISSAGE` in `step5_preparation.py` (options: 'zero', 'forward_fill', 'interpolate')
2. **Adjust SARIMAX order** → Modify `order=(1,1,1), seasonal_order=(1,1,1,12)` in `step6_modeling.py`
3. **Add custom scenarios** → Add sheets to `donnees_externes_tunisie.xlsx` matching pattern (e.g., "scenario_crisis")
4. **Change validation rules** → Edit `COLONNES_REQUISES` in `valider_pipeline.py`

## References

- **Data**: 8 Excel files (2019–2025 sales, external macros)
- **Enrichment**: Segment/country mapping, type standardization (VP/VU)
- **Methods**: SARIMAX (statsmodels), Prophet (Facebook), XGBoost (scikit-learn wrapper)
- **Visualization**: Matplotlib, Seaborn

## License & Contact

**Project**: PFE (Projet de Fin d'Études)  
**Data Source**: ARTES Tunisia automotive market  
**Last Updated**: April 2026

---

## Next Steps

1. ✅ Run full pipeline: `python step3_cleaning.py` → `step6_modeling.py`
2. ✅ Add scenarios: `python add_macro_scenarios.py` + re-run step 6
3. ✅ Generate final report: `python produce_artes_report.py`
4. 📄 Paste outputs into mémoire (CSV tables, PNG charts, text summary)

**Questions?** See README_step6.md or SCENARIO_SETUP.md.
