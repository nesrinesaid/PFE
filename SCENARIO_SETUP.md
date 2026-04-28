# Scenario Setup Guide for Step 6 Modeling

This guide explains how to set up and use macroeconomic scenarios in the Step 6 forecasting pipeline.

## Overview

Scenarios allow you to run the same forecasting model under different macroeconomic assumptions (baseline, optimistic, pessimistic). This produces multiple S1 2026 forecasts reflecting different economic conditions.

## How It Works

1. **Scenario sheets** are detected automatically in `data/donnees_externes_tunisie.xlsx`
2. If scenarios are found, `step6_modeling.py` runs the forecast **once per scenario**
3. Each scenario run produces separate CSV files with metrics
4. `produce_artes_report.py` consolidates these into a single summary report

## Setting Up Scenarios

### Step 1: Populate Excel Workbook

Use the helper script to create example scenarios:

```powershell
python add_macro_scenarios.py
```

This generates three sheets:
- **baseline** — Reference case (mild growth, moderate inflation)
- **optimiste** — Strong growth, low inflation
- **prudent** — Weak growth, high inflation

Each sheet has columns: `Date`, `PIB_INDEX`, `INFLATION`

### Step 2: Customize Scenarios (Optional)

Edit `data/donnees_externos_tunisie.xlsx` in Excel or LibreOffice:

**Baseline scenario example:**

| ANNEE | MOIS | PIB_INDEX | INFLATION | TAUX_CHMG |
|-------|------|-----------|-----------|-----------|
| 2025  | 1    | 103.2     | 5.1       | 17.5      |
| 2025  | 2    | 103.5     | 5.0       | 17.4      |
| ...   | ...  | ...       | ...       | ...       |
| 2026  | 6    | 107.8     | 4.8       | 16.9      |

**Optimiste scenario example** (stronger growth):

| ANNEE | MOIS | PIB_INDEX | INFLATION | TAUX_CHMG |
|-------|------|-----------|-----------|-----------|
| 2025  | 1    | 104.2     | 4.5       | 16.9      |
| 2025  | 2    | 104.7     | 4.3       | 16.7      |
| ...   | ...  | ...       | ...       | ...       |
| 2026  | 6    | 111.0     | 3.8       | 16.0      |

**Prudent scenario example** (weaker growth, inflation):

| ANNEE | MOIS | PIB_INDEX | INFLATION | TAUX_CHMG |
|-------|------|-----------|-----------|-----------|
| 2025  | 1    | 102.1     | 6.2       | 18.5      |
| 2025  | 2    | 102.3     | 6.4       | 18.8      |
| ...   | ...  | ...       | ...       | ...       |
| 2026  | 6    | 104.5     | 5.9       | 19.2      |

### Step 3: Excel Format Requirements

Each scenario sheet must have:

**Required columns (at least one):**
- `DATE` (single column with full date, e.g., 2025-01-01) OR
- `ANNEE` + `MOIS` (year and month separately)

**Optional numeric columns** (up to 6 used):
- `PIB_INDEX`, `INFLATION`, `TAUX_CHMG`, `TAUX_INTERET`, etc.
- Column names are normalized to uppercase, spaces/underscores removed

**Sheet name patterns** (auto-detected):
```
✅ VALID names:
   - baseline (contains "baseline")
   - optimiste (contains "optimiste")
   - prudent (contains "prudent")
   - scenario_HighGrowth (contains "scenario")
   - conservateur (contains "conserv")
   
❌ INVALID names:
   - historical_data (not recognized)
   - metadata (not recognized)
   - settings (not recognized)
```

## Running Forecasts with Scenarios

### Option A: Auto-generate Example Scenarios

```powershell
# Creates baseline, optimiste, prudent with example data
python add_macro_scenarios.py

# Run step6 (will detect and use all three scenarios)
python step6_modeling.py

# Consolidate outputs
python produce_artes_report.py
```

### Option B: Manual Setup

1. Open `data/donnees_externes_tunisie.xlsx` in Excel
2. Create scenario sheets: `baseline`, `optimiste`, `prudent`
3. Fill with your own macro assumptions (2019-2026 monthly data)
4. Run:

```powershell
python step6_modeling.py
python produce_artes_report.py
```

## Scenario Output Files

Once scenarios are detected, you get:

### Per-scenario outputs:
```
step6_forecast_s1_2026_baseline.csv       (6 rows × 2 cols: Date, PREV_TOTAL_MARCHE)
step6_metrics_summary_baseline.csv        (model metrics)
step6_forecast_s1_2026_optimiste.csv
step6_metrics_summary_optimiste.csv
step6_forecast_s1_2026_prudent.csv
step6_metrics_summary_prudent.csv
```

### Consolidated report:
```
rapport_ARTES_s1_2026_by_scenario.csv     (summary: scenario, S1_total, diff_from_baseline)
rapport_ARTES_monthly_by_scenario.csv     (monthly detail across scenarios)
rapport_ARTES_s1_2026_by_scenario.png     (time series chart)
rapport_ARTES_s1_2026_summary.txt         (text for mémoire)
```

## Example Output

### rapport_ARTES_s1_2026_by_scenario.csv

```csv
scenario,S1_2026_TOTAL,DIFF_FROM_BASELINE_ABS,DIFF_FROM_BASELINE_PCT
baseline,42500,0,0.0
optimiste,45800,3300,7.8
prudent,39200,-3300,-7.8
```

### rapport_ARTES_s1_2026_summary.txt

```
Rapport ARTES — Prévisions S1 2026 par scénario

Méthodologie: ensemble SARIMAX + Prophet + XGBoost (shares). 
Scénarios macro fournis dans 'donnees_externes_tunisie.xlsx'.

- Scénario BASELINE: S1 2026 total = 42,500 unités
- Scénario OPTIMISTE: S1 2026 total = 45,800 unités (+3,300 vs baseline, +7.8%)
- Scénario PRUDENT: S1 2026 total = 39,200 unités (-3,300 vs baseline, -7.8%)

Remarques: Ces résultats dépendent fortement des hypothèses macro. 
Voir CSVs et graphique pour détails mensuels.
```

## Common Issues

| Issue | Solution |
|-------|----------|
| "Scenario runs complete. Individual CSVs..." but no scenario CSVs appear | Scenario sheets not detected. Check sheet names match patterns (baseline, optimiste, prudent). |
| Column not found error in scenario merge | Ensure ANNEE/MOIS or DATE column exists and is parseable as date. |
| Very different forecast between scenarios | Expected! Large macro shocks (GDP, inflation) significantly impact vehicle demand. |
| No `rapport_ARTES_*.csv` files | Run `produce_artes_report.py` after `step6_modeling.py`. |
| Macro values look wrong (unrealistic GDP growth) | Check example values in `add_macro_scenarios.py`; adjust to realistic Tunisia macro. |

## Tips & Best Practices

1. **Use Tunisia-specific macro data** — Customize scenarios with actual vs forecasted GDP, inflation, unemployment rates.
2. **Validate date ranges** — Ensure scenarios span 2019-01 through 2026-06 (gaps are forward-filled automatically).
3. **Document assumptions** — Add comments to Excel sheets explaining each scenario's assumptions (e.g., "Optimiste = Favorable inflation control, strong tourism demand").
4. **Test sensitivity** — Run extreme scenarios (e.g., -5% GDP, +10% inflation) to test model sensitivity.
5. **Keep baseline realistic** — Baseline should represent consensus forecasts from central bank or IMF.

## Integration with Mémoire

After running scenarios, the text report is ready to paste:

```markdown
## Scénarios de Prévision (Step 6)

Trois scénarios macroéconomiques ont été testés pour la prévision S1 2026:

(Paste content of rapport_ARTES_s1_2026_summary.txt here)

La figure ci-dessous montre la projection mensuelle par scénario:
(Include rapport_ARTES_s1_2026_by_scenario.png)

Voir fichiers pour détails: rapport_ARTES_*.csv
```

## Advanced: Custom Scenarios Beyond Three

To add more scenarios (e.g., "shock", "recovery"):

1. Create new sheet in Excel: `shock` (or `scenario_shock`)
2. Add 2019-2026 monthly data
3. Run `step6_modeling.py` → new sheet automatically detected
4. Update `produce_artes_report.py` if needed to list new scenario

Naming convention: Sheets matching "scenario|baseline|optimiste|prudent|pessimiste|conserv" (any case) are auto-detected.
