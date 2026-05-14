# PFE Architecture, Logic, and Design Rationale

This document explains the project as an engineering system: architecture, data contracts, modeling logic, why each step exists, and why specific technical choices were made.

It is intended for:
- mémoire/PFE chapter writing,
- technical review,
- future maintenance by another developer.

## 1) Objective and System Boundary

## Business objective
Forecast Tunisian new-vehicle demand and ARTES-related indicators for S1 2026 with a reproducible, auditable pipeline.

## Analytical objective
Move from raw heterogeneous Excel files to decision-grade outputs:
- validated historical datasets,
- interpretable EDA,
- forecast tables and metrics,
- scenario comparison outputs,
- data-mining insights (clusters + association rules).

## Scope
In scope:
- data ingestion, cleaning, enrichment,
- feature engineering,
- monthly forecasting and backtesting,
- scenario simulation,
- pattern discovery (unsupervised and rule mining).

Out of scope:
- real-time serving API,
- MLOps orchestration (Airflow/Kubernetes),
- automatic model retraining scheduler.

## 2) High-Level Architecture

The project follows a staged, file-based architecture where each stage publishes explicit CSV artifacts.

```
Raw Excel files (data/) 
    -> step3_cleaning.py
    -> data_intermediate.csv
    -> step2_5_enrich_data.py
    -> data_cleaned_enriched.csv
    -> step4_eda.py
    -> EDA plots (*.png)
    -> step5_preparation.py
    -> modeling-ready datasets (data_prepared_final*.csv)
    -> verifier_pret.py / valider_pipeline.py
    -> step6_modeling.py
    -> forecasts + metrics + scenario outputs (step6_*.csv)
    -> produce_artes_report.py
    -> rapport_ARTES_*.csv/png/txt
    -> step6_datamining.py
    -> clusters + association rules outputs
```

### Why this architecture?
1. It is transparent: every transformation leaves a persistent artifact.
2. It is debuggable: failures can be isolated by stage.
3. It is report-friendly: chapter tables map directly to files.
4. It tolerates optional dependencies (Prophet/XGBoost) via graceful fallback.

## 3) Data Contracts by Stage

The core design choice is contract-based handoff between scripts.

## Stage: `step3_cleaning.py`
Input:
- raw Excel files from `data/`.

Output:
- `data_intermediate.csv`.

Contract:
- unified schema,
- cleaned column names,
- de-duplicated and standardized rows.

Reasoning:
- upstream files are not guaranteed to be homogeneous; centralizing cleanup prevents repeated defensive code in later steps.

## Stage: `step2_5_enrich_data.py`
Input:
- `data_intermediate.csv`.

Output:
- `data_cleaned_enriched.csv`.

Contract:
- adds dimensions used for analysis/modeling (`SEGMENT`, `SOUS_SEGMENT`, `GROUPE`, `PAYS_DORIGINE`, `CONTINENT`, `TYPE_MARCHE`).

Reasoning:
- enrichment is separated from cleaning to keep responsibilities clear:
  cleaning = correctness, enrichment = analytical context.

## Stage: `step4_eda.py`
Input:
- `data_cleaned_enriched.csv`.

Output:
- descriptive plots (market evolution, seasonality, brand and segment views).

Reasoning:
- EDA validates assumptions before forecasting,
- provides narrative context for the report,
- helps detect anomalies (outliers, abrupt regime changes).

## Stage: `step5_preparation.py`
Input:
- `data_cleaned_enriched.csv`.

Output:
- `data_prepared_final.csv`,
- `data_prepared_final_full.csv`,
- split datasets (`data_train.csv`, `data_validation_2024.csv`, `data_test_2025.csv`, `data_future_2026.csv`).

Contract:
- time-aware features are materialized,
- train/test chronology is respected,
- leakage-sensitive operations are controlled.

Reasoning behind key choices:
1. Lag features: capture short-memory and seasonal inertia.
2. Moving averages: reduce noise and encode local trend.
3. Calendar features: encode known demand rhythms.
4. Ramadan flags: domain-specific seasonality for Tunisia.
5. Explicit future frame for 2026: ensures deterministic forecast shape.

## Stage: validation utilities
Tools:
- `verifier_pret.py`,
- `valider_pipeline.py`.

Reasoning:
- fail early on schema/data issues before expensive modeling.

## Stage: `step6_modeling.py`
Input:
- prepared datasets,
- optional macro scenario sheets from `data/donnees_externes_tunisie.xlsx`.

Output:
- monthly forecasts,
- backtest metrics,
- segment/sous-segment outputs,
- scenario variants.

Reasoning:
- combines classical time-series + ML to balance stability and flexibility.

## Stage: `produce_artes_report.py`
Input:
- step6 scenario outputs.

Output:
- `rapport_ARTES_s1_2026_by_scenario.csv`,
- `rapport_ARTES_monthly_by_scenario.csv`,
- `rapport_ARTES_s1_2026_summary.txt`,
- scenario chart.

Reasoning:
- separates modeling from report assembly,
- guarantees consistent final table formatting and wording.

## Stage: `step6_datamining.py`
Input:
- prepared modeling dataset.

Output:
- `data/datamining_clusters_by_type.csv`,
- `data/datamining_frequent_itemsets.csv`,
- `data/datamining_association_rules.csv`,
- `data/datamining_association_rules_top15.csv`,
- `data/datamining_association_rules_top15.md`,
- `figures/clusters_profile.png`.

Reasoning:
- complements EDA with non-obvious pattern extraction,
- yields explainable business insights before/alongside predictive modeling.

## 4) Modeling Logic and Why Each Choice Was Made

## 4.1 Total market forecast (monthly)
Methods:
- SARIMAX,
- Prophet (if available),
- ensemble blend.

Choice rationale:
1. SARIMAX: robust for autocorrelation + explicit seasonality and exogenous signals.
2. Prophet: flexible trend/season decomposition and holiday-like behavior.
3. Ensemble: reduce model-specific bias variance by weighted averaging.

## 4.2 VP/VU decomposition
Method:
- ML regressor predicts VP share,
- then reconstruct VP and VU volumes.

Choice rationale:
1. Share modeling enforces coherence: VP + VU equals total.
2. Often easier to model proportions than two unconstrained volumes.

## 4.3 ARTES volume/share modeling
Method:
- ML model on ARTES-related target (share or volume depending on data path),
- transformed into final ARTES volume indicators.

Choice rationale:
1. Brand-level dynamics can diverge from total market dynamics.
2. Share-based reasoning remains interpretable for business users.

## 4.4 Segment and sous-segment forecasts
Method:
- group-wise share forecasting at finer granularity.

Choice rationale:
1. supports portfolio planning,
2. provides tactical insights not visible at total-market level.

## 5) Data Mining Logic and Rationale

## 5.1 Clustering
Goal:
- identify behaviorally similar vehicle categories from monthly profiles.

Method:
- pivot by category x month,
- scale features,
- K-Means with robust fallback for small sample cases.

Why K-Means here?
1. fast and stable for low-dimensional monthly profiles,
2. easy to explain in a report,
3. naturally produces discrete behavioral segments.

Current dataset note:
- with only `TYPE_MARCHE` (`VP`, `VU`) available, clustering is intentionally simple (2 samples).
- richer clustering emerges when `CD_TYP_CONS` or finer vehicle types are present.

## 5.2 Association rules (Apriori)
Goal:
- discover co-occurrence structures among market/context attributes.

Method:
- boolean transaction matrix,
- Apriori frequent itemsets,
- confidence/lift-based rules,
- readability constraints (rule-length filtering),
- top-15 clean rule export.

Why Apriori?
1. rules are directly interpretable by non-technical stakeholders,
2. support/confidence/lift provide clear evidence metrics,
3. useful for narrative insights and commercial planning hypotheses.

Why the top-15 export?
1. raw rule files can be large/noisy,
2. report chapters need concise, high-signal rules,
3. standardizes communication.

## 6) Validation Strategy

## Temporal validation
Design:
- train on pre-2025,
- evaluate on 2025 hold-out,
- forecast S1 2026.

Reasoning:
- preserves chronological causality,
- avoids random-split leakage common in time series.

## Metrics
Used:
- MAE,
- RMSE,
- sMAPE.

Reasoning:
1. MAE for average absolute business error,
2. RMSE to penalize large misses,
3. sMAPE for scale-robust percentage comparability.

## 7) Scenario Engine Logic

Mechanism:
1. scenario sheets are auto-detected by name patterns,
2. each scenario runs through same forecasting logic,
3. outputs are generated per scenario,
4. report script consolidates S1 totals and monthly traces.

Reasoning:
1. separates assumptions from model code,
2. supports sensitivity analysis,
3. improves decision robustness under uncertainty.

Important practical note:
- if scenario totals are identical, review scenario sheet values and merge keys; equal outputs usually indicate weak differentiation in exogenous inputs.

## 8) Why This Pipeline Is Defensible in a PFE

## Scientific defensibility
1. explicit hypotheses and transformations,
2. chronological validation,
3. multiple complementary metrics,
4. reproducible artifacts.

## Engineering defensibility
1. modular scripts by responsibility,
2. robust fallbacks and checks,
3. deterministic file outputs,
4. maintainable documentation.

## Business defensibility
1. links model outputs to actionable KPIs,
2. adds scenario stress-testing,
3. adds rule-based market insight beyond pure forecasting.

## 9) Reproduction Steps (End-to-End)

Use from repository root:

```powershell
python step3_cleaning.py
python step2_5_enrich_data.py
python step4_eda.py
python step5_preparation.py
python verifier_pret.py
python step6_modeling.py
python add_macro_scenarios.py
python step6_modeling.py
python produce_artes_report.py
python step6_datamining.py --input data_prepared_final.csv
```

Expected key deliverables:
1. `step6_forecast_s1_2026.csv`
2. `step6_metrics_summary.csv`
3. `rapport_ARTES_s1_2026_by_scenario.csv`
4. `data/datamining_association_rules_top15.csv`
5. `data/datamining_association_rules_top15.md`

## 10) Known Limits and Planned Improvements

Current limits:
1. clustering granularity depends on available category column richness,
2. rules still reflect available feature engineering quality,
3. scenario quality is constrained by macro assumptions quality.

Recommended next upgrades:
1. add richer vehicle taxonomy (`CD_TYP_CONS`) to preparation outputs,
2. add walk-forward evaluation windows,
3. add probabilistic intervals for all forecast outputs,
4. add automated model/result version metadata.

## 11) Decision Log Summary

Core choices and concise justification:
1. File-based modular pipeline: auditability and simplicity.
2. Time-aware feature engineering: captures demand inertia and seasonality.
3. SARIMAX + Prophet + ML blend: complementary strengths.
4. Scenario simulation: decision support under macro uncertainty.
5. Data mining add-on: uncover hidden patterns, not only forecast point values.
6. Top-15 rules export: make outputs directly consumable in report writing.

This architecture intentionally balances academic rigor, practical constraints, and business interpretability.
