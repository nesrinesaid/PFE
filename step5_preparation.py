import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")


def safe_col(df, col_kwd):
    return next((c for c in df.columns if col_kwd in c), None)


def main():
    print("STEP 5 - ML DATA PREPARATION\n")

    project_root = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(project_root, 'data_cleaned_enriched.csv')
    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found. Run step2_5_enrich_data.py first.")
        return

    print("Loading data...")
    df_raw = pd.read_csv(input_file, parse_dates=['DATV'])
    df_raw = df_raw.rename(columns={'DATV': 'DAT_V'})
    df_raw['YEAR_MONTH_dt'] = pd.to_datetime(df_raw['YEAR_MONTH'])
    print(f"  Shape: {df_raw.shape}")

    # ── PART 1: Monthly aggregation with dominant categorical values ───────────
    print("\nPART 1: Monthly aggregation...")
    df_monthly = df_raw.groupby('YEAR_MONTH_dt').size().reset_index(name='Ventes')

    # Dominant mode per month for each categorical dimension
    # These become context features for the models (what was the dominant market segment
    # that month, which continent, which distributor group, etc.)
    cols_to_mode = ['MARQUE', 'GENRE', 'USAGE', 'SEGMENT', 'SOUS_SEGMENT',
                    'MARCHE_TYPE', 'DISTRIBUTEUR', 'CONTINENT', 'GROUPE']
    for col in cols_to_mode:
        matched = safe_col(df_raw, col)
        if matched:
            mode_df = (
                df_raw.groupby('YEAR_MONTH_dt')[matched]
                .apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
                .reset_index(name=f'{col}_dominante')
            )
            df_monthly = df_monthly.merge(mode_df, on='YEAR_MONTH_dt', how='left')

    df_monthly = df_monthly.rename(columns={'YEAR_MONTH_dt': 'Date'})
    df_monthly['Date'] = df_monthly['Date'] + pd.offsets.MonthEnd(0)
    df_monthly = df_monthly.sort_values('Date').reset_index(drop=True)
    print(f"  Monthly records before gap fill: {len(df_monthly)}")

    # ── PART 2: Fill missing months (gaps in data) ────────────────────────────
    print("\nPART 2: Filling missing months...")
    min_date = df_monthly['Date'].min()
    max_date = df_monthly['Date'].max()
    complete_range = pd.date_range(start=min_date.replace(day=1), end=max_date, freq='ME')
    df_complete = pd.DataFrame({'Date': complete_range})
    df_monthly = df_complete.merge(df_monthly, on='Date', how='left')

    missing_before = df_monthly['Ventes'].isna().sum()
    if missing_before > 0:
        print(f"  {missing_before} months with missing Ventes — interpolating...")

    # Numeric interpolation
    df_monthly['Ventes'] = df_monthly['Ventes'].interpolate(method='linear')
    df_monthly['Ventes'] = df_monthly['Ventes'].fillna(df_monthly['Ventes'].mean())

    # Categorical forward/backward fill
    categorical_cols = [c for c in df_monthly.columns if '_dominante' in c]
    for col in categorical_cols:
        df_monthly[col] = df_monthly[col].ffill().bfill()

    # Keep sales numeric as float so MA/outlier replacement can assign decimal values
    df_monthly['Ventes'] = df_monthly['Ventes'].astype(float)

    print(f"  Missing values remaining: {df_monthly.isnull().sum().sum()}")
    print(f"  Total months after gap fill: {len(df_monthly)}")

    # ── PART 3: Moving averages ───────────────────────────────────────────────
    print("\nPART 3: Moving averages & outlier handling...")
    df_monthly['Ventes_MA3']  = df_monthly['Ventes'].rolling(window=3,  min_periods=1).mean()
    df_monthly['Ventes_MA6']  = df_monthly['Ventes'].rolling(window=6,  min_periods=1).mean()
    df_monthly['Ventes_MA12'] = df_monthly['Ventes'].rolling(window=12, min_periods=1).mean()

    # Outlier replacement with MA3 (IQR method)
    Q1  = df_monthly['Ventes'].quantile(0.25)
    Q3  = df_monthly['Ventes'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (
        (df_monthly['Ventes'] < Q1 - 1.5 * IQR) |
        (df_monthly['Ventes'] > Q3 + 1.5 * IQR)
    )
    n_outliers = outlier_mask.sum()
    if n_outliers > 0:
        print(f"  {n_outliers} outlier months replaced with MA3")
        df_monthly.loc[outlier_mask, 'Ventes'] = df_monthly.loc[outlier_mask, 'Ventes_MA3']

    # ── PART 4: Temporal features ─────────────────────────────────────────────
    print("\nPART 4: Temporal features...")
    df_monthly['Year']      = df_monthly['Date'].dt.year
    df_monthly['Month']     = df_monthly['Date'].dt.month
    df_monthly['Quarter']   = df_monthly['Date'].dt.quarter
    df_monthly['DayOfYear'] = df_monthly['Date'].dt.dayofyear

    # One-hot: quarters and months
    for q in range(1, 5):
        df_monthly[f'Is_Q{q}'] = (df_monthly['Quarter'] == q).astype(int)
    for m in range(1, 13):
        df_monthly[f'Month_{m}'] = (df_monthly['Month'] == m).astype(int)

    # ── PART 5: Lag features ──────────────────────────────────────────────────
    print("\nPART 5: Lag features...")
    # IMPORTANT: only ffill (not bfill) to avoid leaking future values into early lags
    lag_cols = ['Ventes_Lag1', 'Ventes_Lag3', 'Ventes_Lag6', 'Ventes_Lag12']
    df_monthly['Ventes_Lag1']  = df_monthly['Ventes'].shift(1)
    df_monthly['Ventes_Lag3']  = df_monthly['Ventes'].shift(3)
    df_monthly['Ventes_Lag6']  = df_monthly['Ventes'].shift(6)
    df_monthly['Ventes_Lag12'] = df_monthly['Ventes'].shift(12)
    # Forward fill only — bfill would leak future values backwards
    df_monthly[lag_cols] = df_monthly[lag_cols].ffill()

    # ── PART 6: Target variable ───────────────────────────────────────────────
    df_monthly['Ventes_Next_Month'] = df_monthly['Ventes'].shift(-1)
    # Drop last row (target is NaN — no future month available)
    df_model = df_monthly.iloc[:-1].copy().reset_index(drop=True)

    # ── PART 7: Train / Validation / Test split (BEFORE scaling) ─────────────
    print("\nPART 7: Train/Val/Test split...")
    # Split on year boundaries — strictly chronological, no shuffling
    train_mask = (df_model['Year'] >= 2019) & (df_model['Year'] <= 2022)
    val_mask   =  df_model['Year'] == 2024
    test_mask  =  df_model['Year'] == 2025

    data_train = df_model[train_mask].copy()
    data_val   = df_model[val_mask].copy()
    data_test  = df_model[test_mask].copy()

    print(f"  Train : {len(data_train)} months ({data_train['Year'].min()}-{data_train['Year'].max()})")
    print(f"  Val   : {len(data_val)} months ({data_val['Year'].min() if len(data_val) else 'N/A'}-{data_val['Year'].max() if len(data_val) else 'N/A'})")
    print(f"  Test  : {len(data_test)} months ({data_test['Year'].min() if len(data_test) else 'N/A'}-{data_test['Year'].max() if len(data_test) else 'N/A'})")

    # ── PART 8: Normalisation — fit on TRAIN only, apply to all ──────────────
    # CRITICAL: fitting the scaler on all data before splitting leaks future
    # statistics (mean, std of 2024-2025) into the training set, inflating
    # model performance. Scaler must be fit on training data only.
    print("\nPART 8: Normalisation (fit on train only)...")
    num_to_scale = ['Ventes'] + lag_cols + ['Ventes_MA3', 'Ventes_MA6', 'Ventes_MA12']

    scaler = StandardScaler()
    scaler.fit(data_train[num_to_scale])  # fit on train ONLY

    # Apply to all splits
    for split_df in [df_model, data_train, data_val, data_test]:
        scaled = scaler.transform(split_df[num_to_scale])
        for i, col in enumerate(num_to_scale):
            split_df[f'{col}_norm'] = scaled[:, i]

    # ── PART 9: Future placeholder (Jan-Jun 2026) ─────────────────────────────
    future_dates = pd.date_range(start='2026-01-31', periods=12, freq='ME')
    data_future  = pd.DataFrame({'Date': future_dates, 'Year': 2026})
    for col in df_model.columns:
        if col not in ['Date', 'Year']:
            data_future[col] = np.nan

    # ── PART 10: Export ───────────────────────────────────────────────────────
    print("\nPART 10: Saving files...")
    output_files = [
        'data_prepared_final.csv',
        'data_train.csv',
        'data_validation_2024.csv',
        'data_test_2025.csv',
        'data_future_2026.csv',
    ]
    for fname, data in [
        ('data_prepared_final.csv', df_model),
        ('data_train.csv', data_train),
        ('data_validation_2024.csv', data_val),
        ('data_test_2025.csv', data_test),
        ('data_future_2026.csv', data_future),
    ]:
        out_path = os.path.join(project_root, fname)
        data.to_csv(out_path, index=False)

    for f in output_files:
        size = os.path.getsize(os.path.join(project_root, f)) / 1024
        print(f"  {f}: {size:.1f} KB")

    # ── PART 11: Summary visualisation ───────────────────────────────────────
    print("\nPART 11: Summary visualisation...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Sales + MAs
    axes[0, 0].plot(df_model['Date'], df_model['Ventes'],
                    label='Ventes', alpha=0.7, linewidth=2, color='steelblue')
    axes[0, 0].plot(df_model['Date'], df_model['Ventes_MA3'],
                    label='MA3', linestyle='--', color='orange')
    axes[0, 0].plot(df_model['Date'], df_model['Ventes_MA12'],
                    label='MA12', linestyle='--', color='red')
    # Shade train/val/test regions
    axes[0, 0].axvspan(data_train['Date'].min(), data_train['Date'].max(),
                       alpha=0.08, color='green', label='Train')
    if len(data_val) > 0:
        axes[0, 0].axvspan(data_val['Date'].min(), data_val['Date'].max(),
                           alpha=0.08, color='orange', label='Val')
    if len(data_test) > 0:
        axes[0, 0].axvspan(data_test['Date'].min(), data_test['Date'].max(),
                           alpha=0.08, color='red', label='Test')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title('Ventes & Moving Averages (with splits)')

    # Panel 2: Annual distribution
    sns.boxplot(ax=axes[0, 1], data=df_model, x='Year', y='Ventes', palette='Set3')
    axes[0, 1].set_title('Distribution annuelle des ventes')

    # Panel 3: Monthly seasonality
    sns.barplot(ax=axes[1, 0], data=df_model, x='Month', y='Ventes',
                errorbar=None, palette='mako')
    axes[1, 0].set_title('Saisonnalité mensuelle')

    # Panel 4: Lag1 autocorrelation
    sns.scatterplot(ax=axes[1, 1], x=df_model['Ventes_Lag1'],
                    y=df_model['Ventes'], color='purple', alpha=0.7)
    axes[1, 1].set_title('Autocorrélation Lag 1')

    plt.suptitle('Step 5 — ML Preparation Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, '11_ML_Preparation_Summary.png'), dpi=300)
    plt.close()
    print("  Saved: 11_ML_Preparation_Summary.png")

    # ── PART 12: Final report ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ML PREPARATION SUMMARY")
    print("=" * 60)
    print(f"  Total months     : {len(df_model)}")
    print(f"  Total features   : {len(df_model.columns)}")
    print(f"  Train months     : {len(data_train)}")
    print(f"  Val months       : {len(data_val)}")
    print(f"  Test months      : {len(data_test)}")
    print(f"  Ventes mean      : {df_model['Ventes'].mean():.1f}")
    print(f"  Ventes std       : {df_model['Ventes'].std():.1f}")
    print(f"  Scaler fit on    : train only (no leakage)")
    print(f"  Lag fill method  : ffill only (no leakage)")

    missing_years = [y for y in [2020, 2023]
                     if y not in df_model['Year'].unique()]
    if missing_years:
        print(f"\n  WARNING: Years {missing_years} missing from data.")
        print(f"  -> Train set covers non-contiguous years.")
        print(f"  -> Request missing data from M. Sami (ARTES).")

    print("\nSTEP 5 COMPLETE -> Ready for Step 6 (Modeling)")


if __name__ == '__main__':
    main()