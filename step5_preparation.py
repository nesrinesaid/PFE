import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

sns.set_theme(style="whitegrid")


def add_ramadan_flag(df, date_col='Date'):
    ramadan_ranges = [('2019-05-06', '2019-06-04'), ('2020-04-24', '2020-05-23'),
                      ('2021-04-13', '2021-05-12'), ('2022-04-02', '2022-05-01'),
                      ('2023-03-23', '2023-04-21'), ('2024-03-11', '2024-04-09'),
                      ('2025-03-01', '2025-03-30'), ('2026-02-18', '2026-03-19')]
    dates = pd.to_datetime(df[date_col])
    flag = pd.Series(False, index=df.index)
    for start, end in ramadan_ranges:
        flag |= dates.between(pd.Timestamp(start), pd.Timestamp(end))
    return flag.astype(int)


def main():
    print("STEP 5 - ML DATA PREPARATION\n")

    project_root = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(project_root, 'data_cleaned_enriched.csv')
    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found. Run step2_5_enrich_data.py first.")
        return

    print("Loading data...")
    df_raw = pd.read_csv(input_file, parse_dates=['DATV'])
    df_raw = df_raw.rename(columns={'DATV': 'Date'})
    print(f"  Shape before filters: {df_raw.shape}")

    df_raw['IM_RI'] = pd.to_numeric(df_raw['IM_RI'], errors='coerce')
    df_raw = df_raw[df_raw['IM_RI'].eq(10)].copy()
    df_raw['MarketType'] = df_raw['Marché'].astype(str).str.upper().str.extract(r'(VP|VU)', expand=False)
    df_raw = df_raw[df_raw['MarketType'].isin(['VP', 'VU'])].copy()
    if df_raw.empty:
        print("ERROR: No rows left after IM_RI=10 and VP/VU filters.")
        return

    df_raw['Date'] = pd.to_datetime(df_raw['Date']).dt.normalize()
    print(f"  Shape after filters : {df_raw.shape}")

    # PART 1: Daily aggregation by market type (separate VP and VU series)
    print("\nPART 1: Daily aggregation by market type...")
    df_daily = df_raw.groupby(['Date', 'MarketType']).size().reset_index(name='Ventes')
    df_daily = df_daily.sort_values(['MarketType', 'Date']).reset_index(drop=True)
    print(f"  Daily records before gap fill: {len(df_daily)}")

    # PART 2: Fill missing days per market type
    print("\nPART 2: Filling missing days...")
    min_date = df_daily['Date'].min()
    max_date = df_daily['Date'].max()
    complete_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    full_index = pd.MultiIndex.from_product(
        [complete_dates, ['VP', 'VU']],
        names=['Date', 'MarketType']
    )
    df_complete = full_index.to_frame(index=False)
    df_daily = df_complete.merge(df_daily, on=['Date', 'MarketType'], how='left')

    missing_before = df_daily['Ventes'].isna().sum()
    if missing_before > 0:
        print(f"  {missing_before} day-market rows were missing, filled with 0")

    df_daily['Ventes'] = df_daily['Ventes'].fillna(0.0).astype(float)
    print(f"  Missing values remaining: {df_daily.isnull().sum().sum()}")
    print(f"  Total day-market rows after gap fill: {len(df_daily)}")

    df_daily['Year'] = df_daily['Date'].dt.year
    df_daily['Month'] = df_daily['Date'].dt.month
    df_daily['IsRamadan'] = add_ramadan_flag(df_daily, 'Date')
    print("  Ramadan flag added")

    # PART 3: Moving averages and outlier handling by market type
    print("\nPART 3: Moving averages & outlier handling...")
    df_daily = df_daily.sort_values(['MarketType', 'Date']).reset_index(drop=True)
    df_daily['Ventes_MA7'] = df_daily.groupby('MarketType')['Ventes'].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )
    df_daily['Ventes_MA30'] = df_daily.groupby('MarketType')['Ventes'].transform(
        lambda s: s.rolling(window=30, min_periods=1).mean()
    )
    df_daily['Ventes_MA90'] = df_daily.groupby('MarketType')['Ventes'].transform(
        lambda s: s.rolling(window=90, min_periods=1).mean()
    )

    n_outliers = 0
    for market in ['VP', 'VU']:
        market_mask = df_daily['MarketType'] == market
        s = df_daily.loc[market_mask, 'Ventes']
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        outlier_mask = market_mask & (
            (df_daily['Ventes'] < q1 - 1.5 * iqr) |
            (df_daily['Ventes'] > q3 + 1.5 * iqr)
        )
        n_outliers += int(outlier_mask.sum())
        df_daily.loc[outlier_mask, 'Ventes'] = df_daily.loc[outlier_mask, 'Ventes_MA7']
    if n_outliers > 0:
        print(f"  {n_outliers} outlier day-market rows replaced with MA7")

    # PART 4: Temporal features
    print("\nPART 4: Temporal features...")
    df_daily['Year'] = df_daily['Date'].dt.year
    df_daily['Month'] = df_daily['Date'].dt.month
    df_daily['Quarter'] = df_daily['Date'].dt.quarter
    df_daily['DayOfYear'] = df_daily['Date'].dt.dayofyear
    df_daily['DayOfWeek'] = df_daily['Date'].dt.dayofweek
    df_daily['IsWeekend'] = (df_daily['DayOfWeek'] >= 5).astype(int)
    df_daily['Is_VU'] = (df_daily['MarketType'] == 'VU').astype(int)
    df_daily['IsRamadan'] = add_ramadan_flag(df_daily, 'Date')

    for q in range(1, 5):
        df_daily[f'Is_Q{q}'] = (df_daily['Quarter'] == q).astype(int)
    for m in range(1, 13):
        df_daily[f'Month_{m}'] = (df_daily['Month'] == m).astype(int)
    for d in range(7):
        df_daily[f'DOW_{d}'] = (df_daily['DayOfWeek'] == d).astype(int)

    # PART 5: Lag features by market type
    print("\nPART 5: Lag features...")
    lag_cols = ['Ventes_Lag1', 'Ventes_Lag7', 'Ventes_Lag30', 'Ventes_Lag90']
    df_daily['Ventes_Lag1'] = df_daily.groupby('MarketType')['Ventes'].shift(1)
    df_daily['Ventes_Lag7'] = df_daily.groupby('MarketType')['Ventes'].shift(7)
    df_daily['Ventes_Lag30'] = df_daily.groupby('MarketType')['Ventes'].shift(30)
    df_daily['Ventes_Lag90'] = df_daily.groupby('MarketType')['Ventes'].shift(90)
    df_daily[lag_cols] = df_daily.groupby('MarketType')[lag_cols].ffill()

    # PART 6: Target variable (next day, per market type)
    df_daily['Ventes_Next_Day'] = df_daily.groupby('MarketType')['Ventes'].shift(-1)
    df_model = df_daily[df_daily['Ventes_Next_Day'].notna()].copy().reset_index(drop=True)

    # ── PART 7: Train / Validation / Test split (BEFORE scaling) ─────────────
    print("\nPART 7: Train/Val/Test split...")
    # Split on year boundaries — strictly chronological, no shuffling
    train_mask = (df_model['Year'] >= 2019) & (df_model['Year'] <= 2023)
    val_mask   =  df_model['Year'] == 2024
    test_mask  =  df_model['Year'] == 2025

    data_train = df_model[train_mask].copy()
    data_val   = df_model[val_mask].copy()
    data_test  = df_model[test_mask].copy()

    # Keep pre-imputation copies for honest availability reporting
    data_train_cov = data_train.copy()
    data_val_cov = data_val.copy()
    data_test_cov = data_test.copy()

    print(f"  Train : {len(data_train)} rows ({data_train['Year'].min()}-{data_train['Year'].max()})")
    print(f"  Val   : {len(data_val)} rows ({data_val['Year'].min() if len(data_val) else 'N/A'}-{data_val['Year'].max() if len(data_val) else 'N/A'})")
    print(f"  Test  : {len(data_test)} rows ({data_test['Year'].min() if len(data_test) else 'N/A'}-{data_test['Year'].max() if len(data_test) else 'N/A'})")

    # Daily model features
    model_features = [
        'Is_VU', 'IsRamadan',
        'Ventes_Lag1', 'Ventes_Lag7', 'Ventes_Lag30', 'Ventes_Lag90',
        'Ventes_MA7', 'Ventes_MA30', 'Ventes_MA90',
        'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6',
        'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12',
        'DOW_0', 'DOW_1', 'DOW_2', 'DOW_3', 'DOW_4', 'DOW_5', 'DOW_6',
        'Is_Q1', 'Is_Q2', 'Is_Q3', 'Is_Q4'
    ]
    print(f"  Train feature count (daily model inputs): {len(model_features)}")

    # ── PART 8: Normalisation — fit on TRAIN only, apply to all ──────────────
    # CRITICAL: fitting the scaler on all data before splitting leaks future
    # statistics (mean, std of 2024-2025) into the training set, inflating
    # model performance. Scaler must be fit on training data only.
    print("\nPART 8: Normalisation (fit on train only)...")
    num_to_scale = ['Ventes'] + lag_cols + ['Ventes_MA7', 'Ventes_MA30', 'Ventes_MA90']

    scaler = StandardScaler()
    scaler.fit(data_train[num_to_scale])  # fit on train ONLY

    # Apply to all splits
    for split_df in [df_model, data_train, data_val, data_test]:
        scaled = scaler.transform(split_df[num_to_scale])
        for i, col in enumerate(num_to_scale):
            split_df[f'{col}_norm'] = scaled[:, i]

    # ── PART 9: Future placeholder (daily 2026 for both VP and VU) ───────────
    future_dates = pd.date_range(start='2026-01-01', end='2026-12-31', freq='D')
    data_future = pd.MultiIndex.from_product(
        [future_dates, ['VP', 'VU']], names=['Date', 'MarketType']
    ).to_frame(index=False)
    data_future['Year'] = data_future['Date'].dt.year
    data_future['Month'] = data_future['Date'].dt.month
    data_future['Quarter'] = data_future['Date'].dt.quarter
    data_future['DayOfYear'] = data_future['Date'].dt.dayofyear
    data_future['DayOfWeek'] = data_future['Date'].dt.dayofweek
    data_future['IsWeekend'] = (data_future['DayOfWeek'] >= 5).astype(int)
    data_future['Is_VU'] = (data_future['MarketType'] == 'VU').astype(int)
    data_future['IsRamadan'] = add_ramadan_flag(data_future, 'Date')
    for q in range(1, 5):
        data_future[f'Is_Q{q}'] = (data_future['Quarter'] == q).astype(int)
    for m in range(1, 13):
        data_future[f'Month_{m}'] = (data_future['Month'] == m).astype(int)
    for d in range(7):
        data_future[f'DOW_{d}'] = (data_future['DayOfWeek'] == d).astype(int)

    for col in df_model.columns:
        if col not in data_future.columns:
            data_future[col] = np.nan
    data_future = data_future[df_model.columns]

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
    vp_series = df_model[df_model['MarketType'] == 'VP']
    vu_series = df_model[df_model['MarketType'] == 'VU']
    axes[0, 0].plot(vp_series['Date'], vp_series['Ventes'],
                    label='VP Ventes', alpha=0.75, linewidth=1.8, color='steelblue')
    axes[0, 0].plot(vp_series['Date'], vp_series['Ventes_MA30'],
                    label='VP MA30', linestyle='--', color='orange')
    axes[0, 0].plot(vu_series['Date'], vu_series['Ventes'],
                    label='VU Ventes', alpha=0.7, linewidth=1.6, color='darkgreen')
    axes[0, 0].plot(vu_series['Date'], vu_series['Ventes_MA30'],
                    label='VU MA30', linestyle='--', color='red')
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
    axes[0, 0].set_title('Daily sales by market type (with splits)')

    # Panel 2: Annual distribution
    sns.boxplot(ax=axes[0, 1], data=df_model, x='Year', y='Ventes', hue='MarketType', palette='Set2')
    axes[0, 1].set_title('Yearly distribution by market type')

    # Panel 3: Monthly seasonality
    sns.barplot(ax=axes[1, 0], data=df_model, x='DayOfWeek', y='Ventes',
                hue='MarketType', errorbar=None, palette='mako')
    axes[1, 0].set_title('Day-of-week seasonality by market type')

    # Panel 4: Lag1 autocorrelation
    sns.scatterplot(ax=axes[1, 1], x=df_model['Ventes_Lag1'],
                    y=df_model['Ventes'], color='purple', alpha=0.7)
    axes[1, 1].set_title('Lag-1 autocorrelation (all markets)')

    plt.suptitle('Step 5 — ML Preparation Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, '11_ML_Preparation_Summary.png'), dpi=300)
    plt.close()
    print("  Saved: 11_ML_Preparation_Summary.png")

    # ── PART 12: Final report ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ML PREPARATION SUMMARY")
    print("=" * 60)
    print(f"  Total rows       : {len(df_model)}")
    print(f"  Total features   : {len(df_model.columns)}")
    print(f"  Train rows       : {len(data_train)}")
    print(f"  Val rows         : {len(data_val)}")
    print(f"  Test rows        : {len(data_test)}")
    print(f"  Markets          : {sorted(df_model['MarketType'].dropna().unique().tolist())}")
    print(f"  Ventes mean      : {df_model['Ventes'].mean():.1f}")
    print(f"  Ventes std       : {df_model['Ventes'].std():.1f}")
    print(f"  Scaler fit on    : train only (no leakage)")
    print(f"  Lag fill method  : ffill only (no leakage)")

    def pct_available(df_split, col):
        if len(df_split) == 0:
            return 0.0
        return 100 * df_split[col].notna().mean()

    print("\nRamadan flag coverage:")
    print(f"  Train (2019-2023)  : {pct_available(data_train, 'IsRamadan'):.1f}%")
    print(f"  Val   (2024)       : {pct_available(data_val, 'IsRamadan'):.1f}%")
    print(f"  Test  (2025)       : {pct_available(data_test, 'IsRamadan'):.1f}%")
    print(f"  Futur (2026)       : {pct_available(data_future, 'IsRamadan'):.1f}%")

    print("\nSTEP 5 COMPLETE -> Ready for Step 6 (Modeling)")


if __name__ == '__main__':
    main()