import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import unicodedata
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")


def safe_col(df, col_kwd):
    aliases = {
        'MARCHE_TYPE': ['Marché', 'MARCHE_TYPE', 'MARCH'],
    }
    for candidate in aliases.get(col_kwd, [col_kwd]):
        if candidate in df.columns:
            return candidate
    return next((c for c in df.columns if col_kwd in c), None)


def normalize_text(value):
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return text


def standardize_external_columns(df):
    out = df.copy()
    out.columns = [normalize_text(c).replace(' ', '_') for c in out.columns]
    return out


def to_numeric_loose(series):
    cleaned = (
        series.astype(str)
        .str.replace('\u00a0', '', regex=False)
        .str.replace(' ', '', regex=False)
        .str.replace(',', '.', regex=False)
    )
    cleaned = cleaned.replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
    return pd.to_numeric(cleaned, errors='coerce')


def build_external_features(ext_file, date_range):
    monthly_index = pd.DatetimeIndex(pd.to_datetime(date_range)).to_period('M').to_timestamp('M')
    monthly_index = pd.DatetimeIndex(sorted(monthly_index.unique()))
    df_external = pd.DataFrame({'Date': monthly_index})
    df_external['annee'] = df_external['Date'].dt.year
    df_external['mois'] = df_external['Date'].dt.month

    # 1) PIB trimestriel -> mensuel (interpolation lineaire sur index mensuel complet)
    df_pib = standardize_external_columns(pd.read_excel(ext_file, sheet_name='PIB_Croissance'))
    df_pib = df_pib.rename(columns={'pib_croissance_pct': 'pib_croissance_pct'})
    quarter_to_month = {1: 3, 2: 6, 3: 9, 4: 12}
    df_pib['annee'] = to_numeric_loose(df_pib['annee'])
    df_pib['trimestre'] = to_numeric_loose(df_pib['trimestre'])
    df_pib['pib_croissance_pct'] = to_numeric_loose(df_pib['pib_croissance_pct'])
    df_pib = df_pib.dropna(subset=['annee', 'trimestre', 'pib_croissance_pct'])
    df_pib['mois'] = df_pib['trimestre'].map(quarter_to_month)
    df_pib['Date'] = pd.to_datetime(
        dict(year=df_pib['annee'].astype(int), month=df_pib['mois'].astype(int), day=1)
    ) + pd.offsets.MonthEnd(0)

    pib_series = (
        df_pib.groupby('Date', as_index=True)['pib_croissance_pct']
        .mean()
        .sort_index()
    )
    pib_monthly = pd.Series(index=monthly_index, dtype=float)
    pib_monthly.loc[pib_series.index] = pib_series.values
    pib_monthly = pib_monthly.interpolate(method='linear', limit_area='inside')
    df_external = df_external.merge(
        pib_monthly.rename('pib_croissance_pct').reset_index().rename(columns={'index': 'Date'}),
        on='Date',
        how='left'
    )

    # 2) Taux de change, inflation, petrole (jointure annee/mois)
    df_change = standardize_external_columns(pd.read_excel(ext_file, sheet_name='Taux_Change'))
    df_change['annee'] = to_numeric_loose(df_change['annee'])
    df_change['mois'] = to_numeric_loose(df_change['mois'])
    for col in ['taux_eur_tnd', 'taux_usd_tnd']:
        df_change[col] = to_numeric_loose(df_change[col])
    df_change = df_change[['annee', 'mois', 'taux_eur_tnd', 'taux_usd_tnd']].drop_duplicates()

    df_infl = standardize_external_columns(pd.read_excel(ext_file, sheet_name='Inflation'))
    df_infl['annee'] = to_numeric_loose(df_infl['annee'])
    df_infl['mois'] = to_numeric_loose(df_infl['mois'])
    df_infl['inflation_pct'] = to_numeric_loose(df_infl['inflation_pct'])
    df_infl = df_infl[['annee', 'mois', 'inflation_pct']].drop_duplicates()

    df_oil = standardize_external_columns(pd.read_excel(ext_file, sheet_name='Prix_Petrole'))
    df_oil['annee'] = to_numeric_loose(df_oil['annee'])
    df_oil['mois'] = to_numeric_loose(df_oil['mois'])
    df_oil['brent_usd_barrel'] = to_numeric_loose(df_oil['brent_usd_barrel'])
    df_oil = df_oil[['annee', 'mois', 'brent_usd_barrel']].drop_duplicates()

    df_external = df_external.merge(df_change, on=['annee', 'mois'], how='left')
    df_external = df_external.merge(df_infl, on=['annee', 'mois'], how='left')
    df_external = df_external.merge(df_oil, on=['annee', 'mois'], how='left')

    # 3) Calendrier des evenements -> features mensuelles
    df_cal = standardize_external_columns(pd.read_excel(ext_file, sheet_name='Calendrier_Evenements'))
    df_cal['date'] = pd.to_datetime(df_cal['date'], errors='coerce')
    df_cal['duration_days'] = to_numeric_loose(df_cal.get('duration_days')).fillna(1)
    df_cal['duration_days'] = df_cal['duration_days'].clip(lower=1).astype(int)

    cal_monthly = pd.DataFrame({'Date': monthly_index})
    cal_monthly['ramadan_jours'] = 0
    cal_monthly['nb_jours_feries'] = 0
    cal_monthly['aid_mois'] = 0

    month_to_idx = {d: i for i, d in enumerate(cal_monthly['Date'])}
    for _, row in df_cal.iterrows():
        if pd.isna(row['date']):
            continue

        start_date = row['date'].normalize()
        end_date = start_date + pd.Timedelta(days=int(row['duration_days']) - 1)
        raw_text = f"{row.get('event_type', '')} {row.get('event_name', '')}"
        text = normalize_text(raw_text)

        is_ramadan = 'ramadan' in text
        is_aid = ('aid' in text) and (('fitr' in text) or ('adha' in text))

        is_holiday_raw = row.get('is_holiday', 0)
        if isinstance(is_holiday_raw, str):
            is_holiday = normalize_text(is_holiday_raw) in {'1', 'true', 'yes', 'oui'}
        else:
            is_holiday = bool(is_holiday_raw)

        for month_period in pd.period_range(start_date.to_period('M'), end_date.to_period('M'), freq='M'):
            month_start = month_period.start_time.normalize()
            month_end = month_period.end_time.normalize()
            overlap_start = max(start_date, month_start)
            overlap_end = min(end_date, month_end)
            overlap_days = (overlap_end - overlap_start).days + 1
            if overlap_days <= 0:
                continue

            month_key = month_period.to_timestamp('M')
            if month_key not in month_to_idx:
                continue
            i = month_to_idx[month_key]

            if is_ramadan:
                cal_monthly.at[i, 'ramadan_jours'] += overlap_days
            if is_holiday:
                cal_monthly.at[i, 'nb_jours_feries'] += overlap_days
            if is_aid:
                cal_monthly.at[i, 'aid_mois'] = 1

    cal_monthly['ramadan_mois'] = (cal_monthly['ramadan_jours'] > 0).astype(int)
    cal_monthly = cal_monthly[['Date', 'ramadan_mois', 'ramadan_jours', 'nb_jours_feries', 'aid_mois']]
    df_external = df_external.merge(cal_monthly, on='Date', how='left')

    for col in ['ramadan_mois', 'ramadan_jours', 'nb_jours_feries', 'aid_mois']:
        df_external[col] = df_external[col].fillna(0)

    df_external = df_external.drop(columns=['annee', 'mois'])
    return df_external


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

    # ── PART 2.5: External macro features (monthly) ─────────────────────────
    print("\nPART 2.5: External macro features integration...")
    external_file = os.path.join(project_root, 'data', 'donnees_externes_tunisie.xlsx')
    external_end = max(max_date, pd.Timestamp('2026-12-31'))
    external_range = pd.date_range(
        start=min_date.replace(day=1),
        end=external_end,
        freq='ME'
    )
    if not os.path.exists(external_file):
        print(f"  WARNING: {external_file} not found. External features left empty.")
        df_external = pd.DataFrame({'Date': external_range})
        for col in [
            'pib_croissance_pct', 'taux_eur_tnd', 'taux_usd_tnd',
            'inflation_pct', 'brent_usd_barrel',
            'ramadan_mois', 'ramadan_jours', 'nb_jours_feries', 'aid_mois'
        ]:
            df_external[col] = np.nan if col == 'pib_croissance_pct' else 0.0
    else:
        df_external = build_external_features(external_file, external_range)

    # Join external monthly series to the already aggregated monthly sales series
    df_monthly = df_monthly.merge(df_external, on='Date', how='left')
    print(f"  External columns added: {len(df_external.columns) - 1}")

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

    # Keep pre-imputation copies for honest availability reporting
    data_train_cov = data_train.copy()
    data_val_cov = data_val.copy()
    data_test_cov = data_test.copy()

    print(f"  Train : {len(data_train)} months ({data_train['Year'].min()}-{data_train['Year'].max()})")
    print(f"  Val   : {len(data_val)} months ({data_val['Year'].min() if len(data_val) else 'N/A'}-{data_val['Year'].max() if len(data_val) else 'N/A'})")
    print(f"  Test  : {len(data_test)} months ({data_test['Year'].min() if len(data_test) else 'N/A'}-{data_test['Year'].max() if len(data_test) else 'N/A'})")

    # Explicit feature sets
    FEATURES_FULL = [
        'Ventes_Lag1', 'Ventes_Lag3', 'Ventes_Lag6', 'Ventes_Lag12',
        'Ventes_MA3', 'Ventes_MA6', 'Ventes_MA12',
        'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6',
        'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12',
        'Is_Q1', 'Is_Q2', 'Is_Q3', 'Is_Q4',
        'ramadan_mois', 'ramadan_jours', 'nb_jours_feries', 'aid_mois',
        'pib_croissance_pct',
    ]
    FEATURES_VAL_TEST_ONLY = [
        'taux_eur_tnd', 'taux_usd_tnd',
        'inflation_pct',
        'brent_usd_barrel',
    ]

    # Train-only imputation to avoid leakage
    train_pib_mean = data_train['pib_croissance_pct'].mean(skipna=True)
    for split_df in [df_model, data_train, data_val, data_test]:
        split_df['pib_croissance_pct'] = split_df['pib_croissance_pct'].fillna(train_pib_mean)

    train_features_for_model = [c for c in FEATURES_FULL if c in data_train.columns]
    val_test_extra_features = [c for c in FEATURES_VAL_TEST_ONLY if c in df_model.columns]
    print(f"  Train feature count (FULL only): {len(train_features_for_model)}")
    print(f"  Val/Test extra feature count   : {len(val_test_extra_features)}")
    print("  Note: taux change / inflation / petrole are excluded from train features.")

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
    data_future  = pd.DataFrame({'Date': future_dates})
    data_future['Year'] = data_future['Date'].dt.year
    data_future['Month'] = data_future['Date'].dt.month
    data_future['Quarter'] = data_future['Date'].dt.quarter
    data_future['DayOfYear'] = data_future['Date'].dt.dayofyear
    for q in range(1, 5):
        data_future[f'Is_Q{q}'] = (data_future['Quarter'] == q).astype(int)
    for m in range(1, 13):
        data_future[f'Month_{m}'] = (data_future['Month'] == m).astype(int)

    data_future = data_future.merge(df_external, on='Date', how='left')
    data_future_cov = data_future.copy()
    data_future['pib_croissance_pct'] = data_future['pib_croissance_pct'].fillna(train_pib_mean)

    for col in df_model.columns:
        if col not in data_future.columns:
            data_future[col] = np.nan
    data_future = data_future[df_model.columns]

    # ── PART 10: Export ───────────────────────────────────────────────────────
    print("\nPART 10: Saving files...")
    output_files = [
        'data_external_features.csv',
        'data_prepared_final.csv',
        'data_train.csv',
        'data_validation_2024.csv',
        'data_test_2025.csv',
        'data_future_2026.csv',
    ]
    for fname, data in [
        ('data_external_features.csv', df_external),
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

    def pct_available(df_split, col):
        if len(df_split) == 0:
            return 0.0
        return 100 * df_split[col].notna().mean()

    train_pib_cov = pct_available(data_train_cov, 'pib_croissance_pct')
    train_ram_cov = pct_available(data_train_cov, 'ramadan_mois')
    train_fx_cov = pct_available(data_train_cov, 'taux_eur_tnd')
    train_inf_cov = pct_available(data_train_cov, 'inflation_pct')

    val_pib_cov = pct_available(data_val_cov, 'pib_croissance_pct')
    val_ram_cov = pct_available(data_val_cov, 'ramadan_mois')
    val_fx_cov = pct_available(data_val_cov, 'taux_eur_tnd')
    val_inf_cov = pct_available(data_val_cov, 'inflation_pct')

    test_pib_cov = pct_available(data_test_cov, 'pib_croissance_pct')
    test_ram_cov = pct_available(data_test_cov, 'ramadan_mois')
    test_fx_cov = pct_available(data_test_cov, 'taux_eur_tnd')
    test_inf_cov = pct_available(data_test_cov, 'inflation_pct')

    future_pib_cov = pct_available(data_future_cov, 'pib_croissance_pct')
    future_ram_cov = pct_available(data_future_cov, 'ramadan_mois')
    future_fx_cov = pct_available(data_future_cov, 'taux_eur_tnd')
    future_inf_cov = pct_available(data_future_cov, 'inflation_pct')

    print("\nCouverture des features externes par split :")
    print(f"  Train (2019-2022)  : PIB={train_pib_cov:.1f}%, ramadan={train_ram_cov:.1f}%, taux_change={train_fx_cov:.1f}%, inflation={train_inf_cov:.1f}%")
    print(f"  Val   (2024)       : PIB={val_pib_cov:.1f}%, ramadan={val_ram_cov:.1f}%, taux_change={val_fx_cov:.1f}%, inflation={val_inf_cov:.1f}%")
    print(f"  Test  (2025)       : PIB={test_pib_cov:.1f}%, ramadan={test_ram_cov:.1f}%, taux_change={test_fx_cov:.1f}%, inflation={test_inf_cov:.1f}%")
    print(f"  Futur (2026)       : PIB={future_pib_cov:.1f}%, ramadan={future_ram_cov:.1f}%, taux_change={future_fx_cov:.1f}%, inflation={future_inf_cov:.1f}%")

    missing_years = [y for y in [2020, 2023]
                     if y not in df_model['Year'].unique()]
    if missing_years:
        print(f"\n  WARNING: Years {missing_years} missing from data.")
        print(f"  -> Train set covers non-contiguous years.")
        print(f"  -> Request missing data from M. Sami (ARTES).")

    print("\nSTEP 5 COMPLETE -> Ready for Step 6 (Modeling)")


if __name__ == '__main__':
    main()