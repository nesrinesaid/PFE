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
    print("🚀 DÉMARRAGE: ÉTAPE 5️⃣ - REAL DATA PREPARATION (POUR ML AVEC ENRICHISSEMENT)\n")
    
    input_file = 'data_cleaned_enriched.csv'
    if not os.path.exists(input_file):
        print(f"❌ {input_file} introuvable.")
        return
        
    print("⏳ Chargement des données...")
    df_raw = pd.read_csv(input_file, parse_dates=['DAT_V'])
    df_raw['YEAR_MONTH_dt'] = pd.to_datetime(df_raw['YEAR_MONTH'])

    print("⏳ PART 1: Agrégation au niveau mensuel avec variables dominantes (Market Dynamics)...")
    df_monthly = df_raw.groupby('YEAR_MONTH_dt').size().reset_index(name='Ventes')
    
    # We dynamically attach the mode of our massive categorical enrichment sheets to identify the trend of the month
    cols_to_mode = ['MARQUE', 'GENRE', 'USAGE', 'SEGMENT', 'DISTRIBUTEUR', 'CONTINENT']
    for basic_col in cols_to_mode:
        matched = safe_col(df_raw, basic_col)
        if matched:
            mode_df = df_raw.groupby('YEAR_MONTH_dt')[matched].apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan).reset_index(name=f'{basic_col}_dominante')
            df_monthly = df_monthly.merge(mode_df, on='YEAR_MONTH_dt', how='left')

    df_monthly = df_monthly.rename(columns={'YEAR_MONTH_dt': 'Date'})
    df_monthly['Date'] = df_monthly['Date'] + pd.offsets.MonthEnd(0)
    df_monthly = df_monthly.sort_values('Date').reset_index(drop=True)
    
    print("⏳ PART 5: Gestion et interpolation des zones vides...")
    min_date = df_monthly['Date'].min()
    max_date = df_monthly['Date'].max()
    complete_range = pd.date_range(start=min_date.replace(day=1), end=max_date, freq='ME')
    df_complete = pd.DataFrame({'Date': complete_range})
    df_monthly = df_complete.merge(df_monthly, on='Date', how='left')
    
    na_before = df_monthly.isnull().sum()[df_monthly.isnull().sum() > 0]
    if not na_before.empty: print(f"Valeurs manquantes AVANT interpolation:\n{na_before}")
    
    # Mathematical Imputations
    df_monthly['Ventes'] = df_monthly['Ventes'].interpolate(method='linear')
    df_monthly['Ventes'] = df_monthly['Ventes'].fillna(df_monthly['Ventes'].mean())
    
    # Categorical string imputations
    categorical_cols = [c for c in df_monthly.columns if '_dominante' in c]
    for col in categorical_cols:
        df_monthly[col] = df_monthly[col].ffill().bfill()
        
    print(f"✅ Remplissage terminé. Valeurs manquantes actuelles: {df_monthly.isnull().sum().sum()}")

    print("⏳ PART 4 & 6: Moyennes mobiles (MAs) et Filtrage des valeurs aberrantes (Outliers)...")
    df_monthly['Ventes_MA3'] = df_monthly['Ventes'].rolling(window=3, min_periods=1).mean()
    df_monthly['Ventes_MA6'] = df_monthly['Ventes'].rolling(window=6, min_periods=1).mean()
    df_monthly['Ventes_MA12'] = df_monthly['Ventes'].rolling(window=12, min_periods=1).mean()

    Q1 = df_monthly['Ventes'].quantile(0.25)
    Q3 = df_monthly['Ventes'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_idx = df_monthly[(df_monthly['Ventes'] < Q1 - 1.5*IQR) | (df_monthly['Ventes'] > Q3 + 1.5*IQR)].index
    if len(outlier_idx) > 0:
        print(f"⚠️ {len(outlier_idx)} outliers détectés. Remplacement restrictif par la MA3.")
        df_monthly.loc[outlier_idx, 'Ventes'] = df_monthly.loc[outlier_idx, 'Ventes_MA3']

    print("⏳ PART 2: Génération des features temporelles standardisées...")
    df_monthly['Year'] = df_monthly['Date'].dt.year
    df_monthly['Month'] = df_monthly['Date'].dt.month
    df_monthly['Quarter'] = df_monthly['Date'].dt.quarter
    df_monthly['DayOfWeek'] = df_monthly['Date'].dt.dayofweek
    df_monthly['DayOfYear'] = df_monthly['Date'].dt.dayofyear
    
    for q in range(1, 5): df_monthly[f'Is_Q{q}'] = (df_monthly['Quarter'] == q).astype(int)
    for m in range(1, 13): df_monthly[f'Month_{m}'] = (df_monthly['Month'] == m).astype(int)

    print("⏳ PART 3: Génération des retards opérationnels (Lags)...")
    lag_cols = ['Ventes_Lag1', 'Ventes_Lag3', 'Ventes_Lag6', 'Ventes_Lag12']
    df_monthly['Ventes_Lag1'] = df_monthly['Ventes'].shift(1)
    df_monthly['Ventes_Lag3'] = df_monthly['Ventes'].shift(3)
    df_monthly['Ventes_Lag6'] = df_monthly['Ventes'].shift(6)
    df_monthly['Ventes_Lag12'] = df_monthly['Ventes'].shift(12)
    df_monthly[lag_cols] = df_monthly[lag_cols].bfill().ffill()

    print("⏳ PART 8: Normalisation des caractéristiques Numériques...")
    scaler = StandardScaler()
    num_to_scale = ['Ventes'] + lag_cols
    df_scaled = scaler.fit_transform(df_monthly[num_to_scale])
    for i, col in enumerate(num_to_scale):
        df_monthly[f'{col}_norm'] = df_scaled[:, i]

    # Target
    df_monthly['Ventes_Next_Month'] = df_monthly['Ventes'].shift(-1)
    df_model = df_monthly.iloc[:-1].copy()

    print("⏳ PART 10: Répartition chronologique de l'apprentissage (Splits)...")
    data_train = df_model[(df_model['Year'] >= 2019) & (df_model['Year'] <= 2023)]
    data_val = df_model[df_model['Year'] == 2024]
    data_test = df_model[df_model['Year'] == 2025]
    
    future_dates = pd.date_range(start='2026-01-31', periods=12, freq='ME')
    data_future = pd.DataFrame({'Date': future_dates, 'Year': 2026})
    for col in df_model.columns:
        if col not in ['Date', 'Year']: data_future[col] = np.nan

    print("⏳ PART 11: Exportation des vecteurs ML...")
    df_model.to_csv('data_prepared_final.csv', index=False)
    data_train.to_csv('data_train_2019_2023.csv', index=False)
    data_val.to_csv('data_validation_2024.csv', index=False)
    data_test.to_csv('data_test_2025.csv', index=False)
    data_future.to_csv('data_future_2026.csv', index=False)

    print("⏳ PART 12: Synthèse Graphique Finale...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes[0, 0].plot(df_model['Date'], df_model['Ventes'], label='Ventes Filtrées', alpha=0.6, linewidth=2)
    axes[0, 0].plot(df_model['Date'], df_model['Ventes_MA3'], label='MA3', linestyle='--')
    axes[0, 0].plot(df_model['Date'], df_model['Ventes_MA12'], label='MA12', linestyle='--')
    axes[0, 0].legend()
    axes[0, 0].set_title('Superposition des Ventes & MAs')
    
    sns.boxplot(ax=axes[0, 1], data=df_model, x='Year', y='Ventes', palette='Set3')
    axes[0, 1].set_title('Distribution annuelle de la concentration des ventes')
    
    sns.barplot(ax=axes[1, 0], data=df_model, x='Month', y='Ventes', ci=None, palette='mako')
    axes[1, 0].set_title('Vue globale Saisonnalité Mensuelle')
    
    sns.scatterplot(ax=axes[1, 1], x=df_model['Ventes_Lag1'], y=df_model['Ventes'], color='purple')
    axes[1, 1].set_title('Autocorrélation séquentielle (Lag 1)')
    
    plt.tight_layout()
    plt.savefig('07_Final_Preparation_Summary.png')
    plt.close()

    print(f"\n--- PART 13: SUMMARY REPORT AUTOMATISÉ ---")
    print(f"Structure de l'espace: {df_model.shape} observations pour {len(df_model.columns)} vecteurs de caractéristiques.")
    print(f"Stats Principales -> Moyenne de Ventes: {df_model['Ventes'].mean():.1f}, Ecart-Type: {df_model['Ventes'].std():.1f}")
    print("\n✅ DATA PIPELINE INTÉGRALEMENT CONSTRUIT -> READY FOR MACHINE LEARNING")

if __name__ == '__main__':
    main()
