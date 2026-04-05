import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

def main():
    print("🚀 DÉMARRAGE: ÉTAPE 5️⃣ - REAL DATA PREPARATION\n")
    
    input_file = 'data_cleaned_step3.csv'
    if not os.path.exists(input_file):
        print(f"❌ {input_file} introuvable.")
        return
        
    print("⏳ Chargement des données...")
    df_raw = pd.read_csv(input_file, parse_dates=['DAT_V'])
    if 'YEAR_MONTH' not in df_raw.columns:
        print("❌ YEAR_MONTH introuvable.")
        return
        
    df_raw['YEAR_MONTH_dt'] = pd.to_datetime(df_raw['YEAR_MONTH'])

    print("⏳ PART 1: Agrégation au niveau mensuel...")
    # Count of vehicles per month
    df_monthly = df_raw.groupby('YEAR_MONTH_dt')['ID'].count().reset_index().rename(columns={'ID': 'Ventes'})
    
    # Identify dominant categorical features
    cols_to_mode = ['MARQUE', 'GENRE', 'ENERGIE']
    for col in cols_to_mode:
        if col in df_raw.columns:
            mode_df = df_raw.groupby('YEAR_MONTH_dt')[col].apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan).reset_index(name=f'{col}_dominante')
            df_monthly = df_monthly.merge(mode_df, on='YEAR_MONTH_dt', how='left')

    df_monthly = df_monthly.rename(columns={'YEAR_MONTH_dt': 'Date'})
    # Set to end of month
    df_monthly['Date'] = df_monthly['Date'] + pd.offsets.MonthEnd(0)
    df_monthly = df_monthly.sort_values('Date').reset_index(drop=True)
    
    print("\n⏳ PART 7: Vérification de la couverture complète (Dates)...")
    min_date = df_monthly['Date'].min()
    max_date = df_monthly['Date'].max()
    complete_range = pd.date_range(start=min_date.replace(day=1), end=max_date, freq='ME')
    
    df_complete = pd.DataFrame({'Date': complete_range})
    df_monthly = df_complete.merge(df_monthly, on='Date', how='left')
    
    missing_val = df_monthly['Ventes'].isnull().sum()
    if missing_val > 0:
        print(f"⚠️ {missing_val} mois manquants trouvés dans la série, lignes créées.")
    else:
        print("✅ Aucun mois manquant, série temporelle continue!")

    print("\n⏳ PART 5: Gestion des valeurs manquantes...")
    print(f"Valeurs manquantes AVANT:\n{df_monthly.isnull().sum()[df_monthly.isnull().sum() > 0]}")
    
    # Interpolation for missing sales
    df_monthly['Ventes'] = df_monthly['Ventes'].interpolate(method='linear')
    # Mean imputation if any remains at boundaries
    df_monthly['Ventes'] = df_monthly['Ventes'].fillna(df_monthly['Ventes'].mean())
    
    # Forward/Backward fill for categorical dominantes
    for col in cols_to_mode:
        if f'{col}_dominante' in df_monthly.columns:
            df_monthly[f'{col}_dominante'] = df_monthly[f'{col}_dominante'].ffill().bfill()
            
    print(f"✅ Remplissage terminé. Valeurs manquantes APRÈS : {df_monthly.isnull().sum().sum()}")

    print("\n⏳ PART 4 & 6: MAs et Détection/Gestion des Outliers...")
    # Moving Averages based on sales
    df_monthly['Ventes_MA3'] = df_monthly['Ventes'].rolling(window=3, min_periods=1).mean()
    df_monthly['Ventes_MA6'] = df_monthly['Ventes'].rolling(window=6, min_periods=1).mean()
    df_monthly['Ventes_MA12'] = df_monthly['Ventes'].rolling(window=12, min_periods=1).mean()

    # Outliers IQR
    Q1 = df_monthly['Ventes'].quantile(0.25)
    Q3 = df_monthly['Ventes'].quantile(0.75)
    IQR = Q3 - Q1
    lb = Q1 - 1.5 * IQR
    ub = Q3 + 1.5 * IQR
    
    outlier_idx = df_monthly[(df_monthly['Ventes'] < lb) | (df_monthly['Ventes'] > ub)].index
    if len(outlier_idx) > 0:
        print(f"⚠️ {len(outlier_idx)} outliers détectés. Remplacement par la moyenne mobile 3 mois (MA3).")
        df_monthly.loc[outlier_idx, 'Ventes'] = df_monthly.loc[outlier_idx, 'Ventes_MA3']
    else:
        print("✅ Aucun outlier majeur détecté.")
        
    print("\n⏳ PART 2: Création de features temporelles...")
    df_monthly['Year'] = df_monthly['Date'].dt.year
    df_monthly['Month'] = df_monthly['Date'].dt.month
    df_monthly['Quarter'] = df_monthly['Date'].dt.quarter
    df_monthly['DayOfWeek'] = df_monthly['Date'].dt.dayofweek
    df_monthly['DayOfYear'] = df_monthly['Date'].dt.dayofyear
    
    for q in range(1, 5):
        df_monthly[f'Is_Q{q}'] = (df_monthly['Quarter'] == q).astype(int)
        
    for m in range(1, 13):
        df_monthly[f'Month_{m}'] = (df_monthly['Month'] == m).astype(int)

    print("\n⏳ PART 3: Création de Features Lag...")
    df_monthly['Ventes_Lag1'] = df_monthly['Ventes'].shift(1)
    df_monthly['Ventes_Lag3'] = df_monthly['Ventes'].shift(3)
    df_monthly['Ventes_Lag6'] = df_monthly['Ventes'].shift(6)
    df_monthly['Ventes_Lag12'] = df_monthly['Ventes'].shift(12)
    
    # Fill NAs in Lags
    lag_cols = ['Ventes_Lag1', 'Ventes_Lag3', 'Ventes_Lag6', 'Ventes_Lag12']
    df_monthly[lag_cols] = df_monthly[lag_cols].bfill().ffill()

    print("\n⏳ PART 8: Normalisation des Features (Optionnel)...")
    scaler = StandardScaler()
    num_to_scale = ['Ventes'] + lag_cols
    df_scaled = scaler.fit_transform(df_monthly[num_to_scale])
    for i, col in enumerate(num_to_scale):
        df_monthly[f'{col}_norm'] = df_scaled[:, i]

    print("\n⏳ PART 9: Création de la Variable Cible (Target)...")
    # Shift sales forward by 1 month to predict the next month
    df_monthly['Ventes_Next_Month'] = df_monthly['Ventes'].shift(-1)
    
    # The last row has NO target, we keep it separate to predict next month directly
    df_model = df_monthly.iloc[:-1].copy()
    predict_row = df_monthly.iloc[[-1]].copy()

    print("\n⏳ PART 10: Splits Train/Validation/Test/Future...")
    # Following user strategy
    data_train = df_model[(df_model['Year'] >= 2019) & (df_model['Year'] <= 2023)]
    data_val = df_model[df_model['Year'] == 2024]
    data_test = df_model[df_model['Year'] == 2025]
    
    # Simulate Future 2026 data based on features we will need to predict sequentially
    future_dates = pd.date_range(start='2026-01-31', periods=12, freq='ME')
    data_future = pd.DataFrame({'Date': future_dates, 'Year': 2026})
    for col in df_model.columns:
        if col not in ['Date', 'Year']:
            data_future[col] = np.nan

    print("\n⏳ PART 11: Sauvegarde de tous les datasets...")
    df_model.to_csv('data_prepared_final.csv', index=False)
    data_train.to_csv('data_train_2019_2023.csv', index=False)
    data_val.to_csv('data_validation_2024.csv', index=False)
    data_test.to_csv('data_test_2025.csv', index=False)
    data_future.to_csv('data_future_2026.csv', index=False)
    print("✅ 5 fichiers Datasets CSV sauvegardés localement.")

    print("\n⏳ PART 12: Création de la visualisation finale...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Sales + MA3 + MA12
    axes[0, 0].plot(df_model['Date'], df_model['Ventes'], label='Ventes Men.', alpha=0.6, linewidth=2)
    axes[0, 0].plot(df_model['Date'], df_model['Ventes_MA3'], label='MA-3 Mois', linestyle='--')
    axes[0, 0].plot(df_model['Date'], df_model['Ventes_MA12'], label='MA-12 Mois', linestyle='--')
    axes[0, 0].set_title('Ventes Régularisées & Moyennes Mobiles')
    axes[0, 0].legend()
    
    # 2. Box plot par année
    sns.boxplot(ax=axes[0, 1], data=df_model, x='Year', y='Ventes', palette='Set3')
    axes[0, 1].set_title('Distribution Mensuelle par Année (Ventes)')
    
    # 3. Saisonnalité
    sns.barplot(ax=axes[1, 0], data=df_model, x='Month', y='Ventes', ci=None, palette='mako')
    axes[1, 0].set_title('Saisonnalité Moyenne Finalisée par Mois')
    
    # 4. Autocorrelation (lag plot)
    sns.scatterplot(ax=axes[1, 1], x=df_model['Ventes_Lag1'], y=df_model['Ventes'], alpha=0.7, color='purple')
    axes[1, 1].set_title('Autocorrélation (Lag 1 vs Ventes)')
    axes[1, 1].set_xlabel('Ventes_Lag1 (Mois Précédent)')
    axes[1, 1].set_ylabel('Ventes (Mois Actuel)')
    
    plt.tight_layout()
    plt.savefig('06_Final_Preparation_Summary.png')
    plt.close()
    print("✅ Résumé Graphique 06_Final_Preparation_Summary.png enregistré.")

    print("\n--- PART 13: Summary Report ---")
    print(f"Shape final du Modèle préparé : {df_model.shape}")
    print(f"Nombre de features extraites : {len(df_model.columns)}")
    print(f"Statistiques Ventes (Imputées) -> Moyenne = {df_model['Ventes'].mean():.1f}, Médiane = {df_model['Ventes'].median():.1f}, Std = {df_model['Ventes'].std():.1f}")
    
    print("\nRépartition Globale pour le Modèle:")
    total_months = len(df_model)
    if total_months > 0:
        print(f"- TRAIN (2019-2023) : {len(data_train)} mois ({len(data_train)/total_months:.1%})")
        print(f"- VALID (2024)      : {len(data_val)} mois ({len(data_val)/total_months:.1%})")
        print(f"- TEST (2025)       : {len(data_test)} mois ({len(data_test)/total_months:.1%})")
    print("- FUTURE (2026)     : 12 mois vides, prêts pour prédiction futures")
    print("\n✅ DATA PIPELINE COMPLET -> PRÊT POUR LA MODÉLISATION MACHING LEARNING !")

if __name__ == '__main__':
    main()
