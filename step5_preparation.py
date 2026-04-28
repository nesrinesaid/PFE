import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from valider_pipeline import valider_colonnes, COLONNES_REQUISES

sns.set_theme(style="whitegrid")

# ─── CONFIGURATION ───
# Strategie de remplissage des lacunes pour les combinaisons jour-marche manquantes
STRATEGIE_REMPLISSAGE = 'zero'  # Options: 'zero', 'forward_fill', 'interpolate'
# zero:         jours manquants = zero ventes
# forward_fill: jours manquants = copier la valeur precedente
# interpolate:  jours manquants = interpolation lineaire


def ajouter_drapeau_ramadan(df, date_col='Date'):
    ramadan_ranges = [
        ('2019-05-06', '2019-06-04'),
        ('2020-04-24', '2020-05-23'),
        ('2021-04-13', '2021-05-12'),
        ('2022-04-02', '2022-05-01'),
        ('2023-03-23', '2023-04-21'),
        ('2024-03-11', '2024-04-09'),
        ('2025-03-01', '2025-03-30'),
        ('2026-02-18', '2026-03-19'),
    ]
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
    print(f"  Shape before filters: {df_raw.shape}")

    # ─── VALIDATE COLONNE IM_RI ───
    if 'IM_RI' not in df_raw.columns:
        print("❌ ERREUR: Colonne IM_RI non trouvee dans data_cleaned_enriched.csv")
        print("   IM_RI est requise pour identifier les ventes neufs (IM_RI=10)")
        print("   Action: Relancer step2_5_enrich_data.py pour ajouter la colonne IM_RI")
        return

    df_raw['IM_RI'] = pd.to_numeric(df_raw['IM_RI'], errors='coerce').round().astype('Int64')
    if df_raw['IM_RI'].isna().all():
        print("❌ ERREUR: Colonne IM_RI est all NaN apres coercion")
        print("   Probleme de qualite donnees: Verifier IM_RI dans data_cleaned_enriched.csv")
        return

    print("✅ Validation IM_RI reussie")
    print(f"   IM_RI=10 (ventes neufs):     {(df_raw['IM_RI'] == 10).sum():,} enregistrements")
    print(f"   IM_RI=20 (ventes occasion):  {(df_raw['IM_RI'] == 20).sum():,} enregistrements")

    # Creation defensive de TYPE_MARCHE si absent
    if 'TYPE_MARCHE' not in df_raw.columns and 'Marché' in df_raw.columns:
        df_raw['TYPE_MARCHE'] = (
            df_raw['Marché'].astype(str).str.upper().str.extract(r'(VP|VU)', expand=False)
        )

    # ─── VALIDER ENTRÉE ───
    try:
        valider_colonnes(df_raw, COLONNES_REQUISES['step5_preparation'], 'step5_preparation', verbose=True)
    except ValueError as e:
        print(str(e))
        return

    if 'TYPE_MARCHE' not in df_raw.columns:
        print("❌ ERREUR: Colonne TYPE_MARCHE non creee")
        print("   Cette colonne doit etre creee depuis la colonne 'Marché' dans step2_5")
        return

    if df_raw['TYPE_MARCHE'].isna().all():
        print("⚠️  AVERTISSEMENT: TYPE_MARCHE est all NaN")
        print("   Verifier que la colonne 'Marché' contient des valeurs 'VP' ou 'VU'")

    print("✅ Validation TYPE_MARCHE reussie")
    print(f"   VP: {(df_raw['TYPE_MARCHE'] == 'VP').sum():,}, VU: {(df_raw['TYPE_MARCHE'] == 'VU').sum():,}")

    df_raw = df_raw[df_raw['IM_RI'].eq(10)].copy()
    df_raw = df_raw[df_raw['TYPE_MARCHE'].isin(['VP', 'VU'])].copy()
    if df_raw.empty:
        print("ERROR: No rows left after IM_RI=10 and VP/VU filters.")
        return

    df_raw['Date'] = pd.to_datetime(df_raw['DATV'], errors='coerce').dt.normalize()
    df_raw = df_raw.dropna(subset=['Date']).copy()
    print(f"  Shape after filters : {df_raw.shape}")

    # PART 1: Daily aggregation by market type
    print("\nPART 1: Daily aggregation by market type...")
    df_jour = df_raw.groupby(['Date', 'TYPE_MARCHE']).size().reset_index(name='VENTES')
    df_jour = df_jour.sort_values(['TYPE_MARCHE', 'Date']).reset_index(drop=True)
    print(f"  Daily records before gap fill: {len(df_jour)}")

    # PART 2: Fill missing days per market type
    print("\nPART 2: Filling missing days...")

    # ─── STRATÉGIE DE REMPLISSAGE DES LACUNES: ZERO-FILL ───
    # HYPOTHESE: si une combinaison jour-marche n'a pas de ventes,
    # cela signifie zero vehicules vendus ce jour (pas une lacune de collecte).
    # Pour IM_RI=10, les ventes devraient etre capturees dans les registres officiels.
    # STRATEGIES alternatives: 'forward_fill', 'interpolate'.
    # CHOIX ACTUEL: STRATEGIE_REMPLISSAGE = 'zero'.
    # ──────────────────────────────────────────────────────────
    date_min = df_jour['Date'].min()
    date_max = df_jour['Date'].max()
    dates_completes = pd.date_range(start=date_min, end=date_max, freq='D')

    index_complet = pd.MultiIndex.from_product(
        [dates_completes, ['VP', 'VU']],
        names=['Date', 'TYPE_MARCHE'],
    )
    df_complet = index_complet.to_frame(index=False)
    df_jour = df_complet.merge(df_jour, on=['Date', 'TYPE_MARCHE'], how='left')

    lacunes_avant = int(df_jour['VENTES'].isna().sum())
    if lacunes_avant > 0:
        pct_lacunes = 100 * lacunes_avant / len(df_jour)
        print("\n⚠️  RAPPORT DE REMPLISSAGE DES LACUNES:")
        print(f"   {lacunes_avant:,} lignes jour-marche ({pct_lacunes:.2f}%) etaient manquantes")
        print(f"   Strategie: {STRATEGIE_REMPLISSAGE.upper()} (hypothese: pas de ventes = zero)")
        if pct_lacunes > 20:
            print("   ⚠️  AVERTISSEMENT: >20% lacunes remplies - verifier qualite donnees")

        lacunes_par_marche = df_jour.groupby('TYPE_MARCHE')['VENTES'].apply(lambda s: int(s.isna().sum()))
        print(f"   Lacunes VP: {int(lacunes_par_marche.get('VP', 0)):,}")
        print(f"   Lacunes VU: {int(lacunes_par_marche.get('VU', 0)):,}")

    if STRATEGIE_REMPLISSAGE == 'forward_fill':
        df_jour['VENTES'] = df_jour.groupby('TYPE_MARCHE')['VENTES'].ffill().fillna(0.0)
    elif STRATEGIE_REMPLISSAGE == 'interpolate':
        df_jour['VENTES'] = (
            df_jour.groupby('TYPE_MARCHE')['VENTES']
            .transform(lambda s: s.interpolate(method='linear').fillna(0.0))
        )
    else:
        df_jour['VENTES'] = df_jour['VENTES'].fillna(0.0)

    df_jour['VENTES'] = df_jour['VENTES'].astype(float)
    print(f"   Lignes apres remplissage: {len(df_jour):,}")
    jours_zero = int((df_jour['VENTES'] == 0).sum())
    print(f"   Jours sans ventes: {jours_zero:,} ({100 * jours_zero / len(df_jour):.1f}%)")

    df_jour['ANNEE'] = df_jour['Date'].dt.year
    df_jour['MOIS'] = df_jour['Date'].dt.month
    df_jour['EST_RAMADAN'] = ajouter_drapeau_ramadan(df_jour, 'Date')
    print("  Ramadan flag added")

    # PART 3: Moving averages and outlier handling
    print("\nPART 3: Moving averages & outlier handling...")
    df_jour = df_jour.sort_values(['TYPE_MARCHE', 'Date']).reset_index(drop=True)
    df_jour['VENTES_MA7'] = df_jour.groupby('TYPE_MARCHE')['VENTES'].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )
    df_jour['VENTES_MA30'] = df_jour.groupby('TYPE_MARCHE')['VENTES'].transform(
        lambda s: s.rolling(window=30, min_periods=1).mean()
    )
    df_jour['VENTES_MA90'] = df_jour.groupby('TYPE_MARCHE')['VENTES'].transform(
        lambda s: s.rolling(window=90, min_periods=1).mean()
    )

    n_outliers = 0
    for marche in ['VP', 'VU']:
        marche_mask = df_jour['TYPE_MARCHE'] == marche
        s = df_jour.loc[marche_mask, 'VENTES']
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        outlier_mask = marche_mask & (
            (df_jour['VENTES'] < q1 - 1.5 * iqr) |
            (df_jour['VENTES'] > q3 + 1.5 * iqr)
        )
        n_outliers += int(outlier_mask.sum())
        df_jour.loc[outlier_mask, 'VENTES'] = df_jour.loc[outlier_mask, 'VENTES_MA7']
    if n_outliers > 0:
        print(f"  {n_outliers} outlier day-market rows replaced with MA7")

    # PART 4: Temporal features
    print("\nPART 4: Temporal features...")
    df_jour['TRIMESTRE'] = df_jour['Date'].dt.quarter
    df_jour['JOUR_ANNEE'] = df_jour['Date'].dt.dayofyear
    df_jour['JOUR_SEMAINE'] = df_jour['Date'].dt.dayofweek
    df_jour['EST_WEEKEND'] = (df_jour['JOUR_SEMAINE'] >= 5).astype(int)
    df_jour['EST_VU'] = (df_jour['TYPE_MARCHE'] == 'VU').astype(int)
    df_jour['EST_RAMADAN'] = ajouter_drapeau_ramadan(df_jour, 'Date')

    for q in range(1, 5):
        df_jour[f'Est_T{q}'] = (df_jour['TRIMESTRE'] == q).astype(int)
    for m in range(1, 13):
        df_jour[f'Mois_{m}'] = (df_jour['MOIS'] == m).astype(int)
    for d in range(7):
        df_jour[f'JS_{d}'] = (df_jour['JOUR_SEMAINE'] == d).astype(int)

    # PART 5: Lag features
    print("\nPART 5: Lag features...")
    lag_cols = ['VENTES_LAG1', 'VENTES_LAG7', 'VENTES_LAG30', 'VENTES_LAG90']
    df_jour['VENTES_LAG1'] = df_jour.groupby('TYPE_MARCHE')['VENTES'].shift(1)
    df_jour['VENTES_LAG7'] = df_jour.groupby('TYPE_MARCHE')['VENTES'].shift(7)
    df_jour['VENTES_LAG30'] = df_jour.groupby('TYPE_MARCHE')['VENTES'].shift(30)
    df_jour['VENTES_LAG90'] = df_jour.groupby('TYPE_MARCHE')['VENTES'].shift(90)
    df_jour[lag_cols] = df_jour.groupby('TYPE_MARCHE')[lag_cols].ffill()

    # PART 6: Target variable
    df_jour['VENTES_JOUR_SUIVANT'] = df_jour.groupby('TYPE_MARCHE')['VENTES'].shift(-1)
    df_modele = df_jour[df_jour['VENTES_JOUR_SUIVANT'].notna()].copy().reset_index(drop=True)

    # PART 7: Split
    print("\nPART 7: Train/Val/Test split...")
    train_mask = (df_modele['ANNEE'] >= 2019) & (df_modele['ANNEE'] <= 2023)
    val_mask = df_modele['ANNEE'] == 2024
    test_mask = df_modele['ANNEE'] == 2025

    data_train = df_modele[train_mask].copy()
    data_val = df_modele[val_mask].copy()
    data_test = df_modele[test_mask].copy()

    print(f"  Train : {len(data_train)} rows ({data_train['ANNEE'].min()}-{data_train['ANNEE'].max()})")
    print(f"  Val   : {len(data_val)} rows ({data_val['ANNEE'].min() if len(data_val) else 'N/A'}-{data_val['ANNEE'].max() if len(data_val) else 'N/A'})")
    print(f"  Test  : {len(data_test)} rows ({data_test['ANNEE'].min() if len(data_test) else 'N/A'}-{data_test['ANNEE'].max() if len(data_test) else 'N/A'})")

    model_features = [
        'EST_VU', 'EST_RAMADAN',
        'VENTES_LAG1', 'VENTES_LAG7', 'VENTES_LAG30', 'VENTES_LAG90',
        'VENTES_MA7', 'VENTES_MA30', 'VENTES_MA90',
        'Mois_1', 'Mois_2', 'Mois_3', 'Mois_4', 'Mois_5', 'Mois_6',
        'Mois_7', 'Mois_8', 'Mois_9', 'Mois_10', 'Mois_11', 'Mois_12',
        'JS_0', 'JS_1', 'JS_2', 'JS_3', 'JS_4', 'JS_5', 'JS_6',
        'Est_T1', 'Est_T2', 'Est_T3', 'Est_T4',
    ]
    print(f"  Train feature count (daily model inputs): {len(model_features)}")

    # PART 8: Normalisation
    print("\nPART 8: Normalisation (fit on train only)...")
    num_to_scale = ['VENTES'] + lag_cols + ['VENTES_MA7', 'VENTES_MA30', 'VENTES_MA90']
    scaler = StandardScaler()
    scaler.fit(data_train[num_to_scale])

    for split_df in [df_modele, data_train, data_val, data_test]:
        scaled = scaler.transform(split_df[num_to_scale])
        for i, col in enumerate(num_to_scale):
            split_df[f'{col}_norm'] = scaled[:, i]

    # PARTIE 9: Placeholder 2026
    print("\nPARTIE 9: Creation du placeholder de verification 2026...")
    print("  ⚠️  NOTE IMPORTANTE:")
    print("      Ce fichier est un TEMPLATE de verification vide.")
    print("      Il sera remplace avec les VRAIES donnees 2026 quand disponibles.")
    print("      NE PAS utiliser ce fichier pour les previsions.\n")

    dates_futures = pd.date_range(start='2026-01-01', end='2026-12-31', freq='D')
    donnees_futures = pd.MultiIndex.from_product(
        [dates_futures, ['VP', 'VU']], names=['Date', 'TYPE_MARCHE']
    ).to_frame(index=False)

    donnees_futures['ANNEE'] = donnees_futures['Date'].dt.year
    donnees_futures['MOIS'] = donnees_futures['Date'].dt.month
    donnees_futures['TRIMESTRE'] = donnees_futures['Date'].dt.quarter
    donnees_futures['JOUR_ANNEE'] = donnees_futures['Date'].dt.dayofyear
    donnees_futures['JOUR_SEMAINE'] = donnees_futures['Date'].dt.dayofweek
    donnees_futures['EST_WEEKEND'] = (donnees_futures['JOUR_SEMAINE'] >= 5).astype(int)
    donnees_futures['EST_VU'] = (donnees_futures['TYPE_MARCHE'] == 'VU').astype(int)
    donnees_futures['EST_RAMADAN'] = ajouter_drapeau_ramadan(donnees_futures, 'Date')

    for q in range(1, 5):
        donnees_futures[f'Est_T{q}'] = (donnees_futures['TRIMESTRE'] == q).astype(int)
    for m in range(1, 13):
        donnees_futures[f'Mois_{m}'] = (donnees_futures['MOIS'] == m).astype(int)
    for d in range(7):
        donnees_futures[f'JS_{d}'] = (donnees_futures['JOUR_SEMAINE'] == d).astype(int)

    print("  Initialisation des colonnes de ventes avec NaN (placeholder):")
    print("    VENTES = NaN")
    print("    VENTES_LAG* = NaN")
    print("    VENTES_MA* = NaN")
    print("    VENTES_JOUR_SUIVANT = NaN")

    donnees_futures['VENTES'] = np.nan
    donnees_futures['VENTES_LAG1'] = np.nan
    donnees_futures['VENTES_LAG7'] = np.nan
    donnees_futures['VENTES_LAG30'] = np.nan
    donnees_futures['VENTES_LAG90'] = np.nan
    donnees_futures['VENTES_MA7'] = np.nan
    donnees_futures['VENTES_MA30'] = np.nan
    donnees_futures['VENTES_MA90'] = np.nan
    donnees_futures['VENTES_JOUR_SUIVANT'] = np.nan

    for col in df_modele.columns:
        if col not in donnees_futures.columns:
            donnees_futures[col] = np.nan
    donnees_futures = donnees_futures[df_modele.columns]

    print("\n" + "=" * 70)
    print("INSTRUCTIONS: REMPLIR DONNEES 2026 REELLES")
    print("=" * 70)
    print("Ce fichier data_future_2026.csv est une STRUCTURE TEMPLATE pour verification.")
    print("1) Remplir VENTES et variables derivees avec donnees reelles 2026.")
    print("2) Executer validation modele (MAE/MAPE/RMSE) sur ces valeurs reelles.")
    print("3) Ne pas utiliser ce fichier vide comme prevision.")
    print("=" * 70)

    print("  Validation du placeholder 2026:")
    print(f"    ✅ Total lignes: {len(donnees_futures):,} (365 jours x 2 marches)")
    print(f"    ✅ Plage dates: {donnees_futures['Date'].min().date()} a {donnees_futures['Date'].max().date()}")
    print(f"    ✅ Marches: {sorted(donnees_futures['TYPE_MARCHE'].dropna().unique().tolist())}")
    print("    ⚠️  Colonnes ventes: ALL NaN (attendu - placeholder vide)")
    print("    ✅ Colonnes temporelles: Completes (structure OK)")

    colonnes_nan = donnees_futures.columns[donnees_futures.isna().all()].tolist()
    print("  Colonnes completement vides (NaN):")
    for col in colonnes_nan:
        print(f"    - {col}")

    # ── PART 9b: Build transaction-level prepared file (keep all enriched columns)
    print("\nPART 9b: Building transaction-level prepared file (full enriched + daily features)...")
    try:
        df_enriched = pd.read_csv(input_file, parse_dates=['DATV'])
        if 'TYPE_MARCHE' not in df_enriched.columns and 'Marché' in df_enriched.columns:
            df_enriched['TYPE_MARCHE'] = df_enriched['Marché'].astype(str).str.upper().str.extract(r'(VP|VU)', expand=False)
        df_enriched['Date'] = pd.to_datetime(df_enriched['DATV'], errors='coerce').dt.normalize()

        join_cols = ['Date', 'TYPE_MARCHE', 'VENTES', 'VENTES_LAG1', 'VENTES_LAG7', 'VENTES_LAG30',
                     'VENTES_LAG90', 'VENTES_MA7', 'VENTES_MA30', 'VENTES_MA90', 'VENTES_JOUR_SUIVANT']
        available = [c for c in join_cols if c in df_modele.columns]
        df_join = df_modele[available].drop_duplicates(subset=['Date', 'TYPE_MARCHE'])

        df_full = df_enriched.merge(df_join, on=['Date', 'TYPE_MARCHE'], how='left')
        print(f"  Transaction-level prepared rows: {len(df_full):,}; daily features attached where available")
    except Exception as e:
        print(f"  WARN: Failed to build transaction-level prepared file: {e}")
        df_full = None

    # PART 10: Export
    print("\nPART 10: Saving files...")
    output_files = [
        'data_prepared_final.csv',
        'data_train.csv',
        'data_validation_2024.csv',
        'data_test_2025.csv',
        'data_future_2026.csv',
    ]

    for fname, data in [
        ('data_prepared_final.csv', df_modele),
        ('data_train.csv', data_train),
        ('data_validation_2024.csv', data_val),
        ('data_test_2025.csv', data_test),
        ('data_future_2026.csv', donnees_futures),
    ]:
        out_path = os.path.join(project_root, fname)
        data.to_csv(out_path, index=False)

    # Save transaction-level prepared file if built
    if df_full is not None:
        out_full = os.path.join(project_root, 'data_prepared_final_full.csv')
        df_full.to_csv(out_full, index=False)
        print(f"  Saved: data_prepared_final_full.csv ({len(df_full):,} rows)")

    for f in output_files:
        size = os.path.getsize(os.path.join(project_root, f)) / 1024
        print(f"  {f}: {size:.1f} KB")
    if df_full is not None:
        size = os.path.getsize(os.path.join(project_root, 'data_prepared_final_full.csv')) / 1024
        print(f"  data_prepared_final_full.csv: {size:.1f} KB")

    # PART 11: Summary visualisation
    print("\nPART 11: Summary visualisation...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    vp_series = df_modele[df_modele['TYPE_MARCHE'] == 'VP']
    vu_series = df_modele[df_modele['TYPE_MARCHE'] == 'VU']
    axes[0, 0].plot(vp_series['Date'], vp_series['VENTES'],
                    label='VP VENTES', alpha=0.75, linewidth=1.8, color='steelblue')
    axes[0, 0].plot(vp_series['Date'], vp_series['VENTES_MA30'],
                    label='VP MA30', linestyle='--', color='orange')
    axes[0, 0].plot(vu_series['Date'], vu_series['VENTES'],
                    label='VU VENTES', alpha=0.7, linewidth=1.6, color='darkgreen')
    axes[0, 0].plot(vu_series['Date'], vu_series['VENTES_MA30'],
                    label='VU MA30', linestyle='--', color='red')

    axes[0, 0].axvspan(data_train['Date'].min(), data_train['Date'].max(),
                       alpha=0.08, color='green', label='Train')
    if len(data_val) > 0:
        axes[0, 0].axvspan(data_val['Date'].min(), data_val['Date'].max(),
                           alpha=0.08, color='orange', label='Val')
    if len(data_test) > 0:
        axes[0, 0].axvspan(data_test['Date'].min(), data_test['Date'].max(),
                           alpha=0.08, color='red', label='Test')

    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title('Ventes journalieres par type de marche')

    sns.boxplot(ax=axes[0, 1], data=df_modele, x='ANNEE', y='VENTES', hue='TYPE_MARCHE', palette='Set2')
    axes[0, 1].set_title('Distribution annuelle par marche')

    sns.barplot(ax=axes[1, 0], data=df_modele, x='JOUR_SEMAINE', y='VENTES',
                hue='TYPE_MARCHE', errorbar=None, palette='mako')
    axes[1, 0].set_title('Saisonnalite jour de semaine')

    sns.scatterplot(ax=axes[1, 1], x=df_modele['VENTES_LAG1'],
                    y=df_modele['VENTES'], color='purple', alpha=0.7)
    axes[1, 1].set_title('Autocorrelation lag-1')

    plt.suptitle('Step 5 — ML Preparation Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, '11_ML_Preparation_Summary.png'), dpi=300)
    plt.close()
    print("  Saved: 11_ML_Preparation_Summary.png")

    # PART 12: Final report
    print("\n" + "=" * 60)
    print("ML PREPARATION SUMMARY")
    print("=" * 60)
    print(f"  Total rows       : {len(df_modele)}")
    print(f"  Total features   : {len(df_modele.columns)}")
    print(f"  Train rows       : {len(data_train)}")
    print(f"  Val rows         : {len(data_val)}")
    print(f"  Test rows        : {len(data_test)}")
    print(f"  Markets          : {sorted(df_modele['TYPE_MARCHE'].dropna().unique().tolist())}")
    print(f"  Ventes mean      : {df_modele['VENTES'].mean():.1f}")
    print(f"  Ventes std       : {df_modele['VENTES'].std():.1f}")
    print("  Scaler fit on    : train only (no leakage)")
    print("  Lag fill method  : ffill only (no leakage)")

    def pct_available(df_split, col):
        if len(df_split) == 0:
            return 0.0
        return 100 * df_split[col].notna().mean()

    print("\nRamadan flag coverage:")
    print(f"  Train (2019-2023)  : {pct_available(data_train, 'EST_RAMADAN'):.1f}%")
    print(f"  Val   (2024)       : {pct_available(data_val, 'EST_RAMADAN'):.1f}%")
    print(f"  Test  (2025)       : {pct_available(data_test, 'EST_RAMADAN'):.1f}%")
    print(f"  Futur (2026)       : {pct_available(donnees_futures, 'EST_RAMADAN'):.1f}%")

    print("\nSTEP 5 COMPLETE -> Ready for Step 6 (Modeling)")


if __name__ == '__main__':
    main()
