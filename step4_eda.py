import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

def identify_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def main():
    print("🚀 DÉMARRAGE: ÉTAPE 4️⃣ - EXPLORATORY DATA ANALYSIS (EDA)\n")
    
    # Load enriched data from step 2.5
    input_file = 'data_cleaned_enriched.csv'
    if not os.path.exists(input_file):
        print(f"❌ {input_file} introuvable. Veuillez exécuter step2_5_enrich_data.py d'abord.")
        return
        
    print("⏳ Chargement des données...")
    df = pd.read_csv(input_file, parse_dates=['DATV'])
    
    # Rename DATV to DAT_V for consistency
    df = df.rename(columns={'DATV': 'DAT_V'})
    
    # Ensure YEAR_MONTH exists
    if 'YEAR_MONTH' not in df.columns:
        print("⚠️ YEAR_MONTH not found, creating it...")
        if 'YEAR' in df.columns and 'MONTH' in df.columns:
            df['YEAR_MONTH'] = df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str).str.zfill(2)
        else:
            print("❌ Cannot create YEAR_MONTH without YEAR and MONTH columns")
            return

    if 'DAT_V' not in df.columns:
        print("❌ DAT_V column not found!")
        return

    print(f"✅ Données chargées: {df.shape}")

    print("\n--- PART 1: General Statistics ---")
    print(f"Total des véhicules : {len(df):,}")
    print(f"Date de début : {df['DAT_V'].min().date()}")
    print(f"Date de fin : {df['DAT_V'].max().date()}")
    if 'YEAR' in df.columns:
        print(f"Années distinctes : {df['YEAR'].nunique()}")
    if 'MARQUE' in df.columns:
        print(f"Marques uniques : {df['MARQUE'].nunique()}")

    print("\n⏳ PART 2: Création de la métrique de Ventes...")
    ventes_mensuelles = df.groupby('YEAR_MONTH').size().reset_index(name='Ventes')
    
    # Convert YEAR_MONTH to datetime
    ventes_mensuelles['YEAR_MONTH'] = pd.to_datetime(ventes_mensuelles['YEAR_MONTH'], errors='coerce')
    
    ventes_mensuelles['YEAR'] = ventes_mensuelles['YEAR_MONTH'].dt.year
    ventes_mensuelles['MONTH'] = ventes_mensuelles['YEAR_MONTH'].dt.month
    print("✅ Métrique Ventes créée par mois.")

    print("\n⏳ PART 3: Analyse Temporelle...")
    # Graph 1: Sales over time
    plt.figure(figsize=(14, 6))
    plt.plot(ventes_mensuelles['YEAR_MONTH'], ventes_mensuelles['Ventes'], marker='o', linestyle='-', color='b', linewidth=2)
    plt.title('Ventes de véhicules dans le temps (2019-2026)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Nombre de véhicules (Ventes)')
    years = ventes_mensuelles['YEAR_MONTH'].dt.year.unique()
    for year in years:
        plt.axvline(pd.to_datetime(f'{year}-01-01'), color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('01_Ventes_Over_Time.png', dpi=300)
    plt.close()
    print("✅ 01_Ventes_Over_Time.png")
    
    # Graph 2: Sales by year
    plt.figure(figsize=(10, 6))
    ventes_annee = df.groupby('YEAR').size()
    ax = ventes_annee.plot(kind='bar', color='skyblue', edgecolor='navy')
    plt.title('Ventes par Année', fontsize=14, fontweight='bold')
    plt.xlabel('Année')
    plt.ylabel('Total de véhicules')
    # Add value labels
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('02_Ventes_Par_Annee.png', dpi=300)
    plt.close()
    print("✅ 02_Ventes_Par_Annee.png")

    print("\n⏳ PART 4: Analyse de Saisonnalité...")
    saisonnalite = ventes_mensuelles.groupby('MONTH')['Ventes'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=saisonnalite, x='MONTH', y='Ventes', palette='viridis')
    plt.title('Ventes Moyennes par Mois (Saisonnalité)', fontsize=14, fontweight='bold')
    plt.xlabel('Mois (1=Jan, 12=Dec)')
    plt.ylabel('Nombre moyen de véhicules')
    plt.tight_layout()
    plt.savefig('03_Saisonnalite.png', dpi=300)
    plt.close()
    print("✅ 03_Saisonnalite.png")
    
    mois_fort = saisonnalite.loc[saisonnalite['Ventes'].idxmax(), 'MONTH']
    mois_faible = saisonnalite.loc[saisonnalite['Ventes'].idxmin(), 'MONTH']
    print(f"   Mois le plus fort : {int(mois_fort)}")
    print(f"   Mois le plus faible : {int(mois_faible)}")

    print("\n⏳ PART 5: Analyse de Marché...")
    if 'MARQUE' in df.columns:
        top_marques = df['MARQUE'].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        ax = top_marques.plot(kind='barh', color='coral', edgecolor='darkred')
        plt.title('Top 10 des Marques par Nombre de Véhicules', fontsize=14, fontweight='bold')
        plt.xlabel('Nombre de véhicules')
        plt.ylabel('Marque')
        plt.gca().invert_yaxis()
        for p in ax.patches:
            ax.annotate(f"{int(p.get_width()):,}", (p.get_width(), p.get_y() + p.get_height()/2), 
                       va='center', ha='left', fontsize=9)
        plt.tight_layout()
        plt.savefig('04_Top_Marques.png', dpi=300)
        plt.close()
        print("✅ 04_Top_Marques.png")
        
        print(f"\n   Top marque : {top_marques.index[0]} ({top_marques.values[0]:,})")
        print(f"   Top 3 : {top_marques.values[:3].sum():,} ({top_marques.values[:3].sum()/len(df)*100:.1f}%)")

    print("\n⏳ PART 6: Détection des Mois Manquants...")
    min_date = ventes_mensuelles['YEAR_MONTH'].min()
    max_date = ventes_mensuelles['YEAR_MONTH'].max()
    complete_months = pd.date_range(start=min_date, end=max_date, freq='MS')
    existing_months = ventes_mensuelles['YEAR_MONTH']
    missing_months = set(complete_months) - set(existing_months)
    
    if len(missing_months) > 0:
        print(f"⚠️ {len(missing_months)} mois manquants détectés:")
        for m in sorted(missing_months):
            print(f"  - {m.date()}")
    else:
        print("✅ Couverture COMPLÈTE: Aucun mois manquant!")

    print("\n⏳ PART 7: Détection des Outliers...")
    l_bound, u_bound = identify_outliers_iqr(ventes_mensuelles['Ventes'])
    outliers = ventes_mensuelles[(ventes_mensuelles['Ventes'] < l_bound) | (ventes_mensuelles['Ventes'] > u_bound)]
    
    print(f"\n   Statistiques Mensuelles:")
    print(f"   - Moyenne : {ventes_mensuelles['Ventes'].mean():.0f} véhicules")
    print(f"   - Médiane : {ventes_mensuelles['Ventes'].median():.0f}")
    print(f"   - Std Dev : {ventes_mensuelles['Ventes'].std():.0f}")
    print(f"   - Min : {ventes_mensuelles['Ventes'].min():.0f}")
    print(f"   - Max : {ventes_mensuelles['Ventes'].max():.0f}")
    
    if len(outliers) > 0:
        print(f"\n   ⚠️ {len(outliers)} mois avec ventes aberrantes (Outliers):")
        for idx, row in outliers.iterrows():
            print(f"     - {row['YEAR_MONTH'].date()} : {row['Ventes']:.0f} véhicules")
    else:
        print("   ✅ Aucun outlier détecté.")
        
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=ventes_mensuelles, x='YEAR', y='Ventes', palette='Set2')
    plt.title('Distribution des Ventes par Année', fontsize=14, fontweight='bold')
    plt.xlabel('Année')
    plt.ylabel('Ventes')
    plt.tight_layout()
    plt.savefig('05_BoxPlot_Outliers.png', dpi=300)
    plt.close()
    print("✅ 05_BoxPlot_Outliers.png")

    print("\n" + "="*70)
    print("📊 KEY INSIGHTS")
    print("="*70)
    
    if 'MARQUE' in df.columns:
        print(f"\n🏆 Marché:")
        print(f"   - Marque dominante : {top_marques.index[0]}")
        print(f"   - Part du marché : {top_marques.values[0]/len(df)*100:.1f}%")
    
    # Growth / Decline trend
    yearly_sales = df.groupby('YEAR').size()
    if len(yearly_sales) >= 2:
        start_year, end_year = yearly_sales.index[0], yearly_sales.index[-1]
        trend = (yearly_sales[end_year] - yearly_sales[start_year]) / yearly_sales[start_year] * 100
        print(f"\n📈 Tendance Globale ({start_year}-{end_year}): {trend:+.1f}%")
        
    print(f"\n📅 Saisonnalité:")
    forte_sales = saisonnalite[saisonnalite['MONTH']==mois_fort]['Ventes'].values[0]
    faible_sales = saisonnalite[saisonnalite['MONTH']==mois_faible]['Ventes'].values[0]
    print(f"   - Mois le plus fort : {int(mois_fort)} (Ventes moyennes : {forte_sales:.0f})")
    print(f"   - Mois le plus faible : {int(mois_faible)} (Ventes moyennes : {faible_sales:.0f})")
    
    print(f"\n📊 Qualité des données:")
    print(f"   - Volatilité : {(ventes_mensuelles['Ventes'].std() / ventes_mensuelles['Ventes'].mean() * 100):.1f}%")
    print(f"   - Mois manquants : {len(missing_months)}")
    print(f"   - Outliers : {len(outliers)}")
    
    print("\n" + "="*70)
    print("✅ EDA TERMINÉE AVEC SUCCÈS!")
    print("="*70)
    print("\n5 visualisations PNG générées:")
    print("   - 01_Ventes_Over_Time.png")
    print("   - 02_Ventes_Par_Annee.png")
    print("   - 03_Saisonnalite.png")
    print("   - 04_Top_Marques.png")
    print("   - 05_BoxPlot_Outliers.png")

if __name__ == '__main__':
    main()