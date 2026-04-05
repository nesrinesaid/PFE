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
    
    input_file = 'data_cleaned_step3.csv'
    if not os.path.exists(input_file):
        print(f"❌ {input_file} introuvable. Veuillez exécuter step3_cleaning.py d'abord.")
        return
        
    print("⏳ Chargement des données...")
    df = pd.read_csv(input_file, parse_dates=['DAT_V'])
    if 'YEAR_MONTH' in df.columns:
        df['YEAR_MONTH'] = pd.to_datetime(df['YEAR_MONTH'])
    else:
        print("❌ Colonne YEAR_MONTH introuvable.")
        return

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
    ventes_mensuelles['YEAR'] = ventes_mensuelles['YEAR_MONTH'].dt.year
    ventes_mensuelles['MONTH'] = ventes_mensuelles['YEAR_MONTH'].dt.month
    print("✅ Métrique Ventes créée par mois.")

    print("\n⏳ PART 3: Analyse Temporelle...")
    # Graph 1
    plt.figure(figsize=(14, 6))
    plt.plot(ventes_mensuelles['YEAR_MONTH'], ventes_mensuelles['Ventes'], marker='o', linestyle='-', color='b')
    plt.title('Ventes de véhicules dans le temps (2019-2026)')
    plt.xlabel('Date')
    plt.ylabel('Nombre de véhicules (Ventes)')
    years = ventes_mensuelles['YEAR_MONTH'].dt.year.unique()
    for year in years:
        plt.axvline(pd.to_datetime(f'{year}-01-01'), color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('01_Ventes_Over_Time.png')
    plt.close()
    
    # Graph 2
    plt.figure(figsize=(10, 6))
    ventes_annee = df.groupby('YEAR').size()
    ax = ventes_annee.plot(kind='bar', color='skyblue')
    plt.title('Ventes par Année')
    plt.xlabel('Année')
    plt.ylabel('Total de véhicules')
    # Add value labels
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('02_Ventes_Par_Annee.png')
    plt.close()
    print("✅ Graphiques temporels générés.")

    print("\n⏳ PART 4: Analyse de Saisonnalité...")
    saisonnalite = ventes_mensuelles.groupby('MONTH')['Ventes'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=saisonnalite, x='MONTH', y='Ventes', palette='viridis')
    plt.title('Ventes Moyennes parois Calendaire (Saisonnalité)')
    plt.xlabel('Mois (1-12)')
    plt.ylabel('Nombre moyen de véhicules')
    plt.tight_layout()
    plt.savefig('03_Saisonnalite.png')
    plt.close()
    
    mois_fort = saisonnalite.loc[saisonnalite['Ventes'].idxmax(), 'MONTH']
    mois_faible = saisonnalite.loc[saisonnalite['Ventes'].idxmin(), 'MONTH']
    print(f"✅ Saisonnalité analysée.")

    print("\n⏳ PART 5: Analyse de Marché...")
    if 'MARQUE' in df.columns:
        top_marques = df['MARQUE'].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        ax = top_marques.plot(kind='barh', color='coral')
        plt.title('Top 10 des Marques par Nombre de Véhicules')
        plt.xlabel('Nombre de véhicules')
        plt.ylabel('Marque')
        plt.gca().invert_yaxis()
        for p in ax.patches:
            ax.annotate(f"{int(p.get_width()):,}", (p.get_width(), p.get_y() + p.get_height()/2), va='center')
        plt.tight_layout()
        plt.savefig('04_Top_Marques.png')
        plt.close()
        print("✅ Graphe Top Marques généré.")

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
        print("✅ Aucun mois manquant!")

    print("\n⏳ PART 7: Détection des Outliers...")
    l_bound, u_bound = identify_outliers_iqr(ventes_mensuelles['Ventes'])
    outliers = ventes_mensuelles[(ventes_mensuelles['Ventes'] < l_bound) | (ventes_mensuelles['Ventes'] > u_bound)]
    print(f"Statistiques Mensuelles -> Moyenne={ventes_mensuelles['Ventes'].mean():.1f}, Médiane={ventes_mensuelles['Ventes'].median():.1f}, Std={ventes_mensuelles['Ventes'].std():.1f}")
    print(f"Limites IQR -> Q1={ventes_mensuelles['Ventes'].quantile(0.25):.1f}, Q3={ventes_mensuelles['Ventes'].quantile(0.75):.1f}")
    
    if len(outliers) > 0:
        print("⚠️ Mois avec des ventes aberrantes (Outliers) détectés:")
        print(outliers[['YEAR_MONTH', 'Ventes']])
    else:
        print("✅ Aucun outlier détecté via IQR.")
        
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=ventes_mensuelles, x='YEAR', y='Ventes', palette='Set2')
    plt.title('Distribution des Ventes par Année avec Outliers')
    plt.tight_layout()
    plt.savefig('05_BoxPlot_Outliers.png')
    plt.close()

    print("\n--- PART 8: Key Insights ---")
    if 'MARQUE' in df.columns:
        print(f"🏆 Marque dominante la plus écoulée : {top_marques.index[0]} ({top_marques.values[0]:,} véhicules)")
    
    # Growth / Decline trend (from first year to last complete year)
    yearly_sales = df.groupby('YEAR').size()
    if len(yearly_sales) >= 2:
        start_year, end_year = yearly_sales.index[0], yearly_sales.index[-1]
        trend = (yearly_sales[end_year] - yearly_sales[start_year]) / yearly_sales[start_year] * 100
        print(f"📊 Tendance Globale ({start_year}-{end_year}): {trend:+.2f}%")
        
    print(f"📈 Volatilité (Coefficient de variation) : {(ventes_mensuelles['Ventes'].std() / ventes_mensuelles['Ventes'].mean() * 100):.2f}%")
    print(f"📅 Le mois proportionnellement le plus fort : {mois_fort} (- Jan=1 ... Dec=12)")
    print(f"📉 Le mois proportionnellement le plus faible : {mois_faible}")
    print("\n🚀 EDA terminée avec succès. 5 Visualisations PNG enregistrées.")

if __name__ == '__main__':
    main()
