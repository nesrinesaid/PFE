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

def safe_col(df, col_kwd):
    return next((c for c in df.columns if col_kwd in c), None)

def main():
    print("🚀 DÉMARRAGE: ÉTAPE 4️⃣ - EXPLORATORY DATA ANALYSIS (EDA) - AVEC ENRICHISSEMENT\n")
    
    input_file = 'data_cleaned_enriched.csv'
    if not os.path.exists(input_file):
        print(f"❌ {input_file} introuvable.")
        return
        
    df = pd.read_csv(input_file, parse_dates=['DAT_V'])
    df['YEAR_MONTH'] = pd.to_datetime(df['YEAR_MONTH'])

    print("\n--- PART 1: General Statistics ---")
    print(f"Total véhicules : {len(df):,}")
    print(f"Date début : {df['DAT_V'].min().date()} | Date fin : {df['DAT_V'].max().date()}")
    
    col_marque = safe_col(df, 'MARQUE')
    if col_marque: print(f"Marques uniques : {df[col_marque].nunique()}")
        
    ventes_mensuelles = df.groupby('YEAR_MONTH').size().reset_index(name='Ventes')
    ventes_mensuelles['YEAR'] = ventes_mensuelles['YEAR_MONTH'].dt.year
    ventes_mensuelles['MONTH'] = ventes_mensuelles['YEAR_MONTH'].dt.month

    # PART 3: Temporal Analysis
    plt.figure(figsize=(14, 6))
    plt.plot(ventes_mensuelles['YEAR_MONTH'], ventes_mensuelles['Ventes'], marker='o', color='b')
    for y in ventes_mensuelles['YEAR'].unique():
        plt.axvline(pd.to_datetime(f'{y}-01-01'), color='gray', linestyle='--', alpha=0.5)
    plt.title("Ventes de véhicules dans le temps")
    plt.savefig('01_Ventes_Over_Time.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    ax = df.groupby('YEAR').size().plot(kind='bar', color='skyblue')
    plt.title("Ventes par Année")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
    plt.savefig('02_Ventes_Par_Annee.png')
    plt.close()

    # PART 4: Seasonality
    saisonnalite = ventes_mensuelles.groupby('MONTH')['Ventes'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=saisonnalite, x='MONTH', y='Ventes', palette='viridis')
    plt.title("Saisonnalité")
    plt.savefig('03_Saisonnalite.png')
    plt.close()
    
    mois_fort = saisonnalite.loc[saisonnalite['Ventes'].idxmax(), 'MONTH']
    mois_faible = saisonnalite.loc[saisonnalite['Ventes'].idxmin(), 'MONTH']

    # PART 5: Top Marques
    if col_marque:
        plt.figure(figsize=(10, 6))
        df[col_marque].value_counts().head(10).plot(kind='barh', color='coral').invert_yaxis()
        plt.title("Top Marques")
        plt.savefig('04_Top_Marques.png')
        plt.close()

    # PART 6: Missing Months
    min_date = ventes_mensuelles['YEAR_MONTH'].min()
    max_date = ventes_mensuelles['YEAR_MONTH'].max()
    complete_months = pd.date_range(start=min_date, end=max_date, freq='MS')
    missing_months = set(complete_months) - set(ventes_mensuelles['YEAR_MONTH'])
    if missing_months:
        print(f"⚠️ {len(missing_months)} mois manquants détectés.")
    else:
        print("✅ Aucun mois manquant!")

    # PART 7: Outliers
    l_bound, u_bound = identify_outliers_iqr(ventes_mensuelles['Ventes'])
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=ventes_mensuelles, x='YEAR', y='Ventes', palette='Set2')
    plt.title("Distribution des ventes")
    plt.savefig('05_BoxPlot_Outliers.png')
    plt.close()

    volatilite = (ventes_mensuelles['Ventes'].std() / ventes_mensuelles['Ventes'].mean()) * 100

    print("\n--- NEW ENRICHMENT ANALYSIS ---")
    
    col_ville = safe_col(df, 'VILLE')
    if col_ville:
        plt.figure(figsize=(10, 6))
        top_v = df[col_ville].value_counts().head(10)
        top_v.plot(kind='bar', color='teal')
        plt.title("Top Villes")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('06A_Ventes_Par_Ville.png')
        plt.close()
        
        conc = top_v.head(3).sum() / len(df.dropna(subset=[col_ville])) * 100
        print(f"📍 Concentration géographique: Les 3 premières villes détiennent {conc:.1f}% du marché!")

    col_seg = safe_col(df, 'SEGMENT')
    if col_seg:
        plt.figure(figsize=(10, 6))
        top_s = df[col_seg].value_counts().head(10)
        top_s.plot(kind='bar', color='purple')
        plt.title("Top Segments")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('06B_Ventes_Par_Segment.png')
        plt.close()
        print(f"🚙 Segment dominant: {top_s.index[0]} avec {top_s.values[0]:,} véhicules")

    col_dist = safe_col(df, 'DISTRIBUTEUR')
    if col_dist:
        plt.figure(figsize=(10, 6))
        df[col_dist].value_counts().head(10).plot(kind='barh', color='orange').invert_yaxis()
        plt.title("Top Distributeurs")
        plt.tight_layout()
        plt.savefig('06C_Ventes_Par_Distributeur.png')
        plt.close()

    col_groupe = safe_col(df, 'GROUPE')
    if col_groupe:
        plt.figure(figsize=(10, 6))
        g_counts = df[col_groupe].value_counts().head(10)
        g_counts.plot(kind='barh', color='darkred').invert_yaxis()
        plt.title("Ventes par Groupe")
        plt.tight_layout()
        plt.savefig('06D_Ventes_Par_Groupe.png')
        plt.close()
        if not g_counts.empty:
            print(f"🏢 Groupe dominant: {g_counts.index[0]} ({g_counts.values[0] / len(df.dropna(subset=[col_groupe])) * 100:.1f}%)")

    col_cont = safe_col(df, 'CONTINENT')
    if col_cont:
        cont_counts = df[col_cont].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(cont_counts, labels=cont_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title("Répartition Continentale (Pie)")
        plt.savefig('06E_Marche_Par_Continent_Pie.png')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        cont_counts.plot(kind='bar', color='green')
        plt.title("Répartition Continentale (Bar)")
        plt.savefig('06E_Marche_Par_Continent_Bar.png')
        plt.close()
        
        if 'Europe' in cont_counts:
            print(f"🌍 L'Europe représente {(cont_counts['Europe'] / cont_counts.sum() * 100):.1f}% des données.")
        
        # Evolution over time
        df_cont_time = df.groupby(['YEAR', col_cont]).size().unstack()
        plt.figure(figsize=(12, 6))
        df_cont_time.plot(marker='o')
        plt.title('Evolution des ventes par continent')
        plt.savefig('06F_Evolution_Continents_Over_Time.png')
        plt.close()
        
    col_marche = safe_col(df, 'MARCHE')
    if col_marche:
        plt.figure(figsize=(8, 6))
        df[col_marche].value_counts().plot(kind='bar', color=['blue', 'gray'])
        plt.title("Marché VP vs Utilitaire")
        plt.savefig('06G_Marche_VP_vs_Utilitaire.png')
        plt.close()

    col_usage = safe_col(df, 'USAGE')
    if col_usage:
        plt.figure(figsize=(10, 6))
        df[col_usage].value_counts().head(10).plot(kind='bar', color='brown')
        plt.title("Ventes par Usage")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('06H_Ventes_Par_Usage.png')
        plt.close()

    print(f"\n✅ RECAP DES INSIGHTS GLOBAUX :")
    print(f"- Mois de pointe annuel: {mois_fort}, Mois le plus faible: {mois_faible}")
    print(f"- Volatilité globale modulaire de {volatilite:.1f}%")
    if col_marque: print(f"- Première marque nationale: {df[col_marque].mode()[0]}")
    print("🚀 NOUVELLE EDA TERMINÉE AVEC SUCCÈS! TOUS LES GRAPHIQUES ONT ÉTÉ SAUVEGARDÉS.")

if __name__ == '__main__':
    main()
