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
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR


def main():
    print("STEP 4 - EXPLORATORY DATA ANALYSIS\n")

    input_file = 'data_cleaned_enriched.csv'
    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found. Run step2_5_enrich_data.py first.")
        return

    print("Loading data...")
    df = pd.read_csv(input_file, parse_dates=['DATV'])
    df = df.rename(columns={'DATV': 'DAT_V'})

    if 'YEAR_MONTH' not in df.columns:
        df['YEAR_MONTH'] = df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str).str.zfill(2)

    print(f"  Shape: {df.shape}")

    # ── PART 1: General statistics ────────────────────────────────────────────
    print("\n--- General Statistics ---")
    print(f"  Total vehicles : {len(df):,}")
    print(f"  Date range     : {df['DAT_V'].min().date()} -> {df['DAT_V'].max().date()}")
    print(f"  Distinct years : {sorted(df['YEAR'].unique())}")
    print(f"  Unique brands  : {df['MARQUE'].nunique()}")

    # ── PART 2: Monthly sales metric ─────────────────────────────────────────
    print("\nBuilding monthly sales metric...")
    ventes = df.groupby('YEAR_MONTH').size().reset_index(name='Ventes')
    ventes['YEAR_MONTH'] = pd.to_datetime(ventes['YEAR_MONTH'], errors='coerce')
    ventes['YEAR']  = ventes['YEAR_MONTH'].dt.year
    ventes['MONTH'] = ventes['YEAR_MONTH'].dt.month
    ventes = ventes.dropna(subset=['YEAR_MONTH']).sort_values('YEAR_MONTH')

    # ── GRAPH 1: Sales over time ──────────────────────────────────────────────
    plt.figure(figsize=(14, 6))
    plt.plot(ventes['YEAR_MONTH'], ventes['Ventes'],
             marker='o', linestyle='-', color='steelblue', linewidth=2, markersize=4)
    plt.title('Ventes de véhicules dans le temps', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Nombre de véhicules')
    for year in ventes['YEAR_MONTH'].dt.year.unique():
        plt.axvline(pd.to_datetime(f'{year}-01-01'), color='gray', linestyle='--', alpha=0.4)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('01_Ventes_Over_Time.png', dpi=300)
    plt.close()
    print("  Saved: 01_Ventes_Over_Time.png")

    # ── GRAPH 2: Sales by year ────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    ventes_annee = df.groupby('YEAR').size()
    ax = ventes_annee.plot(kind='bar', color='steelblue', edgecolor='navy')
    plt.title('Ventes par Année', fontsize=14, fontweight='bold')
    plt.xlabel('Année')
    plt.ylabel('Total de véhicules')
    plt.xticks(rotation=0)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('02_Ventes_Par_Annee.png', dpi=300)
    plt.close()
    print("  Saved: 02_Ventes_Par_Annee.png")

    # ── GRAPH 3: Seasonality ──────────────────────────────────────────────────
    saisonnalite = ventes.groupby('MONTH')['Ventes'].mean().reset_index()
    mois_fort   = int(saisonnalite.loc[saisonnalite['Ventes'].idxmax(), 'MONTH'])
    mois_faible = int(saisonnalite.loc[saisonnalite['Ventes'].idxmin(), 'MONTH'])

    plt.figure(figsize=(10, 6))
    sns.barplot(data=saisonnalite, x='MONTH', y='Ventes', palette='viridis', errorbar=None)
    plt.title('Ventes Moyennes par Mois (Saisonnalité)', fontsize=14, fontweight='bold')
    plt.xlabel('Mois (1=Jan, 12=Déc)')
    plt.ylabel('Nombre moyen de véhicules')
    plt.tight_layout()
    plt.savefig('03_Saisonnalite.png', dpi=300)
    plt.close()
    print(f"  Saved: 03_Saisonnalite.png  (peak: month {mois_fort}, low: month {mois_faible})")

    # ── GRAPH 4: Top 10 brands ────────────────────────────────────────────────
    if 'MARQUE' in df.columns:
        top_marques = df['MARQUE'].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        ax = top_marques.plot(kind='barh', color='coral', edgecolor='darkred')
        plt.title('Top 10 Marques par Nombre de Véhicules', fontsize=14, fontweight='bold')
        plt.xlabel('Nombre de véhicules')
        plt.ylabel('Marque')
        plt.gca().invert_yaxis()
        for p in ax.patches:
            ax.annotate(f"{int(p.get_width()):,}",
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        va='center', ha='left', fontsize=9)
        plt.tight_layout()
        plt.savefig('04_Top_Marques.png', dpi=300)
        plt.close()
        print(f"  Saved: 04_Top_Marques.png  (top: {top_marques.index[0]})")

    # ── GRAPH 5: Outlier boxplot ──────────────────────────────────────────────
    l_bound, u_bound = identify_outliers_iqr(ventes['Ventes'])
    outliers = ventes[(ventes['Ventes'] < l_bound) | (ventes['Ventes'] > u_bound)]

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=ventes, x='YEAR', y='Ventes', palette='Set2')
    plt.title('Distribution des Ventes Mensuelles par Année', fontsize=14, fontweight='bold')
    plt.xlabel('Année')
    plt.ylabel('Ventes mensuelles')
    plt.tight_layout()
    plt.savefig('05_BoxPlot_Outliers.png', dpi=300)
    plt.close()
    print(f"  Saved: 05_BoxPlot_Outliers.png  ({len(outliers)} outlier months)")

    # ── GRAPH 6: Sales by SEGMENT ─────────────────────────────────────────────
    if 'SEGMENT' in df.columns and df['SEGMENT'].notna().sum() > 0:
        seg_counts = df['SEGMENT'].value_counts().head(12)
        plt.figure(figsize=(12, 6))
        ax = seg_counts.plot(kind='barh', color='teal', edgecolor='darkcyan')
        plt.title('Ventes par Segment de Véhicule', fontsize=14, fontweight='bold')
        plt.xlabel('Nombre de véhicules')
        plt.ylabel('Segment')
        plt.gca().invert_yaxis()
        for p in ax.patches:
            ax.annotate(f"{int(p.get_width()):,}",
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        va='center', ha='left', fontsize=8)
        nn = df['SEGMENT'].notna().sum()
        plt.figtext(0.99, 0.01, f'Coverage: {nn:,} / {len(df):,} ({nn/len(df)*100:.1f}%)',
                    ha='right', fontsize=8, color='gray')
        plt.tight_layout()
        plt.savefig('06_Ventes_Par_Segment.png', dpi=300)
        plt.close()
        print(f"  Saved: 06_Ventes_Par_Segment.png")

    # ── GRAPH 7: Sales by SOUS_SEGMENT ───────────────────────────────────────
    if 'SOUS_SEGMENT' in df.columns and df['SOUS_SEGMENT'].notna().sum() > 0:
        sous_counts = df['SOUS_SEGMENT'].value_counts().head(12)
        plt.figure(figsize=(12, 6))
        ax = sous_counts.plot(kind='barh', color='mediumslateblue', edgecolor='darkslateblue')
        plt.title('Ventes par Sous-Segment de Véhicule', fontsize=14, fontweight='bold')
        plt.xlabel('Nombre de véhicules')
        plt.ylabel('Sous-Segment')
        plt.gca().invert_yaxis()
        for p in ax.patches:
            ax.annotate(f"{int(p.get_width()):,}",
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        va='center', ha='left', fontsize=8)
        nn = df['SOUS_SEGMENT'].notna().sum()
        plt.figtext(0.99, 0.01, f'Coverage: {nn:,} / {len(df):,} ({nn/len(df)*100:.1f}%)',
                    ha='right', fontsize=8, color='gray')
        plt.tight_layout()
        plt.savefig('07_Ventes_Par_Sous_Segment.png', dpi=300)
        plt.close()
        print(f"  Saved: 07_Ventes_Par_Sous_Segment.png")

    # ── GRAPH 8: Sales by MARCHE_TYPE (VP / VU / Autre) ──────────────────────
    if 'MARCHE_TYPE' in df.columns and df['MARCHE_TYPE'].notna().sum() > 0:
        marche_counts = df['MARCHE_TYPE'].value_counts()
        plt.figure(figsize=(8, 6))
        colors = ['#2196F3', '#FF9800', '#4CAF50'][:len(marche_counts)]
        wedges, texts, autotexts = plt.pie(
            marche_counts.values,
            labels=marche_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        for t in autotexts:
            t.set_fontsize(10)
        plt.title('Répartition par Type de Marché (VP / VU / Autre)',
                  fontsize=14, fontweight='bold')
        nn = df['MARCHE_TYPE'].notna().sum()
        plt.figtext(0.5, 0.01, f'Coverage: {nn:,} / {len(df):,} ({nn/len(df)*100:.1f}%)',
                    ha='center', fontsize=8, color='gray')
        plt.tight_layout()
        plt.savefig('08_Repartition_Marche.png', dpi=300)
        plt.close()
        print(f"  Saved: 08_Repartition_Marche.png")

    # ── GRAPH 9: Sales by CONTINENT ───────────────────────────────────────────
    if 'CONTINENT' in df.columns and df['CONTINENT'].notna().sum() > 0:
        cont_counts = df['CONTINENT'].value_counts()
        plt.figure(figsize=(8, 6))
        colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12'][:len(cont_counts)]
        wedges, texts, autotexts = plt.pie(
            cont_counts.values,
            labels=cont_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        for t in autotexts:
            t.set_fontsize(10)
        plt.title('Ventes par Continent d\'Origine', fontsize=14, fontweight='bold')
        nn = df['CONTINENT'].notna().sum()
        plt.figtext(0.5, 0.01, f'Coverage: {nn:,} / {len(df):,} ({nn/len(df)*100:.1f}%)',
                    ha='center', fontsize=8, color='gray')
        plt.tight_layout()
        plt.savefig('09_Ventes_Par_Continent.png', dpi=300)
        plt.close()
        print(f"  Saved: 09_Ventes_Par_Continent.png")

    # ── GRAPH 10: Sales by GROUPE (distributor group) ─────────────────────────
    if 'GROUPE' in df.columns and df['GROUPE'].notna().sum() > 0:
        groupe_counts = df['GROUPE'].value_counts().head(10)
        # Shorten long group names for readability
        groupe_counts.index = groupe_counts.index.str.replace(r'\s*\(.*\)', '', regex=True).str.strip()
        plt.figure(figsize=(12, 7))
        ax = groupe_counts.plot(kind='barh', color='darkorange', edgecolor='saddlebrown')
        plt.title('Top 10 Groupes Distributeurs par Volume', fontsize=14, fontweight='bold')
        plt.xlabel('Nombre de véhicules')
        plt.ylabel('Groupe')
        plt.gca().invert_yaxis()
        for p in ax.patches:
            ax.annotate(f"{int(p.get_width()):,}",
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        va='center', ha='left', fontsize=8)
        nn = df['GROUPE'].notna().sum()
        plt.figtext(0.99, 0.01, f'Coverage: {nn:,} / {len(df):,} ({nn/len(df)*100:.1f}%)',
                    ha='right', fontsize=8, color='gray')
        plt.tight_layout()
        plt.savefig('10_Ventes_Par_Groupe.png', dpi=300)
        plt.close()
        print(f"  Saved: 10_Ventes_Par_Groupe.png")

    # ── PART 3: Missing months check ─────────────────────────────────────────
    print("\n--- Data Quality Checks ---")
    complete_range = pd.date_range(
        start=ventes['YEAR_MONTH'].min().replace(day=1),
        end=ventes['YEAR_MONTH'].max(),
        freq='MS'
    )
    missing_months = set(complete_range) - set(ventes['YEAR_MONTH'])
    if missing_months:
        print(f"  WARNING: {len(missing_months)} missing months:")
        for m in sorted(missing_months):
            print(f"    - {m.date()}")
    else:
        print(f"  OK: No missing months in data")

    missing_years = [y for y in [2019, 2020, 2021, 2022, 2023, 2024, 2025]
                     if y not in df['YEAR'].unique()]
    if missing_years:
        print(f"  WARNING: Missing years: {missing_years} -> request from ARTES")

    # ── PART 4: Key insights ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print(f"\n  Market:")
    if 'MARQUE' in df.columns:
        top = df['MARQUE'].value_counts()
        print(f"    Dominant brand    : {top.index[0]} ({top.values[0]/len(df)*100:.1f}% market share)")
        print(f"    Top 3 brands      : {top.values[:3].sum()/len(df)*100:.1f}% of total market")

    yearly = df.groupby('YEAR').size()
    if len(yearly) >= 2:
        trend = (yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] * 100
        print(f"\n  Trend ({yearly.index[0]}-{yearly.index[-1]}): {trend:+.1f}%")

    print(f"\n  Seasonality:")
    print(f"    Peak month   : {mois_fort}")
    print(f"    Weakest month: {mois_faible}")

    print(f"\n  Monthly stats:")
    print(f"    Mean   : {ventes['Ventes'].mean():.0f} vehicles")
    print(f"    Std Dev: {ventes['Ventes'].std():.0f}")
    print(f"    Min    : {ventes['Ventes'].min():.0f}")
    print(f"    Max    : {ventes['Ventes'].max():.0f}")
    print(f"    Outlier months: {len(outliers)}")

    print(f"\n  Enrichment coverage:")
    for col in ['SEGMENT', 'SOUS_SEGMENT', 'MARCHE_TYPE', 'GROUPE', 'PAYS_DORIGINE', 'CONTINENT']:
        if col in df.columns:
            nn = df[col].notna().sum()
            print(f"    {col:20s}: {nn/len(df)*100:.1f}%")

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    print("\nVisualizations saved:")
    for f in sorted(f for f in os.listdir('.') if f.endswith('.png') and f[0].isdigit()):
        print(f"  - {f}")

    print("\nSTEP 4 COMPLETE -> Run step5_preparation.py next")


if __name__ == '__main__':
    main()