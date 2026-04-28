import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from valider_pipeline import valider_colonnes, COLONNES_REQUISES
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")


def identify_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR


def main():
    print("STEP 4 - EXPLORATORY DATA ANALYSIS\n")

    project_root = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(project_root, 'data_cleaned_enriched.csv')
    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found. Run step2_5_enrich_data.py first.")
        return

    print("Loading data...")
    df = pd.read_csv(input_file, parse_dates=['DATV'])

    # ─── VALIDATE COLONNE IM_RI ───
    if 'IM_RI' not in df.columns:
        print("❌ ERREUR: Colonne IM_RI non trouvée dans data_cleaned_enriched.csv")
        print("   IM_RI est requise pour identifier les ventes neufs (IM_RI=10)")
        print("   Action: Relancer step2_5_enrich_data.py pour ajouter la colonne IM_RI")
        return
    df['IM_RI'] = pd.to_numeric(df['IM_RI'], errors='coerce').round().astype('Int64')
    if df['IM_RI'].isna().all():
        print("❌ ERREUR: Colonne IM_RI est all NaN apres coercion")
        print("   Probleme de qualite donnees: Verifier IM_RI dans data_cleaned_enriched.csv")
        return

    print("✅ Validation IM_RI reussie")
    print(f"   IM_RI=10 (ventes neufs):     {(df['IM_RI']==10).sum():,} enregistrements")
    print(f"   IM_RI=20 (ventes occasion):  {(df['IM_RI']==20).sum():,} enregistrements")

    # French temporal aliases for consistency.
    if 'ANNEE' in df.columns and 'YEAR' not in df.columns:
        df['YEAR'] = pd.to_numeric(df['ANNEE'], errors='coerce')
    if 'MOIS' in df.columns and 'MONTH' not in df.columns:
        df['MONTH'] = pd.to_numeric(df['MOIS'], errors='coerce')
    if 'ANNEE_MOIS' in df.columns and 'YEAR_MONTH' not in df.columns:
        df['YEAR_MONTH'] = df['ANNEE_MOIS'].astype(str)

    # ─── VALIDER ENTRÉE ───
    try:
        valider_colonnes(df, COLONNES_REQUISES['step4_eda'], 'step4_eda', verbose=True)
    except ValueError as e:
        print(str(e))
        return

    df = df.rename(columns={'DATV': 'DAT_V'})

    if 'YEAR_MONTH' not in df.columns:
        df['YEAR_MONTH'] = df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str).str.zfill(2)

    print(f"  Shape: {df.shape}")

    # Comparison scope (all records) and analysis scope (new vehicles only)
    df_neuf = df[df['IM_RI'] == 10].copy()
    df_occ = df[df['IM_RI'] == 20].copy()
    if df_neuf.empty:
        print("ERROR: No records with IM_RI=10 (ventes neufs).")
        return

    print(f"  Ventes neufs (IM_RI=10)     : {len(df_neuf):,}")
    print(f"  Ventes d'occasion (IM_RI=20): {len(df_occ):,}")

    # ── PART 1: General statistics ────────────────────────────────────────────
    print("\n--- General Statistics ---")
    print(f"  Total vehicles (all IM_RI) : {len(df):,}")
    print(f"  Vehicles analyzed (neufs)  : {len(df_neuf):,}")
    print(f"  Date range (neufs)         : {df_neuf['DAT_V'].min().date()} -> {df_neuf['DAT_V'].max().date()}")
    print(f"  Distinct years (neufs)     : {sorted(df_neuf['YEAR'].unique())}")
    print(f"  Unique brands (neufs)      : {df_neuf['MARQUE'].nunique()}")

    # ── GRAPH 0: Neuf vs Occasion over time ──────────────────────────────────
    print("\nBuilding monthly sales metrics...")
    ventes_cmp = (df.groupby(['YEAR_MONTH', 'IM_RI'])
                    .size()
                    .reset_index(name='Ventes'))
    ventes_cmp['YEAR_MONTH'] = pd.to_datetime(ventes_cmp['YEAR_MONTH'], errors='coerce')
    ventes_cmp = ventes_cmp.dropna(subset=['YEAR_MONTH'])
    ventes_pivot = (ventes_cmp.pivot_table(index='YEAR_MONTH', columns='IM_RI', values='Ventes', fill_value=0)
                              .sort_index())

    plt.figure(figsize=(14, 6))
    if 10 in ventes_pivot.columns:
        plt.plot(ventes_pivot.index, ventes_pivot[10], marker='o', linewidth=2, markersize=3,
                 label='Ventes Neufs (IM_RI=10)', color='#1976D2')
    if 20 in ventes_pivot.columns:
        plt.plot(ventes_pivot.index, ventes_pivot[20], marker='o', linewidth=2, markersize=3,
                 label="Ventes d'occasion (IM_RI=20)", color='#EF6C00')
    plt.title("Ventes Neufs vs Ventes d'Occasion dans le Temps", fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Nombre de véhicules')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, '00_Ventes_Neufs_vs_Occasion.png'), dpi=300)
    plt.close()
    print("  Saved: 00_Ventes_Neufs_vs_Occasion.png")

    # ── PART 2: Monthly sales metric (neufs only) ────────────────────────────
    ventes = df_neuf.groupby('YEAR_MONTH').size().reset_index(name='Ventes')
    ventes['YEAR_MONTH'] = pd.to_datetime(ventes['YEAR_MONTH'], errors='coerce')
    ventes['YEAR']  = ventes['YEAR_MONTH'].dt.year
    ventes['MONTH'] = ventes['YEAR_MONTH'].dt.month
    ventes = ventes.dropna(subset=['YEAR_MONTH']).sort_values('YEAR_MONTH')

    # ── GRAPH 1: Sales over time ──────────────────────────────────────────────
    plt.figure(figsize=(14, 6))
    plt.plot(ventes['YEAR_MONTH'], ventes['Ventes'],
             marker='o', linestyle='-', color='steelblue', linewidth=2, markersize=4)
    plt.title('Ventes de Véhicules Neufs dans le Temps (IM_RI=10)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Nombre de véhicules')
    for year in ventes['YEAR_MONTH'].dt.year.unique():
        plt.axvline(pd.to_datetime(f'{year}-01-01'), color='gray', linestyle='--', alpha=0.4)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, '01_Ventes_Over_Time.png'), dpi=300)
    plt.close()
    print("  Saved: 01_Ventes_Over_Time.png")

    # ── GRAPH 2: Sales by year ────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    ventes_annee = df_neuf.groupby('YEAR').size()
    ax = ventes_annee.plot(kind='bar', color='steelblue', edgecolor='navy')
    plt.title('Ventes Neufs par Année (IM_RI=10)', fontsize=14, fontweight='bold')
    plt.xlabel('Année')
    plt.ylabel('Total de véhicules')
    plt.xticks(rotation=0)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, '02_Ventes_Par_Annee.png'), dpi=300)
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
    plt.savefig(os.path.join(project_root, '03_Saisonnalite.png'), dpi=300)
    plt.close()
    print(f"  Saved: 03_Saisonnalite.png  (peak: month {mois_fort}, low: month {mois_faible})")

    # ── GRAPH 4: Top 10 brands ────────────────────────────────────────────────
    if 'MARQUE' in df_neuf.columns:
        top_marques = df_neuf['MARQUE'].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        ax = top_marques.plot(kind='barh', color='coral', edgecolor='darkred')
        plt.title('Top 10 Marques (Ventes Neufs, IM_RI=10)', fontsize=14, fontweight='bold')
        plt.xlabel('Nombre de véhicules')
        plt.ylabel('Marque')
        plt.gca().invert_yaxis()
        for p in ax.patches:
            ax.annotate(f"{int(p.get_width()):,}",
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        va='center', ha='left', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(project_root, '04_Top_Marques.png'), dpi=300)
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
    plt.savefig(os.path.join(project_root, '05_BoxPlot_Outliers.png'), dpi=300)
    plt.close()
    print(f"  Saved: 05_BoxPlot_Outliers.png  ({len(outliers)} outlier months)")

    # ── GRAPH 6: Sales by SEGMENT ─────────────────────────────────────────────
    if 'SEGMENT' in df_neuf.columns and df_neuf['SEGMENT'].notna().sum() > 0:
        seg_counts = df_neuf['SEGMENT'].value_counts().head(12)
        plt.figure(figsize=(12, 6))
        ax = seg_counts.plot(kind='barh', color='teal', edgecolor='darkcyan')
        plt.title('Ventes par  sous_segment de Véhicule', fontsize=14, fontweight='bold')
        plt.xlabel('Nombre de véhicules')
        plt.ylabel('Segment')
        plt.gca().invert_yaxis()
        for p in ax.patches:
            ax.annotate(f"{int(p.get_width()):,}",
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        va='center', ha='left', fontsize=8)
        nn = df_neuf['SEGMENT'].notna().sum()
        plt.figtext(0.99, 0.01, f'Coverage: {nn:,} / {len(df_neuf):,} ({nn/len(df_neuf)*100:.1f}%)',
                    ha='right', fontsize=8, color='gray')
        plt.tight_layout()
        plt.savefig(os.path.join(project_root, '06_Ventes_Par_Segment.png'), dpi=300)
        plt.close()
        print(f"  Saved: 06_Ventes_Par_Segment.png")

    # ── GRAPH 7: Sales by SOUS_SEGMENT ───────────────────────────────────────
    if 'SOUS_SEGMENT' in df_neuf.columns and df_neuf['SOUS_SEGMENT'].notna().sum() > 0:
        sous_counts = df_neuf['SOUS_SEGMENT'].value_counts().head(12)
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
        nn = df_neuf['SOUS_SEGMENT'].notna().sum()
        plt.figtext(0.99, 0.01, f'Coverage: {nn:,} / {len(df_neuf):,} ({nn/len(df_neuf)*100:.1f}%)',
                    ha='right', fontsize=8, color='gray')
        plt.tight_layout()
        plt.savefig(os.path.join(project_root, '07_Ventes_Par_Sous_Segment.png'), dpi=300)
        plt.close()
        print(f"  Saved: 07_Ventes_Par_Sous_Segment.png")

    # ── GRAPH 8: Sales by Marché (VP / VU / Autre) ───────────────────────────
    colonne_marche = 'TYPE_MARCHE'
    if colonne_marche not in df_neuf.columns:
        print("❌ ERREUR: Colonne TYPE_MARCHE non trouvee")
        print(f"   Colonnes disponibles: {list(df_neuf.columns)}")
        return

    if df_neuf[colonne_marche].notna().sum() > 0:
        marche_counts = df_neuf[colonne_marche].value_counts()
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
        plt.title('Répartition par Type de Marché (Neufs, IM_RI=10)',
                  fontsize=14, fontweight='bold')
        nn = df_neuf[colonne_marche].notna().sum()
        plt.figtext(0.5, 0.01, f'Coverage: {nn:,} / {len(df_neuf):,} ({nn/len(df_neuf)*100:.1f}%)',
                    ha='center', fontsize=8, color='gray')
        plt.tight_layout()
        plt.savefig(os.path.join(project_root, '08_Repartition_Marche.png'), dpi=300)
        plt.close()
        print(f"  Saved: 08_Repartition_Marche.png")

    # ── GRAPH 9: Sales by CONTINENT ───────────────────────────────────────────
    if 'CONTINENT' in df_neuf.columns and df_neuf['CONTINENT'].notna().sum() > 0:
        cont_counts = df_neuf['CONTINENT'].value_counts()
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
        plt.title('Ventes Neufs par Continent d\'Origine', fontsize=14, fontweight='bold')
        nn = df_neuf['CONTINENT'].notna().sum()
        plt.figtext(0.5, 0.01, f'Coverage: {nn:,} / {len(df_neuf):,} ({nn/len(df_neuf)*100:.1f}%)',
                    ha='center', fontsize=8, color='gray')
        plt.tight_layout()
        plt.savefig(os.path.join(project_root, '09_Ventes_Par_Continent.png'), dpi=300)
        plt.close()
        print(f"  Saved: 09_Ventes_Par_Continent.png")

    # ── GRAPH 10: Sales by GROUPE (distributor group) ─────────────────────────
    if 'GROUPE' in df_neuf.columns and df_neuf['GROUPE'].notna().sum() > 0:
        groupe_counts = df_neuf['GROUPE'].value_counts().head(10)
        # Shorten long group names for readability
        groupe_counts.index = groupe_counts.index.str.replace(r'\s*\(.*\)', '', regex=True).str.strip()
        plt.figure(figsize=(12, 7))
        ax = groupe_counts.plot(kind='barh', color='darkorange', edgecolor='saddlebrown')
        plt.title('Top 10 Groupes Distributeurs (Neufs, IM_RI=10)', fontsize=14, fontweight='bold')
        plt.xlabel('Nombre de véhicules')
        plt.ylabel('Groupe')
        plt.gca().invert_yaxis()
        for p in ax.patches:
            ax.annotate(f"{int(p.get_width()):,}",
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        va='center', ha='left', fontsize=8)
        nn = df_neuf['GROUPE'].notna().sum()
        plt.figtext(0.99, 0.01, f'Coverage: {nn:,} / {len(df_neuf):,} ({nn/len(df_neuf)*100:.1f}%)',
                    ha='right', fontsize=8, color='gray')
        plt.tight_layout()
        plt.savefig(os.path.join(project_root, '10_Ventes_Par_Groupe.png'), dpi=300)
        plt.close()
        print(f"  Saved: 10_Ventes_Par_Groupe.png")

    # ── GRAPH 11: Evolution des marques ARTES (Renault, Nissan, Dacia) ─────
    if 'MARQUE' in df_neuf.columns:
        artes_marques = ['RENAULT', 'NISSAN', 'DACIA']
        df_artes = df_neuf.copy()
        df_artes['MARQUE_UP'] = df_artes['MARQUE'].astype(str).str.strip().str.upper()
        df_artes = df_artes[df_artes['MARQUE_UP'].isin(artes_marques)]

        if not df_artes.empty:
            artes_monthly = (df_artes.groupby(['YEAR_MONTH', 'MARQUE_UP'])
                                     .size()
                                     .reset_index(name='Ventes'))
            artes_monthly['YEAR_MONTH'] = pd.to_datetime(artes_monthly['YEAR_MONTH'], errors='coerce')
            artes_monthly = artes_monthly.dropna(subset=['YEAR_MONTH']).sort_values('YEAR_MONTH')

            plt.figure(figsize=(14, 6))
            for brand, color in [('RENAULT', '#1f77b4'), ('NISSAN', '#ff7f0e'), ('DACIA', '#2ca02c')]:
                b = artes_monthly[artes_monthly['MARQUE_UP'] == brand]
                if not b.empty:
                    plt.plot(b['YEAR_MONTH'], b['Ventes'], marker='o', linewidth=2, markersize=3,
                             label=brand.title(), color=color)

            plt.title('Evolution des Marques ARTES (Renault, Nissan, Dacia) - Ventes Neufs',
                      fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Nombre de véhicules neufs')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(project_root, '11_Evolution_Marques_ARTES.png'), dpi=300)
            plt.close()
            print("  Saved: 11_Evolution_Marques_ARTES.png")

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
                     if y not in df_neuf['YEAR'].unique()]
    if missing_years:
        print(f"  WARNING: Missing years: {missing_years} -> request from ARTES")

    # ── PART 4: Key insights ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print(f"\n  Market:")
    if 'MARQUE' in df_neuf.columns:
        top = df_neuf['MARQUE'].value_counts()
        print(f"    Dominant brand    : {top.index[0]} ({top.values[0]/len(df_neuf)*100:.1f}% market share among neufs)")
        print(f"    Top 3 brands      : {top.values[:3].sum()/len(df_neuf)*100:.1f}% of ventes neufs")

    yearly = df_neuf.groupby('YEAR').size()
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
    for col in ['SEGMENT', 'SOUS_SEGMENT', 'TYPE_MARCHE', 'GROUPE', 'PAYS_DORIGINE', 'CONTINENT']:
        if col in df_neuf.columns:
            nn = df_neuf[col].notna().sum()
            print(f"    {col:20s}: {nn/len(df_neuf)*100:.1f}%")

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    print("\nVisualizations saved:")
    for f in sorted(f for f in os.listdir(project_root) if f.endswith('.png') and f[0].isdigit()):
        print(f"  - {f}")

    print("\nSTEP 4 COMPLETE -> Run step5_preparation.py next")


if __name__ == '__main__':
    main()