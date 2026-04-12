import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

FINAL_COLS = [
    'ID', 'DATV', 'MARQUE', 'MODELE', 'GENRE', 'USAGE',
    'CD_VILLE', 'VILLE', 'ENERGIE', 'PUISSANCE',
    'MARCHE_TYPE', 'SOCIETE', 'DATE_MEC',
    'YEAR', 'MONTH', 'YEAR_MONTH',
    'SEGMENT', 'SOUS_SEGMENT',
    'PAYS_DORIGINE', 'CONTINENT',
    'GROUPE', 'DISTRIBUTEUR',
]

COLS_TO_DROP_BEFORE_ENRICH = [
    'SEGMENT', 'SEGMENT_', 'SOUS_SEGMENT',
    'GROUPE', 'DISTRIBUTEUR',
    'PAYS_DORIGINE', 'CONTINENT',
    'MARQUE.1', 'SOCIT',
]

def norm(series):
    return series.astype(str).str.strip().str.upper()

def main():
    print("STEP 2.5 - DATA ENRICHMENT\n")
    print("Join strategy: CD_TYP_CONS -> MODELE sheet -> Segmentation -> Groupe -> Feuil3\n")

    input_file = 'data_intermediate.csv'
    seg_file   = './data/segmentation.xlsx'

    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found. Run step3_cleaning.py first.")
        return
    if not os.path.exists(seg_file):
        print(f"ERROR: {seg_file} not found.")
        return

    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file, parse_dates=['DATV'], low_memory=False)
    print(f"  Shape: {df.shape}")

    drop_existing = [c for c in COLS_TO_DROP_BEFORE_ENRICH if c in df.columns]
    df = df.drop(columns=drop_existing)
    print(f"  Dropped {len(drop_existing)} pre-existing enrichment columns")

    print("\nReading segmentation.xlsx sheets...")

    # MODELE sheet: CD_TYP_CONS -> MARQUE + MODELE
    df_modele = pd.read_excel(seg_file, sheet_name='MODELE', header=0, usecols=[0, 1, 2])
    df_modele.columns = ['CD_TYP_CONS', 'MARQUE_LK', 'MODELE_LK']
    df_modele = df_modele.dropna(subset=['CD_TYP_CONS'])
    df_modele['CD_TYP_CONS_K'] = norm(df_modele['CD_TYP_CONS'])
    df_modele['MARQUE_K']      = norm(df_modele['MARQUE_LK'])
    df_modele['MODELE_K']      = norm(df_modele['MODELE_LK'])
    df_modele = df_modele.drop_duplicates(subset=['CD_TYP_CONS_K'], keep='first')
    print(f"  OK MODELE sheet: {len(df_modele):,} unique CD_TYP_CONS codes")

    # Segmentation sheet: (MARQUE, MODELE) -> SEGMENT + SOUS_SEGMENT
    df_seg = pd.read_excel(seg_file, sheet_name='Segmentation', header=0)
    df_seg.columns = ['MARQUE', 'MODELE', 'SEGMENT', 'SOUS_SEGMENT', 'COL5', 'COL6', 'PU']
    df_seg = df_seg[['MARQUE', 'MODELE', 'SEGMENT', 'SOUS_SEGMENT']].copy()
    df_seg['MARQUE_K'] = norm(df_seg['MARQUE'])
    df_seg['MODELE_K'] = norm(df_seg['MODELE'])
    df_seg = df_seg.drop_duplicates(subset=['MARQUE_K', 'MODELE_K'], keep='first')
    print(f"  OK Segmentation sheet: {len(df_seg):,} entries")

    # Groupe sheet: MARQUE -> GROUPE + DISTRIBUTEUR
    df_groupe = pd.read_excel(seg_file, sheet_name='Groupe', header=None)
    df_groupe.columns = ['MARQUE', 'GROUPE', 'DISTRIBUTEUR']
    df_groupe = df_groupe.dropna(subset=['MARQUE'])
    df_groupe['MARQUE_K'] = norm(df_groupe['MARQUE'])
    df_groupe = df_groupe.drop_duplicates(subset=['MARQUE_K'], keep='first')
    print(f"  OK Groupe sheet: {len(df_groupe):,} brands")

    # Feuil3 sheet: MARQUE -> PAYS_DORIGINE + CONTINENT
    df_feuil3 = pd.read_excel(seg_file, sheet_name='Feuil3', header=0)
    df_feuil3.columns = ['MARQUE', 'MARQUE_PLUS', 'PAYS_DORIGINE', 'CONTINENT']
    df_feuil3 = df_feuil3[['MARQUE', 'PAYS_DORIGINE', 'CONTINENT']].copy()
    df_feuil3['MARQUE_K'] = norm(df_feuil3['MARQUE'])
    df_feuil3 = df_feuil3.drop_duplicates(subset=['MARQUE_K'], keep='first')
    print(f"  OK Feuil3 sheet: {len(df_feuil3):,} brands")

    # CD GENRE sheet: CD_GENRE numeric -> MARCHE_TYPE
    df_cd_genre = pd.read_excel(seg_file, sheet_name='CD GENRE', header=0)
    df_cd_genre.columns = ['GENRE_LABEL', 'CD_GENRE_NUM', 'MARCHE_TYPE']
    df_cd_genre['CD_GENRE_NUM'] = pd.to_numeric(df_cd_genre['CD_GENRE_NUM'], errors='coerce')
    df_cd_genre = df_cd_genre.dropna(subset=['CD_GENRE_NUM'])
    df_cd_genre = df_cd_genre.drop_duplicates(subset=['CD_GENRE_NUM'], keep='first')
    print(f"  OK CD GENRE sheet: {len(df_cd_genre):,} genre codes")

    # Build master lookup: CD_TYP_CONS -> all enrichment fields
    print("\nBuilding master lookup table...")
    df_lookup = df_modele.merge(
        df_seg[['MARQUE_K', 'MODELE_K', 'SEGMENT', 'SOUS_SEGMENT']],
        on=['MARQUE_K', 'MODELE_K'], how='left'
    )
    df_lookup = df_lookup.merge(
        df_groupe[['MARQUE_K', 'GROUPE', 'DISTRIBUTEUR']],
        on='MARQUE_K', how='left'
    )
    df_lookup = df_lookup.merge(
        df_feuil3[['MARQUE_K', 'PAYS_DORIGINE', 'CONTINENT']],
        on='MARQUE_K', how='left'
    )
    df_lookup = df_lookup.drop_duplicates(subset=['CD_TYP_CONS_K'], keep='first')
    print(f"  Master lookup: {len(df_lookup):,} unique vehicle type codes")
    seg_cov = df_lookup['SEGMENT'].notna().sum()
    print(f"  SEGMENT in lookup: {seg_cov:,} ({seg_cov/len(df_lookup)*100:.1f}%)")

    # Apply enrichments to main data
    print("\nApplying enrichments to registration data...")

    df['CD_TYP_CONS_K'] = norm(df['CD_TYP_CONS'].fillna(''))

    enrich_cols = ['CD_TYP_CONS_K', 'SEGMENT', 'SOUS_SEGMENT',
                   'GROUPE', 'DISTRIBUTEUR', 'PAYS_DORIGINE', 'CONTINENT']
    df = df.merge(df_lookup[enrich_cols], on='CD_TYP_CONS_K', how='left')
    if len(df) != 445513:
        print(f"  WARNING: Row count changed to {len(df):,}! Fixing...")
        df = df.drop_duplicates(subset=['ID'], keep='first')
    print(f"  Row count after CD_TYP_CONS join: {len(df):,}")

    # MARCHE_TYPE via CD_GENRE numeric
    df['CD_GENRE_NUM'] = pd.to_numeric(df['CD_GENRE'].fillna(-1), errors='coerce')
    df = df.merge(
        df_cd_genre[['CD_GENRE_NUM', 'MARCHE_TYPE']],
        on='CD_GENRE_NUM', how='left'
    )
    if len(df) != 445513:
        df = df.drop_duplicates(subset=['ID'], keep='first')

    # Rename source columns to final names
    rename_map = {'SOCIT': 'SOCIETE', 'DMC': 'DATE_MEC'}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # If MARCHE_TYPE mostly null, fill from MARCH (source column renamed by step3)
    if 'MARCH' in df.columns:
        nn_mt = df['MARCHE_TYPE'].notna().sum()
        if nn_mt / len(df) < 0.5:
            df['MARCHE_TYPE'] = df['MARCHE_TYPE'].fillna(df['MARCH'])
        df = df.drop(columns=['MARCH'])

    # Drop helper columns
    df = df.drop(columns=['CD_TYP_CONS_K', 'CD_GENRE_NUM', 'MARQUE_K'], errors='ignore')

    # Select final columns
    available = [c for c in FINAL_COLS if c in df.columns]
    missing   = [c for c in FINAL_COLS if c not in df.columns]
    if missing:
        print(f"\n  NOTE: Some columns absent from source data: {missing}")

    df_out = df[available].copy()

    # Summary
    print("\n" + "=" * 60)
    print("ENRICHMENT SUMMARY")
    print("=" * 60)
    print(f"  Rows      : {len(df_out):,}")
    print(f"  Columns   : {len(df_out.columns)} {list(df_out.columns)}")
    print(f"  Duplicates: {df_out.duplicated('ID').sum()}")

    print(f"\n  Coverage per enrichment field:")
    for col in ['SEGMENT', 'SOUS_SEGMENT', 'MARCHE_TYPE', 'GROUPE', 'DISTRIBUTEUR', 'PAYS_DORIGINE', 'CONTINENT']:
        if col in df_out.columns:
            nn = df_out[col].notna().sum()
            bar = 'X' * int(nn / len(df_out) * 20)
            print(f"    {col:20s}: {nn:,} ({nn/len(df_out)*100:.1f}%) [{bar}]")

    print(f"\n  Year distribution:")
    df_out['_Y'] = pd.to_datetime(df_out['DATV']).dt.year
    for yr, cnt in df_out['_Y'].value_counts().sort_index().items():
        print(f"    {yr}: {cnt:,} vehicles")
    df_out = df_out.drop(columns=['_Y'])

    missing_years = [y for y in [2019,2020,2021,2022,2023,2024,2025]
                     if y not in pd.to_datetime(df_out['DATV']).dt.year.unique()]
    if missing_years:
        print(f"\n  WARNING - Missing years: {missing_years}")
        print(f"  -> Request these files from M. Sami Ben Youssef (ARTES)")

    output_file = 'data_cleaned_enriched.csv'
    df_out.to_csv(output_file, index=False)
    size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"\nSaved: {output_file}")
    print(f"  Rows: {len(df_out):,} | Columns: {len(df_out.columns)} | Size: {size_mb:.1f} MB")
    print("\nSTEP 2.5 COMPLETE -> Run step4_eda.py next")

if __name__ == '__main__':
    main()