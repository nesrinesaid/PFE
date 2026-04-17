import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# GARBAGE COLUMNS — from embedded Excel pivot tables, must be dropped
# ─────────────────────────────────────────────────────────────────────────────
GARBAGE_COLS = [
    'UNNAMED_30','UNNAMED_33','UNNAMED_34','UNNAMED_35','UNNAMED_36',
    'UNNAMED_37','UNNAMED_38','UNNAMED_39','UNNAMED_40','UNNAMED_41',
    'UNNAMED_42','UNNAMED_44','UNNAMED_45','UNNAMED_46','UNNAMED_47','UNNAMED_48',
    'TOUS','TOTAL','MOIS_11','DAT1',
    'VENTES_2019','VENTES_2020','VENTES_2021','VENTES_2022','VENTES_2023','VENTES_2024','VENTES_2025',
    'MARQUE.1','MARQUE_1','DMC.1','DAT_IMMATR',
    'CHA','SSIS',
    'MODLE_POPULAIRE','MODLES','MOIS',
    # Pre-aggregated enrichment columns from pivot — replaced by proper joins below
    'SEGMENT_','MODELE','SOCIT','MARCH','POSITIONS','GPS_COORDS',
]

# ─────────────────────────────────────────────────────────────────────────────
# FINAL OUTPUT — exactly 22 columns
# ─────────────────────────────────────────────────────────────────────────────
FINAL_COLS = [
    'ID','DATV','MARQUE','MODELE','GENRE','USAGE',
    'CD_VILLE','VILLE','ENERGIE','PUISSANCE',
    'MARCHE_TYPE','SOCIETE','DATE_MEC',
    'YEAR','MONTH','YEAR_MONTH',
    'SEGMENT','SOUS_SEGMENT',
    'PAYS_DORIGINE','CONTINENT',
    'GROUPE','DISTRIBUTEUR',
]


def norm(s):
    """Normalize string for joining: strip, uppercase"""
    return s.astype(str).str.strip().str.upper()


def main():
    print("🔗 STEP 2.5 — DATA ENRICHMENT\n")

    project_root = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(project_root, 'data_intermediate.csv')
    seg_file   = os.path.join(project_root, 'data', 'segmentation.xlsx')

    if not os.path.exists(input_file):
        print(f"❌ ERROR: {input_file} not found. Run step3_cleaning.py first.")
        return
    if not os.path.exists(seg_file):
        print(f"❌ ERROR: {seg_file} not found.")
        return

    # ── STEP 1: Load raw intermediate ──────────────────────────────────────────
    print(f"⏳ Loading {os.path.basename(input_file)}...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"  Raw shape: {df.shape}")
    initial_rows = len(df)

    # ── STEP 2: Drop garbage columns ───────────────────────────────────────────
    to_drop = [c for c in GARBAGE_COLS if c in df.columns]
    to_drop += [c for c in df.columns if c.startswith('UNNAMED_')]
    to_drop = list(set(to_drop))
    df = df.drop(columns=to_drop)
    print(f"  After garbage drop: {df.shape[1]} columns")

    # ── STEP 3: Fix core columns ───────────────────────────────────────────────
    print("\n⏳ Fixing core columns...")
    
    # GENRE: the text label is in CD_GENRE or GENRE column
    if 'CD_GENRE' in df.columns and 'GENRE' not in df.columns:
        df['GENRE'] = df['CD_GENRE']
    
    # MODELE: real commercial model name
    if 'TYP_COM' in df.columns:
        df['MODELE'] = df['TYP_COM']
    
    # Rename columns
    df = df.rename(columns={'SOCIT': 'SOCIETE', 'DMC': 'DATE_MEC'}, errors='ignore')
    
    # Ensure datetime
    df['DATV'] = pd.to_datetime(df['DATV'], errors='coerce')
    if 'DATE_MEC' in df.columns:
        df['DATE_MEC'] = pd.to_datetime(df['DATE_MEC'], errors='coerce')
    
    print(f"  ✅ Core columns fixed")

    # ── STEP 4: Create join keys ───────────────────────────────────────────────
    print("\n⏳ Creating join keys...")
    df['CD_TYP_CONS_K'] = norm(df['CD_TYP_CONS'].fillna(''))
    df['MARQUE_K']      = norm(df['MARQUE'].fillna(''))
    df['GENRE_K']       = norm(df['GENRE'].fillna(''))
    print(f"  ✅ Join keys created")
    print(f"  Clean working shape: {df.shape}")

    # ── STEP 5: Read segmentation.xlsx sheets ──────────────────────────────────
    print(f"\n⏳ Reading {os.path.basename(seg_file)}...")

    # Feuil2: CD_TYP_CONS → SEGMENT + SOUS_SEGMENT (direct, precise)
    try:
        df_f2 = pd.read_excel(seg_file, sheet_name='Feuil2', header=0)
        df_f2.columns = ['CD_TYP_CONS','MARQUE_F','MODELE_F','MODELE_POP','SEGMENT','SOUS_SEGMENT']
        df_f2 = df_f2.dropna(subset=['CD_TYP_CONS'])
        df_f2['CD_TYP_CONS_K'] = norm(df_f2['CD_TYP_CONS'])
        df_f2 = df_f2.drop_duplicates(subset=['CD_TYP_CONS_K'], keep='first')
        seg_f2_cov = df_f2['SEGMENT'].notna().sum() / len(df_f2) * 100
        print(f"  ✅ Feuil2:              {len(df_f2):,} codes  —  SEGMENT {seg_f2_cov:.1f}% within sheet")
    except Exception as e:
        print(f"  ⚠️  Feuil2 error: {e}")
        df_f2 = None

    # MODELE sheet: CD_TYP_CONS → MARQUE + MODELE (bridge for fallback)
    try:
        df_modele = pd.read_excel(seg_file, sheet_name='MODELE', header=0, usecols=[0,1,2])
        df_modele.columns = ['CD_TYP_CONS','MK','MLK']
        df_modele = df_modele.dropna(subset=['CD_TYP_CONS'])
        df_modele['CD_TYP_CONS_K'] = norm(df_modele['CD_TYP_CONS'])
        df_modele['MK2'] = norm(df_modele['MK'])
        df_modele['ML2'] = norm(df_modele['MLK'])
        df_modele = df_modele.drop_duplicates(subset=['CD_TYP_CONS_K'], keep='first')
        print(f"  ✅ MODELE bridge:       {len(df_modele):,} codes")
    except Exception as e:
        print(f"  ⚠️  MODELE error: {e}")
        df_modele = None

    # Segmentation sheet: MARQUE+MODELE → SEGMENT + SOUS_SEGMENT (fallback)
    try:
        df_seg = pd.read_excel(seg_file, sheet_name='Segmentation', header=0)
        df_seg.columns = ['MARQUE','MODELE','SEGMENT','SOUS_SEGMENT','C5','C6','PU']
        df_seg = df_seg[['MARQUE','MODELE','SEGMENT','SOUS_SEGMENT']].copy()
        df_seg['MK2'] = norm(df_seg['MARQUE'])
        df_seg['ML2'] = norm(df_seg['MODELE'])
        df_seg = df_seg.drop_duplicates(subset=['MK2','ML2'], keep='first')
        seg_cov = df_seg['SEGMENT'].notna().sum() / len(df_seg) * 100
        print(f"  ✅ Segmentation:        {len(df_seg):,} codes  —  SEGMENT {seg_cov:.1f}% within sheet")
    except Exception as e:
        print(f"  ⚠️  Segmentation error: {e}")
        df_seg = None

    # Build fallback: CD_TYP_CONS → SEGMENT via MODELE→Segmentation
    if df_modele is not None and df_seg is not None:
        df_fallback = df_modele.merge(
            df_seg[['MK2','ML2','SEGMENT','SOUS_SEGMENT']],
            on=['MK2','ML2'], how='left'
        ).drop_duplicates(subset=['CD_TYP_CONS_K'], keep='first')
        fb_cov = df_fallback['SEGMENT'].notna().sum() / len(df_fallback) * 100
        print(f"  ✅ Fallback lookup:     {len(df_fallback):,} codes  —  SEGMENT {fb_cov:.1f}% within sheet")
    else:
        df_fallback = None

    # Groupe: MARQUE → GROUPE + DISTRIBUTEUR
    try:
        df_grp = pd.read_excel(seg_file, sheet_name='Groupe', header=None)
        df_grp.columns = ['MARQUE','GROUPE','DISTRIBUTEUR']
        df_grp = df_grp.dropna(subset=['MARQUE'])
        df_grp['MARQUE_K'] = norm(df_grp['MARQUE'])
        df_grp = df_grp.drop_duplicates(subset=['MARQUE_K'], keep='first')
        print(f"  ✅ Groupe:              {len(df_grp):,} brands")
    except Exception as e:
        print(f"  ⚠️  Groupe error: {e}")
        df_grp = None

    # Feuil3: MARQUE → PAYS_DORIGINE + CONTINENT
    try:
        df_f3 = pd.read_excel(seg_file, sheet_name='Feuil3', header=0)
        df_f3.columns = ['MARQUE','MARQUE_PLUS','PAYS_DORIGINE','CONTINENT']
        df_f3 = df_f3[['MARQUE','PAYS_DORIGINE','CONTINENT']].copy()
        df_f3['MARQUE_K'] = norm(df_f3['MARQUE'])
        df_f3 = df_f3.drop_duplicates(subset=['MARQUE_K'], keep='first')
        print(f"  ✅ Feuil3:              {len(df_f3):,} brands")
    except Exception as e:
        print(f"  ⚠️  Feuil3 error: {e}")
        df_f3 = None

    # CD GENRE: GENRE text label → MARCHE_TYPE
    try:
        df_cdg = pd.read_excel(seg_file, sheet_name='CD GENRE', header=0)
        df_cdg.columns = ['GENRE_LABEL','CD_GENRE_NUM','MARCHE_TYPE']
        df_cdg['GENRE_K'] = norm(df_cdg['GENRE_LABEL'].fillna(''))
        df_cdg = df_cdg.dropna(subset=['GENRE_LABEL'])
        df_cdg = df_cdg.drop_duplicates(subset=['GENRE_K'], keep='first')
        print(f"  ✅ CD GENRE:            {len(df_cdg):,} genre codes")
    except Exception as e:
        print(f"  ⚠️  CD GENRE error: {e}")
        df_cdg = None

    # ── STEP 6: Apply enrichments ──────────────────────────────────────────────
    print("\n⏳ Applying enrichments...\n")

    # SEGMENT / SOUS_SEGMENT — Pass 1: Feuil2 direct join
    if df_f2 is not None:
        df = df.merge(df_f2[['CD_TYP_CONS_K','SEGMENT','SOUS_SEGMENT']],
                      on='CD_TYP_CONS_K', how='left', suffixes=('', '_f2'))
        p1 = df['SEGMENT'].notna().sum()
        print(f"  ✅ Pass 1 (Feuil2):    SEGMENT={p1:,} ({p1/initial_rows*100:.1f}%)")

    # SEGMENT / SOUS_SEGMENT — Pass 2: fallback for remaining NaN
    if df_fallback is not None:
        mask_need = df['SEGMENT'].isna()
        if mask_need.sum() > 0:
            # Get only rows that need segment
            df_need = df.loc[mask_need, ['CD_TYP_CONS_K']].copy()
            df_need = df_need.merge(
                df_fallback[['CD_TYP_CONS_K','SEGMENT','SOUS_SEGMENT']],
                on='CD_TYP_CONS_K', how='left', suffixes=('', '_fb')
            )
            # Update main df using iloc positional indexing
            need_indices = df[mask_need].index
            df.loc[need_indices, 'SEGMENT'] = df_need['SEGMENT'].values
            df.loc[need_indices, 'SOUS_SEGMENT'] = df_need['SOUS_SEGMENT'].values
        
        p2 = df['SEGMENT'].notna().sum()
        print(f"  ✅ Pass 2 (fallback):  SEGMENT={p2:,} ({p2/initial_rows*100:.1f}%)")

    # GROUPE + DISTRIBUTEUR
    if df_grp is not None:
        df = df.merge(df_grp[['MARQUE_K','GROUPE','DISTRIBUTEUR']],
                      on='MARQUE_K', how='left', suffixes=('', '_grp'))
        n_groupe = df['GROUPE'].notna().sum()
        print(f"  ✅ GROUPE:             {n_groupe:,} ({n_groupe/initial_rows*100:.1f}%)")

    # PAYS_DORIGINE + CONTINENT
    if df_f3 is not None:
        df = df.merge(df_f3[['MARQUE_K','PAYS_DORIGINE','CONTINENT']],
                      on='MARQUE_K', how='left', suffixes=('', '_f3'))
        n_pays = df['PAYS_DORIGINE'].notna().sum()
        print(f"  ✅ CONTINENT:         {n_pays:,} ({n_pays/initial_rows*100:.1f}%)")

    # MARCHE_TYPE via GENRE text label
    if df_cdg is not None:
        df = df.merge(df_cdg[['GENRE_K','MARCHE_TYPE']],
                      on='GENRE_K', how='left', suffixes=('', '_cdg'))
        n_marche = df['MARCHE_TYPE'].notna().sum()
        print(f"  ✅ MARCHE_TYPE:        {n_marche:,} ({n_marche/initial_rows*100:.1f}%)")

    # ── STEP 7: Drop helper join keys ──────────────────────────────────────────
    df = df.drop(columns=['CD_TYP_CONS_K','MARQUE_K','GENRE_K'], errors='ignore')
    
    # Drop any duplicate suffixed columns from merges
    cols_to_drop = [c for c in df.columns if c.endswith(('_f2', '_fb', '_grp', '_f3', '_cdg'))]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # ── STEP 8: Fix data types ────────────────────────────────────────────────
    str_cols = ['MARQUE','MODELE','GENRE','USAGE','VILLE','ENERGIE','MARCHE_TYPE',
                'SOCIETE','YEAR_MONTH','SEGMENT','SOUS_SEGMENT',
                'PAYS_DORIGINE','CONTINENT','GROUPE','DISTRIBUTEUR']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].replace({'nan': np.nan, 'NAN': np.nan, '': np.nan})

    for col in ['PUISSANCE','CD_VILLE']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── STEP 9: Select final 22 columns ────────────────────────────────────────
    available = [c for c in FINAL_COLS if c in df.columns]
    missing   = [c for c in FINAL_COLS if c not in df.columns]
    
    for col in missing:
        df[col] = np.nan
    
    df_out = df[FINAL_COLS].copy()

    # ── STEP 10: Validation ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("📊 FINAL VALIDATION")
    print("="*70)
    print(f"  Rows      : {len(df_out):,}")
    print(f"  Columns   : {len(df_out.columns)}")
    print(f"  Duplicate IDs: {df_out.duplicated('ID').sum()}")
    
    print(f"\n  📋 Column coverage:")
    for col in df_out.columns:
        nn  = df_out[col].notna().sum()
        pct = nn / len(df_out) * 100
        dtype_str = str(df_out[col].dtype)
        print(f"    {col:20s}  {dtype_str:12s}  {pct:6.1f}%")

    print(f"\n  📄 Sample enriched row (first with SEGMENT):")
    sample = df_out[df_out['SEGMENT'].notna()].head(1)
    if len(sample) > 0:
        for col in sample.columns:
            val = sample[col].iloc[0]
            if pd.notna(val):
                print(f"    {col:20s} = {val}")

    # ── STEP 11: Save ──────────────────────────────────────────────────────────
    output_file = os.path.join(project_root, 'data_cleaned_enriched.csv')
    df_out.to_csv(output_file, index=False)
    mb = os.path.getsize(output_file) / 1024 / 1024
    
    print(f"\n✅ ENRICHMENT COMPLETE!")
    print(f"   Output: {output_file}")
    print(f"   Rows: {len(df_out):,}")
    print(f"   Columns: {len(df_out.columns)}")
    print(f"   Size: {mb:.1f} MB")
    print(f"\n🚀 Next: Run step4_eda.py")


if __name__ == '__main__':
    main()