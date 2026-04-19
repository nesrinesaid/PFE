import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# LOOKUP STRATEGY — two-pass for maximum SEGMENT coverage:
#
#   Pass 1 — Segment_par_Code_Type.xlsx (all sheets combined)
#             Direct CD_TYP_CONS → SEGMENT + SOUS_SEGMENT
#             2,051 unique codes, 100% with segment within lookup
#             Covers 53% of the 445K records
#
#   Pass 2 — segmentation.xlsx Segmentation sheet fallback
#             MARQUE + MODELE → SEGMENT + SOUS_SEGMENT
#             For records not matched in Pass 1
#             Total combined coverage: ~72%
#
# OTHER ENRICHMENTS (from segmentation.xlsx):
#   Groupe sheet:  MARQUE → GROUPE + DISTRIBUTEUR
#   Feuil3 sheet:  MARQUE → PAYS_DORIGINE + CONTINENT
#   CD GENRE sheet: GENRE text → MARCHE_TYPE (VP / VU / Autre)
# ─────────────────────────────────────────────────────────────────────────────

GARBAGE_COLS = [
    'UNNAMED_30','UNNAMED_33','UNNAMED_34','UNNAMED_35','UNNAMED_36',
    'UNNAMED_37','UNNAMED_38','UNNAMED_39','UNNAMED_40','UNNAMED_41',
    'UNNAMED_42','UNNAMED_44','UNNAMED_45','UNNAMED_46','UNNAMED_47','UNNAMED_48',
    'TOUS','TOTAL','MOIS_11','DAT1',
    'VENTES_2019','VENTES_2021','VENTES_2023','VENTES_2024','VENTES_2025',
    'MARQUE.1','MARQUE_1','DMC.1','DAT_IMMATR','CHA','SSIS',
    'MODLE_POPULAIRE','MODLES','MOIS',
    # Pre-existing enrichment cols from pivot tables — replaced by proper joins
    'SEGMENT_','SEGMENT','GROUPE','PAYS_DORIGINE','CONTINENT',
    'MODELE','SOCIT','MARCH','POSITIONS','GPS_COORDS',
]

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
    return s.astype(str).str.strip().str.upper()


def safe_merge(df, right, on, new_cols, label, n):
    """Left-join guaranteeing row count stays at n."""
    key = [on] if isinstance(on, str) else on
    df = df.merge(right[key + new_cols], on=on, how='left')
    if len(df) != n:
        print(f"  WARNING [{label}]: rows={len(df):,} — deduplicating...")
        df = df.drop_duplicates(subset=['ID'], keep='first')
    return df


def build_master_lookup(new_file):
    """Build master CD_TYP_CONS lookup from all sheets in Segment_par_Code_Type.xlsx."""
    parts = []

    # Year sheets + Feuil5: 6 cols in order CD_TYP_CONS, MARQUE, MODELE, SEGMENT, SOUS_SEGMENT, MODELE_POP
    for sheet in ['2022','2023','2024','2025','Feuil5']:
        df = pd.read_excel(new_file, sheet_name=sheet, header=0).iloc[:, :6]
        df.columns = ['CD_TYP_CONS','MARQUE','MODELE','SEGMENT','SOUS_SEGMENT','MODELE_POP']
        df = df.dropna(subset=['CD_TYP_CONS'])
        parts.append(df)

    # Feuil6, Feuil7: have 2 blank rows before header; column order differs
    # cols: CD_TYP_CONS, MARQUE, MODELE, MODELE_POP, SEGMENT, SOUS_SEGMENT
    for sheet in ['Feuil6','Feuil7']:
        raw = pd.read_excel(new_file, sheet_name=sheet, header=None)
        hi  = next((i for i, row in raw.iterrows()
                    if str(row.iloc[0]).strip() == 'CD_TYP_CONS'), None)
        if hi is not None:
            df = pd.read_excel(new_file, sheet_name=sheet, header=hi).iloc[:, :6]
            df.columns = ['CD_TYP_CONS','MARQUE','MODELE','MODELE_POP','SEGMENT','SOUS_SEGMENT']
            df = df.dropna(subset=['CD_TYP_CONS'])
            df = df[['CD_TYP_CONS','MARQUE','MODELE','SEGMENT','SOUS_SEGMENT','MODELE_POP']]
            parts.append(df)

    df_all = pd.concat(parts, ignore_index=True)
    df_all['CD_TYP_CONS'] = df_all['CD_TYP_CONS'].astype(str).str.strip()
    df_all['CD_TYP_CONS_K'] = norm(df_all['CD_TYP_CONS'])

    # Prioritise rows that have SEGMENT populated
    df_all['_has'] = df_all['SEGMENT'].notna().astype(int)
    df_all = df_all.sort_values('_has', ascending=False)
    df_master = (df_all
                 .drop_duplicates(subset=['CD_TYP_CONS_K'], keep='first')
                 .drop(columns=['_has'])
                 .reset_index(drop=True))
    return df_master


def main():
    print("STEP 2.5 — DATA ENRICHMENT\n")

    input_file = 'data_intermediate.csv'
    seg_file   = './data/segmentation.xlsx'
    new_file   = './data/Segment_par_Code_Type.xlsx'

    for f in [input_file, seg_file, new_file]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found.")
            return

    # ── 0. Load raw intermediate ──────────────────────────────────────────────
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"  Raw shape: {df.shape}")

    # ── 1. Drop all garbage / pivot-table columns ─────────────────────────────
    to_drop = [c for c in GARBAGE_COLS if c in df.columns]
    to_drop += [c for c in df.columns if c.startswith('UNNAMED_')]
    df = df.drop(columns=list(set(to_drop)))
    print(f"  After garbage drop: {df.shape[1]} columns")

    # ── 2. Fix core columns ───────────────────────────────────────────────────
    # GENRE: text label is in CD_GENRE in source, not in empty GENRE column
    if 'CD_GENRE' in df.columns:
        df['GENRE'] = df['CD_GENRE']

    # MODELE: TYP_COM is the real commercial model name
    if 'TYP_COM' in df.columns:
        df['MODELE'] = df['TYP_COM']

    # SOCIETE and DATE_MEC
    df = df.rename(columns={'SOCIT': 'SOCIETE', 'DMC': 'DATE_MEC'}, errors='ignore')

    # Date types
    df['DATV'] = pd.to_datetime(df['DATV'], errors='coerce')
    if 'DATE_MEC' in df.columns:
        df['DATE_MEC'] = pd.to_datetime(df['DATE_MEC'], errors='coerce')

    # ── 3. Build join keys ────────────────────────────────────────────────────
    df['CD_TYP_CONS_K'] = norm(df['CD_TYP_CONS'].fillna(''))
    df['MARQUE_K']      = norm(df['MARQUE'].fillna(''))
    df['MK']            = norm(df['MARQUE'].fillna(''))
    df['ML']            = norm(df['MODELE'].fillna(''))
    df['GENRE_K']       = norm(df['GENRE'].fillna(''))
    n = len(df)
    print(f"  Working shape: {df.shape}")

    # ── 4. Read lookup sheets ─────────────────────────────────────────────────
    print("\nReading lookup files...")

    # Master CD_TYP_CONS lookup from Segment_par_Code_Type.xlsx
    print("  Building master CD_TYP_CONS lookup...")
    df_master = build_master_lookup(new_file)
    print(f"  Master lookup: {len(df_master):,} unique codes, "
          f"SEGMENT {df_master['SEGMENT'].notna().sum()/len(df_master)*100:.1f}% within lookup")

    # Segmentation fallback: MARQUE+MODELE → SEGMENT+SOUS_SEGMENT
    df_seg_fb = pd.read_excel(seg_file, sheet_name='Segmentation', header=0).iloc[:, :4]
    df_seg_fb.columns = ['MARQUE','MODELE','SEGMENT','SOUS_SEGMENT']
    df_seg_fb = df_seg_fb.dropna(subset=['MARQUE'])
    df_seg_fb['MK'] = norm(df_seg_fb['MARQUE'])
    df_seg_fb['ML'] = norm(df_seg_fb['MODELE'])
    df_seg_fb = df_seg_fb.drop_duplicates(subset=['MK','ML'], keep='first')
    print(f"  Segmentation fallback: {len(df_seg_fb):,} MARQUE+MODELE entries")

    # Groupe: MARQUE → GROUPE + DISTRIBUTEUR
    df_grp = pd.read_excel(seg_file, sheet_name='Groupe', header=None)
    df_grp.columns = ['MARQUE','GROUPE','DISTRIBUTEUR']
    df_grp = df_grp.dropna(subset=['MARQUE'])
    df_grp['MARQUE_K'] = norm(df_grp['MARQUE'])
    df_grp = df_grp.drop_duplicates(subset=['MARQUE_K'], keep='first')
    print(f"  Groupe: {len(df_grp):,} brands")

    # Feuil3: MARQUE → PAYS_DORIGINE + CONTINENT
    df_f3 = pd.read_excel(seg_file, sheet_name='Feuil3', header=0)
    df_f3.columns = ['MARQUE','MARQUE_PLUS','PAYS_DORIGINE','CONTINENT']
    df_f3 = df_f3[['MARQUE','PAYS_DORIGINE','CONTINENT']].copy()
    df_f3['MARQUE_K'] = norm(df_f3['MARQUE'])
    df_f3 = df_f3.drop_duplicates(subset=['MARQUE_K'], keep='first')
    print(f"  Feuil3: {len(df_f3):,} brands")

    # CD GENRE: GENRE text → MARCHE_TYPE
    df_cdg = pd.read_excel(seg_file, sheet_name='CD GENRE', header=0)
    df_cdg.columns = ['GENRE_LABEL','CD_GENRE_NUM','MARCHE_TYPE']
    df_cdg['GENRE_K'] = norm(df_cdg['GENRE_LABEL'].fillna(''))
    df_cdg = df_cdg.dropna(subset=['GENRE_LABEL'])
    df_cdg = df_cdg.drop_duplicates(subset=['GENRE_K'], keep='first')
    print(f"  CD GENRE: {len(df_cdg):,} genre codes")

    # ── 5. Apply SEGMENT enrichment (two passes) ──────────────────────────────
    print("\nApplying enrichments...")

    # Pass 1: direct CD_TYP_CONS join
    df = safe_merge(df, df_master[['CD_TYP_CONS_K','SEGMENT','SOUS_SEGMENT']],
                    'CD_TYP_CONS_K', ['SEGMENT','SOUS_SEGMENT'], 'master', n)
    p1 = df['SEGMENT'].notna().sum()
    print(f"  Pass 1 (CD_TYP_CONS direct): {p1:,} ({p1/n*100:.1f}%)")

    # Pass 2: MARQUE+MODELE fallback for remaining NaN
    mask_idx = df.index[df['SEGMENT'].isna()].tolist()
    if mask_idx:
        sub    = df.loc[mask_idx, ['ID','MK','ML']].copy()
        filled = (sub.merge(df_seg_fb[['MK','ML','SEGMENT','SOUS_SEGMENT']],
                            on=['MK','ML'], how='left')
                     .drop_duplicates(subset=['ID'], keep='first'))
        seg_map  = filled.set_index('ID')['SEGMENT']
        sous_map = filled.set_index('ID')['SOUS_SEGMENT']
        ids      = df.loc[mask_idx, 'ID'].values
        df.loc[mask_idx, 'SEGMENT']      = [seg_map.get(i, np.nan)  for i in ids]
        df.loc[mask_idx, 'SOUS_SEGMENT'] = [sous_map.get(i, np.nan) for i in ids]

    p2      = df['SEGMENT'].notna().sum()
    p2_sous = df['SOUS_SEGMENT'].notna().sum()
    print(f"  Pass 2 (MARQUE+MODELE):      {p2:,} ({p2/n*100:.1f}%)")
    print(f"  SOUS_SEGMENT:                {p2_sous:,} ({p2_sous/n*100:.1f}%)")

    # GROUPE + DISTRIBUTEUR
    df = safe_merge(df, df_grp[['MARQUE_K','GROUPE','DISTRIBUTEUR']],
                    'MARQUE_K', ['GROUPE','DISTRIBUTEUR'], 'Groupe', n)
    nn = df['GROUPE'].notna().sum()
    print(f"  GROUPE:                      {nn:,} ({nn/n*100:.1f}%)")

    # PAYS_DORIGINE + CONTINENT
    df = safe_merge(df, df_f3[['MARQUE_K','PAYS_DORIGINE','CONTINENT']],
                    'MARQUE_K', ['PAYS_DORIGINE','CONTINENT'], 'Feuil3', n)
    nn = df['PAYS_DORIGINE'].notna().sum()
    print(f"  PAYS_DORIGINE:               {nn:,} ({nn/n*100:.1f}%)")

    # MARCHE_TYPE
    df = safe_merge(df, df_cdg[['GENRE_K','MARCHE_TYPE']],
                    'GENRE_K', ['MARCHE_TYPE'], 'CD GENRE', n)
    nn = df['MARCHE_TYPE'].notna().sum()
    print(f"  MARCHE_TYPE:                 {nn:,} ({nn/n*100:.1f}%)")

    # ── 6. Clean up helper columns ────────────────────────────────────────────
    df = df.drop(columns=['CD_TYP_CONS_K','MARQUE_K','MK','ML','GENRE_K'], errors='ignore')

    # ── 7. Fix data types ─────────────────────────────────────────────────────
    str_cols = ['MARQUE','MODELE','GENRE','USAGE','VILLE','ENERGIE','MARCHE_TYPE',
                'SOCIETE','YEAR_MONTH','SEGMENT','SOUS_SEGMENT',
                'PAYS_DORIGINE','CONTINENT','GROUPE','DISTRIBUTEUR']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].replace({'nan': np.nan, 'NAN': np.nan, '': np.nan})

    for col in ['PUISSANCE','CD_VILLE']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── 8. Select exactly the 22 final columns ────────────────────────────────
    missing = [c for c in FINAL_COLS if c not in df.columns]
    for col in missing:
        df[col] = np.nan
    df_out = df[FINAL_COLS].copy()

    # ── 9. Validation ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    print(f"  Rows      : {len(df_out):,}")
    print(f"  Columns   : {len(df_out.columns)}")
    print(f"  Dup IDs   : {df_out.duplicated('ID').sum()}")
    print(f"\n  Dtype & coverage per column:")
    for col in df_out.columns:
        nn  = df_out[col].notna().sum()
        pct = nn / len(df_out) * 100
        bar = '█' * int(pct / 5)
        print(f"    {col:20s}  {str(df_out[col].dtype):12s}  {pct:5.1f}%  {bar}")

    print(f"\n  Sample enriched row (first with SEGMENT):")
    sample = df_out[df_out['SEGMENT'].notna()].head(1)
    if len(sample):
        print(sample.iloc[0].to_string())

    # ── 10. Save ──────────────────────────────────────────────────────────────
    df_out.to_csv('data_cleaned_enriched.csv', index=False)
    mb = os.path.getsize('data_cleaned_enriched.csv') / 1024 / 1024
    print(f"\nSaved: data_cleaned_enriched.csv")
    print(f"  {len(df_out):,} rows | {len(df_out.columns)} columns | {mb:.1f} MB")
    print("\nSTEP 2.5 COMPLETE -> Run step4_eda.py next")


if __name__ == '__main__':
    main()