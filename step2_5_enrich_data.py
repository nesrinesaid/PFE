import pandas as pd
import numpy as np
import os

print("🔗 STEP 2.5: MERGE WITH LOOKUP TABLES (CLEAN)\n")

# ============================================================
# LOAD CLEANED DATA FROM STEP 3
# ============================================================

input_file = 'data_intermediate.csv'
if not os.path.exists(input_file):
    print(f"❌ ERROR: {input_file} not found! Run step3_cleaning.py first.")
    exit()

print(f"⏳ Loading {input_file}...")
df_main = pd.read_csv(input_file, parse_dates=['DATV'])

print(f"✅ Loaded: {df_main.shape}")
print(f"   Rows: {df_main.shape[0]:,}")
print(f"   Columns: {df_main.shape[1]}")
print(f"   Date range: {df_main['DATV'].min()} to {df_main['DATV'].max()}")
print(f"\nColumns in main data:")
for col in df_main.columns:
    print(f"   - {col}")

# ============================================================
# LOAD LOOKUP SHEETS
# ============================================================

print("\n" + "="*70)
print("⏳ LOADING LOOKUP SHEETS FROM REFERENCE FILE\n")

ref_file = 'C:/Users/USER/PFE/data/segmentation.xlsx'

if not os.path.exists(ref_file):
    print(f"❌ ERROR: {ref_file} not found!")
    exit()

# Get all sheet names first
xls = pd.ExcelFile(ref_file)
print(f"Available sheets in reference file:")
for i, sheet in enumerate(xls.sheet_names, 1):
    print(f"   {i:2}. {sheet}")

print("\n" + "="*70)
print("⏳ LOADING SPECIFIC SHEETS\n")

lookups = {}

# Sheet 1: Groupe (Distributors & Brand Groups)
try:
    print("Loading Groupe...")
    lookups['groupe'] = pd.read_excel(ref_file, sheet_name='Groupe')
    lookups['groupe'].columns = [col.strip().upper() for col in lookups['groupe'].columns]
    
    # Keep only useful columns
    if 'MARQUE' in lookups['groupe'].columns:
        cols = [c for c in lookups['groupe'].columns if c in ['MARQUE', 'GROUPE', 'DISTRIBUTEUR']]
        lookups['groupe'] = lookups['groupe'][cols]
        lookups['groupe'] = lookups['groupe'].drop_duplicates()
        print(f"   ✅ Groupe: {lookups['groupe'].shape}")
        print(f"      Brands: {lookups['groupe']['MARQUE'].nunique()}")
except Exception as e:
    print(f"   ❌ Groupe failed: {e}")

# Sheet 2: Segmentation (Vehicle segments)
try:
    print("Loading Segmentation...")
    lookups['segment'] = pd.read_excel(ref_file, sheet_name='Segmentation')
    lookups['segment'].columns = [col.strip().upper() for col in lookups['segment'].columns]
    
    if 'MARQUE' in lookups['segment'].columns:
        cols = [c for c in lookups['segment'].columns if c in ['MARQUE', 'MODELE', 'SEGMENT', 'SEGMENT_2']]
        lookups['segment'] = lookups['segment'][cols]
        lookups['segment'] = lookups['segment'].drop_duplicates(subset=['MARQUE', 'MODELE'] if 'MODELE' in lookups['segment'].columns else ['MARQUE'])
        print(f"   ✅ Segmentation: {lookups['segment'].shape}")
except Exception as e:
    print(f"   ❌ Segmentation failed: {e}")

# Sheet 3: Feuil1 (Models & Market Type)
try:
    print("Loading Feuil1 (Models & Market)...")
    lookups['feuil1'] = pd.read_excel(ref_file, sheet_name='Feuil1')
    lookups['feuil1'].columns = [col.strip().upper() for col in lookups['feuil1'].columns]
    
    if 'MARQUE' in lookups['feuil1'].columns:
        cols = [c for c in lookups['feuil1'].columns if c in ['MARQUE', 'MODELE', 'MARCHE']]
        lookups['feuil1'] = lookups['feuil1'][cols]
        lookups['feuil1'] = lookups['feuil1'].drop_duplicates()
        print(f"   ✅ Feuil1: {lookups['feuil1'].shape}")
except Exception as e:
    print(f"   ❌ Feuil1 failed: {e}")

# Sheet 4: Feuil3 (Brand Origin - Country & Continent)
try:
    print("Loading Feuil3 (Brand Origin)...")
    lookups['feuil3'] = pd.read_excel(ref_file, sheet_name='Feuil3')
    lookups['feuil3'].columns = [col.strip().upper() for col in lookups['feuil3'].columns]
    
    # Find MARQUE column (careful: might be MARQUE+)
    marque_col = None
    for col in lookups['feuil3'].columns:
        if col == 'MARQUE':  # Exact match only
            marque_col = col
            break
    
    if marque_col:
        cols = [c for c in lookups['feuil3'].columns if c in [marque_col, 'PAYS_DORIGINE', 'CONTINENT', 'PAYS']]
        if marque_col in cols:
            lookups['feuil3'] = lookups['feuil3'][cols]
            lookups['feuil3'] = lookups['feuil3'].rename(columns={marque_col: 'MARQUE'})
            lookups['feuil3'] = lookups['feuil3'].drop_duplicates(subset=['MARQUE'])
            print(f"   ✅ Feuil3: {lookups['feuil3'].shape}")
    else:
        print(f"   ⚠️  Feuil3: Could not find MARQUE column")
except Exception as e:
    print(f"   ❌ Feuil3 failed: {e}")

# ============================================================
# MERGE LOOKUPS TO MAIN DATA
# ============================================================

print("\n" + "="*70)
print("⏳ MERGING LOOKUPS TO MAIN DATA\n")

df_enriched = df_main.copy()

# MERGE 1: Groupe (by MARQUE)
if 'groupe' in lookups and 'MARQUE' in df_enriched.columns and 'MARQUE' in lookups['groupe'].columns:
    rows_before = len(df_enriched)
    df_enriched = df_enriched.merge(lookups['groupe'], on='MARQUE', how='left')
    rows_after = len(df_enriched)
    
    if rows_before == rows_after:
        matched = df_enriched['GROUPE'].notna().sum()
        print(f"✅ MERGED Groupe:")
        print(f"   Rows: {rows_before:,} → {rows_after:,} (no change ✓)")
        print(f"   Matched: {matched:,} rows ({matched/rows_before*100:.1f}%)")
    else:
        print(f"❌ GROUPE MERGE FAILED: Row count changed!")
        print(f"   {rows_before:,} → {rows_after:,}")
        exit()

# MERGE 2: Segmentation (by MARQUE + MODELE if available)
if 'segment' in lookups and 'MARQUE' in df_enriched.columns and 'MARQUE' in lookups['segment'].columns:
    if 'MODELE' in df_enriched.columns and 'MODELE' in lookups['segment'].columns:
        rows_before = len(df_enriched)
        df_enriched = df_enriched.merge(lookups['segment'], on=['MARQUE', 'MODELE'], how='left')
        rows_after = len(df_enriched)
        
        if rows_before == rows_after:
            print(f"✅ MERGED Segmentation (MARQUE + MODELE):")
            print(f"   Rows: {rows_before:,} (no change ✓)")
        else:
            print(f"❌ SEGMENTATION MERGE FAILED!")
            exit()

# MERGE 3: Feuil1 (by MARQUE + MODELE)
if 'feuil1' in lookups and 'MARQUE' in df_enriched.columns and 'MARQUE' in lookups['feuil1'].columns:
    if 'MODELE' in df_enriched.columns and 'MODELE' in lookups['feuil1'].columns:
        rows_before = len(df_enriched)
        df_enriched = df_enriched.merge(lookups['feuil1'][[col for col in ['MARQUE', 'MODELE', 'MARCHE'] if col in lookups['feuil1'].columns]], 
                                       on=['MARQUE', 'MODELE'], how='left')
        rows_after = len(df_enriched)
        
        if rows_before == rows_after:
            print(f"✅ MERGED Feuil1 (Market Type):")
            print(f"   Rows: {rows_before:,} (no change ✓)")

# MERGE 4: Feuil3 (by MARQUE)
if 'feuil3' in lookups and 'MARQUE' in df_enriched.columns and 'MARQUE' in lookups['feuil3'].columns:
    rows_before = len(df_enriched)
    df_enriched = df_enriched.merge(lookups['feuil3'], on='MARQUE', how='left', suffixes=('', '_origin'))
    rows_after = len(df_enriched)
    
    if rows_before == rows_after:
        print(f"✅ MERGED Feuil3 (Brand Origin):")
        print(f"   Rows: {rows_before:,} (no change ✓)")

# ============================================================
# CLEANUP
# ============================================================

print("\n" + "="*70)
print("⏳ CLEANUP\n")

# Remove duplicate columns
df_enriched = df_enriched.loc[:, ~df_enriched.columns.duplicated()]
print(f"✅ Removed duplicate columns")

# Remove completely empty columns
df_enriched = df_enriched.dropna(axis=1, how='all')
print(f"✅ Removed completely empty columns")

# ============================================================
# FINAL CHECKS
# ============================================================

print("\n" + "="*70)
print("📊 FINAL REPORT\n")

print(f"Shape: {df_enriched.shape}")
print(f"  Rows: {df_enriched.shape[0]:,}")
print(f"  Columns: {df_enriched.shape[1]}")

print(f"\nColumns ({df_enriched.shape[1]}):")
# Just show first few to avoid long logs
for i, col in enumerate(list(df_enriched.columns)[:15], 1):
    print(f"  {i:2}. {col}")
print("  ...")

print(f"\nEnrichment Coverage:")
for col in ['GROUPE', 'DISTRIBUTEUR', 'SEGMENT', 'MARCHE', 'CONTINENT', 'PAYS_DORIGINE']:
    if col in df_enriched.columns:
        matched = df_enriched[col].notna().sum()
        coverage = (matched / len(df_enriched)) * 100
        print(f"  {col:20} {matched:7,} rows ({coverage:5.1f}%)")

# ============================================================
# SAVE
# ============================================================

print("\n" + "="*70)
output_file = 'data_cleaned_enriched.csv'
df_enriched.to_csv(output_file, index=False)
print(f"✅ SAVED: {output_file}")
print(f"   File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")

print("\n✅ ENRICHMENT COMPLETE!")
