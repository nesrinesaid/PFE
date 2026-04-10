import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

def main():
    print("🔗 STEP 2.5: MERGE WITH LOOKUP TABLES (CLEAN)\n")
    
    # Load intermediate data
    input_file = 'data_intermediate.csv'
    if not os.path.exists(input_file):
        print(f"❌ {input_file} not found!")
        return
    
    print("⏳ Loading data_intermediate.csv...")
    df = pd.read_csv(input_file)
    print(f"✅ Loaded: {df.shape}")
    
    # Keep ONLY essential columns for enrichment
    print("\n⏳ STEP 1: Keep only essential columns...")
    essential_cols = ['ID', 'DATV', 'MARQUE', 'MODELE', 'GENRE', 'USAGE', 'CD_VILLE', 'VILLE', 
                     'ENERGIE', 'PUISSANCE', 'YEAR', 'MONTH', 'YEAR_MONTH']
    
    cols_to_keep = [c for c in essential_cols if c in df.columns]
    df_clean = df[cols_to_keep].copy()
    
    # Drop any existing enrichment columns
    cols_to_drop = ['GROUPE', 'SEGMENT', 'PAYS_DORIGINE', 'CONTINENT', 'DISTRIBUTEUR', 'MARCHE']
    cols_to_drop = [c for c in cols_to_drop if c in df_clean.columns]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        print(f"✅ Dropped old enrichment columns: {cols_to_drop}")
    
    print(f"   Clean shape: {df_clean.shape}")
    
    # Load reference file
    print("\n⏳ STEP 2: Load lookup sheets...")
    ref_file = './data/segmentation.xlsx'
    
    lookups = {}
    
    # Load Segmentation (Marque + Modele → Segment)
    try:
        seg = pd.read_excel(ref_file, sheet_name='Segmentation')
        seg.columns = [c.strip().upper().replace('È', 'E').replace('Ç', 'C') for c in seg.columns]
        
        # Keep only: Marque, Modele, Segment
        if 'MARQUE' in seg.columns and 'MODELE' in seg.columns and 'SEGMENT' in seg.columns:
            seg = seg[['MARQUE', 'MODELE', 'SEGMENT']].drop_duplicates()
            seg = seg.dropna(subset=['MARQUE', 'MODELE'])
            lookups['segment'] = seg
            print(f"✅ Segmentation: {seg.shape[0]} rows")
        else:
            print(f"⚠️ Segmentation: Missing required columns")
    except Exception as e:
        print(f"⚠️ Segmentation error: {e}")
    
    # Load Feuil1 (Marque + Modele → Marche/Market)
    try:
        feuil1 = pd.read_excel(ref_file, sheet_name='Feuil1')
        feuil1.columns = [c.strip().upper() for c in feuil1.columns]
        
        if 'MARQUE' in feuil1.columns and 'MODELE' in feuil1.columns and 'MARCHE' in feuil1.columns:
            feuil1 = feuil1[['MARQUE', 'MODELE', 'MARCHE']].drop_duplicates()
            feuil1 = feuil1.dropna(subset=['MARQUE', 'MODELE'])
            lookups['marche'] = feuil1
            print(f"✅ Feuil1 (Market): {feuil1.shape[0]} rows")
        else:
            print(f"⚠️ Feuil1: Missing required columns")
    except Exception as e:
        print(f"⚠️ Feuil1 error: {e}")
    
    # Load Feuil3 (Marque → Country/Continent)
    try:
        feuil3 = pd.read_excel(ref_file, sheet_name='Feuil3')
        feuil3.columns = [c.strip().upper().replace('È', 'E').replace('\'', '').replace(' ', '_') for c in feuil3.columns]
        
        if 'MARQUE' in feuil3.columns:
            # Keep MARQUE and origin/continent columns
            cols_to_use = ['MARQUE']
            if 'PAYS_DORIGINE' in feuil3.columns:
                cols_to_use.append('PAYS_DORIGINE')
            if 'CONTINENT' in feuil3.columns:
                cols_to_use.append('CONTINENT')
            
            feuil3 = feuil3[cols_to_use].drop_duplicates(subset=['MARQUE'])
            feuil3 = feuil3.dropna(subset=['MARQUE'])
            
            lookups['origin'] = feuil3
            print(f"✅ Feuil3 (Origin): {feuil3.shape[0]} rows")
        else:
            print(f"⚠️ Feuil3: Missing MARQUE column")
    except Exception as e:
        print(f"⚠️ Feuil3 error: {e}")
    
    # Load Groupe (Marque → Brand Group)
    try:
        groupe = pd.read_excel(ref_file, sheet_name='Groupe', header=None)
        
        # The sheet structure: Row 0 has headers, Column 0 has MARQUE
        groupe.columns = ['MARQUE', 'GROUPE', 'DISTRIBUTEUR']
        groupe = groupe[1:].reset_index(drop=True)  # Skip header row
        groupe = groupe.dropna(subset=['MARQUE'])
        groupe = groupe[['MARQUE', 'GROUPE', 'DISTRIBUTEUR']].drop_duplicates()
        groupe['MARQUE'] = groupe['MARQUE'].str.strip()
        groupe['GROUPE'] = groupe['GROUPE'].str.strip()
        groupe['DISTRIBUTEUR'] = groupe['DISTRIBUTEUR'].str.strip()
        
        lookups['groupe'] = groupe
        print(f"✅ Groupe: {groupe.shape[0]} rows")
    except Exception as e:
        print(f"⚠️ Groupe error: {e}")
    
    # MERGE LOOKUPS
    print("\n⏳ STEP 3: Merge lookups to main data...\n")
    
    df_enriched = df_clean.copy()
    
    # Merge Segmentation (on MARQUE + MODELE)
    if 'segment' in lookups:
        before = len(df_enriched)
        df_enriched = df_enriched.merge(lookups['segment'], on=['MARQUE', 'MODELE'], how='left')
        matched = df_enriched['SEGMENT'].notna().sum()
        print(f"✅ Segment: {matched:,} rows matched ({matched/before*100:.1f}%)")
    
    # Merge Market (on MARQUE + MODELE)
    if 'marche' in lookups:
        before = len(df_enriched)
        df_enriched = df_enriched.merge(lookups['marche'], on=['MARQUE', 'MODELE'], how='left')
        matched = df_enriched['MARCHE'].notna().sum()
        print(f"✅ Market: {matched:,} rows matched ({matched/before*100:.1f}%)")
    
    # Merge Origin (on MARQUE)
    if 'origin' in lookups:
        before = len(df_enriched)
        df_enriched = df_enriched.merge(lookups['origin'], on='MARQUE', how='left')
        matched = df_enriched['CONTINENT'].notna().sum()
        print(f"✅ Origin: {matched:,} rows matched ({matched/before*100:.1f}%)")
    
    # Merge Groupe (on MARQUE)
    if 'groupe' in lookups:
        before = len(df_enriched)
        df_enriched = df_enriched.merge(lookups['groupe'], on='MARQUE', how='left')
        matched = df_enriched['GROUPE'].notna().sum()
        print(f"✅ Groupe: {matched:,} rows matched ({matched/before*100:.1f}%)")
    
    # Final cleanup
    print("\n⏳ STEP 4: Final cleanup...")
    
    # Remove duplicate columns
    df_enriched = df_enriched.loc[:, ~df_enriched.columns.duplicated()]
    
    # Drop empty columns
    df_enriched = df_enriched.dropna(axis=1, how='all')
    
    print(f"✅ Final shape: {df_enriched.shape}")
    
    # Validation
    print("\n📊 ENRICHMENT COVERAGE:")
    enrichment_cols = ['SEGMENT', 'MARCHE', 'PAYS_DORIGINE', 'CONTINENT', 'GROUPE', 'DISTRIBUTEUR']
    for col in enrichment_cols:
        if col in df_enriched.columns:
            coverage = (df_enriched[col].notna().sum() / len(df_enriched)) * 100
            unique = df_enriched[col].nunique()
            print(f"   {col:20} {coverage:6.1f}% coverage ({unique:3d} unique values)")
    
    # Save
    output_file = 'data_cleaned_enriched.csv'
    df_enriched.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved: {output_file}")
    print(f"   Rows: {len(df_enriched):,}")
    print(f"   Columns: {len(df_enriched.columns)}")
    print(f"   File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
    
    print("\n🎉 ENRICHMENT COMPLETE!")

if __name__ == '__main__':
    main()