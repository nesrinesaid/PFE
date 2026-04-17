import pandas as pd
import numpy as np
import glob
import os
import re
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Columns that come from embedded Excel pivot tables / report sheets
# and must be dropped — they are NOT vehicle registration data
# ─────────────────────────────────────────────────────────────────────────────
GARBAGE_COLS = [
    # Unnamed pivot spillover columns
    'UNNAMED_30', 'UNNAMED_33', 'UNNAMED_34', 'UNNAMED_35', 'UNNAMED_36',
    'UNNAMED_37', 'UNNAMED_38', 'UNNAMED_39', 'UNNAMED_40', 'UNNAMED_41',
    'UNNAMED_42', 'UNNAMED_44', 'UNNAMED_45', 'UNNAMED_46', 'UNNAMED_47', 'UNNAMED_48',
    # Pivot table labels / aggregates
    'TOUS', 'TOTAL', 'MOIS_11', 'DAT1',
    # Pre-aggregated yearly sales columns (belong in a separate report, not here)
    'VENTES_2019', 'VENTES_2020', 'VENTES_2021', 'VENTES_2022', 'VENTES_2023', 'VENTES_2024', 'VENTES_2025',
    # Duplicate / corrupted columns
    'MARQUE.1',   # duplicate of MARQUE
    'DMC.1',      # duplicate of DMC, always null
    'DAT_IMMATR', # always null
    # CHASSIS split into two unusable halves
    'CHA', 'SSIS',
    # Popular model / model list — aggregated summaries, not per-vehicle
    'MODLE_POPULAIRE', 'MODLES', 'MOIS',
    # CHASSIS was already removed in original script — keep excluded
    'CHASSIS',
]

# Columns that exist in the raw Excel with confusing names → rename to clear names
# NOTE: MARCHE_TYPE is derived in step2_5 via the CD GENRE lookup sheet (numeric join).
#       MARCH from the raw Excel is kept as-is so step2_5 can use it as a fallback.
#       SEGMENT_ is kept as-is — step2_5 drops it before enrichment.
RENAME_MAP = {
    'POSITIONS': 'GPS_COORDS',    # GPS coordinates of the registration office
    'DMC':       'DATE_MEC',      # Date de mise en circulation (first registration date)
    'SOCIT':     'SOCIETE',       # Distributor company name
}

# ─────────────────────────────────────────────────────────────────────────────
# Columns whose numeric type should be enforced
# ─────────────────────────────────────────────────────────────────────────────
NUMERIC_COLS = ['PTAC', 'PVID', 'PUISSANCE', 'CYL', 'PLACE_ASSISE']


def clean_column_names(df):
    """Standardise column names: uppercase, spaces→underscore, strip special chars."""
    new_cols = []
    seen = {}
    for col in df.columns:
        clean = re.sub(r'[^A-Z0-9_]', '', str(col).strip().upper().replace(' ', '_'))
        # Handle duplicate column names that pandas suffixes with .1, .2 etc.
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
        new_cols.append(clean)
    df.columns = new_cols
    return df


def drop_garbage_columns(df):
    """Drop all pivot-table / report columns that are not per-vehicle data."""
    to_drop = [c for c in GARBAGE_COLS if c in df.columns]
    # Also drop any remaining UNNAMED_ columns not listed explicitly
    to_drop += [c for c in df.columns if c.startswith('UNNAMED_')]
    to_drop = list(set(to_drop))  # deduplicate
    df = df.drop(columns=to_drop)
    print(f"  Dropped {len(to_drop)} garbage/unnamed columns: {to_drop}")
    return df


def main():
    print("🚀 STEP 3 — INITIAL DATA CLEANING\n")

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"⚠️  Created {data_dir}")

    # ── 1. Find data files (exclude segmentation lookup) ─────────────────────
    excel_files = glob.glob(os.path.join(data_dir, '*.xlsx'))
    data_files = [f for f in excel_files if 'segmentation' not in f.lower()]

    print(f"Files found: {len(data_files)}")
    for f in data_files:
        print(f"  - {os.path.basename(f)}")

    if not data_files:
        print(f"❌ No data files found in {data_dir}")
        return

    # ── 2. Load & concatenate all Excel files ────────────────────────────────
    df_list = []
    print("\n⏳ Loading Excel files...")
    for file in data_files:
        try:
            tmp = pd.read_excel(file)
            print(f"  ✅ {os.path.basename(file)}: {tmp.shape}")
            df_list.append(tmp)
        except Exception as e:
            print(f"  ⚠️  Error reading {file}: {e}")

    if not df_list:
        print("❌ No files loaded.")
        return

    df = pd.concat(df_list, ignore_index=True)
    print(f"\nRaw combined shape: {df.shape}")

    # ── 3. Standardise column names ──────────────────────────────────────────
    print("\n⏳ Cleaning column names...")
    df = clean_column_names(df)

    # ── 4. Drop all garbage / pivot-table columns ────────────────────────────
    print("\n⏳ Dropping garbage columns...")
    df = drop_garbage_columns(df)

    # ── 5. Rename ambiguous columns to clear names ───────────────────────────
    print("\n⏳ Renaming ambiguous columns...")
    actual_rename = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=actual_rename)
    for old, new in actual_rename.items():
        print(f"  '{old}' → '{new}'")

    # ── 6. Remove duplicates ─────────────────────────────────────────────────
    print("\n⏳ Removing duplicates...")
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Removed {before - len(df)} duplicate rows")

    # ── 7. Find & standardise the date column (DATV) ─────────────────────────
    print("\n⏳ Processing date column...")
    date_col = None
    for col in df.columns:
        if 'DATV' in col or 'DAT_V' in col or 'DATE_V' in col:
            date_col = col
            break

    if date_col is None:
        print(f"❌ No date column found! Available: {list(df.columns)}")
        return

    if date_col != 'DATV':
        df = df.rename(columns={date_col: 'DATV'})
    print(f"  Date column: '{date_col}' → 'DATV'")

    df['DATV'] = pd.to_datetime(df['DATV'], errors='coerce').dt.normalize()

    before = len(df)
    df = df.dropna(subset=['DATV'])
    print(f"  Removed {before - len(df)} rows with invalid dates")

    # ── 8. Add primary key ───────────────────────────────────────────────────
    df.insert(0, 'ID', range(1, 1 + len(df)))

    # ── 9. Create temporal features ──────────────────────────────────────────
    print("\n⏳ Creating temporal features...")
    df['YEAR']       = df['DATV'].dt.year
    df['MONTH']      = df['DATV'].dt.month
    df['YEAR_MONTH'] = df['DATV'].dt.to_period('M').astype(str)
    print("  Created: YEAR, MONTH, YEAR_MONTH")

    # ── 10. Enforce numeric types ─────────────────────────────────────────────
    print("\n⏳ Converting numeric columns...")
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"  Converted: {[c for c in NUMERIC_COLS if c in df.columns]}")

    # ── 11. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 CLEANING SUMMARY")
    print("=" * 60)
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {len(df.columns)}")
    print(f"  Period  : {df['DATV'].min().date()} → {df['DATV'].max().date()}")
    print(f"  Years   : {sorted(df['YEAR'].unique())}")

    print(f"\n  Column coverage:")
    key_cols = ['ID', 'DATV', 'MARQUE', 'MODELE', 'GENRE', 'USAGE',
                'VILLE', 'ENERGIE', 'PUISSANCE', 'CD_TYP_CONS',
                'CD_GENRE', 'MARCH', 'SOCIETE', 'DATE_MEC', 'GPS_COORDS',
                'YEAR', 'MONTH', 'YEAR_MONTH']
    for col in key_cols:
        if col in df.columns:
            nn = df[col].notna().sum()
            print(f"    ✅ {col:20s}: {nn:,} ({nn/len(df)*100:.0f}%)")
        else:
            print(f"    ❌ {col:20s}: NOT FOUND")

    missing_years = [y for y in [2019, 2020, 2021, 2022, 2023, 2024, 2025]
                     if y not in df['YEAR'].unique()]
    if missing_years:
        print(f"\n  ⚠️  WARNING: Missing years in data: {missing_years}")
        print(f"     → Request these files from M. Sami Ben Youssef (ARTES)")

    # ── 12. Save ──────────────────────────────────────────────────────────────
    output_file = os.path.join(project_root, 'data_intermediate.csv')
    df.to_csv(output_file, index=False)
    size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"\n✅ Saved: {output_file}")
    print(f"   Rows: {len(df):,} | Columns: {len(df.columns)} | Size: {size_mb:.1f} MB")
    print("\n🎉 STEP 3 COMPLETE → Run step2_5_enrich_data.py next")


if __name__ == '__main__':
    main()