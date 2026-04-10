import pandas as pd

# Check intermediate data
df = pd.read_csv('data_intermediate.csv')
print("=== DATA_INTERMEDIATE ===")
print(f"Shape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
print(df.columns.tolist())
print(f"\nFirst row:")
print(df.iloc[0])

# Check what's in segmentation.xlsx sheets
print("\n\n=== SEGMENTATION.XLSX SHEETS ===")
ref_file = './data/segmentation.xlsx'

for sheet in ['Groupe', 'Segmentation', 'Feuil1', 'Feuil3']:
    try:
        df_sheet = pd.read_excel(ref_file, sheet_name=sheet)
        print(f"\n{sheet}:")
        print(f"  Shape: {df_sheet.shape}")
        print(f"  Columns: {df_sheet.columns.tolist()}")
        print(f"  First 2 rows:")
        print(df_sheet.head(2))
    except Exception as e:
        print(f"  ❌ Error: {e}")