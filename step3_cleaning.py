import pandas as pd
import numpy as np
import glob
import os
import re
import warnings

warnings.filterwarnings('ignore')

def clean_column_names(df):
    """Clean column names - remove special chars, uppercase, spaces to underscore"""
    df.columns = [re.sub(r'[^A-Z0-9_]', '', str(col).strip().upper().replace(' ', '_')) 
                  for col in df.columns]
    return df

def main():
    print("🚀 DÉMARRAGE: ÉTAPE 3️⃣ - INITIAL DATA CLEANING\n")
    
    data_dir = './data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"⚠️ Dossier {data_dir} créé.")
        
    # Find main data files (ONLY cumul files, NOT segmentation.xlsx)
    excel_files = glob.glob(os.path.join(data_dir, '*.xlsx'))
    
    # Exclude segmentation file
    data_files = [f for f in excel_files if 'segmentation' not in f.lower()]
    
    print(f"Fichiers trouvés: {len(data_files)}")
    for f in data_files:
        print(f"  - {os.path.basename(f)}")
    
    if not data_files:
        print("❌ Aucun fichier de données trouvé!")
        return
    
    df_list = []
    
    print("\n⏳ 1-2. Chargement et fusion des fichiers Excel...")
    for file in data_files:
        try:
            print(f"  - Lecture de {os.path.basename(file)}")
            df_list.append(pd.read_excel(file))
        except Exception as e:
            print(f"⚠️ Erreur: {file}: {e}")
                
    if not df_list:
        print("❌ Aucun fichier chargé!")
        return
    
    df = pd.concat(df_list, ignore_index=True)
    print(f"✅ Fusion terminée. Shape brut: {df.shape}")
    
    # Clean column names FIRST
    print("\n⏳ Nettoyage des noms de colonnes...")
    df = clean_column_names(df)
    print(f"✅ Colonnes nettoyées. Colonnes: {list(df.columns)}")
    
    # Find the date column (might be DATV, DAT_V, DATE_V, etc.)
    date_col = None
    for col in df.columns:
        if 'DATV' in col or 'DAT_V' in col or 'DATE_V' in col:
            date_col = col
            break
    
    if date_col is None:
        print(f"❌ Aucune colonne de date trouvée! Colonnes disponibles: {list(df.columns)}")
        return
    
    print(f"✅ Date column trouvée: {date_col}")
    
    # Rename to standard DAT_V
    if date_col != 'DATV':
        df = df.rename(columns={date_col: 'DATV'})
    
    print("\n⏳ 3. Suppression des doublons...")
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"✅ {initial_len - len(df)} doublons supprimés.")
    
    print("\n⏳ 4. Suppression de la colonne CHASSIS...")
    if 'CHASSIS' in df.columns:
        df = df.drop(columns=['CHASSIS'])
        print("✅ CHASSIS supprimée")
    else:
        print("⚠️ CHASSIS non trouvée")
    
    print("\n⏳ 5. Création de la clé primaire ID...")
    df.insert(0, 'ID', range(1, 1 + len(df)))
    
    print("\n⏳ 6. Traitement de DATV...")
    # Convert to datetime
    df['DATV'] = pd.to_datetime(df['DATV'], errors='coerce')
    
    # Remove time component (keep only date)
    df['DATV'] = df['DATV'].dt.normalize()
    
    # Remove rows with invalid dates
    initial_len = len(df)
    df = df.dropna(subset=['DATV'])
    print(f"✅ {initial_len - len(df)} lignes avec dates invalides supprimées")
    
    print("\n⏳ 7. Création des features temporelles...")
    df['YEAR'] = df['DATV'].dt.year
    df['MONTH'] = df['DATV'].dt.month
    df['YEAR_MONTH'] = df['DATV'].dt.to_period('M').astype(str)
    print("✅ YEAR, MONTH, YEAR_MONTH créés")
    
    print("\n⏳ 8. Conversion des types numériques...")
    numeric_cols = ['PTAC', 'PVID', 'PUISSANCE', 'CYL', 'PLACEASSISE']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print("✅ Types numériques convertis")
    
    print("\n📊 RÉSUMÉ DES DONNÉES NETTOYÉES:")
    print(f"- Shape final : {df.shape}")
    print(f"- Rows: {len(df):,}")
    print(f"- Columns: {len(df.columns)}")
    print(f"- Période : {df['DATV'].min().date()} à {df['DATV'].max().date()}")
    print(f"\nColonnes principales:")
    for col in df.columns[:20]:
        print(f"  - {col}")
    
    print(f"\n📋 Vérification des colonnes clés:")
    for col in ['ID', 'DATV', 'YEAR', 'MONTH', 'YEAR_MONTH', 'MARQUE', 'GENRE', 'USAGE']:
        if col in df.columns:
            print(f"  ✅ {col}")
        else:
            print(f"  ❌ {col} MANQUANTE")
    
    # Save
    output_file = 'data_intermediate.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Sauvegardé : {output_file}")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
    
    print("\n🎉 ÉTAPE 3 TERMINÉE!")

if __name__ == '__main__':
    main()