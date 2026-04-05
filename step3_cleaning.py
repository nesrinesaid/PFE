import pandas as pd
import numpy as np
import glob
import os
import re
import warnings

warnings.filterwarnings('ignore')

def clean_column_names(df):
    """Nettoyer les noms de colonnes."""
    df.columns = [re.sub(r'[^A-Z0-9_]', '', str(col).strip().upper().replace(' ', '_')) for col in df.columns]
    return df

def main():
    print("🚀 DÉMARRAGE: ÉTAPE 3️⃣ - INITIAL DATA CLEANING\n")
    
    data_dir = './data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"⚠️ Le dossier {data_dir} n'existait pas et a été créé.")
        
    excel_files = glob.glob(os.path.join(data_dir, '*.xls*'))
    segmentation_file = [f for f in excel_files if 'segmentation' in str(f).lower()]
    data_files = [f for f in excel_files if f not in segmentation_file]
    
    df_list = []
    
    print("⏳ 1-2. Chargement et fusion des fichiers Excel...")
    if not data_files:
        print("⚠️ Aucun fichier de données trouvé dans ./data/.")
    else:
        for file in data_files:
            try:
                print(f"  - Lecture de {os.path.basename(file)}")
                df_list.append(pd.read_excel(file))
            except Exception as e:
                print(f"⚠️ Erreur de lecture pour {file}: {e}")
                
    if not df_list:
        print("❌ Chargement échoué. Placez les fichiers Excel dans ./data/")
        print("⚠️ Création d'un dataset fictif pour vérifier la structure...")
        df = pd.DataFrame(columns=['DAT_V', 'CHASSIS', 'MARQUE', 'GENRE', 'ENERGIE', 'PTAC', 'PVID', 'PUISSANCE', 'CYL', 'PLACE_ASSISE'])
    else:
        df = pd.concat(df_list, ignore_index=True)
        print(f"✅ Fusion terminée. Shape brut: {df.shape}")
        
    print("⏳ 3. Suppression des doublons...")
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"✅ {initial_len - len(df)} doublons supprimés.")
    
    print("⏳ 6. Nettoyage des noms de colonnes...")
    df = clean_column_names(df)
    
    print("⏳ 4. Suppression de la colonne CHASSIS...")
    if 'CHASSIS' in df.columns:
        df = df.drop(columns=['CHASSIS'])
        
    print("⏳ 5. Création de la clé primaire ID...")
    df.insert(0, 'ID', range(1, 1 + len(df)))
    
    print("⏳ 7-10. Traitement de DAT_V et features temporelles...")
    if 'DAT_V' in df.columns:
        df['DAT_V'] = pd.to_datetime(df['DAT_V'], errors='coerce').dt.normalize()
        # Drop naive missing
        missing_dates = df['DAT_V'].isnull().sum()
        df = df.dropna(subset=['DAT_V'])
        print(f"✅ {missing_dates} dates invalides/vides ignorées.")
        
        df['YEAR'] = df['DAT_V'].dt.year
        df['MONTH'] = df['DAT_V'].dt.month
        df['YEAR_MONTH'] = df['DAT_V'].dt.to_period('M')
    else:
        print("❌ Colonne DAT_V introuvable!")
        
    print("⏳ 8. Conversion des types numériques...")
    num_cols = ['PTAC', 'PVID', 'PUISSANCE', 'CYL', 'PLACE_ASSISE']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    print("⏳ 11. Chargement du fichier de segmentation...")
    if segmentation_file:
        try:
            print(f"  - Lecture de {os.path.basename(segmentation_file[0])}")
            df_seg = pd.read_excel(segmentation_file[0])
            df_seg = clean_column_names(df_seg)
            if 'MARQUE' in df.columns and 'MARQUE' in df_seg.columns:
                df = df.merge(df_seg, on='MARQUE', how='left')
                print("✅ Jointure avec la segmentation effectuée.")
        except Exception as e:
            print(f"⚠️ Erreur avec le fichier de segmentation: {e}")
    else:
        print("⚠️ Aucun fichier de segmentation trouvé dans ./data/.")
        
    print("\n📊 13. RÉSUMÉ DES DONNÉES :")
    print(f"- Shape final : {df.shape}")
    print(f"- Colonnes : {list(df.columns)}")
    print("- Valeurs manquantes critiques :")
    print(df.isnull().sum().sort_values(ascending=False).head())
    if not df.empty and 'DAT_V' in df.columns:
        print(f"- Période : {df['DAT_V'].min().date()} à {df['DAT_V'].max().date()}")
    
    output_file = 'data_cleaned_step3.csv'
    # Convert 'YEAR_MONTH' to string properly before saving to CSV
    if 'YEAR_MONTH' in df.columns:
        df['YEAR_MONTH'] = df['YEAR_MONTH'].astype(str)
        
    df.to_csv(output_file, index=False)
    print(f"\n✅ 12. Sauvegardé avec succès : {output_file}")

if __name__ == '__main__':
    main()
