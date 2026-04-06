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
    print("🚀 DÉMARRAGE: ÉTAPE 3️⃣ - INITIAL DATA CLEANING (AVEC ENRICHISSEMENT)\n")
    
    data_dir = './data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        
    excel_files = glob.glob(os.path.join(data_dir, '*.xls*'))
    
    # Locate segmentation (or ref) file specifically
    segmentation_file = [f for f in excel_files if 'segmentation' in str(f).lower() or 'statistique' in str(f).lower()]
    data_files = [f for f in excel_files if f not in segmentation_file]
    
    df_list = []
    
    print("⏳ 1-2. Chargement et fusion des fichiers Excel principaux...")
    for file in data_files:
        try:
            print(f"  - Lecture de {os.path.basename(file)}")
            df_list.append(pd.read_excel(file))
        except Exception as e:
            print(f"⚠️ Erreur de lecture pour {file}: {e}")
            
    if not df_list:
        print("❌ Chargement échoué. Placez les fichiers Excel de données dans ./data/")
        return
        
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
        df = df.dropna(subset=['DAT_V'])
        df['YEAR'] = df['DAT_V'].dt.year
        df['MONTH'] = df['DAT_V'].dt.month
        df['YEAR_MONTH'] = df['DAT_V'].dt.to_period('M')
    
    num_cols = ['PTAC', 'PVID', 'PUISSANCE', 'CYL', 'PLACE_ASSISE']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    print("\n⏳ 11-13. CHARGEMENT ET JOINTURE DU FICHIER DE RÉFÉRENCE (ENRICHISSEMENT)...")
    if not segmentation_file:
        print("❌ Fichier de référence 'segmentation.xlsx' introuvable dans ./data/!")
        return
        
    ref_file = segmentation_file[0]
    print(f"  - Utilisation du fichier de référence: {os.path.basename(ref_file)}")
    
    try:
        xls = pd.ExcelFile(ref_file)
        sheets = xls.sheet_names
        print(f"  - Feuilles détectées : {sheets}")
        
        def safe_read_sheet(sheet_name):
            if sheet_name in sheets:
                temp = pd.read_excel(xls, sheet_name=sheet_name)
                return clean_column_names(temp)
            else:
                for s in sheets:
                    if s.lower() == sheet_name.lower():
                        temp = pd.read_excel(xls, sheet_name=s)
                        return clean_column_names(temp)
                return None
                
        cd_genre = safe_read_sheet('CD_GENRE')
        cd_usage = safe_read_sheet('CD_USAGE')
        cd_marque = safe_read_sheet('CD_MARQUE')
        cd_ville = safe_read_sheet('CD_VILLE')
        groupe = safe_read_sheet('Groupe')
        segmentation = safe_read_sheet('Segmentation')
        modele = safe_read_sheet('MODELE')
        feuil1 = safe_read_sheet('Feuil1')
        feuil3 = safe_read_sheet('Feuil3')
        
        # Left joins
        if cd_genre is not None and 'GENRE' in df.columns and 'GENRE' in cd_genre.columns:
            df = df.merge(cd_genre, on='GENRE', how='left')
        if cd_usage is not None and 'USAGE' in df.columns and 'USAGE' in cd_usage.columns:
            df = df.merge(cd_usage, on='USAGE', how='left')
        if cd_marque is not None and 'MARQUE' in df.columns and 'MARQUE' in cd_marque.columns:
            df = df.merge(cd_marque, on='MARQUE', how='left')
        if cd_ville is not None and 'VILLE' in df.columns and 'VILLE' in cd_ville.columns:
            df = df.merge(cd_ville, on='VILLE', how='left')
        if groupe is not None and 'MARQUE' in df.columns and 'MARQUE' in groupe.columns:
            df = df.merge(groupe, on='MARQUE', how='left')
            
        join_keys_mm = []
        if 'MARQUE' in df.columns and 'MODELE' in df.columns:
            join_keys_mm = ['MARQUE', 'MODELE']
            
        if segmentation is not None and join_keys_mm and all(k in segmentation.columns for k in join_keys_mm):
            df = df.merge(segmentation, on=join_keys_mm, how='left')
        if modele is not None and join_keys_mm and all(k in modele.columns for k in join_keys_mm):
            df = df.merge(modele, on=join_keys_mm, how='left', suffixes=('', '_REF'))
        if feuil1 is not None and join_keys_mm and all(k in feuil1.columns for k in join_keys_mm):
            df = df.merge(feuil1, on=join_keys_mm, how='left', suffixes=('', '_MARCHE'))
        if feuil3 is not None and 'MARQUE' in df.columns and 'MARQUE' in feuil3.columns:
            df = df.merge(feuil3, on='MARQUE', how='left')

        print("✅ Jointures d'enrichissement terminées.")
        
        # Coverage Quality Check
        print("\n📊 14. TAUX DE COUVERTURE DE L'ENRICHISSEMENT :")
        cols_to_check = ['GROUPE', 'DISTRIBUTEUR', 'SEGMENT', 'CONTINENT', 'MARCHE', 'PAYSDORIGINE']
        for col in cols_to_check:
            matched_col = next((c for c in df.columns if col in c), None)
            if matched_col:
                coverage = df[matched_col].notnull().mean() * 100
                print(f"  - {matched_col}: {coverage:.1f}% enrichis ({(100-coverage):.1f}% manquants)")

    except Exception as e:
        print(f"⚠️ Erreur lors de l'enrichissement: {e}")

    print("\n📊 16. RÉSUMÉ DES DONNÉES ENRICHIES :")
    print(f"- Shape final : {df.shape}")
    print(f"- Période : {df['DAT_V'].min().date()} à {df['DAT_V'].max().date()}")
    
    unique_checks = ['GROUPE', 'DISTRIBUTEUR', 'SEGMENT', 'CONTINENT', 'VILLE', 'USAGE']
    for check in unique_checks:
        c_col = next((c for c in df.columns if check in c), None)
        if c_col:
            print(f"- Valeurs distinctes pour {c_col}: {df[c_col].nunique()}")
    
    output_file = 'data_cleaned_enriched.csv'
    if 'YEAR_MONTH' in df.columns:
        df['YEAR_MONTH'] = df['YEAR_MONTH'].astype(str)
        
    df.to_csv(output_file, index=False)
    print(f"\n✅ 15. Sauvegardé avec succès : {output_file}")

if __name__ == '__main__':
    main()
