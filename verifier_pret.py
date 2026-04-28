"""
VERIFICATION DE PRET PRE-STEP-6
Executez ce script pour verifier que les donnees sont pretes pour le modeling.
Usage: python verifier_pret.py
"""

import os
from datetime import datetime

import pandas as pd

VERT = '\033[92m'
JAUNE = '\033[93m'
ROUGE = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'


def verifier_fichier_existe(chemin_fichier, lignes_attendues=None):
    if not os.path.exists(chemin_fichier):
        return False, "FICHIER NON TROUVE"

    try:
        if chemin_fichier.endswith('.csv'):
            df = pd.read_csv(chemin_fichier)
            lignes = len(df)
            if lignes_attendues and lignes != lignes_attendues:
                return True, f"OK ({lignes:,} lignes, attendu ~{lignes_attendues:,})"
            return True, f"OK ({lignes:,} lignes)"
        return True, "OK"
    except Exception as e:
        return False, f"ERREUR: {str(e)}"


def imprimer_statut(statut, message):
    if statut == "PASS":
        print(f"  {VERT}OK{RESET}   {message}")
    elif statut == "WARN":
        print(f"  {JAUNE}WARN{RESET} {message}")
    else:
        print(f"  {ROUGE}FAIL{RESET} {message}")


def main():
    print("\n" + "=" * 70)
    print(f"{BOLD}VERIFICATION DE PRET STEP 6{RESET}")
    print(f"Genere: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    racine_projet = os.path.dirname(os.path.abspath(__file__))
    problemes = []
    avertissements = []

    print(f"\n{BOLD}1. FICHIERS DE DONNEES{RESET}")
    fichiers_a_verifier = [
        ('data_cleaned_enriched.csv', None),
        ('data_prepared_final.csv', None),
        ('data_prepared_final_full.csv', None),
        ('data_train.csv', None),
        ('data_validation_2024.csv', None),
        ('data_test_2025.csv', None),
        ('data_future_2026.csv', None, 'PLACEHOLDER'),
    ]

    for item in fichiers_a_verifier:
        nom_fichier = item[0]
        lignes_attendues = item[1]
        type_fichier = item[2] if len(item) > 2 else None

        chemin = os.path.join(racine_projet, nom_fichier)
        existe, msg = verifier_fichier_existe(chemin, lignes_attendues)
        if existe:
            if type_fichier == 'PLACEHOLDER':
                imprimer_statut("WARN", f"{nom_fichier}: {msg} (PLACEHOLDER 2026)")
            else:
                imprimer_statut("PASS", f"{nom_fichier}: {msg}")
        else:
            imprimer_statut("FAIL", f"{nom_fichier}: {msg}")
            problemes.append(f"Manquant ou invalide: {nom_fichier}")

    print(f"\n{BOLD}2. COLONNES REQUISES{RESET}")
    try:
        df = pd.read_csv(os.path.join(racine_projet, 'data_prepared_final.csv'))

        colonnes_requises = {
            'Noyau': ['Date', 'TYPE_MARCHE', 'VENTES'],
            'Caracteristiques de decalage': ['VENTES_LAG1', 'VENTES_LAG7', 'VENTES_LAG30', 'VENTES_LAG90'],
            'Caracteristiques MA': ['VENTES_MA7', 'VENTES_MA30', 'VENTES_MA90'],
            'Temporel': ['Mois_1', 'JS_0', 'Est_T1'],
            'Special': ['EST_RAMADAN', 'EST_VU', 'EST_WEEKEND'],
        }

        for categorie, cols in colonnes_requises.items():
            manquantes = [c for c in cols if c not in df.columns]
            if not manquantes:
                imprimer_statut("PASS", f"{categorie}: {len(cols)} colonnes presentes")
            else:
                imprimer_statut("FAIL", f"{categorie}: Manquantes {manquantes}")
                problemes.append(f"{categorie} manquant: {manquantes}")

    except Exception as e:
        imprimer_statut("FAIL", f"Impossible de lire data_prepared_final.csv: {str(e)}")
        problemes.append("Impossible de verifier les colonnes")

    # Verify transaction-level prepared file columns (full enriched)
    print(f"\n{BOLD}2b. COLONNES REQUISES - transaction-level (data_prepared_final_full.csv){RESET}")
    try:
        full_path = os.path.join(racine_projet, 'data_prepared_final_full.csv')
        if os.path.exists(full_path):
            df_full = pd.read_csv(full_path, nrows=50)
            required_full = [
                'DATV', 'IM_RI', 'TYPE_MARCHE', 'MARQUE', 'VILLE',
                'SEGMENT', 'SOUS_SEGMENT', 'GROUPE', 'DISTRIBUTEUR',
                'Date', 'VENTES'
            ]
            manquantes = [c for c in required_full if c not in df_full.columns]
            if not manquantes:
                imprimer_statut("PASS", f"Transaction-level: colonnes enrichies presentes")
            else:
                imprimer_statut("FAIL", f"Transaction-level: colonnes manquantes {manquantes}")
                problemes.append(f"data_prepared_final_full.csv colonnes manquantes: {manquantes}")
        else:
            imprimer_statut("WARN", "data_prepared_final_full.csv non trouvee")
            avertissements.append('data_prepared_final_full.csv manquant')
    except Exception as e:
        imprimer_statut("FAIL", f"Impossible de lire data_prepared_final_full.csv: {str(e)}")
        problemes.append('Erreur lecture data_prepared_final_full.csv')

    print(f"\n{BOLD}3. INTEGRITE DES DONNEES{RESET}")
    try:
        df = pd.read_csv(os.path.join(racine_projet, 'data_prepared_final.csv'))

        if 'TYPE_MARCHE' in df.columns:
            vp = (df['TYPE_MARCHE'] == 'VP').sum()
            vu = (df['TYPE_MARCHE'] == 'VU').sum()
            imprimer_statut("PASS", f"Repartition TYPE_MARCHE: VP={vp:,}, VU={vu:,}")
        else:
            imprimer_statut("FAIL", "Colonne TYPE_MARCHE manquante")
            problemes.append("Colonne TYPE_MARCHE requise")

        if 'VENTES' in df.columns:
            couverture_ventes = df['VENTES'].notna().sum() / len(df) * 100
            if couverture_ventes == 100:
                imprimer_statut("PASS", f"VENTES: {couverture_ventes:.1f}% couverture, {df['VENTES'].mean():.1f} moyenne")
            else:
                imprimer_statut("WARN", f"VENTES: {couverture_ventes:.1f}% couverture (certains NaN)")
                avertissements.append(f"VENTES a {100-couverture_ventes:.1f}% NaN")

            zero_ventes = (df['VENTES'] == 0).sum()
            pct_zero = 100 * zero_ventes / len(df)
            if pct_zero < 35:
                imprimer_statut("PASS", f"Jours sans ventes: {pct_zero:.1f}%")
            else:
                imprimer_statut("WARN", f"Jours sans ventes: {pct_zero:.1f}% (eleve)")
                avertissements.append(f"Taux de jours sans ventes eleve: {pct_zero:.1f}%")

    except Exception as e:
        imprimer_statut("FAIL", f"Verification integrite donnees a echoue: {str(e)}")

    print(f"\n{BOLD}4. REPARTITION TRAIN/VAL/TEST{RESET}")
    try:
        donnees_train = pd.read_csv(os.path.join(racine_projet, 'data_train.csv'))
        donnees_val = pd.read_csv(os.path.join(racine_projet, 'data_validation_2024.csv'))
        donnees_test = pd.read_csv(os.path.join(racine_projet, 'data_test_2025.csv'))

        imprimer_statut("PASS", f"Train: {len(donnees_train):,} lignes")
        imprimer_statut("PASS", f"Val:   {len(donnees_val):,} lignes")
        imprimer_statut("PASS", f"Test:  {len(donnees_test):,} lignes")

        if all('Date' in x.columns for x in [donnees_train, donnees_val, donnees_test]):
            donnees_train['Date'] = pd.to_datetime(donnees_train['Date'])
            donnees_val['Date'] = pd.to_datetime(donnees_val['Date'])
            donnees_test['Date'] = pd.to_datetime(donnees_test['Date'])

            chevauchement_val = len(donnees_val) > 0 and donnees_train['Date'].max() >= donnees_val['Date'].min()
            chevauchement_test = len(donnees_test) > 0 and len(donnees_val) > 0 and donnees_val['Date'].max() >= donnees_test['Date'].min()

            if not chevauchement_val and not chevauchement_test:
                imprimer_statut("PASS", "Plages de dates: pas de chevauchement")
            else:
                imprimer_statut("FAIL", "Plages de dates: chevauchement detecte")
                problemes.append("Chevauchement dates train/val/test")

    except Exception as e:
        imprimer_statut("WARN", f"Impossible de verifier les repartitions: {str(e)}")

    print(f"\n{BOLD}5. FICHIERS VISUALISATION{RESET}")
    pngs_attendus = [
        '00_Ventes_Neufs_vs_Occasion.png',
        '01_Ventes_Over_Time.png',
        '02_Ventes_Par_Annee.png',
        '03_Saisonnalite.png',
        '04_Top_Marques.png',
        '05_BoxPlot_Outliers.png',
        '06_Ventes_Par_Segment.png',
        '07_Ventes_Par_Sous_Segment.png',
        '08_Repartition_Marche.png',
        '09_Ventes_Par_Continent.png',
        '10_Ventes_Par_Groupe.png',
        '11_Evolution_Marques_ARTES.png',
        '11_ML_Preparation_Summary.png',
    ]

    pngs_manquants = [
        png for png in pngs_attendus if not os.path.exists(os.path.join(racine_projet, png))
    ]

    if not pngs_manquants:
        imprimer_statut("PASS", f"Les {len(pngs_attendus)} fichiers visualisations presents")
    else:
        imprimer_statut("WARN", f"Manquant {len(pngs_manquants)}/{len(pngs_attendus)} PNG")
        for png in pngs_manquants:
            avertissements.append(f"Manquant: {png}")

    print(f"\n{BOLD}6. STATUT DONNEES 2026{RESET}")
    try:
        donnees_2026 = pd.read_csv(os.path.join(racine_projet, 'data_future_2026.csv'))
        colonnes_ventes = [
            'VENTES', 'VENTES_LAG1', 'VENTES_LAG7', 'VENTES_LAG30', 'VENTES_LAG90',
            'VENTES_MA7', 'VENTES_MA30', 'VENTES_MA90', 'VENTES_JOUR_SUIVANT',
        ]

        est_placeholder = all(
            donnees_2026[col].isna().all() for col in colonnes_ventes if col in donnees_2026.columns
        )

        if est_placeholder:
            imprimer_statut("WARN", "2026: PLACEHOLDER (structure OK, donnees ventes NaN)")
            print("      IMPORTANT: Ce fichier est un template de verification")
            print("      Il sera rempli avec les vraies donnees 2026 ulterieurement")
        else:
            if all(col in donnees_2026.columns for col in ['Date', 'TYPE_MARCHE']):
                imprimer_statut("PASS", f"2026: {len(donnees_2026):,} lignes avec structure temporelle")
                imprimer_statut("WARN", "2026: Donnees ventes partiellement remplies (a verifier)")
            else:
                imprimer_statut("FAIL", "2026: Structure temporelle manquante")
                problemes.append("2026 manque colonnes Date/TYPE_MARCHE")

    except Exception as e:
        imprimer_statut("WARN", f"Impossible de verifier 2026: {str(e)}")

    print(f"\n{BOLD}{'=' * 70}")
    print("RESUME PRET")
    print(f"{'=' * 70}{RESET}")

    if not problemes:
        print(f"\n{VERT}{BOLD}PRET POUR STEP 6 MODELING{RESET}")
        print("\nToutes les verifications critiques sont passees.")
        print(f"Suivant: {BOLD}python step6_modeling.py{RESET}")
        print(f"\n{JAUNE}Note: data_future_2026.csv est un placeholder.")
        print(f"Il sera remplace avec donnees reelles 2026 pour validation finale.{RESET}\n")
    elif len(problemes) <= 2:
        print(f"\n{JAUNE}{BOLD}CORRIGER {len(problemes)} PROBLEME(S) AVANT DE CONTINUER{RESET}")
        for probleme in problemes:
            print(f"  - {probleme}")
        if avertissements:
            print("\nAvertissements (non critique):")
            for aver in avertissements:
                print(f"  - {aver}")
        print()
    else:
        print(f"\n{ROUGE}{BOLD}{len(problemes)} PROBLEME(S) CRITIQUE(S) TROUVE(S){RESET}")
        for probleme in problemes:
            print(f"  - {probleme}")
        print(f"\n{ROUGE}Impossible de proceder avec Step 6 tant que les problemes ne sont pas resolus.{RESET}\n")


if __name__ == '__main__':
    main()
