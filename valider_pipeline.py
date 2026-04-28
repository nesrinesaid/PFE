"""
Utilitaires de validation du pipeline pour le PFE.
Utilise par les etapes 4, 5 et 6 pour verifier l'integrite des donnees.
"""


def valider_colonnes(df, colonnes_requises, nom_etape, verbose=True):
    """
    Valider que le dataframe contient toutes les colonnes requises.

    Args:
        df: DataFrame Pandas a valider
        colonnes_requises: Liste des noms de colonnes qui doivent exister
        nom_etape: Nom de l'etape appelante (ex: 'step4_eda')
        verbose: Imprimer le rapport de validation si True

    Returns:
        True si la validation reussit

    Raises:
        ValueError si des colonnes requises manquent
    """
    manquantes = [col for col in colonnes_requises if col not in df.columns]

    if manquantes:
        disponibles = list(df.columns)
        message_erreur = (
            f"\n❌ ERREUR DE VALIDATION dans {nom_etape}\n"
            f"   Colonnes requises manquantes: {manquantes}\n"
            f"   Colonnes disponibles: {disponibles}\n"
            f"   Cela signifie generalement qu'une etape precedente a echoue ou a ete ignoree.\n"
            f"   Depannage:\n"
            f"   1. Verifier que les etapes precedentes se sont completees avec succes\n"
            f"   2. Verifier le chemin et le contenu du fichier d'entree\n"
            f"   3. Examiner les messages d'erreur des etapes precedentes"
        )
        raise ValueError(message_erreur)

    if verbose:
        print(f"✅ Validation des colonnes {nom_etape} reussie")
        print(f"   Colonnes requises: {colonnes_requises}")
        print(f"   Total colonnes dans les donnees: {len(df.columns)}")

    return True


def valider_completude_donnees(df, col, nom_etape, couverture_min=0.5, verbose=True):
    """
    Valider qu'une colonne a suffisamment de valeurs non-NaN.

    Args:
        df: DataFrame Pandas
        col: Nom de la colonne a verifier
        nom_etape: Nom de l'etape appelante
        couverture_min: Couverture minimale % (0.5 = 50%)
        verbose: Imprimer le rapport si True

    Returns:
        Pourcentage de couverture

    Raises:
        ValueError si la couverture est en-dessous du seuil
    """
    if col not in df.columns:
        raise ValueError(f"Colonne '{col}' non trouvee dans le dataframe")

    if len(df) == 0:
        raise ValueError(f"{nom_etape}: DataFrame vide, impossible de verifier la completude")

    couverture = df[col].notna().sum() / len(df)

    if couverture < couverture_min:
        raise ValueError(
            f"❌ {nom_etape}: Colonne '{col}' a une couverture insuffisante\n"
            f"   Couverture: {couverture*100:.1f}% (minimum requis: {couverture_min*100:.0f}%)\n"
            f"   Cela indique un probleme de qualite des donnees ou d'echec d'enrichissement"
        )

    if verbose:
        print(f"✅ {col}: {couverture*100:.1f}% couverture (requise: {couverture_min*100:.0f}%)")

    return couverture


COLONNES_REQUISES = {
    'step4_eda': [
        'DATV',
        'MARQUE',
        'IM_RI',
        'ANNEE', 'MOIS', 'ANNEE_MOIS',
        'TYPE_MARCHE',
    ],
    'step5_preparation': [
        'DATV',
        'IM_RI',
        'TYPE_MARCHE',
        'MARQUE',
    ],
    'step6_modeling': [
        'Date',
        'TYPE_MARCHE',
        'VENTES',
        'VENTES_LAG1',
        'VENTES_LAG7',
        'VENTES_LAG30',
        'VENTES_LAG90',
        'VENTES_MA7',
        'VENTES_MA30',
        'VENTES_MA90',
        'Mois_1', 'Mois_2',
        'JS_0',
        'EST_RAMADAN',
    ],
    'step5_preparation_full': [
        'DATV', 'IM_RI', 'TYPE_MARCHE', 'MARQUE', 'VILLE',
        'SEGMENT', 'SOUS_SEGMENT', 'GROUPE', 'DISTRIBUTEUR',
        'Date', 'VENTES',
        'VENTES_LAG1', 'VENTES_LAG7', 'VENTES_LAG30', 'VENTES_LAG90',
        'VENTES_MA7', 'VENTES_MA30', 'VENTES_MA90',
    ],
}
