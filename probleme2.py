"""
MATH 60231 - Devoir #4
Problème 2 : estimation par maximum de vraisemblance.

Variable dépendante : excédent de rendement du marché (Mkt-RF).
Régresseurs : SMB, HML, RMW, CMA, Mom.
Données : le même jeu de données mensuel 1980-2021 qu'au Problème 1
(fonction Q1_KenFrenchData de probleme1.py).

Tous les tableaux (.csv) et graphiques (.png) sont enregistrés dans
resultats_probleme2/, sous le nom <question>_<titre>.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from probleme1 import Q1_KenFrenchData

# Dossier où sont enregistrés les tableaux et graphiques du Problème 2
RESULTS_DIR = Path(__file__).resolve().parent / "resultats_probleme2"


def save_table(df, question, titre):
    """Enregistre un tableau dans resultats_probleme2/ sous le nom <question>_<titre>.csv."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{question}_{titre}.csv"
    df.to_csv(path, float_format="%.4f")
    return path


def save_figure(fig, question, titre):
    """Enregistre un graphique dans resultats_probleme2/ sous le nom <question>_<titre>.png."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{question}_{titre}.png"
    # Sans la version de matplotlib dans le PNG : fichier identique d'une machine à l'autre
    fig.savefig(path, dpi=200, bbox_inches="tight", metadata={"Software": None})
    return path


def Q2_Donnees():
    """
    DataFrame du Problème 2 : première colonne = variable dépendante (Mkt-RF),
    colonnes suivantes = régresseurs (SMB, HML, RMW, CMA, Mom), mensuel 1980-2021.
    """
    factors, _ = Q1_KenFrenchData()
    return factors[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]]


if __name__ == "__main__":
    df = Q2_Donnees()
