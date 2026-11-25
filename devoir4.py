"""
MATH 60231 - Devoir #4
Problème 1 a) : chargement des données de la bibliothèque de Ken French.

Source des données : Kenneth R. French Data Library
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
(fichiers CSV téléchargés et placés dans le même dossier que ce script)
"""

import io
import re
from pathlib import Path

import pandas as pd

# Dossier contenant les CSV (le même que ce script)
DATA_DIR = Path(__file__).resolve().parent

# Dossier où sont enregistrés tous les tableaux et graphiques produits par ce code
RESULTS_DIR = DATA_DIR / "resultats"

# Période d'échantillonnage demandée
START, END = "1980-01", "2021-12"

# Ordre des colonnes exigé pour `factors`
FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]

# Expression régulière d'une ligne de données : date AAAAMM ou AAAA, puis une virgule
_DATA_ROW = re.compile(r"^\s*\d{4,6}\s*,")


def save_table(df, question, titre):
    """
    Enregistre un tableau dans resultats/ sous le nom <question>_<titre>.csv
    (p. ex. Q1a_factors_statistiques_descriptives.csv), pour identifier
    clairement la question et le contenu de chaque fichier.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{question}_{titre}.csv"
    df.to_csv(path, float_format="%.4f")  # 4 décimales : évite le bruit de virgule flottante
    print(f"  -> {path.relative_to(DATA_DIR)}")
    return path


def save_figure(fig, question, titre):
    """Enregistre un graphique dans resultats/ sous le nom <question>_<titre>.png."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{question}_{titre}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  -> {path.relative_to(DATA_DIR)}")
    return path


def descriptive_stats(df):
    """Statistiques descriptives usuelles + asymétrie et kurtosis (en excès)."""
    stats = df.describe().T
    stats["asymetrie"] = df.skew()
    stats["kurtosis_exces"] = df.kurt()
    return stats


def _read_ff_block(filename, block=0):
    """
    Lit le `block`-ième tableau (0 = le premier) d'un CSV de Ken French.

    Les fichiers de Ken French empilent plusieurs tableaux dans un même CSV
    (p. ex. mensuel puis annuel ; pondéré par la valeur puis équipondéré, etc.),
    séparés par des lignes de texte. Chaque tableau commence par une ligne
    d'en-tête débutant par une virgule (la colonne de dates n'a pas de nom),
    suivie de lignes dont le premier champ est une date AAAAMM (ou AAAA).
    On isole donc uniquement les lignes du tableau voulu avant de le parser.
    """
    lines = (DATA_DIR / filename).read_text(encoding="latin-1").splitlines()

    # Indices des lignes d'en-tête (une par tableau)
    headers = [i for i, line in enumerate(lines) if line.startswith(",")]
    start = headers[block]

    # Le tableau se termine à la première ligne qui n'est pas une ligne de données
    stop = start + 1
    while stop < len(lines) and _DATA_ROW.match(lines[stop]):
        stop += 1

    df = pd.read_csv(io.StringIO("\n".join(lines[start:stop])),
                     index_col=0, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    # Index de dates mensuel (AAAAMM -> période mensuelle)
    df.index = pd.PeriodIndex(df.index.astype(str).str.strip(), freq="M")
    df.index.name = "Date"

    # Ken French code les valeurs manquantes par -99.99 ou -999
    df = df.mask(df <= -99.99)
    return df


def Q1_KenFrenchData():
    """
    Retourne deux DataFrames mensuels (janvier 1980 à décembre 2021),
    en pourcentage par mois, tous en excédents de rendement :

    factors    : Mkt-RF, SMB, HML, RMW, CMA, Mom
                 - Mkt-RF est déjà le rendement du marché moins le taux sans risque.
                 - SMB, HML, RMW, CMA et Mom sont des portefeuilles longs-courts
                   autofinancés (rendement d'un portefeuille moins celui d'un autre) :
                   le taux sans risque s'annule, ce sont déjà des excédents de rendement.
                 -> aucune transformation nécessaire.
    portfolios : 17 portefeuilles sectoriels (pondérés par la valeur), moins RF.
                 Les rendements bruts doivent être nets du taux sans risque.
    """
    # Premier tableau de chaque fichier = données mensuelles
    ff5 = _read_ff_block("F-F_Research_Data_5_Factors_2x3.csv", block=0)
    mom = _read_ff_block("F-F_Momentum_Factor.csv", block=0)
    # Premier tableau = "Average Value Weighted Returns -- Monthly"
    ind = _read_ff_block("17_Industry_Portfolios.csv", block=0)

    # Restriction à l'échantillon 1980-01 à 2021-12
    ff5, mom, ind = (x.loc[START:END] for x in (ff5, mom, ind))

    # Facteurs : 5 facteurs FF + momentum, dans l'ordre demandé
    factors = ff5.join(mom, how="inner")[FACTOR_COLS]

    # Portefeuilles : excédents de rendement = rendement brut - taux sans risque
    # (soustraction vectorielle ligne par ligne, alignée sur les dates)
    portfolios = ind.sub(ff5["RF"], axis=0)

    # Vérifications : échantillon complet, dates alignées, aucune valeur manquante
    n_months = len(pd.period_range(START, END, freq="M"))
    assert len(factors) == len(portfolios) == n_months
    assert factors.index.equals(portfolios.index)
    assert not factors.isna().any().any() and not portfolios.isna().any().any()

    return factors, portfolios


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    # ------------------------------------------------------------------
    # Problème 1 a) : données de Ken French (excédents de rendement, % par mois)
    # ------------------------------------------------------------------
    factors, portfolios = Q1_KenFrenchData()

    for name, df in [("factors", factors), ("portfolios", portfolios)]:
        print(f"\n===== Q1a - {name} : {df.shape[0]} mois x {df.shape[1]} séries "
              f"({df.index[0]} à {df.index[-1]}), en % par mois =====")
        head, stats = df.head(), descriptive_stats(df)
        print("\n5 premières lignes :")
        print(head.round(2))
        print("\nStatistiques descriptives :")
        print(stats.round(2))

        # Exportation dans resultats/ : données complètes (dont les 5 premières lignes)
        # et statistiques descriptives
        save_table(df, "Q1a", f"{name}_exces_rendements_mensuels_1980-2021")
        save_table(stats, "Q1a", f"{name}_statistiques_descriptives")
