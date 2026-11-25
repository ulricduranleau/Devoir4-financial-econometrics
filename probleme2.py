"""
MATH 60231 - Devoir #4
Problème 2 : estimation par maximum de vraisemblance.

Variable dépendante : excédent de rendement du marché (Mkt-RF).
Régresseurs : SMB, HML, RMW, CMA, Mom.
Données : le même jeu de données mensuel 1980-2021 qu'au Problème 1
(fonction Q1_KenFrenchData de probleme1.py), en % par mois.

Choix de modélisation :
- sigma chapeau = estimateur du maximum de vraisemblance de l'écart-type des
  résidus, sqrt(SSR / T) : cohérent avec le cadre de vraisemblance et traite le
  modèle contraint et le non contraint de façon identique dans le test RV ;
- les résidus standardisés e_t / sigma suivent une loi de Student t(dof)
  standard (non ramenée à une variance de 1), comme dans l'énoncé.
Tous les tableaux (.csv) et graphiques (.png) sont enregistrés dans
resultats_probleme2/, sous le nom <question>_<titre>.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter
from scipy import stats as sps
from scipy.special import gammaln

from probleme1 import BLUE, SERIES_COLORS, TEXT, TEXT_2, FR_AXIS, Q1_KenFrenchData, _fr

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


# ======================================================================
# Problème 2 a) : log-vraisemblance MCO sous des résidus de Student
# ======================================================================

def Q2_OLSStudentT(df, dof):
    """
    (i) MCO de la première colonne de df sur une constante et les autres colonnes ;
    (ii) résidus e_t et leur écart-type estimé sigma = sqrt(SSR / T) (EMV) ;
    (iii) log-vraisemblance du modèle si z_t = e_t / sigma sont i.i.d. t(dof).

    La densité d'un rendement y_t est celle de e_t = sigma * z_t, soit
    f(e_t) = f_t(e_t / sigma; dof) / sigma (changement de variable). D'où :

        llf = sum_t log f_t(z_t; dof) - T log(sigma),

    avec la densité de Student
        log f_t(z; v) = log G((v+1)/2) - log G(v/2) - 0.5 log(v pi)
                        - (v+1)/2 log(1 + z²/v),   G = fonction gamma.

    Le terme -T log(sigma) rend comparables des modèles dont le sigma diffère
    (indispensable pour le test RV de la partie b).
    """
    y = df.iloc[:, 0].to_numpy()
    X = np.column_stack([np.ones(len(df)), df.iloc[:, 1:].to_numpy()])
    T = X.shape[0]

    # (i) MCO : b = (X'X)^-1 X'y
    b = np.linalg.solve(X.T @ X, X.T @ y)
    # (ii) résidus et écart-type estimé par maximum de vraisemblance (division par T)
    e = y - X @ b
    sigma = np.sqrt(e @ e / T)
    # (iii) log-vraisemblance de Student des résidus standardisés
    z = e / sigma
    log_f = (gammaln((dof + 1) / 2) - gammaln(dof / 2) - 0.5 * np.log(dof * np.pi)
             - (dof + 1) / 2 * np.log1p(z ** 2 / dof))
    return float(log_f.sum() - T * np.log(sigma))


# ======================================================================
# Problème 2 b) : test du rapport de vraisemblance
# ======================================================================

def Q2_TestRV(df, dof):
    """
    Test RV de H0 : tous les coefficients autres que la constante sont nuls.

    Modèle non contraint : y sur une constante et les régresseurs de df.
    Modèle contraint (H0) : y sur une constante seulement (df réduit à sa
    première colonne), avec le même dof.

        RV = 2 (llf_non_contraint - llf_contraint) ~ chi²(q) sous H0,

    où q = nombre de régresseurs exclus (ici 5). Retourne (RV, valeur p).
    """
    llf_nc = Q2_OLSStudentT(df, dof)
    llf_c = Q2_OLSStudentT(df.iloc[:, [0]], dof)
    rv = 2 * (llf_nc - llf_c)
    q = df.shape[1] - 1
    return rv, float(sps.chi2.sf(rv, q))


# ======================================================================
# Problème 2 c) : vraisemblance profilée en fonction des degrés de liberté
# ======================================================================

def Q2_VraisemblanceProf(df):
    """
    Évalue Q2_OLSStudentT(df, dof_j) pour dof_j = exp(j/12), j = 0, ..., 50
    (de e^0 = 1 à e^(50/12) ≈ 64,5), trace la log-vraisemblance profilée en fonction des
    degrés de liberté et retourne un tableau 51 x 2 : [dof_j, llf_j].
    """
    dof = np.exp(np.arange(51) / 12)
    llf = np.array([Q2_OLSStudentT(df, d) for d in dof])
    prof = np.column_stack([dof, llf])

    # Graphique : échelle log en abscisse, car la grille est régulière en log(dof)
    j = int(np.argmax(llf))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(dof, llf, color=BLUE, lw=1.6, marker="o", ms=3)
    ax.scatter(dof[j], llf[j], s=60, color=SERIES_COLORS[1], zorder=3,
               edgecolor="white", linewidth=1)
    ax.annotate(f"Maximum : dof ≈ {_fr(dof[j], 1)} (j = {j})\nllf = {_fr(llf[j], 1)}",
                xy=(dof[j], llf[j]), xytext=(0, -38), textcoords="offset points",
                ha="center", fontsize=8, color=TEXT,
                arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.8))
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _fr(v, 0)))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    ax.yaxis.set_major_formatter(FR_AXIS)
    ax.set_xlabel("Degrés de liberté de la loi de Student, dof (échelle log)")
    ax.set_ylabel("Log-vraisemblance")
    ax.set_title("Vraisemblance profilée des degrés de liberté\n"
                 "Régression de Mkt-RF sur SMB, HML, RMW, CMA et Mom, 1980-2021",
                 loc="left", fontsize=9.5)
    save_figure(fig, "Q2c", "fig1_vraisemblance_profilee")
    plt.close(fig)

    return prof


# Nom de la fonction dans la version anglaise de l'énoncé
Q2_ProfileLikelihood = Q2_VraisemblanceProf


if __name__ == "__main__":
    df = Q2_Donnees()

    # a) Log-vraisemblances avec dof = 5 et dof = 30
    llf = pd.DataFrame({"log_vraisemblance": {d: Q2_OLSStudentT(df, d) for d in (5, 30)}})
    llf.index.name = "dof"
    save_table(llf, "Q2a", "log_vraisemblances_dof5_dof30")

    # b) Test RV avec dof = 5 (question principale) et dof = 30 (question iii) ;
    #    les log-vraisemblances des deux modèles servent à répondre à (i)
    rows = {}
    for d in (5, 30):
        rv, p = Q2_TestRV(df, d)
        rows[d] = {"llf_non_contraint": Q2_OLSStudentT(df, d),
                   "llf_contraint_constante_seule": Q2_OLSStudentT(df.iloc[:, [0]], d),
                   "statistique_RV": rv, "ddl": df.shape[1] - 1, "p_valeur": p}
    rv_tab = pd.DataFrame(rows).T
    rv_tab.index.name = "dof"
    # Valeur p en notation scientifique : elle afficherait 0,0000 avec 4 décimales
    rv_tab["p_valeur"] = rv_tab["p_valeur"].map(lambda v: f"{v:.2e}")
    save_table(rv_tab, "Q2b", "test_rapport_vraisemblance")

    # c) Vraisemblance profilée (graphique enregistré par la fonction)
    prof = Q2_VraisemblanceProf(df)
    save_table(pd.DataFrame(prof, columns=["dof", "log_vraisemblance"],
                            index=pd.RangeIndex(51, name="j")),
               "Q2c", "vraisemblance_profilee")
