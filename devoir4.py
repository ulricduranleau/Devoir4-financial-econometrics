"""
MATH 60231 - Devoir #4
Problème 1 a) : chargement des données de la bibliothèque de Ken French.
Problème 1 b) : statistiques descriptives et graphiques.
Problème 1 c) : modèle linéaire d'évaluation d'actifs en deux passes, codé à la
                main et vérifié avec linearmodels (pip install linearmodels).

Tous les tableaux (.csv) et graphiques (.png) sont enregistrés dans resultats/,
sous le nom <question>_<titre> (p. ex. Q1b_fig3_correlations_facteurs_heatmap.png).

Sources :
- Données : Kenneth R. French Data Library
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
  (fichiers CSV téléchargés et placés dans le même dossier que ce script)
- Dates des récessions américaines : NBER Business Cycle Dating Committee
  https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
"""

import io
import re
import textwrap
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy import stats as sps

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


# ======================================================================
# Problème 1 b) : statistiques descriptives et graphiques
# ======================================================================

# Récessions américaines selon le NBER : (mois du pic, mois du creux), sur 1980-2021.
# Source : tableau officiel du NBER (mis à jour le 14/03/2023)
# https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
# Convention du NBER : la récession commence le mois APRÈS le pic et se termine
# le mois du creux (inclus).
NBER_RECESSIONS = [("1980-01", "1980-07"), ("1981-07", "1982-11"), ("1990-07", "1991-03"),
                   ("2001-03", "2001-11"), ("2007-12", "2009-06"), ("2020-02", "2020-04")]

# Libellés lisibles pour les titres des graphiques
FACTOR_LABELS = {"Mkt-RF": "Marché (Mkt-RF)", "SMB": "Taille (SMB)",
                 "HML": "Valeur (HML)", "RMW": "Rentabilité (RMW)",
                 "CMA": "Investissement (CMA)", "Mom": "Momentum (Mom)"}
# Noms officiels (en anglais) des 17 industries de Ken French, tels qu'écrits dans
# le fichier Siccodes17.txt de l'archive
# https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes17.zip
INDUSTRY_NAMES = {
    "Food": "Food",
    "Mines": "Mining and Minerals",
    "Oil": "Oil and Petroleum Products",
    "Clths": "Textiles, Apparel & Footwear",
    "Durbl": "Consumer Durables",
    "Chems": "Chemicals",
    "Cnsum": "Drugs, Soap, Perfumes, Tobacco",
    "Cnstr": "Construction and Construction Materials",
    "Steel": "Steel Works Etc",
    "FabPr": "Fabricated Products",
    "Machn": "Machinery and Business Equipment",
    "Cars": "Automobiles",
    "Trans": "Transportation",
    "Utils": "Utilities",
    "Rtail": "Retail Stores",
    "Finan": "Banks, Insurance Companies, and Other Financials",
    "Other": "Other",
}
# Libellé des graphiques : « nom officiel (code) », ou le code seul s'il est identique
INDUSTRY_LABELS = {c: c if n == c else f"{n} ({c})" for c, n in INDUSTRY_NAMES.items()}

# Couleurs : palette catégorielle validée pour le daltonisme (ordre fixe),
# encres de texte neutres, palette divergente bleu <-> rouge avec milieu gris.
SERIES_COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17"]
BLUE = SERIES_COLORS[0]
TEXT, TEXT_2, GRID, RECESSION = "#191919", "#666666", "#e6e6e6", "#e3e3e3"
CORR_CMAP = LinearSegmentedColormap.from_list(
    "bleu_gris_rouge", ["#0f4a96", "#84b9f3", "#efefef", "#f4a3a3", "#b52a2b"])

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": TEXT_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def _fr(x, d=2):
    """Formate un nombre à la française : virgule décimale et vrai signe moins."""
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


FR_AXIS = FuncFormatter(lambda v, _: _fr(v, 0))  # graduations d'axes à la française


def _corr_colorbar(fig, im, ax):
    """Barre de couleur des corrélations, graduée de -1 à 1 par pas de 0,5."""
    cb = fig.colorbar(im, ax=ax, shrink=0.8, label="Corrélation de Pearson",
                      ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _fr(v, 1)))
    return cb


def annualized_stats(df):
    """
    Statistiques par série (rendements mensuels en %), y compris annualisées :
    moyenne x 12, volatilité x sqrt(12), ratio de Sharpe x sqrt(12) (moyenne de
    l'excédent de rendement / écart-type), statistique t de H0 : moyenne = 0 et
    intervalle de confiance à 95 % de la moyenne annualisée. L'asymétrie et la
    kurtosis en excès décrivent l'épaisseur des queues (cf. Session 1 : les
    rendements financiers ont des queues plus épaisses que la loi normale).
    """
    T = len(df)
    m, s = df.mean(), df.std()
    se = s / np.sqrt(T)
    t = m / se
    q = sps.t.ppf(0.975, T - 1)
    return pd.DataFrame({
        "moyenne_mens_%": m,
        "ecart_type_mens_%": s,
        "moyenne_ann_%": 12 * m,
        "volatilite_ann_%": np.sqrt(12) * s,
        "ratio_Sharpe_ann": np.sqrt(12) * m / s,  # séries déjà en excédents de rendement
        "t_stat_moyenne": t,
        "p_valeur_moyenne": 2 * sps.t.sf(t.abs(), T - 1),
        "IC95_bas_moyenne_ann_%": 12 * (m - q * se),
        "IC95_haut_moyenne_ann_%": 12 * (m + q * se),
        "asymetrie": df.skew(),
        "kurtosis_exces": df.kurt(),
        "pire_mois_%": df.min(),
        "date_pire_mois": df.idxmin().astype(str),
        "meilleur_mois_%": df.max(),
        "date_meilleur_mois": df.idxmax().astype(str),
    })


def _shade_recessions(ax):
    """Ombre les récessions NBER (du mois suivant le pic à la fin du mois du creux)."""
    for a, b in NBER_RECESSIONS:
        ax.axvspan((pd.Period(a, "M") + 1).to_timestamp(),
                   pd.Period(b, "M").to_timestamp(how="end"),
                   color=RECESSION, lw=0, zorder=0)


RECESSION_HANDLE = Patch(color=RECESSION, label="Récession (NBER)")


def _small_multiples(df, labels, ncols, title, panel_w=3.6):
    """
    Un panneau par série (rendements mensuels en %), même échelle verticale pour
    toutes, afin de comparer directement les volatilités. Chaque panneau indique
    la volatilité annualisée et étiquette le pire mois.
    """
    n = df.shape[1]
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, 1.75 * nrows + 0.9),
                             sharex=True, sharey=True)
    axes = axes.ravel()
    x = df.index.to_timestamp()
    vol = np.sqrt(12) * df.std()

    for ax, col in zip(axes, df.columns):
        _shade_recessions(ax)
        ax.axhline(0, color=TEXT_2, lw=0.6)
        ax.plot(x, df[col], color=BLUE, lw=0.6)
        # Titre sur deux lignes s'il est trop long pour la largeur du panneau
        ax.set_title(textwrap.fill(labels[col], 50), loc="left")
        ax.text(0.99, 0.97, f"σ annualisée = {_fr(vol[col], 1)} %", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color=TEXT)
        # Étiquette sélective : pire mois de la série
        worst = df[col].idxmin()
        right = worst.year > 2008  # texte à gauche du point s'il est en fin d'échantillon
        ax.annotate(f"{_fr(df[col].min(), 1)} % ({worst})",
                    xy=(worst.to_timestamp(), df[col].min()),
                    xytext=(-5 if right else 5, 0), textcoords="offset points",
                    ha="right" if right else "left", va="center", fontsize=7, color=TEXT_2)

    # Panneaux vides (grille incomplète) : on les masque et on remet les dates
    # sous le panneau du dessus.
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
        axes[i - ncols].tick_params(labelbottom=True)
    for ax in axes[::ncols]:
        ax.set_ylabel("% par mois")
    # Marge en haut pour que l'étiquette de volatilité ne chevauche pas les pics
    lo, hi = df.min().min(), df.max().max()
    axes[0].set_ylim(lo - 0.08 * (hi - lo), hi + 0.30 * (hi - lo))
    axes[0].yaxis.set_major_formatter(FR_AXIS)
    axes[0].xaxis.set_major_locator(mdates.YearLocator(10))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11, color=TEXT)
    fig.legend(handles=[RECESSION_HANDLE], loc="lower left", frameon=False,
               bbox_to_anchor=(0.01, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


def _corr_heatmap(ax, corr, title):
    """Carte de chaleur annotée (triangle inférieur), échelle divergente de -1 à 1."""
    k = len(corr)
    mask = np.triu(np.ones((k, k), dtype=bool), k=1)
    im = ax.imshow(np.ma.array(corr.values, mask=mask), cmap=CORR_CMAP, vmin=-1, vmax=1)
    for i in range(k):
        for j in range(i + 1):
            v = corr.iat[i, j]
            ax.text(j, i, _fr(v, 2), ha="center", va="center", fontsize=8.5,
                    color="white" if abs(v) > 0.6 else TEXT)
    ax.set_xticks(range(k), corr.columns)
    ax.set_yticks(range(k), corr.index)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left")
    return im


def _label_line_ends(ax, x_end, values, labels, min_gap=0.075):
    """
    Étiquettes directes en bout de courbe (échelle log), décalées verticalement
    au besoin pour ne pas se chevaucher.
    """
    order = np.argsort(values)
    pos = np.log10(np.asarray(values, dtype=float))
    placed = []
    for i in order:
        y = pos[i] if not placed else max(pos[i], placed[-1] + min_gap)
        placed.append(y)
        ax.text(x_end, 10 ** y, labels[i], fontsize=8, color=TEXT, va="center", ha="left")


def Q1b_StatistiquesGraphiques(factors, portfolios):
    """
    Problème 1 b) : statistiques descriptives et graphiques.

    Graphiques (resultats/Q1b_fig*.png) :
      1. séries temporelles des 6 facteurs                      [exigé (i)]
      2. séries temporelles des 17 portefeuilles sectoriels     [exigé (ii)]
      3. carte de chaleur des corrélations entre facteurs       [exigé (iii)]
      4. corrélations par sous-période (1980-2000 vs 2001-2021) : stabilité
      5. volatilité annualisée des facteurs : « quels facteurs sont les plus volatils ? »
      6. excédent moyen des portefeuilles avec IC 95 % : « plus élevés / plus bas ? »
      7. rendements cumulés des facteurs : primes réalisées dans le temps
    Tableaux (resultats/Q1b_*.csv) : statistiques annualisées des facteurs et des
    portefeuilles, matrice de corrélation des facteurs et corrélations par paire
    sur deux sous-périodes (« colinéarité potentielle » dans les régressions).
    """
    Q = "Q1b"
    period = f"{factors.index[0]} à {factors.index[-1]}"
    x = factors.index.to_timestamp()

    # ---------------- Tableaux ----------------
    stats_f = annualized_stats(factors).sort_values("volatilite_ann_%", ascending=False)
    stats_p = annualized_stats(portfolios).sort_values("moyenne_ann_%", ascending=False)
    save_table(stats_f, Q, "statistiques_annualisees_facteurs")
    save_table(stats_p, Q, "statistiques_annualisees_portefeuilles")

    # Corrélations : toute la période et deux sous-périodes de 21 ans
    corr = factors.corr()
    sub = {"1980-2000": factors.loc[:"2000"].corr(), "2001-2021": factors.loc["2001":].corr()}
    save_table(corr, Q, "matrice_correlation_facteurs_1980-2021")
    pairs = pd.DataFrame({
        name: c.where(np.tril(np.ones(c.shape, dtype=bool), k=-1)).stack()
        for name, c in {"1980-2021": corr, **sub}.items()})
    pairs.index = [f"{a} / {b}" for a, b in pairs.index]
    pairs["variation_2001-2021_moins_1980-2000"] = pairs["2001-2021"] - pairs["1980-2000"]
    pairs = pairs.reindex(pairs["1980-2021"].abs().sort_values(ascending=False).index)
    save_table(pairs, Q, "correlations_par_paire_sous_periodes")

    # ---------------- Graphiques ----------------
    # Fig. 1 (i) : séries temporelles des 6 facteurs
    fig = _small_multiples(factors, FACTOR_LABELS, 2,
                           f"Rendements mensuels des six facteurs, {period} (% par mois)")
    save_figure(fig, Q, "fig1_series_temporelles_facteurs")
    plt.close(fig)

    # Fig. 2 (ii) : séries temporelles des excédents des 17 portefeuilles
    fig = _small_multiples(portfolios, INDUSTRY_LABELS, 3,
                           "Excédents de rendement mensuels des 17 portefeuilles sectoriels, "
                           f"{period} (% par mois)", panel_w=4.4)
    save_figure(fig, Q, "fig2_series_temporelles_portefeuilles")
    plt.close(fig)

    # Fig. 3 (iii) : carte de chaleur des corrélations entre facteurs
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = _corr_heatmap(ax, corr, f"Corrélations entre les six facteurs, {period}")
    _corr_colorbar(fig, im, ax)
    save_figure(fig, Q, "fig3_correlations_facteurs_heatmap")
    plt.close(fig)

    # Fig. 4 : stabilité des corrélations, deux sous-périodes côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, (name, c) in zip(axes, sub.items()):
        im = _corr_heatmap(ax, c, name.replace("-", "–"))
    _corr_colorbar(fig, im, axes)
    fig.suptitle("Corrélations entre les facteurs par sous-période", x=0.01, ha="left",
                 fontweight="bold", fontsize=11, color=TEXT)
    save_figure(fig, Q, "fig4_correlations_facteurs_sous_periodes")
    plt.close(fig)

    # Fig. 5 : volatilité annualisée des facteurs (triée)
    v = stats_f["volatilite_ann_%"].sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.barh([FACTOR_LABELS[c] for c in v.index], v.values, color=BLUE, height=0.6)
    for i, val in enumerate(v.values):
        ax.text(val + 0.2, i, f"{_fr(val, 1)} %", va="center", fontsize=8, color=TEXT)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FR_AXIS)
    ax.set_xlabel("Volatilité annualisée (%)  =  écart-type mensuel × √12")
    ax.set_title(f"Volatilité des six facteurs, {period}", loc="left")
    save_figure(fig, Q, "fig5_volatilite_facteurs")
    plt.close(fig)

    # Fig. 6 : excédent de rendement moyen annualisé des portefeuilles, IC 95 %
    s = stats_p.sort_values("moyenne_ann_%")
    m = s["moyenne_ann_%"]
    err = np.vstack([m - s["IC95_bas_moyenne_ann_%"], s["IC95_haut_moyenne_ann_%"] - m])
    fig, ax = plt.subplots(figsize=(7, 5.4))
    y = np.arange(len(s))
    ax.barh(y, m.values, color=BLUE, height=0.6)
    ax.errorbar(m.values, y, xerr=err, fmt="none", ecolor=TEXT_2, elinewidth=1, capsize=2.5)
    ax.axvline(0, color=TEXT_2, lw=0.8)
    ax.set_yticks(y, [INDUSTRY_LABELS[c] for c in s.index])
    for i in (0, len(s) - 1):  # étiquettes sélectives : le plus bas et le plus haut
        ax.text(s["IC95_haut_moyenne_ann_%"].iloc[i] + 0.4, i, f"{_fr(m.iloc[i], 1)} %",
                va="center", fontsize=8, color=TEXT)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FR_AXIS)
    ax.set_xlabel("Excédent de rendement moyen annualisé (%), avec intervalle de confiance à 95 %")
    ax.set_title(f"Excédent de rendement moyen des 17 portefeuilles sectoriels, {period}",
                 loc="left")
    save_figure(fig, Q, "fig6_rendement_moyen_portefeuilles")
    plt.close(fig)

    # Fig. 7 : valeur de 1 $ investi dans chaque facteur (cumul des rendements), échelle log
    cum = (1 + factors / 100).cumprod()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    _shade_recessions(ax)
    for col, color in zip(factors.columns, SERIES_COLORS):
        ax.plot(x, cum[col], color=color, lw=1.6, label=FACTOR_LABELS[col])
    ax.set_yscale("log")
    ax.axhline(1, color=TEXT_2, lw=0.8)
    ax.set_yticks([0.5, 1, 2, 5, 10, 20])
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v_, _: _fr(v_, 0 if v_ >= 1 else 1)))
    end = cum.iloc[-1]
    _label_line_ends(ax, x[-1] + pd.Timedelta(days=150), end.values,
                     [f"{c} : {_fr(val, 1)} $" for c, val in end.items()])
    ax.set_xlim(x[0], x[-1] + pd.Timedelta(days=2600))
    ax.spines["bottom"].set_bounds(mdates.date2num(x[0]), mdates.date2num(x[-1]))
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Valeur de 1 $ investi en janvier 1980 (échelle log)")
    ax.set_title(f"Rendements cumulés des six facteurs, {period}", loc="left")
    ax.legend(handles=ax.get_lines()[:6] + [RECESSION_HANDLE], frameon=False,
              loc="upper left", fontsize=8, ncol=2)
    save_figure(fig, Q, "fig7_rendements_cumules_facteurs")
    plt.close(fig)

    # ---------------- Synthèse pour la discussion ----------------
    print("\n--- Q1b : éléments de réponse ---")
    print("Volatilité annualisée des facteurs (%), du plus au moins volatil :")
    print(stats_f["volatilite_ann_%"].round(2).to_string())
    print("\nExcédent moyen annualisé des portefeuilles (%), du plus haut au plus bas :")
    print(stats_p[["moyenne_ann_%", "IC95_bas_moyenne_ann_%", "IC95_haut_moyenne_ann_%",
                   "t_stat_moyenne", "volatilite_ann_%", "ratio_Sharpe_ann"]].round(2).to_string())
    print("\nCorrélations par paire (triées par |corr| sur 1980-2021) :")
    print(pairs.round(2).head(6).to_string())

    return {"stats_facteurs": stats_f, "stats_portefeuilles": stats_p,
            "correlation": corr, "correlations_sous_periodes": pairs}


# ======================================================================
# Problème 1 c) : modèle linéaire d'évaluation d'actifs en deux passes
# ======================================================================
#
# Références : Session 8 (« Pricing Financial Assets ») : modèle (6), tests
# t de tarification des facteurs, tests J, et distinction rendements excédentaires
# / facteurs non négociés ; Cochrane (2005), Asset Pricing, ch. 12-13 (GMM),
# cité dans la Session 8.
#
# Notation : T mois, N actifs, K facteurs. R (T x N) : excédents de rendement des
# actifs ; F (T x K) : facteurs ; X = [1, F] (T x (K+1)).
#
# Erreurs-types robustes à l'hétéroscédasticité (White) avec correction des degrés
# de liberté : la matrice « viande » S = g'g / T est divisée par T - (K+1), comme
# dans linearmodels (cov_type="robust", debiased=True), qui sert de vérification.


def _first_pass(F, R):
    """
    (i) Passe en séries temporelles : pour chaque actif i, MCO de r_it sur une
    constante et les K facteurs. Les N régressions ont les mêmes régresseurs X,
    donc on les résout en une seule opération matricielle : B = (X'X)^-1 X'R.

    Retourne B ((K+1) x N : ligne 0 = alphas, lignes 1.. = bêtas), les résidus
    E (T x N), X et A = (X'X / T)^-1.
    """
    T = F.shape[0]
    X = np.column_stack([np.ones(T), F])
    B = np.linalg.solve(X.T @ X, X.T @ R)
    E = R - X @ B
    A = np.linalg.inv(X.T @ X / T)
    return B, E, X, A


def _two_pass(Factors, Assets, Traded):
    """
    Estimation complète du modèle en deux passes (calculs « à la main »).
    Retourne un dictionnaire avec toutes les quantités (y compris les alphas,
    utiles pour l'interprétation) ; Q1_AssetPricingModel en extrait les sorties
    demandées.
    """
    F, R = Factors.values, Assets.values
    T, K = F.shape
    N = R.shape[1]
    dof = T - (K + 1)                      # correction des degrés de liberté
    rbar = R.mean(axis=0)                  # moyennes temporelles des excédents

    # ---- (i) Passe en séries temporelles ----
    B, E, X, A = _first_pass(F, R)
    alpha_ts, beta = B[0], B[1:].T         # alpha (N,), bêtas (N x K)

    # Covariance robuste (White) des coefficients de chaque régression :
    # V(b_i) = A [ sum_t x_t x_t' e_it² / T ] A / (T-K-1). Vectorisé : avec
    # Z = X A, la variance du coefficient k de l'actif i est sum_t Z_tk² e_it² / T / dof.
    Z = X @ A
    se_B = np.sqrt((Z ** 2).T @ (E ** 2) / T / dof)     # (K+1) x N
    beta_t = (B[1:] / se_B[1:]).T                        # N x K

    if Traded:
        # ---- (ii) Facteurs négociés (excédents de rendement d'actifs) ----
        # Chaque facteur est lui-même un actif : sa prime de risque est son
        # rendement moyen, lambda = moyenne de F (le modèle doit « tarifer » les
        # facteurs eux-mêmes avec alpha = 0). Écart-type robuste de la moyenne.
        lam = F.mean(axis=0)
        lam_se = F.std(axis=0, ddof=1) / np.sqrt(T)
        lam_vcv = np.diag(lam_se ** 2)
        # Erreurs de valorisation = alphas de la passe temporelle, puisque
        # r̄_i = alpha_i + beta_i' F̄ exactement (propriété des MCO).
        alpha = alpha_ts
        # Covariance robuste conjointe des N alphas : u_it = (x_t' a0) e_it, où
        # a0 est la 1re colonne de A ; V(alpha) = U'U / T / dof.
        U = Z[:, [0]] * E
        alpha_vcv = U.T @ U / T / dof
        # Test J : H0 : alpha_i = 0 pour tout i ; J = alpha' V^-1 alpha ~ chi²(N)
        J = float(alpha @ np.linalg.solve(alpha_vcv, alpha))
        J_df = N
    else:
        # ---- (ii) Passe transversale (facteurs traités comme non négociés) ----
        # MCO des N moyennes r̄_i sur les bêtas estimés, SANS constante : les
        # actifs sont des excédents de rendement, donc la constante (taux sans
        # risque / taux zéro-bêta) vaut 0 sous le modèle (Session 8, éq. (9)).
        lam = np.linalg.lstsq(beta, rbar, rcond=None)[0]
        alpha = rbar - beta @ lam          # erreurs de valorisation transversales

        # Erreurs-types : les bêtas sont estimés (problème d'erreurs sur les
        # variables), donc on écrit les deux passes comme un GMM exactement
        # identifié (Cochrane, 2005, ch. 12-13), paramètres theta = (b, lambda, alpha) :
        #   g1_t = x_t ⊗ e_t              (N(K+1) moments, passe temporelle)
        #   g2_t = beta' (r_t - beta·lambda)   (K moments, passe transversale)
        #   g3_t = (r_t - beta·lambda) - alpha (N moments, erreurs de valorisation)
        # V(theta) = D^-1 S D^-1' / dof, avec S = g'g / T et D = E[dg/dtheta'].
        P = R - beta @ lam                              # T x N
        g1 = (X[:, None, :] * E[:, :, None]).reshape(T, N * (K + 1))
        g2 = P @ beta
        g3 = P - alpha
        G = np.hstack([g1, g2, g3])
        S = G.T @ G / T

        nb = N * (K + 1)
        D = np.zeros((nb + K + N, nb + K + N))
        D[:nb, :nb] = -np.kron(np.eye(N), X.T @ X / T)          # dg1/db
        for i in range(N):
            cols = slice(i * (K + 1) + 1, (i + 1) * (K + 1))    # bêtas de l'actif i
            D[nb:nb + K, cols] = alpha[i] * np.eye(K) - np.outer(beta[i], lam)  # dg2/dbeta_i
            D[nb + K + i, cols] = -lam                          # dg3_i/dbeta_i
        D[nb:nb + K, nb:nb + K] = -beta.T @ beta                # dg2/dlambda
        D[nb + K:, nb:nb + K] = -beta                           # dg3/dlambda
        D[nb + K:, nb + K:] = -np.eye(N)                        # dg3/dalpha
        Di = np.linalg.inv(D)
        V = Di @ S @ Di.T / dof
        lam_vcv = V[nb:nb + K, nb:nb + K]
        lam_se = np.sqrt(np.diag(lam_vcv))
        alpha_vcv = V[nb + K:, nb + K:]

        # Test J : les K conditions beta' alpha = 0 sont imposées par la passe
        # transversale, donc les erreurs de valorisation vivent dans un espace de
        # dimension N - K. On projette sur une base orthonormée Q de cet espace :
        # J = (Q'alpha)' (Q' V(alpha) Q)^-1 (Q'alpha) ~ chi²(N - K).
        Q = np.linalg.svd(beta)[0][:, K:]
        qa = Q.T @ alpha
        J = float(qa @ np.linalg.solve(Q.T @ alpha_vcv @ Q, qa))
        J_df = N - K

    # R² de la régression transversale : part de la dispersion des rendements
    # moyens r̄_i expliquée par les rendements prédits beta_i' lambda
    # R² = 1 - sum(alpha_i²) / sum((r̄_i - moyenne de r̄)²)
    r2 = 1 - np.sum(alpha ** 2) / np.sum((rbar - rbar.mean()) ** 2)

    f_names, a_names = list(Factors.columns), list(Assets.columns)
    return {
        "lambda": pd.Series(lam, index=f_names), "lambda_se": pd.Series(lam_se, index=f_names),
        "lambda_vcv": lam_vcv,
        "beta": pd.DataFrame(beta, index=a_names, columns=f_names),
        "beta_t": pd.DataFrame(beta_t, index=a_names, columns=f_names),
        "alpha": pd.Series(alpha, index=a_names),
        "alpha_t": pd.Series(alpha / np.sqrt(np.diag(alpha_vcv)), index=a_names),
        "rbar": pd.Series(rbar, index=a_names),
        "predit": pd.Series(beta @ lam, index=a_names),
        "r2": float(r2), "J": J, "J_df": J_df, "J_p": float(sps.chi2.sf(J, J_df)),
        "T": T, "N": N, "K": K, "Traded": Traded,
    }


def Q1_AssetPricingModel(Factors, Assets, Traded):
    """
    Problème 1 c) : modèle linéaire d'évaluation d'actifs en deux passes.

    Factors : DataFrame T x K de rendements de facteurs
    Assets  : DataFrame T x N d'excédents de rendement
    Traded  : True si les facteurs sont des excédents de rendement d'actifs négociés

    Affiche un résumé et retourne, dans l'ordre :
      lambda   (K x 1) primes de risque estimées (% par mois)
      lambda_t (K x 1) statistiques t robustes à l'hétéroscédasticité
      beta     (N x K) expositions aux facteurs
      beta_t   (N x K) statistiques t robustes
      r2       scalaire, R² de la régression transversale
      jtest    (3 x 1) statistique J, valeur p, degrés de liberté sous H0
    """
    res = _two_pass(Factors, Assets, Traded)
    lam = res["lambda"].to_frame("lambda")
    lam_t = (res["lambda"] / res["lambda_se"]).to_frame("t_stat")
    jtest = pd.DataFrame({"valeur": [res["J"], res["J_p"], res["J_df"]]},
                         index=["statistique_J", "p_valeur", "ddl"])

    # ---- Résumé ----
    titre = "facteurs négociés (Traded=True)" if Traded else "facteurs non négociés (Traded=False)"
    print(f"\n--- Modèle en deux passes, {titre} : T = {res['T']}, N = {res['N']}, "
          f"K = {res['K']} ---")
    print("Primes de risque lambda (% par mois), t robustes :")
    print(pd.DataFrame({"lambda_mens_%": res["lambda"], "lambda_ann_%": 12 * res["lambda"],
                        "t_stat": lam_t["t_stat"]}).round(3).to_string())
    print("\nBêtas (expositions), * si |t| > 1,96 :")
    stars = res["beta"].round(2).astype(str) + np.where(res["beta_t"].abs() > 1.96, "*", " ")
    print(stars.to_string())
    print(f"\nR² transversal = {res['r2']:.3f}")
    h0 = "tous les alphas = 0" if Traded else "toutes les erreurs de valorisation = 0"
    print(f"Test J (H0 : {h0}) : J = {res['J']:.2f}, ddl = {res['J_df']}, "
          f"p = {res['J_p']:.4f}")
    print("Erreurs de valorisation alpha (% par mois) et t robustes :")
    print(pd.DataFrame({"alpha": res["alpha"], "t": res["alpha_t"]}).round(3).T.to_string())

    return lam, lam_t, res["beta"], res["beta_t"], res["r2"], jtest


def verify_with_linearmodels(Factors, Assets, Traded):
    """
    Vérifie les calculs « à la main » avec linearmodels (Session 8) :
    TradedFactorModel si Traded=True, LinearFactorModel(risk_free=False) sinon,
    covariance robuste à l'hétéroscédasticité avec correction des degrés de liberté.
    Retourne l'écart absolu maximal pour chaque quantité.
    """
    from linearmodels.asset_pricing import LinearFactorModel, TradedFactorModel

    mine = _two_pass(Factors, Assets, Traded)
    model = (TradedFactorModel(Assets, Factors) if Traded
             else LinearFactorModel(Assets, Factors, risk_free=False))
    lm = model.fit(cov_type="robust", debiased=True)

    # params et tstats : tableaux N x (alpha + K facteurs)
    lm_beta_t = lm.tstats[list(Factors.columns)].values
    lm_alpha = lm.params["alpha"].values
    lm_alpha_t = lm.tstats["alpha"].values
    ecarts = {
        "lambda": np.abs(mine["lambda"].values - lm.risk_premia.values).max(),
        "t(lambda)": np.abs((mine["lambda"] / mine["lambda_se"]).values
                            - lm.risk_premia_tstats.values).max(),
        "beta": np.abs(mine["beta"].values - lm.betas.values).max(),
        "t(beta)": np.abs(mine["beta_t"].values - lm_beta_t).max(),
        "alpha": np.abs(mine["alpha"].values - lm_alpha).max(),
        "t(alpha)": np.abs(mine["alpha_t"].values - lm_alpha_t).max(),
        "J": abs(mine["J"] - lm.j_statistic.stat),
        "ddl J": abs(mine["J_df"] - lm.j_statistic.df),
        "p(J)": abs(mine["J_p"] - lm.j_statistic.pval),
    }
    return pd.Series(ecarts, name=f"Traded={Traded}")


def Q1c_Resultats(factors, portfolios):
    """
    Applique Q1_AssetPricingModel avec Traded=True et Traded=False, vérifie avec
    linearmodels, puis enregistre les tableaux et graphiques de la question 1 c).
    """
    Q = "Q1c"
    out = {tr: Q1_AssetPricingModel(factors, portfolios, Traded=tr) for tr in (True, False)}
    res = {tr: _two_pass(factors, portfolios, tr) for tr in (True, False)}
    lab = {True: "Traded=True", False: "Traded=False"}

    # ---- Vérification avec linearmodels ----
    verif = pd.concat([verify_with_linearmodels(factors, portfolios, tr)
                       for tr in (True, False)], axis=1)
    print("\nVérification avec linearmodels (écart absolu maximal, calcul à la main vs package) :")
    print(verif.map(lambda v: f"{v:.1e}").to_string())
    assert (verif < 1e-6).all().all(), "Les calculs à la main diffèrent de linearmodels"
    save_table(verif, Q, "verification_linearmodels_ecarts_max")

    # ---- Tableaux ----
    # Primes de risque des deux versions côte à côte (mensuelles et annualisées)
    lam_tab = pd.concat({
        lab[tr]: pd.DataFrame({
            "lambda_mens_%": out[tr][0]["lambda"],
            "lambda_ann_%": 12 * out[tr][0]["lambda"],
            "t_stat": out[tr][1]["t_stat"],
            "p_valeur": 2 * sps.norm.sf(out[tr][1]["t_stat"].abs()),
        }) for tr in (True, False)}, axis=1)
    save_table(lam_tab, Q, "primes_de_risque_lambda")

    # Bêtas et t : identiques pour les deux versions (même passe temporelle)
    assert np.allclose(out[True][2], out[False][2]) and np.allclose(out[True][3], out[False][3])
    save_table(out[True][2], Q, "betas_expositions")
    save_table(out[True][3], Q, "betas_t_stats_robustes")

    # Erreurs de valorisation (alphas) des deux versions
    alpha_tab = pd.concat({
        lab[tr]: pd.DataFrame({
            "rendement_moyen_realise_%": res[tr]["rbar"],
            "rendement_moyen_predit_%": res[tr]["predit"],
            "alpha_mens_%": res[tr]["alpha"],
            "alpha_ann_%": 12 * res[tr]["alpha"],
            "t_stat": res[tr]["alpha_t"],
        }) for tr in (True, False)}, axis=1)
    save_table(alpha_tab, Q, "alphas_erreurs_de_valorisation")

    # R² et tests J
    jr = pd.DataFrame({
        lab[tr]: {"R2_transversal": out[tr][4], "statistique_J": out[tr][5].loc["statistique_J", "valeur"],
                  "p_valeur_J": out[tr][5].loc["p_valeur", "valeur"],
                  "ddl_J": out[tr][5].loc["ddl", "valeur"],
                  # Taille moyenne des erreurs de valorisation (% par an)
                  "alpha_absolu_moyen_ann_%": 12 * res[tr]["alpha"].abs().mean()}
        for tr in (True, False)})
    save_table(jr, Q, "R2_et_tests_J")

    # ---- Graphiques ----
    period = f"{factors.index[0]} à {factors.index[-1]}"

    # Fig. 1 : rendements moyens réalisés vs prédits (annualisés), deux versions
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    lim = 12 * np.array([min(r[k].min() for r in res.values() for k in ("rbar", "predit")),
                         max(r[k].max() for r in res.values() for k in ("rbar", "predit"))])
    lim = lim + np.array([-1, 1]) * 0.08 * (lim[1] - lim[0])
    for ax, tr in zip(axes, (True, False)):
        r = res[tr]
        ax.plot(lim, lim, color=TEXT_2, lw=0.8, ls="--", label="Droite à 45° (alpha = 0)")
        ax.scatter(12 * r["predit"], 12 * r["rbar"], s=36, color=BLUE,
                   edgecolor="white", linewidth=1, zorder=3)
        for name in r["rbar"].index:
            ax.annotate(name, (12 * r["predit"][name], 12 * r["rbar"][name]),
                        xytext=(4, 3), textcoords="offset points", fontsize=7, color=TEXT)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_title(f"{lab[tr]} : R² = {_fr(r['r2'], 2)} ; J = {_fr(r['J'], 1)} "
                     f"(ddl = {r['J_df']}, p = {_fr(r['J_p'], 3)})", loc="left", fontsize=9.5)
        ax.set_xlabel("Rendement excédentaire moyen prédit β̂ᵢ'λ̂ (% par an)")
        ax.xaxis.set_major_formatter(FR_AXIS)
        ax.yaxis.set_major_formatter(FR_AXIS)
    axes[0].set_ylabel("Rendement excédentaire moyen réalisé r̄ᵢ (% par an)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle(f"Rendements moyens réalisés et prédits des 17 portefeuilles sectoriels, {period}",
                 x=0.01, ha="left", fontweight="bold", fontsize=11, color=TEXT)
    fig.tight_layout()
    save_figure(fig, Q, "fig1_rendements_realises_vs_predits")
    plt.close(fig)

    # Fig. 2 : primes de risque annualisées avec IC à 95 %, deux versions
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(factors.columns))[::-1]
    for j, tr in enumerate((True, False)):
        lam_a = 12 * res[tr]["lambda"].values
        ci = 1.96 * 12 * res[tr]["lambda_se"].values
        yy = y + (0.15 if tr else -0.15)
        ax.errorbar(lam_a, yy, xerr=ci, fmt="o", ms=6, color=SERIES_COLORS[j],
                    ecolor=SERIES_COLORS[j], elinewidth=1.5, capsize=3, label=lab[tr],
                    markeredgecolor="white", zorder=3)
    ax.axvline(0, color=TEXT_2, lw=0.8)
    ax.set_yticks(y, [FACTOR_LABELS[c] for c in factors.columns])
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FR_AXIS)
    ax.set_xlabel("Prime de risque λ̂ annualisée (% par an), avec intervalle de confiance à 95 %")
    ax.set_title(f"Primes de risque estimées des six facteurs, {period}", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    save_figure(fig, Q, "fig2_primes_de_risque_IC95")
    plt.close(fig)

    # Fig. 3 : carte des bêtas (* si |t| > 1,96)
    b, bt = out[True][2], out[True][3]
    vmax = np.abs(b.values).max()
    fig, ax = plt.subplots(figsize=(7.5, 8))
    im = ax.imshow(b.values, cmap=CORR_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            v = b.iat[i, j]
            ax.text(j, i, _fr(v, 2) + ("*" if abs(bt.iat[i, j]) > 1.96 else ""),
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 * vmax else TEXT)
    ax.set_xticks(range(b.shape[1]), b.columns)
    ax.set_yticks(range(b.shape[0]), [INDUSTRY_LABELS[c] for c in b.index])
    ax.tick_params(length=0)
    ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, label="Bêta estimé")
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _fr(v, 1)))
    ax.set_title(f"Expositions des portefeuilles aux facteurs (β̂), {period}\n"
                 "* : |t robuste| > 1,96 (significatif à 5 %)", loc="left")
    save_figure(fig, Q, "fig3_betas_heatmap")
    plt.close(fig)

    # Fig. 4 : erreurs de valorisation (alphas) annualisées avec IC à 95 %
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharex=True, sharey=True)
    order = res[True]["alpha"].sort_values().index
    yy = np.arange(len(order))
    for ax, tr in zip(axes, (True, False)):
        a = 12 * res[tr]["alpha"][order]
        ci = 1.96 * np.abs(a / res[tr]["alpha_t"][order])
        sig = res[tr]["alpha_t"][order].abs() > 1.96
        ax.errorbar(a, yy, xerr=ci, fmt="none", ecolor=TEXT_2, elinewidth=1, capsize=2.5)
        ax.scatter(a[~sig], yy[~sig], s=30, color=BLUE, zorder=3, label="non significatif à 5 %")
        ax.scatter(a[sig], yy[sig], s=30, color=SERIES_COLORS[1], zorder=3, label="significatif à 5 %")
        ax.axvline(0, color=TEXT_2, lw=0.8)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(FR_AXIS)
        ax.set_title(lab[tr], loc="left")
        ax.set_xlabel("Alpha annualisé (% par an), IC à 95 %")
    axes[0].set_yticks(yy, [INDUSTRY_LABELS[c] for c in order])
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.6, -0.01))
    fig.suptitle(f"Erreurs de valorisation (alphas) des 17 portefeuilles sectoriels, {period}",
                 x=0.01, ha="left", fontweight="bold", fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, Q, "fig4_alphas_IC95")
    plt.close(fig)

    return out, res, verif


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

    # ------------------------------------------------------------------
    # Problème 1 b) : statistiques descriptives et graphiques
    # ------------------------------------------------------------------
    print("\n===== Q1b - statistiques descriptives et graphiques =====")
    q1b = Q1b_StatistiquesGraphiques(factors, portfolios)

    # ------------------------------------------------------------------
    # Problème 1 c) : modèle en deux passes, Traded=True et Traded=False
    # ------------------------------------------------------------------
    print("\n===== Q1c - modèle linéaire d'évaluation d'actifs en deux passes =====")
    q1c_out, q1c_res, q1c_verif = Q1c_Resultats(factors, portfolios)
