"""
MATH 60231 - Devoir #4
Problème 1 a) : chargement des données de la bibliothèque de Ken French.
Problème 1 b) : statistiques descriptives et graphiques.

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
