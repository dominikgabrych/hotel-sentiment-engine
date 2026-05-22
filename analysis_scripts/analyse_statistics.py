"""
statistics.py — Statistical Analysis Pipeline (PB1, PB2, PB3)

Runs the full statistical verification workflow on the cleaned review dataset:
  - PB1: Star category vs. overall guest rating (3★ / 4★ / 5★)
  - PB2: Traveler type vs. overall guest rating (Business / Couple / Family / Solo / Other)
  - PB3: Country of origin (Poland vs. Other) vs. overall guest rating

Decision path for each research problem:
  1. Shapiro-Wilk normality test per group
  2. Levene's test for homogeneity of variances
  3. If ALL groups are normal → ANOVA (f_oneway / ttest_ind)
     Otherwise            → Kruskal-Wallis / Mann-Whitney U
  4. Post-hoc tests if omnibus test is significant (p < 0.05):
     ANOVA  → Tukey HSD
     K-W    → Dunn with Bonferroni correction

Outputs saved to:
  data/04_results/tables/   — CSV result tables
  data/04_results/figures/  — PNG publication-quality charts (DPI 300)

Usage:
  python statistics.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script use

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from scipy.stats import shapiro, levene, kruskal, mannwhitneyu, f_oneway, ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from matplotlib.figure import Figure

warnings.filterwarnings("ignore")

# ─────────────────────────── configuration ───────────────────────────

INPUT_PATH   = "data/02_cleaned/reviews_stats.csv"
TABLES_DIR   = "data/04_results/tables"
FIGURES_DIR  = "data/04_results/figures"
ALPHA        = 0.05
DPI          = 300

# Publication-quality style
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       DPI,
})

PALETTE = {
    "stars":    ["#4C9BE8", "#2C6FAC", "#1A3F6F"],   # Blues 3★→5★
    "traveler": ["#E87B4C", "#E8B44C", "#6BBF6B", "#9B6BE8", "#A0A0A0"],
    "country":  ["#4C9BE8", "#E8634C"],               # Blue = Poland, Red = Other
}

# ─────────────────────────── helpers ─────────────────────────────────


def ensure_dirs() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def save_table(df: pd.DataFrame, filename: str) -> None:
    path = os.path.join(TABLES_DIR, filename)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  [✓] Table saved → {path}")


from matplotlib.figure import Figure

def save_figure(fig: Figure, filename: str) -> None:
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Figure saved → {path}")


def normality_report(groups: dict) -> tuple[bool, pd.DataFrame]:
    """
    Runs Shapiro-Wilk per group.
    Returns (all_normal: bool, results_df: DataFrame).
    """
    rows = []
    all_normal = True
    for label, data in groups.items():
        stat, p = shapiro(data)
        normal = p > ALPHA
        if not normal:
            all_normal = False
        rows.append({
            "Group":       label,
            "N":           len(data),
            "W statistic": round(stat, 4),
            "p-value":     round(p, 4),
            "Normal (α=0.05)": "Yes" if normal else "No",
        })
    return all_normal, pd.DataFrame(rows)


def levene_report(groups: dict) -> tuple[bool, float, float]:
    """Runs Levene's test. Returns (homogeneous, stat, p)."""
    arrays = list(groups.values())
    stat, p = levene(*arrays)
    return p > ALPHA, round(stat, 4), round(p, 4)


def decide_and_run(groups: dict, all_normal: bool, label: str) -> dict:
    """
    Runs ANOVA or Kruskal-Wallis depending on normality.
    Returns a result dict with test name, statistic, p-value, decision.
    """
    arrays = list(groups.values())
    if all_normal:
        stat, p = f_oneway(*arrays)
        test_name = "One-Way ANOVA"
        stat_label = "F"
    else:
        stat, p = kruskal(*arrays)
        test_name = "Kruskal-Wallis"
        stat_label = "H"

    decision = f"Reject H₀ (p < {ALPHA})" if p < ALPHA else f"Fail to reject H₀ (p ≥ {ALPHA})"
    print(f"  {test_name}: {stat_label} = {stat:.4f}, p = {p:.4f} → {decision}")

    return {
        "Research Problem": label,
        "Test":             test_name,
        "Statistic":        round(stat, 4),
        "Statistic label":  stat_label,
        "p-value":          round(p, 4),
        "Decision (α=0.05)": decision,
    }


def dunn_posthoc(groups: dict) -> pd.DataFrame:
    """Dunn test with Bonferroni correction (after Kruskal-Wallis)."""
    data  = list(groups.values())
    names = list(groups.keys())
    result = sp.posthoc_dunn(data, p_adjust="bonferroni")
    result.index   = names
    result.columns = names
    return result.round(4)


def tukey_posthoc(df: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    """Tukey HSD test (after ANOVA)."""
    result = pairwise_tukeyhsd(df[value_col], df[group_col], alpha=ALPHA)
    return pd.DataFrame(data=result._results_table.data[1:],
                        columns=result._results_table.data[0])


# ─────────────────────────── loading & normalizing ───────────────────


def load_data() -> pd.DataFrame:
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  Records loaded: {len(df):,}\n")

    # Normalize traveler_type to five canonical categories
    traveler_map = {
        "solo traveller": "Solo",
        "solo traveler":  "Solo",
        "couple":         "Couple",
        "family":         "Family",
        "business":       "Business",
        "group":          "Group of friends",
    }
    df["traveler_type"] = (
        df["traveler_type"]
        .str.strip()
        .str.lower()
        .map(traveler_map)
        .fillna("Group of friends")
    )

    # Normalize reviewer_country to binary: Poland / Other
    df["origin"] = df["reviewer_country"].apply(
        lambda c: "Poland" if str(c).strip().lower() == "poland" else "Other"
    )

    # Ensure hotel_stars is integer
    df["hotel_stars"] = df["hotel_stars"].astype(int)

    return df


# ─────────────────────────── descriptive stats ───────────────────────


def descriptive_statistics(df: pd.DataFrame) -> None:
    print("─" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("─" * 60)

    # Overall rating_score stats
    overall = df["rating_score"].agg(["count", "mean", "median", "std", "min", "max"])
    overall.index = ["N", "Mean", "Median", "Std Dev", "Min", "Max"]
    overall_df = overall.round(3).reset_index()
    overall_df.columns = ["Statistic", "Value"]

    # Per-group breakdowns
    stars_dist    = df.groupby("hotel_stars")["rating_score"].agg(
        N="count", Mean="mean", Median="median", Std="std").round(3).reset_index()
    traveler_dist = df.groupby("traveler_type")["rating_score"].agg(
        N="count", Mean="mean", Median="median", Std="std").round(3).reset_index()
    country_dist  = df.groupby("origin")["rating_score"].agg(
        N="count", Mean="mean", Median="median", Std="std").round(3).reset_index()

    save_table(overall_df,    "tab_01a_descriptive_overall.csv")
    save_table(stars_dist,    "tab_01b_descriptive_by_stars.csv")
    save_table(traveler_dist, "tab_01c_descriptive_by_traveler.csv")
    save_table(country_dist,  "tab_01d_descriptive_by_country.csv")

    # fig_01 — rating distribution histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["rating_score"], bins=20, color="#4C9BE8", edgecolor="white", linewidth=0.6)
    ax.set_title("Rozkład ocen gości", fontweight="bold", pad=12)
    ax.set_xlabel("Ocena ogólna (1–10)")
    ax.set_ylabel("Liczba recenzji")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    mean_val = df["rating_score"].mean()
    label_text = f"Średnia = {mean_val:.2f}".replace('.', ',')
    ax.axvline(mean_val, color="#E8634C", linestyle="--", linewidth=1.4,
               label=label_text)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig_01_rating_distribution.png")
    print()


# ─────────────────────────── PB1 ─────────────────────────────────────


def analyse_pb1(df: pd.DataFrame) -> None:
    print("─" * 60)
    print("PB1 — Star Category vs. Rating Score")
    print("─" * 60)

    groups = {
        "3★": df.loc[df["hotel_stars"] == 3, "rating_score"].dropna().values,
        "4★": df.loc[df["hotel_stars"] == 4, "rating_score"].dropna().values,
        "5★": df.loc[df["hotel_stars"] == 5, "rating_score"].dropna().values,
    }

    # Normality
    all_normal, normality_df = normality_report(groups)
    save_table(normality_df, "tab_02a_pb1_shapiro.csv")
    print(f"  Shapiro-Wilk: all groups normal = {all_normal}")

    # Levene
    hom, lev_stat, lev_p = levene_report(groups)
    print(f"  Levene:       homogeneous variances = {hom} (W={lev_stat}, p={lev_p})")

    # Main test
    result = decide_and_run(groups, all_normal, "PB1")
    result_df = pd.DataFrame([result])
    save_table(result_df, "tab_02b_pb1_main_test.csv")

    # Post-hoc (only if significant)
    if result["p-value"] < ALPHA:
        if all_normal:
            ph_df = tukey_posthoc(df.dropna(subset=["rating_score"]), "rating_score", "hotel_stars")
            save_table(ph_df, "tab_02c_pb1_posthoc_tukey.csv")
        else:
            ph_df = dunn_posthoc(groups)
            save_table(ph_df, "tab_02c_pb1_posthoc_dunn.csv")
        print(f"  Post-hoc saved.")

    # fig_02 — boxplot per star category
    fig, ax = plt.subplots(figsize=(8, 6))
    data_for_plot = [groups["3★"], groups["4★"], groups["5★"]]
    bp = ax.boxplot(data_for_plot, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE["stars"]):
        patch.set_facecolor(color)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Hotele 3★", "Hotele 4★", "Hotele 5★"])
    ax.set_title("Ocena ogólna a standard hotelu (PB1)", fontweight="bold", pad=12)
    ax.set_ylabel("Ocena ogólna (1–10)")
    ax.set_ylim(0, 11)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))

    p_str = f"p = {result['p-value']:.4f}" if result["p-value"] >= 0.0001 else "p < 0.0001"
    annot_str = f"{result['Test']}: {result['Statistic label']} = {result['Statistic']:.2f}, {p_str}".replace('.', ',')
    ax.annotate(annot_str,
                xy=(0.5, 0.02), xycoords="axes fraction", ha="center",
                fontsize=10, color="#555555")
    fig.tight_layout()
    save_figure(fig, "fig_02_pb1_stars_boxplot.png")
    print()


# ─────────────────────────── PB2 ─────────────────────────────────────


def analyse_pb2(df: pd.DataFrame) -> None:
    print("─" * 60)
    print("PB2 — Traveler Type vs. Rating Score")
    print("─" * 60)

    categories = ["Business", "Couple", "Family", "Solo", "Group of friends"]
    groups = {
        cat: df.loc[df["traveler_type"] == cat, "rating_score"].dropna().values
        for cat in categories
    }
    # Drop any empty groups
    groups = {k: v for k, v in groups.items() if len(v) >= 5}

    # Normality
    all_normal, normality_df = normality_report(groups)
    save_table(normality_df, "tab_03a_pb2_shapiro.csv")
    print(f"  Shapiro-Wilk: all groups normal = {all_normal}")

    # Levene
    hom, lev_stat, lev_p = levene_report(groups)
    print(f"  Levene:       homogeneous variances = {hom} (W={lev_stat}, p={lev_p})")

    # Main test
    result = decide_and_run(groups, all_normal, "PB2")
    result_df = pd.DataFrame([result])
    save_table(result_df, "tab_03b_pb2_main_test.csv")

    # Post-hoc
    if result["p-value"] < ALPHA:
        if all_normal:
            subset = df[df["traveler_type"].isin(groups.keys())].dropna(subset=["rating_score"])
            ph_df = tukey_posthoc(subset, "rating_score", "traveler_type")
            save_table(ph_df, "tab_03c_pb2_posthoc_tukey.csv")
        else:
            ph_df = dunn_posthoc(groups)
            save_table(ph_df, "tab_03c_pb2_posthoc_dunn.csv")
        print(f"  Post-hoc saved.")

    # fig_03 — boxplot per traveler type
    labels  = list(groups.keys())
    colors  = PALETTE["traveler"][:len(labels)]
    fig, ax = plt.subplots(figsize=(10, 6))
    data_for_plot = [groups[k] for k in labels]
    bp = ax.boxplot(data_for_plot, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(["Biznes", "Pary", "Rodziny", "Solo", "Grupa znajomych"])
    ax.set_title("Ocena ogólna a profil podróżnego (PB2)", fontweight="bold", pad=12)
    ax.set_ylabel("Ocena ogólna (1–10)")
    ax.set_ylim(0, 11)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))

    p_str = f"p = {result['p-value']:.4f}" if result["p-value"] >= 0.0001 else "p < 0.0001"
    annot_str = f"{result['Test']}: {result['Statistic label']} = {result['Statistic']:.2f}, {p_str}".replace('.', ',')
    ax.annotate(annot_str,
                xy=(0.5, 0.02), xycoords="axes fraction", ha="center",
                fontsize=10, color="#555555")
    fig.tight_layout()
    save_figure(fig, "fig_03_pb2_traveler_boxplot.png")
    print()


# ─────────────────────────── PB3 ─────────────────────────────────────


def analyse_pb3(df: pd.DataFrame) -> None:
    print("─" * 60)
    print("PB3 — Country of Origin vs. Rating Score (Poland vs. Other)")
    print("─" * 60)

    groups = {
        "Poland": df.loc[df["origin"] == "Poland", "rating_score"].dropna().values,
        "Other":  df.loc[df["origin"] == "Other",  "rating_score"].dropna().values,
    }

    # Normality
    all_normal, normality_df = normality_report(groups)
    save_table(normality_df, "tab_04a_pb3_shapiro.csv")
    print(f"  Shapiro-Wilk: all groups normal = {all_normal}")

    # Levene
    hom, lev_stat, lev_p = levene_report(groups)
    print(f"  Levene:       homogeneous variances = {hom} (W={lev_stat}, p={lev_p})")

    # Main test (two-sample)
    if all_normal:
        stat, p = ttest_ind(groups["Poland"], groups["Other"], equal_var=hom)
        test_name, stat_label = "Welch's t-test", "t"
    else:
        stat, p = mannwhitneyu(groups["Poland"], groups["Other"], alternative="two-sided")
        test_name, stat_label = "Mann-Whitney U", "U"

    decision = f"Reject H₀ (p < {ALPHA})" if p < ALPHA else f"Fail to reject H₀ (p ≥ {ALPHA})"
    print(f"  {test_name}: {stat_label} = {stat:.4f}, p = {p:.4f} → {decision}")

    result_df = pd.DataFrame([{
        "Research Problem": "PB3",
        "Test":             test_name,
        "Statistic":        round(stat, 4),
        "Statistic label":  stat_label,
        "p-value":          round(p, 4),
        "Decision (α=0.05)": decision,
    }])
    save_table(result_df, "tab_04b_pb3_main_test.csv")

    # fig_04 — boxplot per country (changed from bar chart due to non-parametric Mann-Whitney)
    fig, ax = plt.subplots(figsize=(7, 6))
    data_for_plot = [groups["Poland"], groups["Other"]]
    bp = ax.boxplot(data_for_plot, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE["country"]):
        patch.set_facecolor(color)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Goście z Polski", "Goście z zagranicy"], fontsize=12)
    ax.set_title("Ocena ogólna a kraj pochodzenia (PB3)", fontweight="bold", pad=12)
    ax.set_ylabel("Ocena ogólna (1–10)")
    ax.set_ylim(0, 11)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))

    p_str = f"p = {p:.4f}" if p >= 0.0001 else "p < 0.0001"
    annot_str = f"{test_name}: {stat_label} = {stat:.2f}, {p_str}".replace('.', ',')
    ax.annotate(annot_str,
                xy=(0.5, 0.02), xycoords="axes fraction", ha="center",
                fontsize=10, color="#555555")

    fig.tight_layout()
    save_figure(fig, "fig_04_pb3_country_comparison.png")
    print()


# ─────────────────────────── entry point ─────────────────────────────


def main() -> None:
    print("=" * 60)
    print("Statistical Analysis Pipeline — PB1 / PB2 / PB3")
    print("=" * 60 + "\n")

    ensure_dirs()
    df = load_data()

    descriptive_statistics(df)
    analyse_pb1(df)
    analyse_pb2(df)
    analyse_pb3(df)

    print("=" * 60)
    print("All outputs saved. Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
