"""
visualisation.py — Dissertation Chart Generator
================================================
Reads results_full_benchmark.csv and convergence_ga.csv produced by
compare_static.py and generates five publication-ready figures.

Figure 4.1  Total Revenue (BYN)       — all 4 algorithm+pricing combos
Figure 4.2  Passengers Assigned        — all 4 combos
Figure 4.3  Seat Occupancy (%)         — all 4 combos
Figure 4.4  Pricing Uplift             — Dynamic minus Static revenue gain
                                         split by algorithm across scenarios
Figure 4.5  GA Convergence Curves      — best passenger count per generation,
                                         Static vs Dynamic, per scenario

Run order:
  1. python generator_dataset_static.py
  2. python generator.py + train_pricing.py
  3. python compare_static.py   →  results_full_benchmark.csv + convergence_ga.csv
  4. python visualisation.py    →  5 PNG files ready for your dissertation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
RESULTS_CSV     = "results_full_benchmark.csv"
CONVERGENCE_CSV = "convergence_ga.csv"

# ─────────────────────────────────────────────────────────────
# Visual identity — consistent across all five figures
# ─────────────────────────────────────────────────────────────
# Four colours: algorithm (dark/light) × pricing (muted/vivid)
COMBO_ORDER  = ["Greedy / Static", "Greedy / Dynamic", "GA / Static", "GA / Dynamic"]
COMBO_COLORS = {
    "Greedy / Static":  "#7f7f7f",   # neutral grey  — baseline
    "Greedy / Dynamic": "#4878d0",   # blue          — pricing upgrade on Greedy
    "GA / Static":      "#d65f5f",   # coral/red     — better algorithm, old pricing
    "GA / Dynamic":     "#2ca02c",   # green         — best of both worlds
}

SCENARIO_ORDER  = ["Low_Demand", "Medium_Urban", "High_Congestion", "Extreme_Peak"]
SCENARIO_LABELS = ["Low demand", "Medium urban", "High congestion", "Extreme peak"]

ALGO_COLORS = {"Greedy": "#4878d0", "GA": "#2ca02c"}


# ─────────────────────────────────────────────────────────────
# Data loading & preparation
# ─────────────────────────────────────────────────────────────

def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    df_raw   : full CSV with one row per (Scenario, Trial, Algorithm, Pricing)
    df_agg   : averaged over Dataset_Trial_ID — one row per (Scenario, Combo)
    """
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(
            f"'{RESULTS_CSV}' not found.\n"
            "Run  python compare_static.py  first to generate results."
        )
    df = pd.read_csv(RESULTS_CSV)

    # Create human-readable combo label
    df["Algorithm_Label"] = df["Algorithm"].map(
        {"Greedy": "Greedy", "GeneticAlg": "GA"}
    )
    df["Combo"] = df["Algorithm_Label"] + " / " + df["Pricing_Type"]

    # Enforce display ordering
    df["Scenario"] = pd.Categorical(df["Scenario"], categories=SCENARIO_ORDER, ordered=True)
    df["Combo"]    = pd.Categorical(df["Combo"],    categories=COMBO_ORDER,    ordered=True)

    # Average across the 5 dataset trials for each (Scenario, Combo)
    df_agg = (
        df.groupby(["Scenario", "Combo"], observed=True)
        .agg(
            Passengers_Assigned = ("Passengers_Assigned", "mean"),
            Total_Passengers    = ("Total_Passengers",    "mean"),
            Total_Revenue_BYN   = ("Total_Revenue_BYN",   "mean"),
            Seat_Occupancy_Pct  = ("Seat_Occupancy_Pct",  "mean"),
            Runtime_MS          = ("Runtime_MS",           "mean"),
        )
        .reset_index()
        .sort_values(["Scenario", "Combo"])
    )
    return df, df_agg


# ─────────────────────────────────────────────────────────────
# Helper: grouped bar chart (Figures 4.1 – 4.3)
# ─────────────────────────────────────────────────────────────

def grouped_bar_chart(
    df_agg: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    fignum: int,
) -> None:
    """
    Four bars per scenario group, one bar per combo, averaged over trials.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    n_scen   = len(SCENARIO_ORDER)
    n_combos = len(COMBO_ORDER)
    x        = np.arange(n_scen)
    width    = 0.18
    offsets  = np.linspace(-(n_combos - 1) / 2,
                            (n_combos - 1) / 2,
                            n_combos) * width

    for offset, combo in zip(offsets, COMBO_ORDER):
        values = []
        for sc in SCENARIO_ORDER:
            row = df_agg[(df_agg["Scenario"] == sc) & (df_agg["Combo"] == combo)]
            values.append(float(row[metric].values[0]) if len(row) else 0.0)
        ax.bar(
            x + offset, values,
            width=width,
            label=combo,
            color=COMBO_COLORS[combo],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_LABELS, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_xlabel("Demand scenario", fontsize=12, fontweight="bold")
    ax.set_title(f"Figure 4.{fignum}: {title}", fontsize=13, pad=14)
    ax.legend(
        title="Algorithm / Pricing model",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=10,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, left=False, bottom=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] {filename}")


# ─────────────────────────────────────────────────────────────
# Figure 4.4: Pricing Uplift
# ─────────────────────────────────────────────────────────────

def pricing_uplift_chart(df_raw: pd.DataFrame, fignum: int) -> None:
    """
    For each (Scenario, Algorithm), compute:
        Uplift = mean Dynamic revenue − mean Static revenue

    Positive bars → dynamic pricing earns more.
    Split by algorithm (Greedy vs GA) to show which benefits more.

    Dissertation question answered: "Does dynamic pricing help, and does
    it help equally for both algorithms?"
    """
    # Average over trials for each (Scenario, Algorithm, Pricing_Type)
    agg = (
        df_raw
        .groupby(["Scenario", "Algorithm_Label", "Pricing_Type"], observed=True)["Total_Revenue_BYN"]
        .mean()
        .reset_index()
    )

    pivot = agg.pivot_table(
        index=["Scenario", "Algorithm_Label"],
        columns="Pricing_Type",
        values="Total_Revenue_BYN",
    ).reset_index()

    # Handle possible missing columns gracefully
    if "Dynamic" not in pivot.columns or "Static" not in pivot.columns:
        print("[WARN] Pricing uplift chart skipped — missing Static or Dynamic column.")
        return

    pivot["Uplift_BYN"] = pivot["Dynamic"] - pivot["Static"]

    fig, ax = plt.subplots(figsize=(11, 6))

    x       = np.arange(len(SCENARIO_ORDER))
    width   = 0.30
    algos   = ["Greedy", "GA"]
    offsets = [-width / 2, width / 2]

    for offset, algo in zip(offsets, algos):
        values = []
        for sc in SCENARIO_ORDER:
            row = pivot[(pivot["Scenario"] == sc) & (pivot["Algorithm_Label"] == algo)]
            values.append(float(row["Uplift_BYN"].values[0]) if len(row) else 0.0)
        ax.bar(
            x + offset, values,
            width=width,
            label=algo,
            color=ALGO_COLORS[algo],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.axhline(0, color="black", linewidth=0.9, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_LABELS, fontsize=11)
    ax.set_ylabel("Revenue uplift (Dynamic − Static, BYN)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Demand scenario", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Figure 4.{fignum}: Pricing uplift — revenue gain from dynamic pricing by algorithm",
        fontsize=13, pad=14,
    )
    ax.legend(title="Algorithm", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, left=False, bottom=False)
    plt.tight_layout()
    plt.savefig("dissertation_pricing_uplift.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] dissertation_pricing_uplift.png")


# ─────────────────────────────────────────────────────────────
# Figure 4.5: GA Convergence
# ─────────────────────────────────────────────────────────────

def convergence_chart(fignum: int) -> None:
    """
    Two subplots (Static | Dynamic), one line per scenario.
    Each line = mean best passenger count per generation, averaged over
    all dataset trials and GA runs.

    Dissertation question answered: "How quickly does the GA converge,
    and does the pricing model affect learning speed?"
    """
    if not os.path.exists(CONVERGENCE_CSV):
        print(
            f"[WARN] '{CONVERGENCE_CSV}' not found — skipping Figure 4.{fignum}.\n"
            "       Run compare_static.py first."
        )
        return

    conv = pd.read_csv(CONVERGENCE_CSV)

    # Average over Dataset_Trial_ID and GA_Run
    agg = (
        conv
        .groupby(["Scenario", "Pricing_Type", "Generation"])["Best_Pax_Count"]
        .mean()
        .reset_index()
    )

    sc_colors = dict(zip(
        SCENARIO_ORDER,
        ["#4878d0", "#2ca02c", "#d65f5f", "#7f7f7f"],
    ))
    sc_labels = dict(zip(SCENARIO_ORDER, SCENARIO_LABELS))

    pricing_types = ["Static", "Dynamic"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, pricing in zip(axes, pricing_types):
        subset = agg[agg["Pricing_Type"] == pricing]
        for sc in SCENARIO_ORDER:
            sc_data = subset[subset["Scenario"] == sc].sort_values("Generation")
            if len(sc_data):
                ax.plot(
                    sc_data["Generation"],
                    sc_data["Best_Pax_Count"],
                    label=sc_labels[sc],
                    color=sc_colors.get(sc, "gray"),
                    linewidth=2,
                )
        ax.set_title(f"{pricing} pricing", fontsize=12, pad=8)
        ax.set_xlabel("Generation", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title="Scenario", fontsize=9)
        sns.despine(ax=ax)

    axes[0].set_ylabel("Best passengers assigned (avg over trials)", fontsize=11, fontweight="bold")
    fig.suptitle(
        f"Figure 4.{fignum}: GA convergence — best passenger count per generation",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    plt.savefig("dissertation_convergence.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] dissertation_convergence.png")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def create_dissertation_charts() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

    df_raw, df_agg = load_results()

    # Figure 4.1 — Revenue
    grouped_bar_chart(
        df_agg,
        metric   = "Total_Revenue_BYN",
        ylabel   = "Total revenue (BYN)",
        title    = "Revenue comparison — all algorithm and pricing combinations",
        filename = "dissertation_revenue_comparison.png",
        fignum   = 1,
    )

    # Figure 4.2 — Passengers Assigned
    grouped_bar_chart(
        df_agg,
        metric   = "Passengers_Assigned",
        ylabel   = "Passengers assigned",
        title    = "Passenger throughput — all algorithm and pricing combinations",
        filename = "dissertation_passengers_comparison.png",
        fignum   = 2,
    )

    # Figure 4.3 — Seat Occupancy
    grouped_bar_chart(
        df_agg,
        metric   = "Seat_Occupancy_Pct",
        ylabel   = "Seat occupancy (%)",
        title    = "Fleet utilisation — all algorithm and pricing combinations",
        filename = "dissertation_occupancy_comparison.png",
        fignum   = 3,
    )

    # Figure 4.4 — Pricing Uplift
    pricing_uplift_chart(df_raw, fignum=4)

    # Figure 4.5 — GA Convergence
    convergence_chart(fignum=5)

    print(
        "\n[SUCCESS] All dissertation figures generated:\n"
        "  dissertation_revenue_comparison.png    (Figure 4.1)\n"
        "  dissertation_passengers_comparison.png (Figure 4.2)\n"
        "  dissertation_occupancy_comparison.png  (Figure 4.3)\n"
        "  dissertation_pricing_uplift.png        (Figure 4.4)\n"
        "  dissertation_convergence.png           (Figure 4.5)\n"
        "\nAll PNG files are 300 DPI — ready to embed in Word or LaTeX."
    )


if __name__ == "__main__":
    create_dissertation_charts()
