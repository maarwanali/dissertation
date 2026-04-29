"""
compare_static.py — Master Benchmark: Full 2×2 Dissertation Experiment
=======================================================================
Covers every algorithm × pricing combination on the SAME fixed datasets:

  ┌──────────────────┬──────────────────┬──────────────────┐
  │                  │  Static pricing  │ Dynamic pricing  │
  ├──────────────────┼──────────────────┼──────────────────┤
  │ Greedy (BestFit) │      RUN 1       │      RUN 2       │
  │ Genetic Alg.     │      RUN 3       │      RUN 4       │
  └──────────────────┴──────────────────┴──────────────────┘

Using fixed_datasets.json for all four cells guarantees every number
in the dissertation comes from identical passenger data — required for
a controlled experimental comparison.

Run order
---------
  1. python generator_dataset_static.py  →  fixed_datasets.json
  2. python generator.py                 →  data/historical_prices_belarus.csv
  3. python train_pricing.py             →  pricing_model.pkl
  4. python compare_static.py            →  results_full_benchmark.csv
                                             convergence_ga.csv
  5. python visualisation.py             →  5 dissertation figures (PNG)

Output columns — results_full_benchmark.csv
-------------------------------------------
  Scenario, Dataset_Trial_ID, Algorithm, Pricing_Type,
  Passengers_Assigned, Total_Passengers,
  Total_Revenue_BYN, Seat_Occupancy_Pct, Runtime_MS,
  StdDev_Passengers, StdDev_Revenue, Num_Trials,
  T_Statistic, P_Value, Significance

  Greedy rows: StdDev = 0 (deterministic — 1 run).
  GA rows    : mean over GA_TRIALS; StdDev = trial-to-trial variance.
"""

import time
import json
import statistics
import csv
import concurrent.futures  # <--- NEW: Python's native parallel processing library
from scipy import stats as scipy_stats

from greedy import Driver, Passenger, Problem, GreedySolver
from genetic import GeneticAlgorithm
from pricing import PricingEngine

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
DATASET_FILE = "fixed_datasets.json"
# 10 So out computer could handle the laod in a reasonable time.
GA_TRIALS = 10
GA_POP = 100
GA_GENS = 30
STATIC_BASE_FEE = 2.0
STATIC_RATE = 0.07   # BYN per km — within passenger budget band [0.06, 0.08]

# Demand context per scenario
SCENARIO_CONTEXT = {
    "Low_Demand":      {"demand_score": 0.8,  "is_weekend": 0},
    "Medium_Urban":    {"demand_score": 1.5,  "is_weekend": 0},
    "High_Congestion": {"demand_score": 2.2,  "is_weekend": 1},
    "Extreme_Peak":    {"demand_score": 3.0,  "is_weekend": 1},
}


# ─────────────────────────────────────────────────────────────
# Static Pricing Engine
# ─────────────────────────────────────────────────────────────

class StaticPricingEngine:
    """
    A baseline pricing engine that rigidly applies a fixed base fee + flat rate. 
    It does NOT offer discounts for empty seats. 
    """

    def __init__(self, base_fee: float, rate_per_km: float):
        self.base_fee = base_fee
        self.rate_per_km = rate_per_km

    def precompute_baselines(self, passengers, demand_score, is_weekend):
        # Apply the exact same economic structure (Base + Rate * Dist)
        for p in passengers:
            p.base_price = self.base_fee + (self.rate_per_km * p.dist_km)

    def calculate_price(self, base_price, current_occupancy_rate, max_budget=None):
        # Ignores occupancy. Stubbornly returns the fixed static price.
        return base_price

# ─────────────────────────────────────────────────────────────
# Guard: abort if mock pricing model is active
# ─────────────────────────────────────────────────────────────


def _assert_real_pricing_model(engine: PricingEngine) -> None:
    if engine.using_mock:
        raise RuntimeError(
            "\n[ABORT] PricingEngine is using the built-in mock model.\n"
            "  pricing_model.pkl was not found.\n"
            "  Dissertation results must use the trained model.\n"
            "  Run:  python train_pricing.py  then re-run this script.\n"
        )
    print("[PricingEngine] ✓ Trained model confirmed — proceeding.\n")


# ─────────────────────────────────────────────────────────────
# Problem builder & Metrics
# ─────────────────────────────────────────────────────────────

def load_problem(dataset_instance, pricing_engine, demand_score, is_weekend):
    drivers = [Driver(d["id"], d["capacity"])
               for d in dataset_instance["drivers"]]
    passengers = []
    for p_data in dataset_instance["passengers"]:
        p = Passenger(p_data["id"], p_data["start"], p_data["end"])
        p.dist_km = p_data["dist_km"]
        p.max_budget = p_data["max_budget"]
        passengers.append(p)

    return Problem(
        drivers, passengers, dataset_instance["stops"],
        pricing_engine=pricing_engine,
        demand_score=demand_score,
        is_weekend=is_weekend,
    )


def assigned_ids_from(assignment) -> set:
    return {pid for route in assignment for pid in route}


def seat_occupancy(assigned_ids: set, passengers, drivers, num_stops: int) -> float:
    used = sum(p.distance for p in passengers if p.id in assigned_ids)
    total = sum(d.capacity for d in drivers) * (num_stops - 1)
    return (used / total * 100) if total else 0.0


def one_sample_ttest(sample: list[float], pop_mean: float) -> tuple[float, float]:
    """H0: μ_GA = greedy_count. p < 0.05 → significant difference."""
    if len(sample) < 2:
        return float("nan"), float("nan")
    t_stat, p_val = scipy_stats.ttest_1samp(sample, pop_mean)
    return round(float(t_stat), 4), round(float(p_val), 4)

# ─────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────


def run_single_ga_trial(ga_run, prob, pop_size, generations, num_stops):
    """This is the isolated job that one CPU core will execute."""
    ga = GeneticAlgorithm(prob, pop_size=pop_size, generations=generations)

    t0 = time.perf_counter()
    best_ind, convergence = ga.run(verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1_000

    asgn, _, _, rev, _ = prob.decode(best_ind.perm)
    ids = assigned_ids_from(asgn)

    # Return a dictionary of results for this specific run
    return {
        "run_id": ga_run + 1,
        "count": len(ids),
        "rev": rev,
        "occ": seat_occupancy(ids, prob.passengers, prob.drivers, num_stops),
        "time": elapsed_ms,
        "convergence": convergence
    }


def run_benchmarks():
    dynamic_engine = PricingEngine()
    _assert_real_pricing_model(dynamic_engine)
    static_engine = StaticPricingEngine(
        base_fee=STATIC_BASE_FEE, rate_per_km=STATIC_RATE)

    with open(DATASET_FILE) as f:
        all_datasets = json.load(f)

    result_rows = []
    convergence_rows = []

    PRICING_CONFIGS = [
        ("Static",  static_engine),
        ("Dynamic", dynamic_engine),
    ]

    col = "{:<18} | {:>5} | {:<10} | {:<8} | {:>20} | {:>11} | {:>9} | {:>10} | {}"
    hdr = col.format(
        "Scenario", "Trial", "Algorithm", "Pricing",
        "Pax (assigned / total)", "Rev (BYN)", "Occ (%)", "Time (ms)", "Significance"
    )
    sep = "=" * len(hdr)
    print(hdr)
    print(sep)

    for scenario_name, instances in all_datasets.items():
        ctx = SCENARIO_CONTEXT.get(
            scenario_name, {"demand_score": 1.5, "is_weekend": 0})
        demand_score = ctx["demand_score"]
        is_weekend = ctx["is_weekend"]

        for instance in instances:
            trial_id = instance["trial_id"]
            num_stops = instance["stops"]
            total_pax = len(instance["passengers"])

            for pricing_type, engine in PRICING_CONFIGS:

                prob = load_problem(instance, engine, demand_score, is_weekend)
                solver = GreedySolver(prob)

                # ── GREEDY ─────────────
                t0 = time.perf_counter()
                g_asgn, _, _, g_rev, _ = solver.best_fit()
                g_ms = (time.perf_counter() - t0) * 1_000

                g_ids = assigned_ids_from(g_asgn)
                g_count = len(g_ids)
                g_occ = seat_occupancy(
                    g_ids, prob.passengers, prob.drivers, num_stops)

                result_rows.append([
                    scenario_name, trial_id, "Greedy", pricing_type,
                    g_count, total_pax,
                    round(g_rev, 2), round(g_occ, 2), round(g_ms, 2),
                    0.0, 0.0, 1,
                    "", "", "Baseline"
                ])
                print(col.format(
                    scenario_name, trial_id, "Greedy", pricing_type,
                    f"{g_count} / {total_pax}",
                    f"{g_rev:.2f}", f"{g_occ:.1f}", f"{g_ms:.1f}",
                    "Baseline"
                ))

                # ── GENETIC ALGORITHM ────────
                ga_counts, ga_revs, ga_occs, ga_times = [], [], [], []

                # NEW: Spin up the CPU cores!
                # max_workers=None tells Python to use 100% of the available CPU cores
                with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:

                    # Submit all 30 jobs to the CPU cores at the same time
                    futures = [
                        executor.submit(
                            run_single_ga_trial,
                            ga_run, prob, GA_POP, GA_GENS, num_stops
                        )
                        for ga_run in range(GA_TRIALS)
                    ]

                    # Gather the results as each core finishes its assigned job
                    for future in concurrent.futures.as_completed(futures):
                        res = future.result()
                        ga_counts.append(res["count"])
                        ga_revs.append(res["rev"])
                        ga_occs.append(res["occ"])
                        ga_times.append(res["time"])

                        for gen_idx, pax_count in enumerate(res["convergence"]):
                            convergence_rows.append([
                                scenario_name, trial_id, pricing_type,
                                res["run_id"], gen_idx, pax_count,
                            ])
                avg_count = statistics.mean(ga_counts)
                std_count = statistics.stdev(
                    ga_counts) if len(ga_counts) > 1 else 0.0
                avg_rev = statistics.mean(ga_revs)
                std_rev = statistics.stdev(
                    ga_revs) if len(ga_revs) > 1 else 0.0
                avg_occ = statistics.mean(ga_occs)
                avg_ms = statistics.mean(ga_times)

                # Statistical significance
                t_stat, p_val = one_sample_ttest(ga_counts, g_count)
                sig_flag = (
                    "p<0.05 ✓ significant" if p_val < 0.05 else
                    "p<0.10 marginal" if p_val < 0.10 else
                    "not significant"
                )

                result_rows.append([
                    scenario_name, trial_id, "GeneticAlg", pricing_type,
                    round(avg_count, 2), total_pax,
                    round(avg_rev, 2), round(avg_occ, 2), round(avg_ms, 2),
                    round(std_count, 2), round(std_rev, 2), GA_TRIALS,
                    t_stat, p_val, sig_flag
                ])
                print(col.format(
                    "", "", f"GA ×{GA_TRIALS}", pricing_type,
                    f"{avg_count:.1f}±{std_count:.1f} / {total_pax}",
                    f"{avg_rev:.2f}", f"{avg_occ:.1f}", f"{avg_ms:.0f}",
                    sig_flag
                ))

        print("-" * len(hdr))

    print(sep)

    # ── Export ────────────────────────────────────
    results_file = "results_full_benchmark.csv"
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Scenario", "Dataset_Trial_ID", "Algorithm", "Pricing_Type",
            "Passengers_Assigned", "Total_Passengers",
            "Total_Revenue_BYN", "Seat_Occupancy_Pct", "Runtime_MS",
            "StdDev_Passengers", "StdDev_Revenue", "Num_Trials",
            "T_Statistic", "P_Value", "Significance"
        ])
        writer.writerows(result_rows)
    print(f"\n[RESULTS]     → {results_file}")

    conv_file = "convergence_ga.csv"
    with open(conv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Scenario", "Dataset_Trial_ID", "Pricing_Type",
            "GA_Run", "Generation", "Best_Pax_Count",
        ])
        writer.writerows(convergence_rows)
    print(f"[CONVERGENCE] → {conv_file}")
    print("\nAll done — next step: python visualisation.py")


if __name__ == "__main__":
    run_benchmarks()
