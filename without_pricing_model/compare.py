import time
import json
import statistics
import csv
from datetime import datetime

from greedy import Driver, Passenger, Problem, GreedySolver
from genatic import GeneticAlgorithm

# Configuration
DATASET_FILE = "fixed_datasets.json"
GA_TRIALS = 10     # Running GA 10 times for academic averaging
GA_POP = 80
GA_GENS = 100


def load_problem_from_json(dataset_instance):
    """Reconstructs the Problem object from the JSON dictionary."""
    drivers = [Driver(d["id"], d["capacity"])
               for d in dataset_instance["drivers"]]
    passengers = [Passenger(p["id"], p["start"], p["end"])
                  for p in dataset_instance["passengers"]]
    return Problem(drivers, passengers, dataset_instance["stops"])


def seat_occupancy(assigned_pax_ids: set, passengers, drivers, num_stops: int) -> float:
    used = sum(p.distance for p in passengers if p.id in assigned_pax_ids)
    total = sum(d.capacity for d in drivers) * (num_stops - 1)
    return (used / total * 100) if total else 0.0


def assigned_ids_from(assignment) -> set:
    return {pid for route in assignment for pid in route}


def run_benchmarks():
    # Load the fixed datasets
    with open(DATASET_FILE, "r") as f:
        all_datasets = json.load(f)

    csv_rows = []

    for scenario_name, instances in all_datasets.items():
        print(f"\nEvaluating Scenario: {scenario_name}")

        for instance in instances:
            trial_id = instance["trial_id"]
            prob = load_problem_from_json(instance)
            total_pax = len(prob.passengers)
            num_stops = instance["stops"]

            # ── 1. GREEDY (Deterministic - runs 1 time) ──
            solver = GreedySolver(prob)
            t0 = time.perf_counter()
            g_asgn, _, g_dist = solver.best_fit()
            g_ms = (time.perf_counter() - t0) * 1_000

            g_ids = assigned_ids_from(g_asgn)
            g_count = len(g_ids)
            g_occ = seat_occupancy(
                g_ids, prob.passengers, prob.drivers, num_stops)

            # Save Greedy Result
            csv_rows.append([
                scenario_name, trial_id, "Greedy", g_count, total_pax,
                round(g_dist, 2), round(g_occ, 2), round(g_ms, 2)
            ])

            # ── 2. GENETIC ALGORITHM (Stochastic - runs 10 times, takes average) ──
            ga_counts, ga_dists, ga_occs, ga_times = [], [], [], []

            for _ in range(GA_TRIALS):
                ga = GeneticAlgorithm(
                    prob, pop_size=GA_POP, generations=GA_GENS)
                t0 = time.perf_counter()
                best_ind = ga.run(verbose=False)
                ga_times.append((time.perf_counter() - t0) * 1_000)

                asgn, _, dist = prob.decode(best_ind.perm)
                ids = assigned_ids_from(asgn)
                ga_counts.append(len(ids))
                ga_dists.append(dist)
                ga_occs.append(seat_occupancy(
                    ids, prob.passengers, prob.drivers, num_stops))

            # Average the 10 GA runs
            avg_count = statistics.mean(ga_counts)
            avg_dist = statistics.mean(ga_dists)
            avg_occ = statistics.mean(ga_occs)
            avg_ms = statistics.mean(ga_times)

            # Save Averaged GA Result
            csv_rows.append([
                scenario_name, trial_id, "GeneticAlg", round(
                    avg_count, 2), total_pax,
                round(avg_dist, 2), round(avg_occ, 2), round(avg_ms, 2)
            ])

            print(
                f"  Trial {trial_id}/5 completed. GA Average Pax: {avg_count:.1f} vs Greedy Pax: {g_count}")

    # ── Export CSV ──
    filename = f"final_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Scenario", "Dataset_Trial_ID", "Algorithm", "Passengers_Assigned",
            "Total_Passengers", "Distance_KM", "Seat_Occupancy_Pct", "Runtime_MS"
        ])
        writer.writerows(csv_rows)

    print(f"\n[SUCCESS] Final averaged results saved to: {filename}")


if __name__ == "__main__":
    run_benchmarks()
