import json
import random

# Scenario Definitions
SCENARIOS = [
    {"name": "Low_Demand",      "drivers": 5,  "pax": 45,  "stops": 6},
    {"name": "Medium_Urban",    "drivers": 15, "pax": 130, "stops": 10},
    {"name": "High_Congestion", "drivers": 25, "pax": 250, "stops": 15},
    {"name": "Extreme_Peak",    "drivers": 40, "pax": 450, "stops": 20},
]

DATASETS_PER_SCENARIO = 5  # Generate 5 different datasets per scenario
BASE_SEED = 42
KM_PER_SEGMENT = 28.5      # Approx km per route segment (matches compare_static.py)


def generate_datasets():
    all_datasets = {}

    for sc in SCENARIOS:
        scenario_name = sc["name"]
        all_datasets[scenario_name] = []

        for trial in range(DATASETS_PER_SCENARIO):
            # Unique seed for each trial so datasets differ, but are reproducible
            current_seed = BASE_SEED + trial
            random.seed(current_seed)

            drivers = [
                {"id": f"D{i}", "capacity": random.randint(2, 4)}
                for i in range(sc["drivers"])
            ]

            passengers = []
            for i in range(sc["pax"]):
                s = random.randint(0, sc["stops"] - 2)
                e = random.randint(s + 1, sc["stops"] - 1)
                dist_km = round((e - s) * KM_PER_SEGMENT, 2)

                # Budget logic: passengers expect to pay a fuel share + base fee.
                # Rate in [0.06, 0.08] BYN/km mirrors generator.py's simulation data,
                # ensuring compare_static.py uses identical budgets every run.
                expected_rate = random.uniform(0.06, 0.08)
                max_budget = round(max(2.0 + expected_rate * dist_km, 5.0), 2)

                passengers.append({
                    "id": f"P{i}",
                    "start": s,
                    "end": e,
                    "distance": e - s,      # segment hops (used by Problem decoder)
                    "dist_km": dist_km,     # real-world km (used by PricingEngine)
                    "max_budget": max_budget,  # willingness-to-pay ceiling (BYN)
                })

            dataset_instance = {
                "trial_id": trial + 1,
                "seed": current_seed,
                "stops": sc["stops"],
                "drivers": drivers,
                "passengers": passengers,
            }
            all_datasets[scenario_name].append(dataset_instance)

    with open("fixed_datasets.json", "w") as f:
        json.dump(all_datasets, f, indent=4)

    print("[SUCCESS] 20 Fixed Datasets generated and saved to 'fixed_datasets.json'")
    print("         Each passenger now includes dist_km and max_budget for")
    print("         reproducible, comparable Static vs Dynamic pricing benchmarks.")


if __name__ == "__main__":
    generate_datasets()
