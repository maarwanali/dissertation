import json
import random

# Экономические параметры (Таблица 3.1)
SEGMENT_KM = 28.5
BASE_FEE = 2.0
MIN_PRICE = 5.0

SCENARIOS = {
    "Low_Demand":      {"stops": 6,  "num_vehicles": 5,  "num_passengers": 45},
    "Medium_Urban":    {"stops": 10, "num_vehicles": 15, "num_passengers": 130},
    "High_Congestion": {"stops": 15, "num_vehicles": 25, "num_passengers": 250},
    "Extreme_Peak":    {"stops": 20, "num_vehicles": 40, "num_passengers": 450}
}

NUM_TRIALS = 5
BASE_SEED = 42


def generate_unified_data():
    all_datasets = {}

    for scenario_name, params in SCENARIOS.items():
        all_datasets[scenario_name] = []

        for trial in range(1, NUM_TRIALS + 1):
            # Фиксированный seed для воспроизводимости
            random.seed(BASE_SEED + trial)

            # Генерация водителей
            drivers = []
            for v_id in range(params['num_vehicles']):
                capacity = random.randint(2, 4)
                drivers.append({"id": v_id, "capacity": capacity})

            # Генерация пассажиров
            passengers = []
            for p_id in range(params['num_passengers']):
                start = random.randint(0, params['stops'] - 2)
                end = random.randint(start + 1, params['stops'] - 1)

                dist_segments = end - start
                dist_km = dist_segments * SEGMENT_KM

                # Формула бюджета (3.10)
                r_i = random.uniform(0.06, 0.08)
                budget_raw = BASE_FEE + (r_i * dist_km)
                max_budget = round(max(budget_raw, MIN_PRICE), 2)

                passengers.append({
                    "id": p_id,
                    "start": start,
                    "end": end,
                    "dist_km": dist_km,
                    "max_budget": max_budget
                })

            # Структура, которую ожидает ваш код
            all_datasets[scenario_name].append({
                "trial_id": trial,
                "stops": params['stops'],
                "drivers": drivers,
                "passengers": passengers
            })

    with open('data/fixed_datasets.json', 'w', encoding='utf-8') as f:
        json.dump(all_datasets, f, indent=4)
    print("✅ Unified dataset successfully created: fixed_datasets.json")


if __name__ == "__main__":
    generate_unified_data()
