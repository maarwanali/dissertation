import pandas as pd
import numpy as np
import random
import os

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Output directory — compare.py reads from data/
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# --- UPDATED BELARUS TRANSPORT CONSTANTS ---
FUEL_PRICE_BYN = 2.63
KM_PER_LITER = 15.0  # Average of 12-18 km/L range
FUEL_COST_PER_KM_TOTAL = FUEL_PRICE_BYN / KM_PER_LITER  # ~0.175 BYN/km
PASSENGER_CAPACITY = 3.5  # Average of 3-4 passengers
# target range: 0.05 - 0.07 BYN/km
FUEL_SHARE_PER_KM = FUEL_COST_PER_KM_TOTAL / PASSENGER_CAPACITY


def generate_historical_market_data(n_samples=2000):
    """
    SECTION 4.3: Training Data (Belarus Context)
    Simulates historical prices based on cost-recovery for intercity trips.
    """
    # Features (X)
    distances = np.random.uniform(50, 400, n_samples)  # 50km to 400km
    demand_level = np.random.uniform(0.8, 2.0, n_samples)
    is_weekend = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

    # The Logic:
    b0 = 2.0   # Base transaction fee (BYN)
    b1 = 0.065  # Distance coefficient (BYN/km) - aligned with fuel share
    b2 = 1.2   # Demand impact (BYN)
    b3 = 0.8   # Weekend impact (BYN)

    # Price = B0 + B1*Dist + B2*Demand + B3*Weekend
    perfect_price = b0 + (b1 * distances) + \
        (b2 * demand_level) + (b3 * is_weekend)

    # Add realistic variation (Epsilon)
    epsilon = np.random.normal(0, 1.2, n_samples)
    market_price = perfect_price + epsilon

    # Ensure price covers at least the raw fuel share (Floor Price)
    min_fuel_share = distances * FUEL_SHARE_PER_KM
    market_price = np.maximum(market_price, min_fuel_share + b0)

    df = pd.DataFrame({
        'distance_km': np.round(distances, 2),
        'demand_score': np.round(demand_level, 2),
        'is_weekend': is_weekend,
        'observed_price_BYN': np.round(market_price, 2)
    })

    out = os.path.join(DATA_DIR, 'historical_prices_belarus.csv')
    df.to_csv(out, index=False)
    print(f"✓ Created {out} (Avg Price: {df['observed_price_BYN'].mean():.2f} BYN)")


def generate_simulation_requests(n_pax=1000, n_stops=15):
    """
    SECTION 4.4: Simulation Data
    Generates passengers with budget constraints based on fuel-sharing expectations.
    """
    passengers = []
    # Route distance logic (Assume ~28km per stop for a 400km route)
    for i in range(n_pax):
        start = random.randint(0, n_stops - 2)
        end = random.randint(start + 1, n_stops - 1)
        dist_km = (end - start) * 28.5

        # Budget logic: Passengers are willing to pay a share of fuel + base fee
        # Expected rate between 0.06 and 0.08 BYN/km
        expected_rate = np.random.uniform(0.06, 0.08)
        max_budget = 2.0 + (expected_rate * dist_km)

        passengers.append({
            'pax_id': f"P{i:03d}",
            'start': start,
            'end': end,
            'dist_km': round(dist_km, 2),
            # Minimum logical trip price
            'max_budget': round(max(max_budget, 5.0), 2)
        })

    df = pd.DataFrame(passengers)
    out = os.path.join(DATA_DIR, 'passenger_requests_belarus.csv')
    df.to_csv(out, index=False)
    print(f"✓ Created {out}")


if __name__ == "__main__":
    generate_historical_market_data()
    generate_simulation_requests()
