import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib  # To save the model for the GA to use


def train_offline_model(csv_file):
    # 1. Load the historical data
    df = pd.read_csv(csv_file)

    # 2. Define our Features (X) and our Target (y)
    # Price = B0 + B1(dist) + B2(demand) + B3(weekend)
    X = df[['distance_km', 'demand_score', 'is_weekend']]
    y = df['observed_price_BYN']

    # 3. Create and train the model
    model = LinearRegression()
    model.fit(X, y)

    # 4. Extract the "Learning" for your dissertation
    b0 = model.intercept_
    b1, b2, b3 = model.coef_

    print("--- Model Parameters Found ---")
    print(f"Intercept (B0): {b0:.4f}  (Base fee)")
    print(f"Distance  (B1): {b1:.4f}  (Cost per km)")
    print(f"Demand    (B2): {b2:.4f}  (Surge impact)")
    print(f"Weekend   (B3): {b3:.4f}  (Holiday impact)")

    # 5. Evaluate accuracy
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"\nModel Accuracy (R-squared): {r2:.4f}")

    # 6. Save the model to a file
    joblib.dump(model, 'pricing_model.pkl')
    print("\n✓ Model saved as pricing_model.pkl")


if __name__ == "__main__":
    train_offline_model('data/historical_prices_belarus.csv')
