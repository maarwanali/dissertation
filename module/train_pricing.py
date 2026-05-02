import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib


def train_offline_model(csv_file):
    # 1. Load the historical data
    df = pd.read_csv(csv_file)

    # 2. Define our Features (X) and our Target (y)
    X = df[['distance_km', 'demand_score', 'is_weekend']]
    y = df['observed_price_BYN']

    # 3. SPLIT THE DATA (80% Training, 20% Testing) - CRITICAL FOR DISSERTATION
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 4. Create and train the model ONLY on the training set
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Extract the parameters
    b0 = model.intercept_
    b1, b2, b3 = model.coef_

    print("--- Model Parameters Found ---")
    print(f"Intercept (B0): {b0:.4f}  (Base fee)")
    print(f"Distance  (B1): {b1:.4f}  (Cost per km)")
    print(f"Demand    (B2): {b2:.4f}  (Surge impact)")
    print(f"Weekend   (B3): {b3:.4f}  (Holiday impact)")

    # 6. Evaluate accuracy ONLY on the unseen test set
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- ML Model Validation Metrics (Test Set) ---")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f} BYN")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} BYN")

    # 7. Save the model
    # Note: adjusted path to save to root
    joblib.dump(model, './pricing_model.pkl')
    print("\n✓ Model saved as pricing_model.pkl")


if __name__ == "__main__":
    # Adjusted to match folder structure
    train_offline_model('data/historical_prices_belarus.csv')
