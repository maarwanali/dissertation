"""
Pricing Engine — Dynamic Rideshare Fare Calculator
===================================================
Uses a trained regression model (pricing_model.pkl) to set a market
baseline, then applies an occupancy-aware discount to convert passengers
whose max_budget would otherwise be too low.

Performance design
------------------
The market baseline (Step 1) depends only on dist_km, demand_score, and
is_weekend — none of which change during a run.  Calling the sklearn
model inside a per-passenger, per-driver, per-individual loop with 1000
passengers × 80 individuals × 100 generations = ~8 million calls is the
main source of slowness.

Solution: Problem.precompute_baselines() calls the model ONCE in bulk
(one numpy predict over all passengers) and stores the result on each
Passenger object as p.base_price.  calculate_price() then skips the
model call entirely and just does the two cheap arithmetic steps.

Dissertation core logic
-----------------------
max_budget is a BEHAVIOURAL THRESHOLD — the fare ceiling above which a
simulated passenger "closes the app."  The engine's job is to quote a
price that:
  1. Does not exceed max_budget           (passenger says YES)
  2. Does not drop below min_price        (operator covers fuel/costs)
  3. Scales with occupancy:
       Empty car → deep discount → convert budget-sensitive passengers
       Full  car → market price  → no need to incentivise

Pricing formula
---------------
  base_price    = regression_model(dist_km, demand_score, is_weekend)
                  [computed ONCE per passenger via precompute_baselines]
  discount      = (1 - occupancy_rate) × MAX_DISCOUNT      [0 – 30 %]
  dynamic_price = base_price × (1 - discount)

  If max_budget supplied and dynamic_price > max_budget
  AND occupancy < 0.50 (car genuinely needs filling):
      → quote max_budget  (meet the passenger at their ceiling)
  final_price = max(dynamic_price_or_budget, min_price)

Public API
----------
  engine.precompute_baselines(passengers, demand_score, is_weekend)
      → sets p.base_price on every Passenger; call once after CSV load.

  engine.calculate_price(base_price, current_occupancy_rate,
                          max_budget=None) -> float
      → pure arithmetic, no model call, safe in tight loops.
"""

import numpy as np
import pandas as pd

try:
    import joblib
    _JOBLIB_OK = True
except ImportError:
    _JOBLIB_OK = False

_FEATURE_NAMES = ["distance_km", "demand_score", "is_weekend"]
MAX_DISCOUNT   = 0.30   # maximum occupancy-driven discount (30 %)


# ─────────────────────────────────────────────────────────────
# Mock model — used when pricing_model.pkl is absent
# ─────────────────────────────────────────────────────────────

class _MockModel:
    """
    Linear stand-in calibrated to typical rideshare data ranges.
    Accepts a named DataFrame (production path) or a numpy array (tests).

        base = 2.50 + 0.12 × dist_km + 1.80 × demand_score
               + 1.50 × is_weekend
    """
    def predict(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            d = X["distance_km"].values.astype(float)
            s = X["demand_score"].values.astype(float)
            w = X["is_weekend"].values.astype(float)
        else:
            arr = np.asarray(X, dtype=float)
            d, s, w = arr[:, 0], arr[:, 1], arr[:, 2]
        return 2.50 + 0.12 * d + 1.80 * s + 1.50 * w


# ─────────────────────────────────────────────────────────────
# PricingEngine
# ─────────────────────────────────────────────────────────────

class PricingEngine:
    """
    Parameters
    ----------
    model_path : str   — path to joblib-serialised sklearn regression model
    min_price  : float — hard floor; no fare is ever quoted below this
    """

    MIN_PRICE_DEFAULT = 5.0

    def __init__(self, model_path: str = "pricing_model.pkl",
                 min_price: float = MIN_PRICE_DEFAULT):
        self.min_price = min_price
        self.using_mock = False

        if _JOBLIB_OK:
            try:
                self.model = joblib.load(model_path)
                print(f"[PricingEngine] Loaded model from '{model_path}'")
            except FileNotFoundError:
                print(f"[PricingEngine] '{model_path}' not found — "
                      "using built-in linear mock model.")
                self.model = _MockModel()
                self.using_mock = True
        else:
            print("[PricingEngine] joblib not available — "
                  "using built-in linear mock model.")
            self.model = _MockModel()
            self.using_mock = True

    # ------------------------------------------------------------------
    # Bulk baseline computation  (call ONCE per Problem, not per decode)
    # ------------------------------------------------------------------

    def precompute_baselines(
        self,
        passengers: list,
        demand_score: float,
        is_weekend: int,
    ) -> None:
        """
        Run the regression model ONCE over all passengers in bulk and
        store the result as p.base_price on each Passenger object.

        This is the performance-critical change: instead of calling
        pd.DataFrame + model.predict inside the inner decode loop
        (millions of times), we do one vectorised numpy call here and
        cache the result.  calculate_price() then does only two
        multiplications per call.

        Call this after constructing Problem, before any solver runs.
        Problem.__init__ calls it automatically when pricing_engine is set.
        """
        if not passengers:
            return

        # Build feature matrix in one go — shape (N, 3)
        n = len(passengers)
        X = np.empty((n, 3), dtype=float)
        for i, p in enumerate(passengers):
            X[i, 0] = p.dist_km
            X[i, 1] = demand_score
            X[i, 2] = is_weekend

        # One model call for all N passengers
        X_df = pd.DataFrame(X, columns=_FEATURE_NAMES)
        baselines = self.model.predict(X_df).astype(float)

        for p, base in zip(passengers, baselines):
            p.base_price = float(base)

    # ------------------------------------------------------------------
    # Per-boarding price calculation  (pure arithmetic — no model call)
    # ------------------------------------------------------------------

    def calculate_price(
        self,
        base_price: float,
        current_occupancy_rate: float,
        max_budget: float = None,
    ) -> float:
        """
        Quote a dynamic fare for one passenger boarding decision.

        Parameters
        ----------
        base_price            : pre-computed market baseline (p.base_price)
        current_occupancy_rate: seats already filled / total capacity [0, 1]
        max_budget            : passenger's willingness-to-pay ceiling

        Returns
        -------
        float — quoted fare (always >= min_price)
        """
        # Step 1 — occupancy discount (empty car → cheaper)
        discount      = (1.0 - current_occupancy_rate) * MAX_DISCOUNT
        dynamic_price = base_price * (1.0 - discount)

        # Step 2 — budget-aware adjustment
        #   If the discounted price is still above budget AND the car is
        #   under 50 % full, drop to the passenger's exact budget ceiling.
        #   This is the "fill the empty seat" behaviour from the dissertation.
        if max_budget is not None and dynamic_price > max_budget:
            if current_occupancy_rate < 0.50:
                dynamic_price = max_budget

        # Step 3 — hard floor (never below fuel/operating cost)
        return max(dynamic_price, self.min_price)

    def occupancy_rate(self, seats_used: int, total_seats: int) -> float:
        """Convenience helper: safe division → [0.0, 1.0]."""
        return (seats_used / total_seats) if total_seats > 0 else 0.0


# ─────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = PricingEngine()

    # Simulate precompute for a few passengers
    class _FakePax:
        def __init__(self, dist_km): self.dist_km = dist_km
    pax = [_FakePax(28.5), _FakePax(100.0), _FakePax(342.0)]
    engine.precompute_baselines(pax, demand_score=1.0, is_weekend=0)

    print("\nDissertation scenario: Minsk → Gomel, 28.5 km, budget = 20 BYN")
    print("─" * 62)
    base = pax[0].base_price
    for label, occ, budget in [
        ("Empty car,  no  budget hint",  0.0,  None),
        ("Empty car,  budget = 20 BYN",  0.0,  20.0),
        ("Half  car,  budget = 20 BYN",  0.5,  20.0),
        ("Full  car,  budget = 20 BYN",  1.0,  20.0),
        ("Empty car,  budget =  5 BYN",  0.0,   5.0),
    ]:
        p = engine.calculate_price(base, occ, budget)
        boards = "✓ BOARDS" if (budget is None or p <= budget) else "✗ walks"
        print(f"  {label:<35} occ={occ:.0%}  →  {p:>6.2f} BYN  {boards}")

    print(f"\n  Precomputed baselines: "
          f"28.5km={pax[0].base_price:.2f}  "
          f"100km={pax[1].base_price:.2f}  "
          f"342km={pax[2].base_price:.2f}")
