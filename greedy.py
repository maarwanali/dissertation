"""
Rideshare Greedy Baseline (Segment-Aware MKP) — with Pricing Integration
=========================================================================
Goal: Maximise the NUMBER of passengers assigned to vehicles while
      strictly respecting per-segment capacity on a fixed route.

Pricing integration
-------------------
Each Passenger now carries two extra fields:
  • dist_km     — real-world distance used by the pricing engine
  • max_budget  — the maximum fare this passenger is willing to pay

During assignment the decoder:
  1. Computes the *current* occupancy rate of the candidate driver
     (seats already filled / total capacity).
  2. Calls PricingEngine.calculate_price() to obtain the quoted fare
     for (passenger.dist_km, scenario demand, is_weekend, occupancy).
  3. Accepts the passenger only if quoted_price <= passenger.max_budget.

Because low occupancy triggers a discount, under-utilised drivers
become cheaper and attract more passengers — directly tying pricing
to occupancy maximisation.

Strategies implemented:
  1. FirstFit  — assigns passenger to the first affordable driver
  2. BestFit   — assigns passenger to the driver with the tightest
                 remaining capacity (still subject to affordability)
  3. RandomFit — stochastic shuffle + FirstFit, best of N trials

Return contract (all public methods):
    (assignment, fitness_score, total_distance, total_revenue, price_map)
    ─ assignment    : List[List[str]]   — passenger IDs per driver
    ─ fitness_score : int               — (count × 10 000) + distance
    ─ total_distance: int
    ─ total_revenue : float             — sum of fares collected
    ─ price_map     : Dict[str, float]  — {pax_id: quoted_price}
"""

import random
from typing import Dict, List, Optional, Tuple

from pricing import PricingEngine


# ─────────────────────────────────────────────────────────────
# Domain Objects
# ─────────────────────────────────────────────────────────────


class Driver:
    __slots__ = ("id", "capacity")

    def __init__(self, driver_id: str, capacity: int):
        self.id = driver_id
        self.capacity = capacity


class Passenger:
    """
    Route passenger with budget constraint.

    Parameters
    ----------
    pid        : unique identifier
    start      : boarding stop index
    end        : alighting stop index
    dist_km    : real-world trip distance fed to the pricing engine
                 (defaults to (end - start) × 25 km if not provided)
    max_budget : maximum fare this passenger is willing to pay ($)
                 (defaults to a generous $999 so legacy code still works)
    """
    __slots__ = ("id", "start", "end", "distance",
                 "dist_km", "max_budget", "base_price")

    def __init__(
        self,
        pid: str,
        start: int,
        end: int,
        dist_km: Optional[float] = None,
        max_budget: float = 999.0,
    ):
        self.id = pid
        self.start = start
        self.end = end
        self.distance = end - start          # segment hops
        # If no real-world distance supplied, approximate from segment count
        self.dist_km = dist_km if dist_km is not None else float(
            self.distance * 25)
        self.max_budget = max_budget
        self.base_price = 0.0   # set by PricingEngine.precompute_baselines()


class Problem:
    """
    Encapsulates drivers, passengers, route topology, and (optionally)
    the pricing engine + scenario context.

    Parameters
    ----------
    drivers       : list of Driver
    passengers    : list of Passenger
    num_stops     : total stops on the fixed route
    pricing_engine: PricingEngine instance (or None to disable pricing)
    demand_score  : market demand index passed to the pricing engine
    is_weekend    : 1 = weekend, 0 = weekday (passed to pricing engine)
    """

    def __init__(
        self,
        drivers: list,
        passengers: list,
        num_stops: int,
        pricing_engine: Optional[PricingEngine] = None,
        demand_score: float = 1.0,
        is_weekend: int = 0,
    ):
        self.drivers = drivers
        self.passengers = passengers
        self.num_stops = num_stops
        self.num_segments = num_stops - 1
        self.pricing_engine = pricing_engine
        self.demand_score = demand_score
        self.is_weekend = is_weekend

        # Build a fast lookup: pax_id → Passenger
        self._pax_lookup: Dict[str, Passenger] = {
            p.id: p for p in passengers
        }

        # Pre-compute the market baseline for every passenger in one bulk
        # model call.  This avoids calling pd.DataFrame + model.predict
        # inside the hot decode loop (millions of times with 1000 passengers).
        if pricing_engine is not None:
            pricing_engine.precompute_baselines(
                passengers, demand_score, is_weekend
            )

    # ------------------------------------------------------------------
    # Internal helper — quoted price for a passenger given a driver's
    # current seat usage across all its segments.
    # ------------------------------------------------------------------

    def _quoted_price(
        self,
        passenger: Passenger,
        driver_idx: int,
        route_usage: List[List[int]],
    ) -> float:
        """
        Calculate the dynamic price quoted to *passenger* if they board
        *driver_idx* right now.

        Occupancy rate = max segment utilisation on passenger's route
        (worst-case: if the bottleneck segment is full we won't assign
        anyway, but we use the average for a fairer price signal).
        """
        if self.pricing_engine is None:
            return 0.0          # no pricing → always affordable

        driver = self.drivers[driver_idx]
        total_capacity = driver.capacity

        if total_capacity == 0:
            return 0.0

        # Average occupancy across the passenger's segments
        segs = list(range(passenger.start, passenger.end))
        if not segs:
            return 0.0

        avg_used = sum(route_usage[driver_idx][s] for s in segs) / len(segs)
        occupancy_rate = avg_used / total_capacity

        return self.pricing_engine.calculate_price(
            base_price=passenger.base_price,   # pre-computed — no model call here
            current_occupancy_rate=occupancy_rate,
            max_budget=passenger.max_budget,
        )

    # ------------------------------------------------------------------
    # Shared decoder — used by both Greedy (via permutation) and GA
    # ------------------------------------------------------------------

    def decode(self, perm: list) -> tuple:
        """
        Translate a passenger permutation into a price-aware assignment.

        A passenger is accepted by a driver only if:
          (a) the driver has a free seat on every required segment, AND
          (b) the dynamically-quoted price <= passenger.max_budget.

        When occupancy is low the pricing engine applies a discount,
        making it more likely that budget-constrained passengers can
        afford a seat — directly linking pricing to occupancy uplift.

        Returns
        -------
        assignment    : List[List[str]]
        fitness_score : int   — (count × 10 000) + total_distance
        total_distance: int
        total_revenue : float — sum of fares collected
        price_map     : Dict[str, float] — {pax_id: quoted price}
        """
        n_drivers = len(self.drivers)
        route_usage = [[0] * self.num_segments for _ in range(n_drivers)]
        assignment = [[] for _ in range(n_drivers)]

        total_seg_hops = 0    # segment hops — used only as fitness tiebreaker
        total_dist_km = 0.0  # real kilometres — returned for reporting
        total_count = 0
        total_revenue = 0.0
        price_map: Dict[str, float] = {}

        for p in perm:
            for di, driver in enumerate(self.drivers):
                # ── Capacity check ────────────────────────────────────
                can_fit = all(
                    route_usage[di][seg] < driver.capacity
                    for seg in range(p.start, p.end)
                )
                if not can_fit:
                    continue

                # ── Affordability check ───────────────────────────────
                quoted = self._quoted_price(p, di, route_usage)
                if self.pricing_engine is not None and quoted > p.max_budget:
                    # Too expensive for this passenger on this driver right
                    # now — try the next driver (might be emptier / cheaper)
                    continue

                # ── Accept ────────────────────────────────────────────
                assignment[di].append(p.id)
                total_seg_hops += p.distance   # hops (tiebreaker only)
                total_dist_km += p.dist_km    # real km (reporting)
                total_count += 1
                total_revenue += quoted
                price_map[p.id] = round(quoted, 2)
                for seg in range(p.start, p.end):
                    route_usage[di][seg] += 1
                break   # passenger boarded — move to next

        fitness_score = (total_count * 10_000) + total_seg_hops
        return assignment, fitness_score, round(total_dist_km, 2), total_revenue, price_map

    def theoretical_max_passengers(self) -> int:
        """Upper-bound: total fleet seats (ignores overlap conflicts)."""
        return sum(d.capacity for d in self.drivers)


# ─────────────────────────────────────────────────────────────
# Greedy Solver
# ─────────────────────────────────────────────────────────────


class GreedySolver:
    def __init__(self, problem: Problem):
        self.problem = problem

    # ------------------------------------------------------------------
    # Internal assignment engine
    # ------------------------------------------------------------------

    def _assign(
        self,
        ordered_passengers: List[Passenger],
        strategy: str,
    ) -> Tuple[List[List[str]], int, int, float, Dict[str, float]]:
        """
        Core segment-aware, price-aware assignment.

        Returns
        -------
        assignment     : List[List[str]]
        fitness_score  : int
        total_distance : int
        total_revenue  : float
        price_map      : Dict[str, float]
        """
        num_drivers = len(self.problem.drivers)
        num_segments = self.problem.num_segments
        pricing = self.problem.pricing_engine

        route_usage = [[0] * num_segments for _ in range(num_drivers)]
        assignment = [[] for _ in range(num_drivers)]
        total_seg_hops = 0    # hops — fitness tiebreaker only
        total_dist_km = 0.0  # real km — for reporting
        total_count = 0
        total_revenue = 0.0
        price_map: Dict[str, float] = {}

        for p in ordered_passengers:
            best_driver_idx = None
            best_quoted_price = 0.0
            tightest_fit_space = float("inf")

            for di, driver in enumerate(self.problem.drivers):
                can_fit = True
                bottleneck_space = driver.capacity

                for seg in range(p.start, p.end):
                    used = route_usage[di][seg]
                    if used >= driver.capacity:
                        can_fit = False
                        break
                    available = driver.capacity - used
                    if available < bottleneck_space:
                        bottleneck_space = available

                if not can_fit:
                    continue

                # ── Affordability gate ────────────────────────────────
                quoted = self.problem._quoted_price(p, di, route_usage)
                if pricing is not None and quoted > p.max_budget:
                    continue    # unaffordable on this driver — try next

                # ── Strategy selection ────────────────────────────────
                if strategy == "first_fit":
                    best_driver_idx = di
                    best_quoted_price = quoted
                    break

                elif strategy == "best_fit":
                    if bottleneck_space < tightest_fit_space:
                        tightest_fit_space = bottleneck_space
                        best_driver_idx = di
                        best_quoted_price = quoted

            if best_driver_idx is not None:
                assignment[best_driver_idx].append(p.id)
                total_seg_hops += p.distance    # hops (tiebreaker)
                total_dist_km += p.dist_km     # real km (reporting)
                total_count += 1
                total_revenue += best_quoted_price
                price_map[p.id] = round(best_quoted_price, 2)
                for seg in range(p.start, p.end):
                    route_usage[best_driver_idx][seg] += 1

        fitness_score = (total_count * 10_000) + total_seg_hops
        return assignment, fitness_score, round(total_dist_km, 2), total_revenue, price_map

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def first_fit(self):
        """Longest trips first → first affordable driver."""
        sorted_pax = sorted(
            self.problem.passengers, key=lambda p: p.distance, reverse=True
        )
        return self._assign(sorted_pax, "first_fit")

    def best_fit(self):
        """Longest trips first → tightest affordable driver."""
        sorted_pax = sorted(
            self.problem.passengers, key=lambda p: p.distance, reverse=True
        )
        return self._assign(sorted_pax, "best_fit")

    def random_fit(self, trials: int = 100):
        """Stochastic shuffle + FirstFit × N trials; best fitness returned."""
        best_result = None
        best_score = -1

        for _ in range(trials):
            shuffled = self.problem.passengers[:]
            random.shuffle(shuffled)
            result = self._assign(shuffled, "first_fit")
            if result[1] > best_score:
                best_score = result[1]
                best_result = result

        return best_result


# ─────────────────────────────────────────────────────────────
# Pretty-print helper
# ─────────────────────────────────────────────────────────────

def _print_result(
    label: str,
    drivers,
    passengers,
    assignment,
    fitness,
    distance,
    revenue: float = 0.0,
    price_map: Dict[str, float] = None,
):
    price_map = price_map or {}
    assigned_ids = {pid for route in assignment for pid in route}
    count = len(assigned_ids)
    unassigned = [p.id for p in passengers if p.id not in assigned_ids]

    # Segment-aware occupancy — the only formula that stays within [0, 100]%.
    # Each assigned passenger consumes (end - start) seat-segments.
    # Each driver offers (capacity x num_segments) seat-segments in total.
    # Dividing passengers-assigned by seat-count (the old formula) ignores
    # trip length and can exceed 100%, which is physically meaningless.
    used_segments = sum(p.distance for p in passengers if p.id in assigned_ids)
    num_segments = max((p.end for p in passengers), default=1)  # 0-based stops
    total_segments = sum(d.capacity for d in drivers) * num_segments
    occupancy = (used_segments / total_segments * 100) if total_segments else 0

    print(f"\n── {label} " + "─" * max(0, 50 - len(label)))
    for driver, route in zip(drivers, assignment):
        detail = ", ".join(
            f"{pid}(${price_map[pid]:.2f})" if pid in price_map else pid
            for pid in route
        )
        print(f"  {driver.id} (cap {driver.capacity}) → [{detail}]")
    print(f"\n  Passengers Assigned : {count} / {len(passengers)}")
    print(f"  Fleet Occupancy     : {occupancy:.1f}%")
    print(f"  Total Distance (km) : {distance}")
    print(f"  Total Revenue ($)   : {revenue:.2f}")
    print(f"  Fitness Score       : {fitness}")
    print(f"  Unassigned          : {unassigned if unassigned else 'none'}")


# ─────────────────────────────────────────────────────────────
# Example / smoke-test  (uses sample CSV data format)
# ─────────────────────────────────────────────────────────────

def example_run():
    pricing = PricingEngine()

    drivers = [
        Driver("CarA", capacity=2),
        Driver("CarB", capacity=2),
        Driver("CarC", capacity=3),
    ]
    num_stops = 6

    # Mirrors the CSV example data (pax_id, start, end, dist_km, max_budget)
    passengers = [
        Passenger("P000", 3, 4, dist_km=25,  max_budget=9.39),
        Passenger("P001", 0, 4, dist_km=100, max_budget=15.98),
        Passenger("P002", 1, 4, dist_km=75,  max_budget=17.86),
        Passenger("P003", 0, 3, dist_km=75,  max_budget=17.44),
        Passenger("P004", 4, 5, dist_km=25,  max_budget=12.05),
        Passenger("P1",   0, 5, dist_km=125, max_budget=30.00),
        Passenger("P2",   0, 2, dist_km=50,  max_budget=12.00),
        Passenger("P3",   2, 5, dist_km=75,  max_budget=18.00),
        Passenger("P4",   1, 4, dist_km=75,  max_budget=20.00),
        Passenger("P5",   0, 3, dist_km=75,  max_budget=16.00),
        Passenger("P6",   3, 5, dist_km=50,  max_budget=14.00),
        Passenger("P7",   1, 5, dist_km=100, max_budget=25.00),
        Passenger("P8",   0, 1, dist_km=25,  max_budget=8.00),
    ]

    prob = Problem(
        drivers, passengers, num_stops,
        pricing_engine=pricing,
        demand_score=1.5,
        is_weekend=0,
    )
    solver = GreedySolver(prob)

    print(f"Theoretical max passengers (fleet seats): "
          f"{prob.theoretical_max_passengers()}\n")

    for label, result in [
        ("FirstFit",              solver.first_fit()),
        ("BestFit",               solver.best_fit()),
        ("RandomFit (50 trials)", solver.random_fit(trials=50)),
    ]:
        _print_result(label, drivers, passengers, *result)


if __name__ == "__main__":
    example_run()
