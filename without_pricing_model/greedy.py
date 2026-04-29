"""
Rideshare Greedy Baseline (Segment-Aware MKP)
==============================================
Goal: Maximise the NUMBER of passengers assigned to vehicles while
      strictly respecting per-segment capacity on a fixed route.

Strategies implemented:
  1. FirstFit  — assigns passenger to the first driver with available
                 seats on all required segments.
  2. BestFit   — assigns passenger to the driver with the LEAST remaining
                 capacity on the bottleneck segment (tightest fit).
  3. RandomFit — shuffles passengers randomly, then applies FirstFit.
                 Runs multiple times and returns the best result.

Return contract (public methods first_fit / best_fit / random_fit):
    (assignment, fitness_score, total_distance)
    ─ assignment      : List[List[str]]  — passenger IDs per driver
    ─ fitness_score   : int              — (count × 10 000) + distance
    ─ total_distance  : int              — sum of passenger-km assigned
"""

import random
import time
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────
# Domain Objects
# ─────────────────────────────────────────────────────────────


class Driver:
    __slots__ = ("id", "capacity")

    def __init__(self, driver_id: str, capacity: int):
        self.id = driver_id
        self.capacity = capacity


class Passenger:
    __slots__ = ("id", "start", "end", "distance")

    def __init__(self, pid: str, start: int, end: int):
        self.id = pid
        self.start = start
        self.end = end
        self.distance = end - start  # segment-km this passenger occupies


class Problem:
    def __init__(self, drivers: list, passengers: list, num_stops: int):
        self.drivers = drivers
        self.passengers = passengers
        self.num_stops = num_stops
        self.num_segments = num_stops - 1

    # ------------------------------------------------------------------
    # Shared decoder — used by both Greedy (via permutation) and GA
    # ------------------------------------------------------------------
    def decode(self, perm: list) -> tuple:
        """
        Translate a passenger permutation into an assignment.

        Priority: maximise COUNT of passengers assigned, then distance.

        Returns
        -------
        assignment    : List[List[str]]
        fitness_score : int   — (count × 10 000) + total_distance
        total_distance: int
        """
        n_drivers = len(self.drivers)
        route_usage = [[0] * self.num_segments for _ in range(n_drivers)]
        assignment = [[] for _ in range(n_drivers)]

        total_distance = 0
        total_count = 0

        for p in perm:
            for di, driver in enumerate(self.drivers):
                can_fit = all(
                    route_usage[di][seg] < driver.capacity
                    for seg in range(p.start, p.end)
                )
                if can_fit:
                    assignment[di].append(p.id)
                    total_distance += p.distance
                    total_count += 1
                    for seg in range(p.start, p.end):
                        route_usage[di][seg] += 1
                    break  # passenger assigned — move to next

        # Fitness heavily weights count so GA/greedy prefer more passengers
        fitness_score = (total_count * 10_000) + total_distance
        return assignment, fitness_score, total_distance

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
    ) -> Tuple[List[List[str]], int, int]:
        """
        Core segment-aware assignment.

        Returns
        -------
        assignment     : List[List[str]]
        fitness_score  : int   — (count × 10 000) + total_distance
        total_distance : int
        """
        num_drivers = len(self.problem.drivers)
        num_segments = self.problem.num_segments

        route_usage = [[0] * num_segments for _ in range(num_drivers)]
        assignment = [[] for _ in range(num_drivers)]
        total_distance = 0
        total_count = 0

        for p in ordered_passengers:
            best_driver_idx = None
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

                if can_fit:
                    if strategy == "first_fit":
                        best_driver_idx = di
                        break  # take first valid driver immediately

                    elif strategy == "best_fit":
                        # prefer driver with least leftover capacity (tighter pack)
                        if bottleneck_space < tightest_fit_space:
                            tightest_fit_space = bottleneck_space
                            best_driver_idx = di

            if best_driver_idx is not None:
                assignment[best_driver_idx].append(p.id)
                total_distance += p.distance
                total_count += 1
                for seg in range(p.start, p.end):
                    route_usage[best_driver_idx][seg] += 1

        fitness_score = (total_count * 10_000) + total_distance
        return assignment, fitness_score, total_distance

    # ------------------------------------------------------------------
    # Public API  ─ all return (assignment, fitness_score, total_distance)
    # ------------------------------------------------------------------

    def first_fit(self) -> Tuple[List[List[str]], int, int]:
        """
        Sort passengers by trip length (longest first) then assign to the
        first available vehicle on every required segment.
        """
        sorted_pax = sorted(
            self.problem.passengers, key=lambda p: p.distance, reverse=True
        )
        return self._assign(sorted_pax, "first_fit")

    def best_fit(self) -> Tuple[List[List[str]], int, int]:
        """
        Sort passengers by trip length (longest first) then assign each to
        the vehicle with the tightest remaining capacity (minimises waste).
        """
        sorted_pax = sorted(
            self.problem.passengers, key=lambda p: p.distance, reverse=True
        )
        return self._assign(sorted_pax, "best_fit")

    def random_fit(self, trials: int = 100) -> Tuple[List[List[str]], int, int]:
        """
        Stochastic greedy: shuffle passengers and apply FirstFit repeatedly.
        Returns the trial with the highest fitness score.
        """
        best_result = None
        best_score = -1

        for _ in range(trials):
            shuffled = self.problem.passengers[:]
            random.shuffle(shuffled)
            result = self._assign(shuffled, "first_fit")
            if result[1] > best_score:   # index 1 = fitness_score
                best_score = result[1]
                best_result = result

        return best_result


# ─────────────────────────────────────────────────────────────
# Pretty-print helper (standalone use / debugging)
# ─────────────────────────────────────────────────────────────

def _print_result(label: str, drivers, passengers, assignment, fitness, distance):
    assigned_ids = {pid for route in assignment for pid in route}
    count = len(assigned_ids)
    unassigned = [p.id for p in passengers if p.id not in assigned_ids]
    total_seats = sum(d.capacity for d in drivers)
    occupancy = (count / total_seats * 100) if total_seats else 0

    print(f"\n── {label} " + "─" * max(0, 50 - len(label)))
    for driver, route in zip(drivers, assignment):
        print(f"  {driver.id} (cap {driver.capacity}) → {route or '[]'}")
    print(f"\n  Passengers Assigned : {count} / {len(passengers)}")
    print(f"  Fleet Occupancy     : {occupancy:.1f}%")
    print(f"  Total Distance (km) : {distance}")
    print(f"  Fitness Score       : {fitness}")
    print(f"  Unassigned          : {unassigned if unassigned else 'none'}")


# ─────────────────────────────────────────────────────────────
# Example / smoke-test
# ─────────────────────────────────────────────────────────────

def example_run():
    drivers = [
        Driver("CarA", capacity=2),
        Driver("CarB", capacity=2),
        Driver("CarC", capacity=3),
    ]
    num_stops = 6
    passengers = [
        Passenger("P1", 0, 5),
        Passenger("P2", 0, 2),
        Passenger("P3", 2, 5),
        Passenger("P4", 1, 4),
        Passenger("P5", 0, 3),
        Passenger("P6", 3, 5),
        Passenger("P7", 1, 5),
        Passenger("P8", 0, 1),
    ]

    prob = Problem(drivers, passengers, num_stops)
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
