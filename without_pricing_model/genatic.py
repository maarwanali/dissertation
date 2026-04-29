"""
Rideshare Genetic Algorithm (Segment-Aware MKP)
==============================================
Goal: Maximise the NUMBER of passengers assigned across a fixed route
      with multiple stops, respecting vehicle capacity on every segment.

Features:
  • Representation : Permutation-based chromosome (Passenger objects).
  • Decoder        : Deterministic segment-aware assignment (shared with Greedy).
  • Crossover      : Partially Mapped Crossover (PMX).
  • Mutation       : Adaptive weights — Swap, Relocate, Reverse.
  • Selection      : Tournament selection with Elitism (top-2 carry-over).

Return contract for GeneticAlgorithm.run():
    Individual  — best_ind.perm  : List[Passenger]
                  best_ind.fitness : int  (count × 10 000 + distance)
"""

import random
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────
# Domain Objects  (mirrors greedy.py — kept in sync)
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
        self.distance = end - start


class Problem:
    def __init__(self, drivers: list, passengers: list, num_stops: int):
        self.drivers = drivers
        self.passengers = passengers
        self.num_stops = num_stops
        self.num_segments = num_stops - 1

    def decode(self, perm: list) -> tuple:
        """
        Translate a passenger permutation into a segment-aware assignment.

        Priority: maximise COUNT of passengers, then total distance.

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
                    break

        fitness_score = (total_count * 10_000) + total_distance
        return assignment, fitness_score, total_distance

    def theoretical_max_passengers(self) -> int:
        """Upper-bound: total fleet seats (ignores segment conflicts)."""
        return sum(d.capacity for d in self.drivers)


# ─────────────────────────────────────────────────────────────
# Genetic Algorithm Components
# ─────────────────────────────────────────────────────────────


class Individual:
    __slots__ = ("perm", "fitness")

    def __init__(self, perm: List[Passenger], fitness: int):
        self.perm = perm
        self.fitness = fitness


class GeneticAlgorithm:
    def __init__(
        self,
        problem: Problem,
        pop_size: int = 100,
        generations: int = 150,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.20,
        tournament_k: int = 5,
        elitism: int = 2,
    ):
        self.problem = problem
        self.pop_size = pop_size
        self.generations = generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.tournament_k = tournament_k
        self.elitism = elitism

    # ------------------------------------------------------------------
    # Individual factory
    # ------------------------------------------------------------------
    def _make_individual(self, perm: List[Passenger]) -> Individual:
        # decode returns (assignment, fitness, distance) — we only need fitness here
        _, fitness, _ = self.problem.decode(perm)
        return Individual(perm, fitness)

    # ------------------------------------------------------------------
    # Population initialisation
    # ------------------------------------------------------------------
    def _initial_population(self) -> List[Individual]:
        pop = []

        # Seed 1 — shortest trips first (maximises number of passengers fitting)
        short_first = sorted(self.problem.passengers,
                             key=lambda p: p.distance)
        pop.append(self._make_individual(short_first))

        # Seed 2 — longest trips first (classic greedy heuristic)
        long_first = sorted(self.problem.passengers,
                            key=lambda p: p.distance, reverse=True)
        pop.append(self._make_individual(long_first))

        # Fill the rest randomly
        for _ in range(self.pop_size - 2):
            perm = self.problem.passengers[:]
            random.shuffle(perm)
            pop.append(self._make_individual(perm))

        return pop

    # ------------------------------------------------------------------
    # Crossover — Partially Mapped Crossover (PMX)
    # ------------------------------------------------------------------
    def _pmx_crossover(
        self,
        p1: List[Passenger],
        p2: List[Passenger],
    ) -> Tuple[List[Passenger], List[Passenger]]:
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))

        def _child(main, other):
            child = [None] * size
            child[a: b + 1] = main[a: b + 1]
            segment_set = set(id(x) for x in child[a: b + 1])

            for i in range(a, b + 1):
                candidate = other[i]
                if id(candidate) not in segment_set:
                    # Walk the mapping chain to find an empty slot
                    pos = i
                    while a <= pos <= b:
                        val = main[pos]
                        pos = next(
                            j for j, x in enumerate(other) if id(x) == id(val)
                        )
                    child[pos] = candidate
                    segment_set.add(id(candidate))

            # Fill remaining None slots in order from `other`
            other_iter = (x for x in other if id(x) not in segment_set)
            for i in range(size):
                if child[i] is None:
                    child[i] = next(other_iter)
            return child

        return _child(p1, p2), _child(p2, p1)

    # ------------------------------------------------------------------
    # Mutation — randomly pick Swap / Relocate / Reverse
    # ------------------------------------------------------------------
    def _mutate(self, perm: List[Passenger]) -> List[Passenger]:
        n = len(perm)
        if n < 2:
            return perm
        a, b = random.sample(range(n), 2)
        r = random.random()

        if r < 0.33:                        # Swap
            perm[a], perm[b] = perm[b], perm[a]
        elif r < 0.66:                      # Relocate
            item = perm.pop(a)
            perm.insert(b, item)
        else:                               # Reverse sub-sequence
            i, j = min(a, b), max(a, b)
            perm[i: j + 1] = reversed(perm[i: j + 1])

        return perm

    # ------------------------------------------------------------------
    # Tournament selection
    # ------------------------------------------------------------------
    def _tournament(self, population: List[Individual]) -> Individual:
        contestants = random.sample(population, min(
            self.tournament_k, len(population)))
        return max(contestants, key=lambda x: x.fitness)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, verbose: bool = False) -> Individual:
        population = self._initial_population()
        best_overall = max(population, key=lambda x: x.fitness)

        for gen in range(self.generations):
            population.sort(key=lambda x: x.fitness, reverse=True)

            # Elitism — carry best individuals unchanged
            new_pop: List[Individual] = population[: self.elitism]

            while len(new_pop) < self.pop_size:
                p1 = self._tournament(population)
                p2 = self._tournament(population)

                c1_perm, c2_perm = p1.perm[:], p2.perm[:]

                if random.random() < self.cx_rate:
                    c1_perm, c2_perm = self._pmx_crossover(c1_perm, c2_perm)

                if random.random() < self.mut_rate:
                    c1_perm = self._mutate(c1_perm)
                if random.random() < self.mut_rate:
                    c2_perm = self._mutate(c2_perm)

                new_pop.append(self._make_individual(c1_perm))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._make_individual(c2_perm))

            population = new_pop
            gen_best = max(population, key=lambda x: x.fitness)
            if gen_best.fitness > best_overall.fitness:
                best_overall = gen_best

            if verbose and gen % 20 == 0:
                pax_count = best_overall.fitness // 10_000
                print(f"  Gen {gen:03d} | Best fitness: {best_overall.fitness} "
                      f"({pax_count} passengers)")

        return best_overall


# ─────────────────────────────────────────────────────────────
# Standalone smoke-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    drivers = [Driver("Car1", 3), Driver("Car2", 2)]
    num_stops = 10
    passengers = []
    for i in range(20):
        s = random.randint(0, num_stops - 2)
        e = random.randint(s + 1, num_stops - 1)
        passengers.append(Passenger(f"P{i}", s, e))

    prob = Problem(drivers, passengers, num_stops)

    print("Starting Genetic Algorithm (goal = max passengers)…")
    ga = GeneticAlgorithm(prob, pop_size=100, generations=100)
    best_ind = ga.run(verbose=True)

    asgn, fitness, dist = prob.decode(best_ind.perm)
    assigned_count = sum(len(r) for r in asgn)
    print("\n" + "=" * 45)
    print(f"Passengers assigned : {assigned_count} / {len(passengers)}")
    print(f"Total distance (km) : {dist}")
    print(f"Fitness score       : {fitness}")
    for i, d in enumerate(drivers):
        print(f"  {d.id} (cap {d.capacity}): {asgn[i]}")
    print("=" * 45)
