"""
Rideshare Genetic Algorithm (Segment-Aware MKP) — with Pricing Integration
===========================================================================
Goal: Maximise the NUMBER of passengers assigned across a fixed route
      with multiple stops, respecting vehicle capacity AND passenger
      budget constraints on every segment.

Domain objects (Driver, Passenger, Problem) live exclusively in greedy.py.
This module imports them — a single source of truth prevents silent
divergence if the decode logic is ever updated.

Pricing integration
-------------------
  • Problem.decode() (defined in greedy.py) is shared by both algorithms.
  • A passenger is only accepted if the dynamically-quoted fare
    (which drops when occupancy is low) fits within their max_budget.
  • Placing a passenger earlier in the permutation means the car is
    emptier → bigger discount → more passengers can board.

Features
--------
  • Representation   : Permutation-based chromosome (Passenger objects).
  • Decoder          : Deterministic price-aware segment assignment
                       (shared with Greedy via Problem.decode).
  • Crossover        : Partially Mapped Crossover (PMX).
  • Mutation         : Adaptive weights — Swap, Relocate, Reverse.
  • Selection        : Tournament selection with Elitism (top-2 carry-over).
  • Convergence      : run() returns per-generation best passenger count
                       alongside the best Individual, enabling convergence
                       plots without re-running the algorithm.

Return contract for GeneticAlgorithm.run():
    (Individual, List[int])
      Individual.perm    : List[Passenger]
      Individual.fitness : int  (count × 10 000 + hop tiebreaker)
      List[int]          : best passenger count at each generation
                           (length == self.generations)
"""

import random
from typing import List, Tuple

# ── Single source of truth: domain objects live in greedy.py ─────────────────
from greedy import Driver, Passenger, Problem  # noqa: F401 (re-exported for smoke-test)


# ─────────────────────────────────────────────────────────────────────────────
# Chromosome wrapper
# ─────────────────────────────────────────────────────────────────────────────

class Individual:
    __slots__ = ("perm", "fitness")

    def __init__(self, perm: List[Passenger], fitness: int):
        self.perm = perm
        self.fitness = fitness


# ─────────────────────────────────────────────────────────────────────────────
# Genetic Algorithm
# ─────────────────────────────────────────────────────────────────────────────

class GeneticAlgorithm:
    """
    Parameters
    ----------
    problem        : Problem instance (from greedy.py)
    pop_size       : number of individuals in the population
    generations    : number of evolution cycles
    crossover_rate : probability of applying PMX crossover to a parent pair
    mutation_rate  : probability of mutating each offspring
    tournament_k   : number of contestants in each selection tournament
    elitism        : number of top individuals carried over unchanged
    """

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

    # ── Individual factory ────────────────────────────────────────────────────

    def _make_individual(self, perm: List[Passenger]) -> Individual:
        _, fitness, _, _, _ = self.problem.decode(perm)
        return Individual(perm, fitness)

    # ── Population initialisation ─────────────────────────────────────────────

    def _initial_population(self) -> List[Individual]:
        pop = []

        # Seed 1 — shortest trips first → maximises passenger count
        short_first = sorted(self.problem.passengers, key=lambda p: p.distance)
        pop.append(self._make_individual(short_first))

        # Seed 2 — longest trips first (classic greedy heuristic)
        long_first = sorted(
            self.problem.passengers, key=lambda p: p.distance, reverse=True
        )
        pop.append(self._make_individual(long_first))

        # Seed 3 — cheapest passengers first (budget-aware seed)
        #   Placing low-budget pax early means cars are emptier → bigger
        #   discount → they can actually afford a seat.
        budget_first = sorted(self.problem.passengers,
                              key=lambda p: p.max_budget)
        pop.append(self._make_individual(budget_first))

        # Fill remaining slots randomly
        for _ in range(self.pop_size - 3):
            perm = self.problem.passengers[:]
            random.shuffle(perm)
            pop.append(self._make_individual(perm))

        return pop

    # ── Crossover — Partially Mapped Crossover (PMX) ──────────────────────────

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
                    pos = i
                    while a <= pos <= b:
                        val = main[pos]
                        pos = next(
                            j for j, x in enumerate(other) if id(x) == id(val)
                        )
                    child[pos] = candidate
                    segment_set.add(id(candidate))

            other_iter = (x for x in other if id(x) not in segment_set)
            for i in range(size):
                if child[i] is None:
                    child[i] = next(other_iter)
            return child

        return _child(p1, p2), _child(p2, p1)

    # ── Mutation — Swap / Relocate / Reverse ──────────────────────────────────

    def _mutate(self, perm: List[Passenger]) -> List[Passenger]:
        n = len(perm)
        if n < 2:
            return perm
        a, b = random.sample(range(n), 2)
        r = random.random()

        if r < 0.33:
            # Swap two positions
            perm[a], perm[b] = perm[b], perm[a]
        elif r < 0.66:
            # Relocate: remove from a, insert at b
            item = perm.pop(a)
            perm.insert(b, item)
        else:
            # Reverse a sub-segment
            i, j = min(a, b), max(a, b)
            perm[i: j + 1] = reversed(perm[i: j + 1])

        return perm

    # ── Tournament selection ──────────────────────────────────────────────────

    def _tournament(self, population: List[Individual]) -> Individual:
        contestants = random.sample(
            population, min(self.tournament_k, len(population))
        )
        return max(contestants, key=lambda x: x.fitness)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, verbose: bool = False) -> Tuple[Individual, List[int]]:
        """
        Evolve the population and return the best solution found.

        Returns
        -------
        best_individual : Individual
            The chromosome with the highest fitness seen across all generations.
        convergence     : List[int]
            Best passenger count at each generation (index 0 = generation 0).
            Length equals self.generations.  Use this to plot learning curves.
        """
        population = self._initial_population()
        best_overall = max(population, key=lambda x: x.fitness)
        convergence: List[int] = []

        for gen in range(self.generations):
            population.sort(key=lambda x: x.fitness, reverse=True)

            # Elitism: carry top individuals unchanged
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

            best_pax_count = best_overall.fitness // 10_000
            convergence.append(best_pax_count)

            if verbose and gen % 20 == 0:
                print(f"  Gen {gen:03d} | Best fitness: {best_overall.fitness} "
                      f"({best_pax_count} passengers)")

        return best_overall, convergence


# ─────────────────────────────────────────────────────────────────────────────
# Standalone smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pricing import PricingEngine

    pricing = PricingEngine()
    drivers = [Driver("Car1", 3), Driver("Car2", 2)]
    num_stops = 10
    passengers = []
    for i in range(20):
        s = random.randint(0, num_stops - 2)
        e = random.randint(s + 1, num_stops - 1)
        dist = (e - s) * 25.0
        budget = random.uniform(8.0, 30.0)
        passengers.append(
            Passenger(f"P{i}", s, e, dist_km=dist, max_budget=budget)
        )

    prob = Problem(
        drivers, passengers, num_stops,
        pricing_engine=pricing,
        demand_score=1.2,
        is_weekend=0,
    )

    print("Starting Genetic Algorithm (goal = max passengers with pricing)…")
    ga = GeneticAlgorithm(prob, pop_size=100, generations=100)
    best_ind, convergence = ga.run(verbose=True)

    asgn, fitness, dist_km, revenue, price_map = prob.decode(best_ind.perm)
    assigned_count = sum(len(r) for r in asgn)
    print("\n" + "=" * 50)
    print(f"Passengers assigned : {assigned_count} / {len(passengers)}")
    print(f"Total distance (km) : {dist_km:.1f}")
    print(f"Total revenue ($)   : {revenue:.2f}")
    print(f"Fitness score       : {fitness}")
    for i, d in enumerate(drivers):
        route_detail = [
            f"{pid}(${price_map.get(pid, 0):.2f})" for pid in asgn[i]
        ]
        print(f"  {d.id} (cap {d.capacity}): {route_detail}")
    print(f"Convergence (last 5 gens): {convergence[-5:]}")
    print("=" * 50)
