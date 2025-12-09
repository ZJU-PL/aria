#!/usr/bin/env python3
"""Genetic Algorithm for Z3 tactic optimization."""

import os
import random
import json
from .models import TacticSeq, CustomJsonEncoder
from .evaluator import run_tests


class GA:
    """Genetic Algorithm for evolving optimal Z3 tactic sequences."""
    _population_size = 64
    _retain_percentage = 0.25
    _retain_size = int(_retain_percentage * _population_size) + 1
    _rand = random.Random()
    _rand.seed()

    def __init__(self, population_size=None, retain_percentage=None):
        if population_size is not None:
            self._population_size = population_size
            self._retain_size = int(retain_percentage * population_size) + 1 if retain_percentage else self._retain_size
        if retain_percentage is not None:
            self._retain_percentage = retain_percentage
            self._retain_size = int(retain_percentage * self._population_size) + 1

        self._population = [TacticSeq.random().mutate() for _ in range(self._population_size)]
        self._new = []
        self._retained = []

    def evaluate(self, mode=None, smtlib_file=None, timeout=8):
        """Evaluate fitness of all individuals in the current population."""
        for tactics in self._population:
            tactics.fitness = run_tests(tactics, mode, smtlib_file, timeout)
            print(f"Fitness: {tactics.fitness}")
            self._new.append(tactics)

    def _save_sequences(self, sequences, output_dir, prefix, suffix="tactics"):
        """Helper to save sequences to files."""
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, f"{prefix}.{suffix}")
        with open(results_file, "w") as results:
            for i, tactics in enumerate(sequences):
                tactic_file = os.path.join(output_dir, f"{prefix}.{suffix}.{i}")
                with open(tactic_file, "w") as f:
                    json.dump(tactics, f, cls=CustomJsonEncoder, indent=2)
                results.write(f"{i}: {tactics.fitness}\n")

    def dump_results(self, output_dir=".", prefix="z3"):
        """Save the best tactic sequences from the current generation."""
        sorted_new = sorted(self._new, key=lambda e: e.fitness)
        self._save_sequences(sorted_new, output_dir, prefix, "results")

    def save_retained_elite(self, output_dir=".", prefix="z3"):
        """Save the elite individuals from previous generation."""
        self._save_sequences(self._retained, output_dir, prefix, "elite")

    def repopulate(self):
        """Create next generation through selection, crossover, and mutation."""
        self._new.extend(self._retained)
        self._new = sorted(self._new, key=lambda e: e.fitness)
        self._population, self._retained = [], self._new[:self._retain_size]

        PENALTY = 4294967295
        while len(self._population) < self._population_size:
            i1, i2 = self._rand.randint(0, len(self._new) - 1), self._rand.randint(0, len(self._new) - 1)
            if i1 != i2 and self._new[i1].fitness != PENALTY and self._new[i2].fitness != PENALTY:
                self._population.append(TacticSeq.crossover(self._new[i1], self._new[i2]).mutate())
        self._new = []

    def get_best_sequence(self):
        """Get the best tactic sequence from the current population."""
        all_individuals = self._new + self._retained
        return min(all_individuals, key=lambda x: x.fitness) if all_individuals else None

    def get_population_stats(self):
        """Get statistics about the current population."""
        all_individuals = self._new + self._retained
        if not all_individuals:
            return {}
        fitness_values = [ind.fitness for ind in all_individuals]
        return {
            'population_size': len(all_individuals),
            'best_fitness': min(fitness_values),
            'worst_fitness': max(fitness_values),
            'avg_fitness': sum(fitness_values) / len(fitness_values),
            'elite_count': len(self._retained)
        }

    def run_generation(self, mode=None, smtlib_file=None, timeout=8, output_dir="."):
        """Run one complete generation of the genetic algorithm."""
        self.evaluate(mode, smtlib_file, timeout)
        self.dump_results(output_dir)
        stats_before = self.get_population_stats()
        self.repopulate()
        return {
            'generation': getattr(self, '_generation', 0) + 1,
            'before_repopulation': stats_before,
            'after_repopulation': self.get_population_stats(),
            'best_sequence': self.get_best_sequence()
        }

    def run_evolution(self, generations=128, mode=None, smtlib_file=None,
                     timeout=8, output_dir=".", save_interval=10):
        """Run the complete genetic algorithm evolution process."""
        self._generation = 0
        for gen in range(generations):
            self._generation = gen + 1
            gen_stats = self.run_generation(mode, smtlib_file, timeout, output_dir)
            print(f"Generation {gen + 1}: Best fitness = {gen_stats['before_repopulation']['best_fitness']}")
            if (gen + 1) % save_interval == 0:
                self.save_retained_elite(output_dir, f"z3_gen{gen + 1}")

        self.save_retained_elite(output_dir, "z3_final")
        return {
            'generations_run': generations,
            'final_stats': self.get_population_stats(),
            'best_sequence': self.get_best_sequence()
        }
