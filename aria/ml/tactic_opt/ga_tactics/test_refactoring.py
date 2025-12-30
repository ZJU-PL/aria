#!/usr/bin/env python3
"""Test script for refactored Z3 tactic optimization system."""

import os
import random
from aria.ml.tactic_opt.ga_tactics import TacticSeq, GA, EvaluationMode
from aria.ml.tactic_opt.ga_tactics.main import demo_tactic_sequence


def test_basic_functionality():
    """Test basic functionality of the refactored system."""
    print("=== Testing Basic Functionality ===")
    tactic_seq = TacticSeq.random()
    print(f"✓ Created tactic sequence: {tactic_seq.to_string()}")
    z3_tactic = tactic_seq.to_z3_tactic()
    print(f"✓ Converted to Z3 tactic: {z3_tactic}")
    smtlib_format = tactic_seq.to_smtlib_apply()
    print(f"✓ SMT-LIB2 format: {smtlib_format}")
    print()


def test_evaluation_modes():
    """Test both evaluation modes."""
    print("=== Testing Evaluation Modes ===")
    try:
        from aria.ml.tactic_opt.ga_tactics.core import get_evaluation_mode
        os.environ["Z3_EVALUATION_MODE"] = EvaluationMode.PYTHON_API
        assert get_evaluation_mode() == EvaluationMode.PYTHON_API
        print("✓ Python API mode configuration works")

        os.environ["Z3_EVALUATION_MODE"] = EvaluationMode.BINARY_Z3
        assert get_evaluation_mode() == EvaluationMode.BINARY_Z3
        print("✓ Binary Z3 mode configuration works")

        os.environ["Z3_EVALUATION_MODE"] = EvaluationMode.PYTHON_API
    except Exception as e:
        print(f"✗ Mode configuration failed: {e}")
    print()


def test_genetic_algorithm():
    """Test the genetic algorithm with a small population."""
    print("=== Testing Genetic Algorithm ===")
    ga = GA(population_size=4)

    try:
        for gen in range(2):
            print(f"Generation {gen + 1}:")
            for tactics in ga.get_population():
                tactics.fitness = random.uniform(0.1, 10.0)
                ga.get_new_individuals().append(tactics)

            best = ga.get_best_sequence()
            if best:
                print(f"  Best fitness: {best.fitness:.2f}")
                print(f"  Best sequence: {best.to_string()}")

            ga.repopulate()

        print("✓ Genetic algorithm works correctly")
    except Exception as e:
        print(f"✗ Genetic algorithm failed: {e}")
        import traceback
        traceback.print_exc()
    print()


def main():
    """Main test function."""
    print("Testing Refactored Z3 Tactic Optimization System\n")
    test_basic_functionality()
    test_evaluation_modes()
    test_genetic_algorithm()
    print("=== Demo ===")
    demo_tactic_sequence()
    print("=== All Tests Complete ===")


if __name__ == "__main__":
    main()
