#!/usr/bin/env python3
"""Main entry point for Z3 tactic optimization."""

import argparse
import os
import sys
from .core import GA, TacticSeq, EvaluationMode, pretty_print_tactic


def demo_tactic_sequence():
    """Demonstrate tactic sequence creation and visualization."""
    print("=== Z3 Tactic Optimization Demo ===\n")
    tactic_seq = TacticSeq.random()
    print("Randomly generated tactic sequence:")
    print(tactic_seq.to_string())
    print("\nConverting to Z3 tactic object...")
    z3_tactic = tactic_seq.to_z3_tactic()
    print("✓ Conversion successful\n")
    print("Demonstrating tactic on test formula...")
    pretty_print_tactic(z3_tactic)
    print("\nSMT-LIB2 format for binary Z3:")
    print(tactic_seq.to_smtlib_apply())
    print()


def run_genetic_algorithm(args):
    """Run the genetic algorithm with specified parameters."""
    print("=== Z3 Tactic Optimization - Genetic Algorithm ===\n")

    mode_name = "binary Z3" if args.mode == EvaluationMode.BINARY_Z3 else "Python API"
    os.environ["Z3_EVALUATION_MODE"] = args.mode
    print(f"Using {mode_name} evaluation mode")
    print(f"Population: {args.population}, Generations: {args.generations}, Timeout: {args.timeout}s\n")

    ga = GA(population_size=args.population)

    try:
        results = ga.run_evolution(
            generations=args.generations, mode=args.mode, smtlib_file=args.smtlib_file,
            timeout=args.timeout, output_dir=args.output_dir, save_interval=args.save_interval
        )

        print("\n=== Evolution Complete ===")
        print(f"Generations: {results['generations_run']}, Best fitness: {results['final_stats']['best_fitness']}")

        if results['best_sequence']:
            print(f"\nBest tactic sequence:\n{results['best_sequence'].to_string()}")
            print(f"\nSMT-LIB2 format:\n{results['best_sequence'].to_smtlib_apply()}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving current best results...")
        if ga.get_best_sequence():
            os.makedirs(args.output_dir, exist_ok=True)
            ga.save_retained_elite(args.output_dir, "z3_interrupted")
            print(f"Results saved to {args.output_dir}/")

    except Exception as e:
        print(f"\nError during evolution: {e}")
        return 1

    return 0


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Z3 Tactic Optimization using Genetic Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python main.py --mode binary\n  python main.py --generations 50\n  python main.py --demo"
    )

    parser.add_argument('--mode', choices=[EvaluationMode.PYTHON_API, EvaluationMode.BINARY_Z3],
                       default=EvaluationMode.PYTHON_API, help='Evaluation mode (default: python_api)')
    parser.add_argument('--population', '-p', type=int, default=64, help='Population size (default: 64)')
    parser.add_argument('--generations', '-g', type=int, default=128, help='Generations (default: 128)')
    parser.add_argument('--timeout', '-t', type=int, default=8, help='Timeout in seconds (default: 8)')
    parser.add_argument('--smtlib-file', help='SMT-LIB2 file for evaluation')
    parser.add_argument('--output-dir', '-o', default='.', help='Output directory (default: .)')
    parser.add_argument('--save-interval', type=int, default=10, help='Save elite every N generations (default: 10)')
    parser.add_argument('--demo', action='store_true', help='Show demo instead of running GA')

    args = parser.parse_args()

    if args.demo:
        demo_tactic_sequence()
        return 0

    return run_genetic_algorithm(args)


if __name__ == "__main__":
    sys.exit(main())
