"""Script for generating and running examples for the MCAI project."""
import argparse
import os
import sys

import z3

from aria.tests.formula_generator import FormulaGenerator
from aria.utils.z3_expr_utils import get_variables


def gen_examples(tot: int, output_directory: str) -> None:
    """Generate example formulas and save them as SMT2 files."""
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    for i in range(tot):
        x, y, z = z3.BitVecs("x y z", 8)
        variables = [x, y, z]
        formula = FormulaGenerator(variables).generate_formula()
        sol = z3.Solver()
        sol.add(formula)
        var = get_variables(formula)
        if sol.check() == z3.sat and len(var) == len(variables):
            filepath = f"{output_directory}/formula_{i}.smt2"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(sol.sexpr())


def run_examples(output_directory: str) -> None:
    """Run examples from the output directory."""
    for root, _, files in os.walk(output_directory):
        for file in files:
            if file.endswith(".smt2"):
                path = os.path.abspath(os.path.join(root, file))
                log_path = path.replace(".smt2", ".log")
                cmd = (f"python3 bv_mcai.py -f={path} "
                       f"-l={log_path} -c={output_directory}/results.csv")
                print(cmd)
                os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate examples for the MCAI project.")
    parser.add_argument(
        "-n", "--num",
        type=int,
        help="Number of examples to generate.",
        default=100
    )
    parser.add_argument(
        "-d", "--dir",
        type=str,
        help="Directory to save/load smt2 files.",
        default="smt2"
    )
    parser.add_argument(
        "-r", "--run",
        action="store_true",
        help="Run the examples after generation."
    )
    parser.add_argument(
        "-g", "--gen",
        action="store_true",
        help="Generate random examples."
    )
    parser.add_argument(
        "--c-dir",
        type=str,
        help="Directory to load C programs.",
        default="c"
    )

    args = parser.parse_args()
    cnt = args.num
    output_dir = args.dir
    try:
        if args.gen:
            gen_examples(cnt, output_dir)
        if args.run:
            run_examples(output_dir)
        if not args.gen and not args.run:
            print("No action specified. Please specify either -g or -r.")
    except Exception as exc:
        print(exc)
        sys.exit(1)
