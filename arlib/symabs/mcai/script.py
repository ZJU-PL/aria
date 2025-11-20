import z3
import sys
import os
import argparse
from arlib.tests.formula_generator import FormulaGenerator
from arlib.utils.z3_expr_utils import get_variables


def gen_examples(tot: int, output_dir: str):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i in range(tot):
        x, y, z = z3.BitVecs("x y z", 8)
        variables = [x, y, z]
        formula = FormulaGenerator(variables).generate_formula()
        sol = z3.Solver()
        sol.add(formula)
        var = get_variables(formula)
        if sol.check() == z3.sat and len(var) == len(variables):
            with open(f"{output_dir}/formula_{i}.smt2", "w") as f:
                f.write(sol.sexpr())

def run_examples(output_dir: str):
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".smt2"):
                path = os.path.abspath(os.path.join(root, file))
                cmd = f"python3 bv_mcai.py -f={path} -l={path.replace('.smt2', '.log')} -c={output_dir}/results.csv"
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
    except Exception as e:
        print(e)
        sys.exit(1)
