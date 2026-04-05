import sys
from typing import cast

import z3

from aria.smt.bv.mapped_blast import translate_smt2formula_to_cnf


def convert_smt2_to_dimacs(input_path, output_path):
    formula = cast(z3.ExprRef, z3.And(z3.parse_smt2_file(input_path)))
    _, _, header, clauses = translate_smt2formula_to_cnf(formula)

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(header) + "\n")
        for clause in clauses:
            output_file.write(f"{clause} 0\n")

    return output_path


def main(argv):
    if len(argv) < 3:
        print(f"usage {argv[0]} INPUT_SMT2 OUTPUT_CNF")
        return 1
    convert_smt2_to_dimacs(argv[1], argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
