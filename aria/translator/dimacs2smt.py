"""
Converting DIMACS (CNF format) to SMT2
"""

import sys
import argparse
from typing import List, Optional, TextIO

from aria.translator.parsing import parse_dimacs_file


def parse_header(line: str) -> int:
    """
    Parse the DIMACS header line to extract the number of variables.

    Args:
        line: The header line starting with 'p cnf'

    Returns:
        The number of variables declared in the problem
    """
    try:
        parts = line.split()
        if len(parts) < 4 or parts[0] != "p" or parts[1] != "cnf":
            raise ValueError(f"Invalid DIMACS header format: {line}")
        return int(parts[2])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse DIMACS header: {line}") from e


def declare_variables(num_vars: int, output: TextIO, prefix: str = "v_") -> None:
    """
    Write variable declarations to the SMT2 output.

    Args:
        num_vars: Number of Boolean variables to declare
        output: Output file handle
        prefix: Prefix to use for variable names
    """
    for i in range(1, num_vars + 1):
        output.write(f"(declare-const {prefix}{i} Bool)\n")


def write_clause(literals: List[int], output: TextIO, prefix: str = "v_") -> None:
    if not literals:
        output.write("(assert false)\n")
        return

    output.write("(assert (or ")
    for lit in literals:
        if lit < 0:
            output.write(f"(not {prefix}{abs(lit)}) ")
        else:
            output.write(f"{prefix}{lit} ")
    output.write("))\n")


def convert_dimacs_to_smt2(
    input_path: str,
    output_path: Optional[str] = None,
    logic: str = "QF_UF",
    var_prefix: str = "v_",
) -> str:
    """
    Convert a DIMACS CNF file to SMT2 format.

    Args:
        input_path: Path to input DIMACS file
        output_path: Path to output SMT2 file (default: input_path + ".smt2")
        logic: SMT2 logic to use (default: QF_UF)
        var_prefix: Prefix for variable names

    Returns:
        Path to the created SMT2 file
    """
    if output_path is None:
        output_path = f"{input_path}.smt2"

    try:
        num_vars, _, clauses = parse_dimacs_file(input_path)

        # Write SMT2 file
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(f"(set-logic {logic})\n")
            declare_variables(num_vars, output_file, var_prefix)

            for clause in clauses:
                write_clause(clause, output_file, var_prefix)

            output_file.write("(check-sat)\n")
            output_file.write("(get-model)\n")

        return output_path

    except (ValueError, IOError, OSError) as e:
        print(f"Error converting {input_path}: {str(e)}", file=sys.stderr)
        raise


def main():
    """
    Main function for command-line execution.
    """
    parser = argparse.ArgumentParser(
        description="Convert DIMACS CNF files to SMT2 format"
    )
    parser.add_argument("input", help="Input DIMACS file path")
    parser.add_argument(
        "-o", "--output", help="Output SMT2 file path (default: input_path.smt2)"
    )
    parser.add_argument(
        "-l",
        "--logic",
        choices=["QF_UF", "QF_BV"],
        default="QF_UF",
        help="SMT2 logic to use (default: QF_UF)",
    )
    parser.add_argument(
        "-p", "--prefix", default="v_", help="Variable name prefix (default: v_)"
    )

    args = parser.parse_args()

    try:
        output_path = convert_dimacs_to_smt2(
            args.input, args.output, args.logic, args.prefix
        )
        print(f"Successfully converted {args.input} to {output_path}")
    except (ValueError, IOError, OSError) as e:
        print(f"Conversion failed: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
