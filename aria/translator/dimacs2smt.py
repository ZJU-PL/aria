"""
Converting DIMACS (CNF format) to SMT2
"""
import sys
import argparse
import os
from typing import TextIO, List, Set


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
        if len(parts) < 4 or parts[0] != 'p' or parts[1] != 'cnf':
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


def parse_clause(line: str, output: TextIO, prefix: str = "v_") -> None:
    """
    Parse a single clause and write the corresponding
    compound logic expression to the output.
    
    Args:
        line: DIMACS clause line (space-separated literals ending with 0)
        output: Output file handle
        prefix: Prefix to use for variable names
    """
    literals = line.split()
    if not literals or literals[-1] != '0':
        raise ValueError(f"Invalid clause format (must end with 0): {line}")

    # Skip empty clauses
    if len(literals) == 1 and literals[0] == '0':
        return

    output.write("(assert (or ")
    for lit in literals:
        if lit == '0':
            break
        try:
            var_num = int(lit)
            if var_num == 0:
                continue
            if var_num < 0:
                output.write(f"(not {prefix}{abs(var_num)}) ")
            else:
                output.write(f"{prefix}{var_num} ")
        except ValueError:
            # Skip non-integer tokens (could be comments)
            continue
    output.write("))\n")


def convert_dimacs_to_smt2(
        input_path: str,
        output_path: str = None,
        logic: str = "QF_UF",
        var_prefix: str = "v_"
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
        with open(input_path, 'r') as input_file:
            # Skip comments and find header
            header_line = None
            for line in input_file:
                line = line.strip()
                if not line or line.startswith('c'):
                    continue
                if line.startswith('p'):
                    header_line = line
                    break

            if header_line is None:
                raise ValueError(f"No problem header found in {input_path}")

            num_vars = parse_header(header_line)

            # Read all clauses (skipping comments)
            clauses = []
            for line in input_file:
                line = line.strip()
                if not line or line.startswith('c'):
                    continue
                clauses.append(line)

        # Write SMT2 file
        with open(output_path, 'w') as output_file:
            output_file.write(f"(set-logic {logic})\n")
            declare_variables(num_vars, output_file, var_prefix)

            for clause in clauses:
                parse_clause(clause, output_file, var_prefix)

            output_file.write("(check-sat)\n")
            output_file.write("(get-model)\n")

        return output_path

    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}", file=sys.stderr)
        raise


def main():
    """
    Main function for command-line execution.
    """
    parser = argparse.ArgumentParser(description='Convert DIMACS CNF files to SMT2 format')
    parser.add_argument('input', help='Input DIMACS file path')
    parser.add_argument('-o', '--output', help='Output SMT2 file path (default: input_path.smt2)')
    parser.add_argument('-l', '--logic', choices=['QF_UF', 'QF_BV'], default='QF_UF',
                        help='SMT2 logic to use (default: QF_UF)')
    parser.add_argument('-p', '--prefix', default='v_',
                        help='Variable name prefix (default: v_)')

    args = parser.parse_args()

    try:
        output_path = convert_dimacs_to_smt2(
            args.input,
            args.output,
            args.logic,
            args.prefix
        )
        print(f"Successfully converted {args.input} to {output_path}")
    except Exception as e:
        print(f"Conversion failed: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
