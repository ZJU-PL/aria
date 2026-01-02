#!/usr/bin/env python3
"""Command line tool for CNF formula simplification
TODO: to be tested
"""

import argparse
import sys
from typing import List

from aria.bool.cnfsimplifier.cnf import Cnf
from aria.bool.cnfsimplifier.clause import Clause


def parse_dimacs(input_file) -> Cnf:
    """Parse DIMACS format CNF file"""
    clauses = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("c") or line.startswith("p"):
                continue
            if not line:
                continue
            # Parse clause
            lits = [int(x) for x in line.split()[:-1]]  # Skip trailing 0
            if lits:
                clause = Clause([])
                for lit in lits:
                    clause.add_literal(lit)
                clauses.append(clause)
    return Cnf(clauses)


def write_dimacs(cnf: Cnf, output_file=None):
    """Write CNF to DIMACS format"""
    if output_file is None:
        out = sys.stdout
        should_close = False
    else:
        out = open(output_file, "w", encoding="utf-8")
        should_close = True

    # Write header
    num_vars = cnf.get_number_of_literals()
    num_clauses = cnf.get_number_of_clauses()
    print(f"p cnf {num_vars} {num_clauses}", file=out)

    # Write clauses
    for clause in cnf.get_clauses():
        lits = [str(lit) for lit in clause.literals_set]
        print(" ".join(lits + ["0"]), file=out)

    if should_close:
        out.close()


def phase_ordering() -> List[int]:
    """Phase ordering for CNF formula simplification"""
    # TODO: implement phase ordering
    return []


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="CNF Formula Simplification Tool")
    parser.add_argument("input", help="Input CNF file in DIMACS format")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument(
        "--tautology", action="store_true", help="Remove tautological clauses"
    )
    parser.add_argument(
        "--hidden-tautology",
        action="store_true",
        help="Remove hidden tautological clauses",
    )
    parser.add_argument(
        "--asymmetric-tautology",
        action="store_true",
        help="Remove asymmetric tautological clauses",
    )
    parser.add_argument(
        "--subsumption", action="store_true", help="Remove subsumed clauses"
    )
    parser.add_argument(
        "--hidden-subsumption",
        action="store_true",
        help="Remove hidden subsumed clauses",
    )
    parser.add_argument(
        "--asymmetric-subsumption",
        action="store_true",
        help="Remove asymmetric subsumed clauses",
    )
    parser.add_argument("--blocked", action="store_true", help="Remove blocked clauses")
    parser.add_argument(
        "--hidden-blocked", action="store_true", help="Remove hidden blocked clauses"
    )
    parser.add_argument(
        "--asymmetric-blocked",
        action="store_true",
        help="Remove asymmetric blocked clauses",
    )
    parser.add_argument(
        "--all", action="store_true", help="Apply all simplification techniques"
    )

    args = parser.parse_args()

    # Read input CNF
    cnf = parse_dimacs(args.input)

    # Apply simplifications
    if args.all or args.tautology:
        cnf = cnf.tautology_elimination()
    if args.all or args.hidden_tautology:
        cnf = cnf.hidden_tautology_elimination()
    if args.all or args.asymmetric_tautology:
        cnf = cnf.asymmetric_tautology_elimination()
    if args.all or args.subsumption:
        cnf = cnf.subsumption_elimination()
    if args.all or args.hidden_subsumption:
        cnf = cnf.hidden_subsumption_elimination()
    if args.all or args.asymmetric_subsumption:
        cnf = cnf.asymmetric_subsumption_elimination()
    if args.all or args.blocked:
        cnf = cnf.blocked_clause_elimination()
    if args.all or args.hidden_blocked:
        cnf = cnf.hidden_blocked_clause_elimination()
    if args.all or args.asymmetric_blocked:
        cnf = cnf.asymmetric_blocked_clause_elimination()

    # Write output
    write_dimacs(cnf, args.output)


if __name__ == "__main__":
    main()
