"""
Bigtop - Command Line Interface for SRK.

This module provides a command-line interface for SRK (Symbolic Reasoning Kit)
that implements various analysis commands similar to the original OCaml bigtop tool.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
import sys
import argparse
import random
import re

from aria.utils.srk.core.syntax import (
    Context,
    Symbol,
    Type,
    ExpressionBuilder,
    mk_symbol,
    mk_const,
    mk_int,
    mk_real,
    mk_mul,
    symbols,
)
from aria.utils.srk.logic.smt import SMTInterface, SMTResult
from aria.utils.srk.logic.srkSimplify import Simplifier, make_simplifier
from aria.utils.srk.logic.abstract import SignDomain, AbstractValue
from aria.utils.srk.linalg.polyhedron import Polyhedron, Constraint
from aria.utils.srk.linalg.linear import QQVector
from aria.utils.srk.linalg.qQ import QQ
from fractions import Fraction


class BigtopCLI:
    """Main CLI class for Bigtop commands."""

    def __init__(self):
        """Initialize the CLI."""
        self.context = Context()
        self.builder = ExpressionBuilder(self.context)
        self.smt = SMTInterface(self.context)
        self.simplifier = make_simplifier(self.context)

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            prog="bigtop",
            description="Symbolic Reasoning Kit - Command Line Interface",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  bigtop -simsat "x > 0"
  bigtop -convex-hull "x >= 0" "y <= 1"
  bigtop -qe "∃x. x > 0 ∧ x < 1"
  bigtop -stats "x > 0 ∧ y < 1"
  bigtop -random 3 2
            """,
        )

        parser.add_argument(
            "-simsat", "--simsat", help="Check satisfiability using simplex"
        )

        parser.add_argument(
            "-nlsat", "--nlsat", help="Check satisfiability using non-linear solver"
        )

        parser.add_argument(
            "-convex-hull",
            "--convex-hull",
            nargs="+",
            help="Compute convex hull of constraints",
        )

        parser.add_argument(
            "-wedge-hull",
            "--wedge-hull",
            nargs="+",
            help="Compute wedge hull of constraints",
        )

        parser.add_argument(
            "-affine-hull",
            "--affine-hull",
            nargs="+",
            help="Compute affine hull of constraints",
        )

        parser.add_argument(
            "-qe", "--quantifier-elimination", help="Eliminate quantifiers from formula"
        )

        parser.add_argument(
            "-stats", "--statistics", help="Show statistics for formula"
        )

        parser.add_argument(
            "-random",
            "--random",
            nargs=2,
            type=int,
            metavar=("NUM_VARS", "DEPTH"),
            help="Generate random formula with NUM_VARS variables and given DEPTH",
        )

        parser.add_argument(
            "-v", "--verbose", action="store_true", help="Verbose output"
        )

        return parser

    def parse_simple_formula(self, formula_str: str) -> Optional[Any]:
        """Parse a simple formula string."""
        formula_str = formula_str.strip()

        # Handle basic boolean constants
        if formula_str == "true":
            return self.builder.mk_true()
        elif formula_str == "false":
            return self.builder.mk_false()

        # Handle conjunctions before atoms.
        for separator in ("&&", " and "):
            if separator in formula_str:
                parts = [part.strip() for part in formula_str.split(separator)]
                formulas = [self.parse_simple_formula(part) for part in parts]
                if any(formula is None for formula in formulas):
                    return None
                return self.builder.mk_and(formulas)

        # Handle simple comparisons like "x >= 0".  Multi-character operators
        # must be checked first so x >= 0 does not get split as x > "= 0".
        for token, op in ((">=", "geq"), ("<=", "leq"), ("=", "eq"), (">", "gt"), ("<", "lt")):
            if token in formula_str:
                parts = formula_str.split(token, 1)
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    return self._parse_comparison(left, right, op)

        print(f"Cannot parse formula: {formula_str}", file=sys.stderr)
        return None

    def _parse_comparison(self, left: str, right: str, op: str) -> Optional[Any]:
        """Parse a comparison expression."""
        try:
            left_expr = self._parse_term(left)
            right_expr = self._parse_term(right)

            if left_expr is None or right_expr is None:
                return None

            if op == "gt":
                return self.builder.mk_lt(right_expr, left_expr)  # x > 0 becomes 0 < x
            elif op == "lt":
                return self.builder.mk_lt(left_expr, right_expr)  # x < 0
            elif op == "geq":
                return self.builder.mk_leq(
                    right_expr, left_expr
                )  # x >= 0 becomes 0 <= x
            elif op == "leq":
                return self.builder.mk_leq(left_expr, right_expr)  # x <= 0
            elif op == "eq":
                return self.builder.mk_eq(left_expr, right_expr)

        except Exception as e:
            print(f"Error parsing comparison {left} {op} {right}: {e}", file=sys.stderr)
            return None

        return None

    def _parse_term(self, term: str) -> Optional[Any]:
        """Parse a term (variable or constant)."""
        term = term.strip()

        # Parenthesized term
        if term.startswith("(") and term.endswith(")"):
            return self._parse_term(term[1:-1])

        # Small infix multiplication support is enough for bigtop's nlsat smoke
        # path and prevents nonlinear terms from being silently rejected.
        mul_parts = [part.strip() for part in term.split("*")]
        if len(mul_parts) > 1:
            factors = [self._parse_term(part) for part in mul_parts]
            if any(factor is None for factor in factors):
                return None
            return mk_mul(self.context, factors)

        # Handle constants
        if re.fullmatch(r"-?\d+", term):
            return mk_int(self.context, int(term))
        elif re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", term):
            return mk_real(self.context, Fraction(term))

        # Handle variables.  The current lightweight bigtop parser only knows
        # single-letter variables; richer identifiers belong to the SMT-LIB path.
        elif re.fullmatch(r"[A-Za-z]", term):
            var_symbol = self.context._named_symbols.get(term)
            if var_symbol is None:
                var_symbol = self.context.mk_symbol(term, Type.INT)
            return self.builder.mk_var(var_symbol.id, var_symbol.typ)

        print(f"Cannot parse term: {term}", file=sys.stderr)
        return None

    def cmd_simsat(self, formula_str: str) -> None:
        """Check satisfiability using simplex."""
        print(f"Checking satisfiability (simplex): {formula_str}")

        formula = self.parse_simple_formula(formula_str)
        if formula is None:
            print("ERROR: Could not parse formula")
            return

        result = self.smt.is_sat(formula)

        if result == SMTResult.SAT:
            print("SATISFIABLE")
            model = self.smt.get_model(formula)
            if model:
                print("Model:")
                for symbol, value in model.interpretations.items():
                    print(f"  {symbol} = {value}")
        elif result == SMTResult.UNSAT:
            print("UNSATISFIABLE")
        else:
            print("UNKNOWN")

    def cmd_nlsat(self, formula_str: str) -> None:
        """Check satisfiability using non-linear solver."""
        print(f"Checking satisfiability (non-linear): {formula_str}")
        formula = self.parse_simple_formula(formula_str)
        if formula is None:
            print("ERROR: Could not parse formula")
            return

        result = self.smt.is_sat(formula)
        if result == SMTResult.SAT:
            print("SATISFIABLE")
        elif result == SMTResult.UNSAT:
            print("UNSATISFIABLE")
        else:
            print("UNKNOWN")

    def cmd_convex_hull(self, constraints: List[str]) -> None:
        """Compute convex hull of constraints."""
        print(f"Computing convex hull of {len(constraints)} constraints")

        constraint_objects = []
        for constraint_str in constraints:
            # Parse constraint like "x >= 0"
            if ">=" in constraint_str:
                parts = constraint_str.split(">=")
                if len(parts) == 2:
                    var = parts[0].strip()
                    const = parts[1].strip()
                    if var.isalpha() and const.isdigit():
                        # x >= 0 becomes constraint x >= 0
                        coeff = QQVector({0: QQ(1)})  # Simple case for single variable
                        constant = QQ(int(const))
                        constraint = Constraint(coeff, constant, False)
                        constraint_objects.append(constraint)

        if constraint_objects:
            try:
                polyhedron = Polyhedron(constraint_objects)
                print(f"Convex hull computed: {polyhedron}")
            except Exception as e:
                print(f"Error computing convex hull: {e}")
        else:
            print("No valid constraints to compute hull")

    def cmd_wedge_hull(self, constraints: List[str]) -> None:
        """Compute wedge hull of constraints."""
        print(f"Computing wedge hull of {len(constraints)} constraints")
        print(
            "ERROR: wedge hull requires the OCaml Wedge/APRON migration surface, "
            "which is not implemented in this Python port"
        )

    def cmd_affine_hull(self, constraints: List[str]) -> None:
        """Compute affine hull of constraints."""
        print(f"Computing affine hull of {len(constraints)} constraints")

        # Parse constraints into a formula
        formula_parts = []
        symbols = set()
        for constraint_str in constraints:
            # Simple parsing for x >= c, x <= c, etc.
            constraint_str = constraint_str.strip()
            if ">=" in constraint_str:
                parts = constraint_str.split(">=")
                if len(parts) == 2:
                    var = parts[0].strip()
                    const = parts[1].strip()
                    if var and const.replace('.', '').replace('-', '').isdigit():
                        symbols.add(var)
                        # x >= c becomes x - c >= 0
                        formula_parts.append(f"({var} - {const}) >= 0")
            elif "<=" in constraint_str:
                parts = constraint_str.split("<=")
                if len(parts) == 2:
                    var = parts[0].strip()
                    const = parts[1].strip()
                    if var and const.replace('.', '').replace('-', '').isdigit():
                        symbols.add(var)
                        formula_parts.append(f"({var} - {const}) <= 0")

        if not formula_parts:
            print("No valid constraints parsed")
            return

        formula_str = " and ".join(formula_parts)
        formula = self.parse_simple_formula(formula_str)
        if formula is None:
            print("ERROR: Could not parse constraints into formula")
            return

        symbol_list = [self.srk.mk_symbol(name, Type.REAL) for name in symbols]

        try:
            from ..logic.abstract import affine_hull
            hull = affine_hull(self.srk, formula, symbol_list)
            print(f"Affine hull: {hull}")
        except Exception as e:
            print(f"Error computing affine hull: {e}")

    def cmd_quantifier_elimination(self, formula_str: str) -> None:
        """Eliminate quantifiers from formula."""
        print(f"Quantifier elimination: {formula_str}")

        formula = self.parse_simple_formula(formula_str)
        if formula is None:
            print("ERROR: Could not parse formula")
            return

        if symbols(formula):
            print(
                "ERROR: quantifier elimination for parsed free-variable formulas "
                "requires the full Quantifier.qe_mbp migration surface"
            )
            return

        print(formula)

    def cmd_statistics(self, formula_str: str) -> None:
        """Show statistics for formula."""
        print(f"Formula statistics: {formula_str}")

        formula = self.parse_simple_formula(formula_str)
        if formula is None:
            print("ERROR: Could not parse formula")
            return

        # Basic statistics
        print(f"Formula type: {type(formula).__name__}")
        print("Basic analysis: formula parsed successfully")

    def cmd_random(self, num_vars: int, depth: int) -> None:
        """Generate random formula."""
        print(f"Generating random formula with {num_vars} variables, depth {depth}")

        # Simple random formula generation
        variables = [chr(ord("a") + i) for i in range(num_vars)]

        # Generate a simple random expression
        if num_vars > 0:
            var = random.choice(variables)
            const_val = random.randint(0, 10)
            op = random.choice([">", "<", ">=", "<="])

            formula = f"{var} {op} {const_val}"
            print(f"Generated: {formula}")
        else:
            print("Generated: true")


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for bigtop CLI."""
    cli = BigtopCLI()
    parser = cli.create_parser()

    if argv is None:
        argv = sys.argv[1:]

    # Handle case where no arguments are provided
    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    # Handle the different commands
    if args.simsat:
        cli.cmd_simsat(args.simsat)
    elif args.nlsat:
        cli.cmd_nlsat(args.nlsat)
    elif args.convex_hull:
        cli.cmd_convex_hull(args.convex_hull)
    elif args.wedge_hull:
        cli.cmd_wedge_hull(args.wedge_hull)
    elif args.affine_hull:
        cli.cmd_affine_hull(args.affine_hull)
    elif args.quantifier_elimination:
        cli.cmd_quantifier_elimination(args.quantifier_elimination)
    elif args.statistics:
        cli.cmd_statistics(args.statistics)
    elif args.random:
        num_vars, depth = args.random
        cli.cmd_random(num_vars, depth)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
