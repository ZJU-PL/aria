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

from aria.srk.syntax import Context, Symbol, Type, ExpressionBuilder, mk_symbol, mk_const
from aria.srk.smt import SMTInterface, SMTResult
from aria.srk.srkSimplify import Simplifier, make_simplifier
from aria.srk.abstract import SignDomain, AbstractValue
from aria.srk.polyhedron import Polyhedron, Constraint
from aria.srk.linear import QQVector
from aria.srk.qQ import QQ
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
            """
        )

        parser.add_argument(
            "-simsat", "--simsat",
            help="Check satisfiability using simplex"
        )

        parser.add_argument(
            "-nlsat", "--nlsat",
            help="Check satisfiability using non-linear solver"
        )

        parser.add_argument(
            "-convex-hull", "--convex-hull",
            nargs="+",
            help="Compute convex hull of constraints"
        )

        parser.add_argument(
            "-wedge-hull", "--wedge-hull",
            nargs="+",
            help="Compute wedge hull of constraints"
        )

        parser.add_argument(
            "-affine-hull", "--affine-hull",
            nargs="+",
            help="Compute affine hull of constraints"
        )

        parser.add_argument(
            "-qe", "--quantifier-elimination",
            help="Eliminate quantifiers from formula"
        )

        parser.add_argument(
            "-stats", "--statistics",
            help="Show statistics for formula"
        )

        parser.add_argument(
            "-random", "--random",
            nargs=2, type=int,
            metavar=("NUM_VARS", "DEPTH"),
            help="Generate random formula with NUM_VARS variables and given DEPTH"
        )

        parser.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="Verbose output"
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

        # Handle simple comparisons like "x > 0"
        if ">" in formula_str and "<" not in formula_str:
            parts = formula_str.split(">")
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                return self._parse_comparison(left, right, "gt")
        elif "<" in formula_str and ">" not in formula_str:
            parts = formula_str.split("<")
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                return self._parse_comparison(left, right, "lt")
        elif ">=" in formula_str:
            parts = formula_str.split(">=")
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                return self._parse_comparison(left, right, "geq")
        elif "<=" in formula_str:
            parts = formula_str.split("<=")
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                return self._parse_comparison(left, right, "leq")

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
                return self.builder.mk_leq(right_expr, left_expr)  # x >= 0 becomes 0 <= x
            elif op == "leq":
                return self.builder.mk_leq(left_expr, right_expr)  # x <= 0

        except Exception as e:
            print(f"Error parsing comparison {left} {op} {right}: {e}", file=sys.stderr)
            return None

        return None

    def _parse_term(self, term: str) -> Optional[Any]:
        """Parse a term (variable or constant)."""
        term = term.strip()

        # Handle constants
        if term.isdigit() or (term.startswith('-') and term[1:].isdigit()):
            # Create a constant symbol for the integer
            const_symbol = self.context.mk_symbol(f"const_{term}", Type.INT)
            return self.builder.mk_const(const_symbol)
        elif term.replace('.', '').replace('-', '').isdigit():
            # Handle floats
            const_symbol = self.context.mk_symbol(f"const_{term}", Type.REAL)
            return self.builder.mk_const(const_symbol)

        # Handle variables (assume single letters for now)
        elif term.isalpha() and len(term) == 1:
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
        # For now, delegate to simplex
        self.cmd_simsat(formula_str)

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
        # For now, delegate to convex hull
        self.cmd_convex_hull(constraints)

    def cmd_affine_hull(self, constraints: List[str]) -> None:
        """Compute affine hull of constraints."""
        print(f"Computing affine hull of {len(constraints)} constraints")
        # Simplified implementation
        print("Affine hull computation not fully implemented")

    def cmd_quantifier_elimination(self, formula_str: str) -> None:
        """Eliminate quantifiers from formula."""
        print(f"Quantifier elimination: {formula_str}")

        # Check if formula starts with quantifier
        if formula_str.startswith("∃") or formula_str.startswith("forall"):
            print("Quantifier found - elimination not fully implemented in this version")
        else:
            print("No quantifiers detected")

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
        variables = [chr(ord('a') + i) for i in range(num_vars)]

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
