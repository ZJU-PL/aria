"""SMT-based verification for program synthesis.

This module provides SMT-based verification and counterexample generation
for expressions in the Version Space Algebra.
"""

import z3
from typing import Dict, Any, List, Optional, Set, Tuple
from .expressions import Expression, Theory, Variable
from .expression_to_smt import SMTConverter, expression_to_smt


class SMTVerifier:
    """SMT-based verifier for expressions."""

    def __init__(self):
        self.converter = SMTConverter()
        self.solver = z3.Solver(ctx=self.converter.context)

    def verify_expression(self, expr: Expression, examples: List[Dict[str, Any]],
                         var_types: Dict[str, str] = None) -> bool:
        """Verify that an expression is consistent with all examples using SMT."""
        try:
            # Create SMT formula for the expression
            smt_expr = expression_to_smt(expr, var_types)

            # For each example, check if it satisfies the expression
            for example in examples:
                # Create constraints for input variables
                constraints = []
                output_var = None

                for var_name, value in example.items():
                    if var_name == 'output':
                        output_var = value
                        continue

                    var_smt = self.converter.variable_map.get(var_name)
                    if var_smt is None:
                        # Variable not yet in context, create it
                        var_smt = self._create_variable(var_name, expr.theory, var_types)
                        self.converter.variable_map[var_name] = var_smt

                    constraints.append(var_smt == value)

                # Add constraints to solver
                self.solver.push()
                self.solver.add(constraints)

                # Check if the expression equals the expected output
                if output_var is not None:
                    self.solver.add(smt_expr != output_var)
                    result = self.solver.check()

                    # If UNSAT, then the expression DOES equal the output (constraint is false)
                    # If SAT, then the expression does NOT equal the output (constraint is satisfiable)
                    if result == z3.sat:
                        self.solver.pop()
                        return False

                self.solver.pop()

            return True

        except Exception as e:
            print(f"SMT verification failed: {e}")
            return False

    def find_counterexample(self, expressions: List[Expression],
                           examples: List[Dict[str, Any]],
                           var_types: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
        """Find a counterexample that distinguishes between expressions using SMT."""
        try:
            if not expressions:
                return None

            theory = expressions[0].theory

            # Get all variables used in expressions
            all_variables = set()
            for expr in expressions:
                all_variables.update(expr.get_variables())

            # Create SMT expressions for all candidates
            smt_expressions = []
            for expr in expressions:
                smt_expr = expression_to_smt(expr, var_types)
                smt_expressions.append(smt_expr)

            # Create variables for counterexample search
            counterexample_vars = {}
            for var_name in all_variables:
                var_smt = self._create_variable(var_name, theory, var_types)
                counterexample_vars[var_name] = var_smt

            # Try to find an assignment where expressions produce different outputs
            for attempt in range(10):  # Limit attempts
                self.solver.push()

                # Add constraints that expressions must be different
                if len(smt_expressions) >= 2:
                    # At least two expressions must differ
                    diff_constraints = []
                    for i in range(len(smt_expressions)):
                        for j in range(i+1, len(smt_expressions)):
                            diff_constraints.append(smt_expressions[i] != smt_expressions[j])

                    # Require at least one difference
                    self.solver.add(z3.Or(diff_constraints))

                # Add some basic constraints to avoid trivial cases
                for var_name, var_smt in counterexample_vars.items():
                    if theory == Theory.LIA:
                        # Reasonable integer range
                        self.solver.add(var_smt >= -100)
                        self.solver.add(var_smt <= 100)
                    elif theory == Theory.BV:
                        # Reasonable bitvector range
                        self.solver.add(z3.ULT(var_smt, 2**16))  # Less than 65536

                result = self.solver.check()

                if result == z3.sat:
                    model = self.solver.model()

                    # Extract counterexample from model
                    counterexample = {}
                    for var_name, var_smt in counterexample_vars.items():
                        value = model[var_smt]
                        if theory == Theory.LIA:
                            counterexample[var_name] = int(value.as_long())
                        elif theory == Theory.BV:
                            counterexample[var_name] = int(value.as_long())
                        elif theory == Theory.STRING:
                            counterexample[var_name] = str(value)

                    # Verify this is actually a good counterexample
                    outputs = []
                    for expr in expressions:
                        try:
                            output = expr.evaluate(counterexample)
                            outputs.append(output)
                        except:
                            continue

                    if len(set(outputs)) > 1:  # Different outputs
                        self.solver.pop()
                        return counterexample

                self.solver.pop()

            return None

        except Exception as e:
            print(f"SMT counterexample generation failed: {e}")
            return None

    def _create_variable(self, var_name: str, theory: Theory, var_types: Dict[str, str] = None) -> z3.ExprRef:
        """Create a variable in the SMT context."""
        var_types = var_types or {}

        if theory == Theory.LIA:
            return z3.Int(var_name, self.converter.context)
        elif theory == Theory.BV:
            bitwidth = var_types.get(var_name, 32)
            return z3.BitVec(var_name, bitwidth, self.converter.context)
        elif theory == Theory.STRING:
            return z3.String(var_name, self.converter.context)
        else:
            raise ValueError(f"Unsupported theory: {theory}")

    def prove_equivalence(self, expr1: Expression, expr2: Expression,
                         var_types: Dict[str, str] = None) -> bool:
        """Prove that two expressions are equivalent using SMT."""
        try:
            smt_expr1 = expression_to_smt(expr1, var_types)
            smt_expr2 = expression_to_smt(expr2, var_types)

            # Check if expr1 != expr2 is unsatisfiable (i.e., they are equivalent)
            self.solver.push()
            self.solver.add(smt_expr1 != smt_expr2)

            result = self.solver.check()

            is_equivalent = (result == z3.unsat)
            self.solver.pop()

            return is_equivalent

        except Exception as e:
            print(f"SMT equivalence check failed: {e}")
            return False

    def get_smt_formula(self, expr: Expression, var_types: Dict[str, str] = None) -> str:
        """Get SMT-LIB format string for an expression."""
        smt_expr = expression_to_smt(expr, var_types)
        return smt_expr.sexpr()
