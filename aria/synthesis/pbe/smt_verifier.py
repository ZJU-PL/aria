"""SMT-based verification utilities for PBE."""

from typing import Any, Dict, List, Optional

import z3

from .expression_to_smt import SMTConverter
from .expressions import Expression, Theory, ValueType


class SMTVerifier:
    """SMT-based verifier for candidate expressions."""

    def verify_expression(
        self,
        expr: Expression,
        examples: List[Dict[str, Any]],
        var_types: Optional[Dict[str, int]] = None,
    ) -> bool:
        """Verify that an expression matches all examples."""
        try:
            context = z3.Context()
            converter = SMTConverter(context)
            smt_expr = converter.convert_expression(expr, var_types)

            for example in examples:
                solver = z3.Solver(ctx=context)
                for var_name, value in example.items():
                    if var_name == "output":
                        continue
                    variable = converter.get_variables().get(var_name)
                    if variable is None:
                        continue
                    solver.add(
                        variable
                        == self._python_value_to_smt(
                            value, expr, context, variable
                        )
                    )

                solver.add(
                    smt_expr
                    != self._python_output_to_smt(example["output"], expr, context)
                )
                if solver.check() == z3.sat:
                    return False

            return True
        except Exception:
            return False

    def find_counterexample(
        self,
        expressions: List[Expression],
        examples: List[Dict[str, Any]],
        var_types: Optional[Dict[str, int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find an unlabeled input on which the expressions disagree."""
        if not expressions:
            return None

        try:
            context = z3.Context()
            converter = SMTConverter(context)
            smt_expressions = [
                converter.convert_expression(expr, var_types) for expr in expressions
            ]
            variables = converter.get_variables()
            solver = z3.Solver(ctx=context)

            differences = []
            for index, first in enumerate(smt_expressions):
                for second in smt_expressions[index + 1 :]:
                    differences.append(first != second)
            if not differences:
                return None
            solver.add(z3.Or(differences))

            for example in examples:
                equalities = []
                for name, value in example.items():
                    if name == "output" or name not in variables:
                        continue
                    equalities.append(
                        variables[name]
                        == self._python_value_to_smt(
                            value, expressions[0], context, variables[name]
                        )
                    )
                if equalities:
                    solver.add(z3.Not(z3.And(equalities)))

            for variable in variables.values():
                solver.add(
                    *self._domain_constraints(
                        variable, expressions[0].theory, context
                    )
                )

            if solver.check() != z3.sat:
                return None

            model = solver.model()
            return {
                name: self._model_value_to_python(
                    model.eval(symbol, model_completion=True)
                )
                for name, symbol in variables.items()
            }
        except Exception:
            return None

    def prove_equivalence(
        self,
        expr1: Expression,
        expr2: Expression,
        var_types: Optional[Dict[str, int]] = None,
    ) -> bool:
        """Prove two expressions equivalent with SMT."""
        try:
            context = z3.Context()
            converter = SMTConverter(context)
            left = converter.convert_expression(expr1, var_types)
            right = converter.convert_expression(expr2, var_types)
            solver = z3.Solver(ctx=context)
            solver.add(left != right)
            return solver.check() == z3.unsat
        except Exception:
            return False

    def get_smt_formula(
        self, expr: Expression, var_types: Optional[Dict[str, int]] = None
    ) -> str:
        """Return the SMT-LIB body for an expression."""
        context = z3.Context()
        converter = SMTConverter(context)
        return converter.convert_expression(expr, var_types).sexpr()

    def _python_output_to_smt(
        self, value: Any, expr: Expression, context: z3.Context
    ) -> z3.ExprRef:
        if expr.value_type == ValueType.BOOL:
            return z3.BoolVal(bool(value), context)
        if expr.value_type == ValueType.STRING:
            return z3.StringVal(str(value), context)
        if expr.value_type == ValueType.BV:
            return z3.BitVecVal(int(value), expr.bitwidth or 32, context)
        return z3.IntVal(int(value), context)

    def _python_value_to_smt(
        self,
        value: Any,
        expr: Expression,
        context: z3.Context,
        symbol: z3.ExprRef,
    ) -> z3.ExprRef:
        if z3.is_bool(symbol):
            return z3.BoolVal(bool(value), context)
        if z3.is_string(symbol):
            return z3.StringVal(str(value), context)
        if z3.is_bv(symbol):
            return z3.BitVecVal(int(value), symbol.size(), context)
        del expr
        return z3.IntVal(int(value), context)

    def _domain_constraints(
        self, symbol: z3.ExprRef, theory: Theory, context: z3.Context
    ) -> List[z3.ExprRef]:
        if z3.is_int(symbol):
            return [
                symbol >= z3.IntVal(-64, context),
                symbol <= z3.IntVal(64, context),
            ]
        if z3.is_bv(symbol):
            return [z3.ULE(symbol, z3.BitVecVal(255, symbol.size(), context))]
        if theory == Theory.STRING and z3.is_string(symbol):
            return [z3.Length(symbol) <= z3.IntVal(8, context)]
        return []

    def _model_value_to_python(self, value: z3.ExprRef) -> Any:
        if z3.is_true(value) or z3.is_false(value):
            return z3.is_true(value)
        if z3.is_int_value(value) or z3.is_bv_value(value):
            return value.as_long()
        if z3.is_string_value(value):
            return value.as_string()
        return str(value)
