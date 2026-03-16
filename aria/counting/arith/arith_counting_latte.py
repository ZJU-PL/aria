import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import z3

from aria.utils.z3_expr_utils import get_variables, z3_value_to_python


@dataclass
class ArithCountResult:
    status: str
    count: Optional[int]
    backend: str
    reason: str


@dataclass
class ArithAnalysis:
    status: str
    logic: str
    int_variables: List[z3.ExprRef]
    bounds: Dict[str, Tuple[int, int]]
    reason: str


class ArithModelCounter:
    def __init__(self, latte_path: Optional[str] = None):
        self.latte_path = latte_path or self._find_latte()
        self.formula: Optional[object] = None

    def _find_latte(self) -> Optional[str]:
        for cmd in ("count", "latte-count", "latte-int"):
            path = shutil.which(cmd)
            if path:
                return path
        return None

    def init_from_file(self, filename: str) -> None:
        assertions = z3.parse_smt2_file(filename)
        self.formula = z3.And(*assertions)

    def init_from_fml(self, formula: z3.ExprRef) -> None:
        self.formula = formula

    def _has_quantifier(self, expr: z3.ExprRef) -> bool:
        stack = [expr]
        while stack:
            node = stack.pop()
            if z3.is_quantifier(node):
                return True
            stack.extend(node.children())
        return False

    def _is_lia_formula(self, formula: z3.ExprRef) -> Tuple[bool, str]:
        if self._has_quantifier(formula):
            return False, "quantified formulas are not supported"

        all_vars = get_variables(formula)
        for var in all_vars:
            if z3.is_real(var):
                return False, "real-valued variables are not supported"
            if not z3.is_int(var) and not z3.is_bool(var):
                return False, f"unsupported variable sort: {var.sort()}"

        forbidden_op_names = [
            "Z3_OP_DIV",
            "Z3_OP_IDIV",
            "Z3_OP_MOD",
            "Z3_OP_REM",
            "Z3_OP_POWER",
            "Z3_OP_TO_REAL",
            "Z3_OP_TO_INT",
            "Z3_OP_IS_INT",
        ]
        forbidden_ops = {
            getattr(z3, name) for name in forbidden_op_names if hasattr(z3, name)
        }

        stack = [formula]
        while stack:
            node = stack.pop()
            if z3.is_quantifier(node):
                return False, "quantified formulas are not supported"
            if z3.is_app(node):
                kind = node.decl().kind()
                if kind in forbidden_ops:
                    return False, "unsupported arithmetic operator"
                if kind == z3.Z3_OP_MUL:
                    non_constant = [c for c in node.children() if not z3.is_int_value(c)]
                    if len(non_constant) > 1:
                        return False, "nonlinear arithmetic is not supported"
            stack.extend(node.children())

        return True, ""

    def _bound_for_var(
        self, formula: z3.ExprRef, var: z3.ExprRef
    ) -> Tuple[Optional[int], Optional[int]]:
        opt_min = z3.Optimize()
        opt_min.add(formula)
        hmin = opt_min.minimize(var)
        if opt_min.check() != z3.sat:
            return None, None
        min_val = opt_min.lower(hmin)

        opt_max = z3.Optimize()
        opt_max.add(formula)
        hmax = opt_max.maximize(var)
        if opt_max.check() != z3.sat:
            return None, None
        max_val = opt_max.upper(hmax)

        if not (z3.is_int_value(min_val) and z3.is_int_value(max_val)):
            return None, None

        min_python = z3_value_to_python(min_val)
        max_python = z3_value_to_python(max_val)
        if not isinstance(min_python, int) or not isinstance(max_python, int):
            return None, None

        return min_python, max_python

    def analyze(
        self,
        formula: Optional[z3.ExprRef] = None,
        variables: Optional[List[z3.ExprRef]] = None,
    ) -> ArithAnalysis:
        raw_formula: Optional[object]
        if formula is not None:
            raw_formula = formula
        else:
            raw_formula = self.formula

        if raw_formula is None or not isinstance(raw_formula, z3.ExprRef):
            return ArithAnalysis(
                status="unsupported",
                logic="ALL",
                int_variables=[],
                bounds={},
                reason="formula is not initialized",
            )

        fml = raw_formula

        ok_lia, reason = self._is_lia_formula(fml)
        logic = "QF_LIA"
        if not ok_lia:
            return ArithAnalysis(
                status="unsupported",
                logic=logic,
                int_variables=[],
                bounds={},
                reason=reason,
            )

        int_vars = [v for v in get_variables(fml) if z3.is_int(v)]
        if variables is not None:
            int_var_set = {v.get_id() for v in int_vars}
            filtered = []
            for var in variables:
                if not z3.is_int(var):
                    return ArithAnalysis(
                        status="unsupported",
                        logic=logic,
                        int_variables=[],
                        bounds={},
                        reason="projection variables must be Int",
                    )
                if var.get_id() not in int_var_set:
                    return ArithAnalysis(
                        status="unsupported",
                        logic=logic,
                        int_variables=[],
                        bounds={},
                        reason="projection variable not found in formula",
                    )
                filtered.append(var)
            int_vars = filtered

        solver = z3.Solver()
        solver.add(fml)
        if solver.check() == z3.unsat:
            return ArithAnalysis(
                status="exact",
                logic=logic,
                int_variables=int_vars,
                bounds={},
                reason="formula is unsatisfiable",
            )

        bounds: Dict[str, Tuple[int, int]] = {}
        for var in int_vars:
            lower, upper = self._bound_for_var(fml, var)
            if lower is None or upper is None:
                return ArithAnalysis(
                    status="unbounded",
                    logic=logic,
                    int_variables=int_vars,
                    bounds={},
                    reason=f"could not prove finite bounds for {var}",
                )
            bounds[str(var)] = (lower, upper)

        return ArithAnalysis(
            status="exact",
            logic=logic,
            int_variables=int_vars,
            bounds=bounds,
            reason="",
        )

    def count_models_by_enumeration(
        self,
        formula: Optional[z3.ExprRef] = None,
        variables: Optional[List[z3.ExprRef]] = None,
    ) -> ArithCountResult:
        raw_formula: Optional[object]
        if formula is not None:
            raw_formula = formula
        else:
            raw_formula = self.formula

        if raw_formula is None or not isinstance(raw_formula, z3.ExprRef):
            return ArithCountResult(
                status="unsupported",
                count=None,
                backend="enumeration",
                reason="formula is not initialized",
            )

        fml = raw_formula

        analysis = self.analyze(fml, variables=variables)
        if analysis.status != "exact":
            return ArithCountResult(
                status=analysis.status,
                count=None,
                backend="enumeration",
                reason=analysis.reason,
            )

        solver = z3.Solver()
        solver.add(fml)

        if solver.check() == z3.unsat:
            return ArithCountResult(
                status="exact",
                count=0,
                backend="enumeration",
                reason="formula is unsatisfiable",
            )

        vars_to_count = analysis.int_variables
        if len(vars_to_count) == 0:
            return ArithCountResult(
                status="exact",
                count=1,
                backend="enumeration",
                reason="no Int variables to count",
            )

        count = 0
        while solver.check() == z3.sat:
            count += 1
            model = solver.model()
            block = []
            for var in vars_to_count:
                val = model.eval(var, model_completion=True)
                block.append(var != val)
            solver.add(z3.Or(block))

        return ArithCountResult(
            status="exact",
            count=count,
            backend="enumeration",
            reason="",
        )

    def count_models_by_latte(
        self,
        formula: Optional[z3.ExprRef] = None,
        variables: Optional[List[z3.ExprRef]] = None,
    ) -> ArithCountResult:
        _ = formula
        _ = variables
        if not self.latte_path:
            return ArithCountResult(
                status="unsupported",
                count=None,
                backend="latte",
                reason="LattE executable not found",
            )
        return ArithCountResult(
            status="unsupported",
            count=None,
            backend="latte",
            reason="LattE backend is not implemented yet",
        )

    def count_models(
        self,
        formula: Optional[z3.ExprRef] = None,
        variables: Optional[List[z3.ExprRef]] = None,
        method: str = "auto",
    ) -> ArithCountResult:
        if method == "enumeration":
            return self.count_models_by_enumeration(formula, variables)
        if method == "latte":
            return self.count_models_by_latte(formula, variables)

        if method != "auto":
            return ArithCountResult(
                status="unsupported",
                count=None,
                backend="none",
                reason=f"unknown arithmetic counting method: {method}",
            )

        return self.count_models_by_enumeration(formula, variables)


def count_lia_models(
    formula: z3.ExprRef,
    variables: Optional[List[z3.ExprRef]] = None,
    method: str = "auto",
) -> int:
    counter = ArithModelCounter()
    result = counter.count_models(formula=formula, variables=variables, method=method)
    if result.status != "exact" or result.count is None:
        raise ValueError(f"Arithmetic counting failed: {result.status}: {result.reason}")
    return result.count
