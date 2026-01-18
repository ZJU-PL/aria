"""Trigger selection for E-matching with optional LLM assistance."""

from typing import Dict, Iterable, List, Sequence, Tuple

import z3
from z3 import Const, Exists, ExprRef, ForAll, Solver

from aria.ml.ematching.llm_trigger import (
    LLMTriggerGenerator,
    TriggerCandidate,
)


class TriggerSelector:
    """
    Select appropriate triggers for quantifiers in SMT formulas.

    The selector extracts candidate trigger terms, optionally asks an LLM to pick
    combinations, and falls back to a simple heuristic when the LLM is not
    available.
    """

    def __init__(
        self,
        formula: ExprRef,
        llm_generator: LLMTriggerGenerator | None = None,
        max_groups: int = 3,
        verbose: bool = False,
    ):
        self.formula = formula
        self.solver = Solver()
        self.solver.add(formula)
        self.quantifiers: List[ExprRef] = []
        self.llm_generator = llm_generator
        self.max_groups = max_groups
        self.verbose = verbose

        self.collect_quantifiers(formula)
        self._log(f"Collected {len(self.quantifiers)} quantifiers")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[trigger-select] {message}")  # noqa: T201

    def collect_quantifiers(self, expr: ExprRef) -> None:
        """Collect all quantifiers in the given expression."""
        if z3.is_quantifier(expr):
            self.quantifiers.append(expr)
            return
        for child in expr.children():
            self.collect_quantifiers(child)

    def _build_bound_vars(self, quantifier: ExprRef) -> List[ExprRef]:
        return [
            Const(quantifier.var_name(i), quantifier.var_sort(i))
            for i in range(quantifier.num_vars())
        ]

    def _collect_candidates(
        self,
        quantifier: ExprRef,
        bound_vars: Sequence[ExprRef],
        bound_var_map: Dict[str, ExprRef],
    ) -> List[TriggerCandidate]:
        """
        Extract candidate trigger terms from the quantifier body.
        Only uninterpreted function applications (and array selects/stores) that
        mention bound variables are kept.
        """
        seen: set[str] = set()
        candidates: List[TriggerCandidate] = []
        # Replace de Bruijn indices with the canonical bound variables so that
        # variable lookup works during trigger extraction.
        body = z3.substitute_vars(quantifier.body(), *reversed(bound_vars))
        bound_names = set(bound_var_map.keys())

        def visit(expr: ExprRef) -> None:
            if z3.is_quantifier(expr):
                return
            if self._is_candidate_app(expr, bound_names):
                normalized = self._normalize_with_bound_vars(expr, bound_var_map)
                key = normalized.sexpr()
                if key not in seen:
                    seen.add(key)
                    vars_in_term = sorted(self._collect_var_names(normalized, bound_names))
                    candidates.append(
                        TriggerCandidate(
                            expr=normalized,
                            text=normalized.sexpr(),
                            variables=vars_in_term,
                        )
                    )
            for child in expr.children():
                visit(child)

        visit(body)
        self._log(f"Found {len(candidates)} trigger candidates.")
        return candidates

    def _is_candidate_app(self, expr: ExprRef, bound_names: set[str]) -> bool:
        if not z3.is_app(expr):
            return False
        if expr.num_args() == 0:
            return False
        if self._is_boolean_op(expr) or self._is_arithmetic_op(expr):
            return False
        if not self._mentions_bound_var(expr, bound_names):
            return False

        decl_kind = expr.decl().kind()
        if decl_kind in (
            z3.Z3_OP_UNINTERPRETED,
            z3.Z3_OP_SELECT,
            z3.Z3_OP_STORE,
        ):
            return True
        # Allow other non-boolean applications that are not obvious arith/boolean ops.
        return not expr.is_bool()

    def _normalize_with_bound_vars(
        self, expr: ExprRef, bound_var_map: Dict[str, ExprRef]
    ) -> ExprRef:
        """
        Rewrite expr so it refers to the canonical bound variable objects.
        """
        replacements = []
        for var in self._iter_consts(expr):
            name = str(var)
            if name in bound_var_map:
                replacements.append((var, bound_var_map[name]))
        if replacements:
            return z3.substitute(expr, replacements)
        return expr

    def _mentions_bound_var(self, expr: ExprRef, bound_names: set[str]) -> bool:
        return bool(self._collect_var_names(expr, bound_names))

    def _collect_var_names(self, expr: ExprRef, whitelist: set[str]) -> set[str]:
        names: set[str] = set()
        for var in self._iter_consts(expr):
            if str(var) in whitelist:
                names.add(str(var))
        return names

    def _iter_consts(self, expr: ExprRef) -> Iterable[ExprRef]:
        stack = [expr]
        while stack:
            current = stack.pop()
            if z3.is_const(current) and current.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                yield current
            stack.extend(current.children())

    def _is_boolean_op(self, expr: ExprRef) -> bool:
        """Check if expr is a boolean operation like And, Or, Not."""
        if not z3.is_app(expr):
            return False
        op = str(expr.decl())
        return op in ["and", "or", "not", "implies", "ite", "=", "<", "<=", ">", ">="]

    def _is_arithmetic_op(self, expr: ExprRef) -> bool:
        """Check if expr is an arithmetic operation like +, -, *, /."""
        if not z3.is_app(expr):
            return False
        op = str(expr.decl())
        return op in ["+", "-", "*", "/", "div", "mod"]

    def rank_triggers(
        self, candidates: List[TriggerCandidate]
    ) -> List[Tuple[TriggerCandidate, float]]:
        """
        Rank potential triggers based on heuristics.
        Returns a list of (trigger, score) pairs sorted by score (higher is better).
        """
        if not candidates:
            return []

        ranked_triggers = []

        for candidate in candidates:
            # Score based on depth and how many bound variables appear.
            depth = self._calculate_depth(candidate.expr)
            var_bonus = len(set(candidate.variables)) * 2
            arity_bonus = min(candidate.expr.num_args(), 3) * 0.5
            score = depth + var_bonus + arity_bonus
            ranked_triggers.append((candidate, score))

        return sorted(ranked_triggers, key=lambda x: x[1], reverse=True)

    def _calculate_depth(self, expr: ExprRef) -> int:
        """Calculate the depth of the expression tree."""
        if not expr.children():
            return 1

        max_child_depth = 0
        for child in expr.children():
            depth = self._calculate_depth(child)
            max_child_depth = max(max_child_depth, depth)

        return max_child_depth + 1

    def select_triggers(
        self, quantifier: ExprRef, bound_vars: Sequence[ExprRef] | None = None
    ) -> List[List[ExprRef]]:
        """
        Select appropriate trigger groups for a quantifier.
        """
        if bound_vars is None:
            bound_vars = self._build_bound_vars(quantifier)
        bound_var_map = {str(var): var for var in bound_vars}

        candidates = self._collect_candidates(quantifier, bound_vars, bound_var_map)
        if not candidates:
            self._log("No potential triggers found.")
            return []

        # LLM suggestion first.
        if self.llm_generator:
            if getattr(self.llm_generator, "direct_terms", False):
                llm_groups = self.llm_generator.suggest_direct_trigger_groups(
                    quantifier, bound_vars
                )
                if llm_groups:
                    self._log("Using LLM-proposed triggers (direct).")
                    return llm_groups[: self.max_groups]
                self._log("LLM direct triggers unavailable or invalid.")
                return []
            llm_groups = self.llm_generator.suggest_trigger_groups(
                quantifier, candidates, list(bound_var_map.keys())
            )
            if llm_groups:
                self._log("Using LLM-proposed triggers.")
                return llm_groups[: self.max_groups]

        ranked_triggers = self.rank_triggers(candidates)
        selected_groups: List[List[ExprRef]] = []
        for candidate, _ in ranked_triggers:
            selected_groups.append([candidate.expr])
            if len(selected_groups) >= self.max_groups:
                break

        if not self._covers_all_bound_vars(selected_groups, bound_var_map):
            selected_groups = self._repair_coverage(
                selected_groups, candidates, bound_var_map
            )

        self._log(f"Selected {len(selected_groups)} trigger group(s).")
        return selected_groups

    def _repair_coverage(
        self,
        selected_groups: List[List[ExprRef]],
        candidates: List[TriggerCandidate],
        bound_var_map: Dict[str, ExprRef],
    ) -> List[List[ExprRef]]:
        needed = set(bound_var_map.keys()) - self._covered_vars(selected_groups, bound_var_map)
        if not needed:
            return selected_groups

        for candidate in candidates:
            if needed.intersection(candidate.variables):
                selected_groups.append([candidate.expr])
                needed -= set(candidate.variables)
                if not needed or len(selected_groups) >= self.max_groups:
                    break
        return selected_groups

    def _covered_vars(
        self, trigger_groups: List[List[ExprRef]], bound_var_map: Dict[str, ExprRef]
    ) -> set[str]:
        covered: set[str] = set()
        names = set(bound_var_map.keys())
        for group in trigger_groups:
            for expr in group:
                covered.update(self._collect_var_names(expr, names))
        return covered

    def _covers_all_bound_vars(
        self, trigger_groups: List[List[ExprRef]], bound_var_map: Dict[str, ExprRef]
    ) -> bool:
        names = set(bound_var_map.keys())
        if not names:
            return True
        covered = self._covered_vars(trigger_groups, bound_var_map)
        return names.issubset(covered)

    def get_triggers_for_all_quantifiers(self) -> Dict[ExprRef, List[List[ExprRef]]]:
        """
        Get selected triggers for all quantifiers in the formula.
        """
        result = {}
        for quantifier in self.quantifiers:
            result[quantifier] = self.select_triggers(quantifier)
        return result

    def annotate_with_triggers(self) -> ExprRef:
        """
        Returns a new formula with trigger annotations added to quantifiers.
        """
        return self._annotate_expr(self.formula)

    def _annotate_expr(self, expr: ExprRef) -> ExprRef:
        """
        Recursively annotate quantifiers in the expression with selected triggers.
        """
        if z3.is_quantifier(expr):
            bound_vars = self._build_bound_vars(expr)
            triggers = self.select_triggers(expr, bound_vars)
            # Recurse into the body before re-wrapping the quantifier
            body = self._annotate_expr(expr.body())

            kwargs = {}
            if triggers:
                try:
                    patterns = [z3.MultiPattern(*group) for group in triggers]
                    if patterns:
                        kwargs["patterns"] = patterns
                except z3.Z3Exception as exc:  # pragma: no cover - defensive
                    self._log(f"Failed to build patterns: {exc}")

            if expr.is_forall():
                return ForAll(bound_vars, body, **kwargs)
            return Exists(bound_vars, body, **kwargs)

        if z3.is_app(expr):
            args = [self._annotate_expr(child) for child in expr.children()]
            return expr.decl()(*args) if args else expr

        return expr


def example_usage():
    """
    Example demonstrating the use of TriggerSelector (heuristic fallback).
    """
    from z3 import Function, IntSort, Int  # pylint: disable=import-outside-toplevel

    # Define sorts and functions
    int_sort = IntSort()
    f = Function("f", int_sort, int_sort)

    # Define variables
    x = Int("x")

    # Create a simpler formula with a quantifier
    # ForAll x. f(x) > 0
    formula = ForAll([x], f(x) > 0)

    print("Created formula:", formula)

    # Create a TriggerSelector
    print("Creating TriggerSelector...")
    selector = TriggerSelector(formula, verbose=True)

    # Get and print potential triggers for all quantifiers
    print("Getting triggers for all quantifiers...")
    triggers_dict = selector.get_triggers_for_all_quantifiers()

    print("\nFound quantifiers:", len(selector.quantifiers))
    for quantifier, triggers in triggers_dict.items():
        print("\nQuantifier:", quantifier)
        print("Selected triggers:")
        for i, trigger in enumerate(triggers):
            print(f"  {i + 1}. {trigger}")

    # Annotate the formula with selected triggers
    print("\nAnnotating formula with triggers...")
    annotated_formula = selector.annotate_with_triggers()
    print("Annotated formula:", annotated_formula)

    # Example of solving with the annotated formula
    print("\nSolving annotated formula...")
    s = Solver()
    s.add(annotated_formula)
    check_result = s.check()
    if check_result == z3.sat:
        print("Formula is satisfiable. Model:")
        print(s.model())
    else:
        print(f"Formula is {check_result}.")


if __name__ == "__main__":
    example_usage()
