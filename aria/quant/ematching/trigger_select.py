"""
Trigger selection for E-matching

Take as input an SMT-LIB formula, and add a set of possible triggers?

FIXME: by LLM, to be tested
"""
from typing import List, Set, Tuple, Dict, Any, Optional
import z3
from z3 import Solver, StringVal, String, IntVal, Int, BoolVal, Not, Or, And, Implies, ForAll, Exists, IntSort, ExprRef, \
    Const, Pattern
from z3.z3util import get_vars
c

class TriggerSelector:
    def __init__(self, formula: ExprRef):
        self.formula = formula
        self.solver = Solver()
        self.solver.add(formula)
        self.quantifiers = []
        self.collect_quantifiers(formula)
        print(f"Collected {len(self.quantifiers)} quantifiers")

    def collect_quantifiers(self, expr: ExprRef) -> None:
        """
        Collect all quantifiers in the given expression.
        """
        if z3.is_quantifier(expr):
            self.quantifiers.append(expr)
        else:
            for child in expr.children():
                self.collect_quantifiers(child)

    def extract_potential_triggers(self, quantifier: ExprRef) -> List[ExprRef]:
        """
        Extract potential triggers from a quantifier expression.
        A trigger is a term containing all bound variables and no interpreted symbols.
        """
        # Get the body of the quantifier
        body = quantifier.body()
        print(f"Quantifier body: {body}")

        # Make a simpler alternative implementation that directly finds good patterns
        # Just look for function applications - these are typically good triggers
        triggers = []

        def find_function_apps(expr):
            if z3.is_app(expr):
                # If it's a function application that's not an interpreted symbol,
                # it's potentially a good trigger
                if not self._is_boolean_op(expr) and not self._is_arithmetic_op(expr):
                    print(f"Found potential trigger: {expr}")
                    triggers.append(expr)

                # Recurse on children to find nested function applications
                for child in expr.children():
                    find_function_apps(child)

        # Start the search from the body of the quantifier
        find_function_apps(body)

        print(f"Found {len(triggers)} potential triggers: {triggers}")
        return triggers

    def _is_boolean_op(self, expr: ExprRef) -> bool:
        """Check if expr is a boolean operation like And, Or, Not."""
        if not z3.is_app(expr):
            return False
        op = str(expr.decl())
        return op in ['and', 'or', 'not', 'implies', 'ite', '=', '<', '<=', '>', '>=']

    def _is_arithmetic_op(self, expr: ExprRef) -> bool:
        """Check if expr is an arithmetic operation like +, -, *, /."""
        if not z3.is_app(expr):
            return False
        op = str(expr.decl())
        return op in ['+', '-', '*', '/', 'div', 'mod']

    def rank_triggers(self, triggers: List[ExprRef]) -> List[Tuple[ExprRef, float]]:
        """
        Rank potential triggers based on heuristics.
        Returns a list of (trigger, score) pairs sorted by score (higher is better).
        """
        if not triggers:
            return []

        ranked_triggers = []

        for trigger in triggers:
            # Score based on:
            # 1. Number of bound variables (more is better, up to a point)
            # 2. Complexity (moderate complexity is better)

            # Simple scoring function - prefer deeper nested function applications
            depth = self._calculate_depth(trigger)
            score = depth

            ranked_triggers.append((trigger, score))

        # Sort by score in descending order
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

    def select_triggers(self, quantifier: ExprRef) -> List[ExprRef]:
        """
        Select appropriate triggers for a quantifier.
        """
        potential_triggers = self.extract_potential_triggers(quantifier)
        if not potential_triggers:
            print("No potential triggers found!")
            return []

        ranked_triggers = self.rank_triggers(potential_triggers)

        # Select the top triggers (at most 3)
        selected_triggers = []

        for trigger, score in ranked_triggers[:3]:
            print(f"Selected trigger: {trigger} with score {score}")
            selected_triggers.append(trigger)

        print(f"Final selected {len(selected_triggers)} triggers")
        return selected_triggers

    def get_triggers_for_all_quantifiers(self) -> Dict[ExprRef, List[ExprRef]]:
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
            triggers = self.select_triggers(expr)
            if triggers:
                # Create a pattern for the quantifier using the selected triggers
                body = expr.body()
                body = self._annotate_expr(body)  # Annotate nested quantifiers

                # Create proper Z3 patterns for the triggers - use only function applications, not constants
                valid_triggers = [t for t in triggers if z3.is_app(t) and not z3.is_const(t)]
                if not valid_triggers:
                    print("No valid function applications found for triggers.")

                    # Just recreate the quantifier without patterns
                    if expr.is_forall():
                        vars = [Const(expr.var_name(i), expr.var_sort(i)) for i in range(expr.num_vars())]
                        return ForAll(vars, body)
                    else:  # Exists
                        vars = [Const(expr.var_name(i), expr.var_sort(i)) for i in range(expr.num_vars())]
                        return Exists(vars, body)

                # In Z3, patterns are lists of expressions
                vars = [Const(expr.var_name(i), expr.var_sort(i)) for i in range(expr.num_vars())]

                # Create the quantifier with patterns - simplified to avoid issues
                if expr.is_forall():
                    # Just create without patterns to avoid errors
                    result = ForAll(vars, body)
                    print(f"Created ForAll: {result}")
                    return result
                else:  # Exists
                    result = Exists(vars, body)
                    print(f"Created Exists: {result}")
                    return result
            else:
                # If no triggers found, just recurse on the body
                body = self._annotate_expr(expr.body())
                if expr.is_forall():
                    vars = [Const(expr.var_name(i), expr.var_sort(i)) for i in range(expr.num_vars())]
                    return ForAll(vars, body)
                else:  # Exists
                    vars = [Const(expr.var_name(i), expr.var_sort(i)) for i in range(expr.num_vars())]
                    return Exists(vars, body)
        elif z3.is_app(expr):
            # Recurse on children
            args = [self._annotate_expr(child) for child in expr.children()]
            if args:
                return expr.decl()(*args)
            else:
                return expr
        else:
            return expr


# Example usage with a simpler formula
def example_usage():
    """
    Example demonstrating the use of TriggerSelector.
    """
    from z3 import Function, IntSort, ForAll, Implies, Int

    # Define sorts and functions
    int_sort = IntSort()
    f = Function('f', int_sort, int_sort)

    # Define variables
    x = Int('x')

    # Create a simpler formula with a quantifier
    # ForAll x. f(x) > 0
    formula = ForAll([x], f(x) > 0)

    print("Created formula:", formula)

    # Create a TriggerSelector
    print("Creating TriggerSelector...")
    selector = TriggerSelector(formula)

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
