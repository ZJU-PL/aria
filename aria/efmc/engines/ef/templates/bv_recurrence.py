"""
Recurrence set templates for bit-vector programs to verify non-termination.

This module implements template-based recurrence set synthesis for bit-vector
programs. Recurrence sets are used to prove program non-termination by showing
that there exists a set of states from which the program can loop infinitely.

For a loop with guard G and body B, a recurrence set S proves non-termination if:
1. S is reachable from initial states: ∃x. init(x) ∧ S(x)
2. S is closed under transitions: ∀x,x'. S(x) ∧ G(x) ∧ B(x,x') ⇒ S(x')
3. All states in S satisfy the guard: ∀x. S(x) ⇒ G(x)
4. S is non-empty: ∃x. S(x)

The templates support various forms of recurrence sets:
- Linear: a₁*x₁ + a₂*x₂ + ... + aₙ*xₙ + c ≥ 0
- Interval: l₁ ≤ x₁ ≤ u₁ ∧ l₂ ≤ x₂ ≤ u₂ ∧ ... ∧ lₙ ≤ xₙ ≤ uₙ
- Disjunctive: S₁ ∨ S₂ ∨ ... ∨ Sₖ where each Sᵢ is a simpler recurrence set
"""

import z3
from typing import List, Dict, Tuple, Optional, Any
from aria.efmc.engines.ef.templates.abstract_template import Template, TemplateType
from aria.utils.bv_utils import Signedness
from aria.efmc.sts import TransitionSystem
from aria.utils.z3_expr_utils import big_and, big_or


class BitVecLinearRecurrenceTemplate(Template):
    """
    Linear recurrence set template for bit-vector programs.

    Generates recurrence sets of the form:
    S(x₁, x₂, ..., xₙ) ≡ a₁*x₁ + a₂*x₂ + ... + aₙ*xₙ + c ≥ 0

    where aᵢ and c are template variables (coefficients) to be synthesized.
    """

    def __init__(self, sts: TransitionSystem, **kwargs):
        self.template_type = TemplateType.BV_INTERVAL  # Reusing existing type for now

        # Handle signedness
        signedness_val = self._init_signedness_from_sts(sts)
        if signedness_val is not None:
            self.signedness = signedness_val
        else:
            self.signedness = Signedness.UNSIGNED  # Default to unsigned

        self.sts = sts
        self.arity = len(self.sts.variables)

        # Template variables: coefficients for each program variable + constant
        self.template_vars = []
        self.template_index = 0

        # Constraints for recurrence set properties
        self.template_cnt_init_and_post = None
        self.template_cnt_trans = None

        # Initialize template variables and constraints
        self.add_template_vars()
        self.add_template_cnts()

    def add_template_vars(self) -> None:
        """Initialize template variables for linear recurrence set coefficients."""
        self.template_index += 1

        # Create coefficient variables for each program variable
        coeffs = []
        for i, var in enumerate(self.sts.variables):
            # Coefficient for variable i
            coeff_name = f"recur_coeff_{i}_{str(var)}"
            coeff = z3.BitVec(coeff_name, var.sort().size())
            coeffs.append(coeff)

        # Constant term
        if self.sts.variables:
            const_size = self.sts.variables[0].sort().size()
        else:
            const_size = 32  # Default bit-width
        const_name = f"recur_const_{self.template_index}"
        const = z3.BitVec(const_name, const_size)
        coeffs.append(const)

        self.template_vars.append(coeffs)

    def add_template_cnts(self) -> None:
        """
        Add constraints for recurrence set properties.

        The main constraints are handled in the verification condition generation,
        but we can add template-specific constraints here if needed.
        """
        constraints_init_post = []
        constraints_trans = []

        # For linear recurrence sets, we typically don't need additional
        # template-specific constraints beyond what's in the VC

        self.template_cnt_init_and_post = z3.BoolVal(True)
        self.template_cnt_trans = z3.BoolVal(True)

    def _build_recurrence_expr(self, variables: List[z3.ExprRef]) -> z3.ExprRef:
        """
        Build recurrence set expression: a₁*x₁ + a₂*x₂ + ... + aₙ*xₙ + c ≥ 0

        Args:
            variables: List of program variables (either current or primed)

        Returns:
            Z3 expression representing the recurrence set
        """
        if not self.template_vars or not self.template_vars[0]:
            return z3.BoolVal(True)  # Default to true if no template vars

        coeffs = self.template_vars[0]

        # Start with constant term
        expr = coeffs[-1]  # Last element is the constant

        # Add coefficient * variable terms
        for i, var in enumerate(variables):
            if i < len(coeffs) - 1:  # Exclude the constant term
                coeff = coeffs[i]
                # Ensure coefficient and variable have same bit-width
                if coeff.sort().size() != var.sort().size():
                    if coeff.sort().size() < var.sort().size():
                        coeff = z3.ZeroExt(
                            var.sort().size() - coeff.sort().size(), coeff
                        )
                    else:
                        coeff = z3.Extract(var.sort().size() - 1, 0, coeff)

                expr = expr + coeff * var

        # Return the constraint: linear expression ≥ 0
        if self.signedness == Signedness.UNSIGNED:
            # For unsigned, we use UGE (unsigned greater or equal)
            return z3.UGE(expr, z3.BitVecVal(0, expr.sort().size()))
        else:
            # For signed, we use regular >=
            return expr >= z3.BitVecVal(0, expr.sort().size())

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables: bool = False
    ) -> z3.ExprRef:
        """
        Build recurrence set expression from a model.

        Args:
            model: Z3 model containing values for template variables
            use_prime_variables: Whether to use primed variables

        Returns:
            Z3 expression representing the concrete recurrence set
        """
        variables = (
            self.sts.prime_variables if use_prime_variables else self.sts.variables
        )

        if not self.template_vars or not self.template_vars[0]:
            return z3.BoolVal(True)

        coeffs = self.template_vars[0]

        # Start with evaluated constant term
        expr = model.eval(coeffs[-1])

        # Add evaluated coefficient * variable terms
        for i, var in enumerate(variables):
            if i < len(coeffs) - 1:
                coeff_val = model.eval(coeffs[i])
                # Ensure coefficient and variable have same bit-width
                if coeff_val.sort().size() != var.sort().size():
                    if coeff_val.sort().size() < var.sort().size():
                        coeff_val = z3.ZeroExt(
                            var.sort().size() - coeff_val.sort().size(), coeff_val
                        )
                    else:
                        coeff_val = z3.Extract(var.sort().size() - 1, 0, coeff_val)

                expr = expr + coeff_val * var

        # Return the constraint: linear expression ≥ 0
        if self.signedness == Signedness.UNSIGNED:
            return z3.UGE(expr, z3.BitVecVal(0, expr.sort().size()))
        else:
            return expr >= z3.BitVecVal(0, expr.sort().size())


class BitVecIntervalRecurrenceTemplate(Template):
    """
    Interval recurrence set template for bit-vector programs.

    Generates recurrence sets of the form:
    S(x₁, x₂, ..., xₙ) ≡ l₁ ≤ x₁ ≤ u₁ ∧ l₂ ≤ x₂ ≤ u₂ ∧ ... ∧ lₙ ≤ xₙ ≤ uₙ

    where lᵢ and uᵢ are lower and upper bounds to be synthesized.
    """

    def __init__(self, sts: TransitionSystem, **kwargs):
        self.template_type = TemplateType.BV_INTERVAL

        if sts.signedness == "signed":
            self.signedness = Signedness.SIGNED
        elif sts.signedness == "unsigned":
            self.signedness = Signedness.UNSIGNED
        else:
            self.signedness = Signedness.UNSIGNED

        self.sts = sts
        self.arity = len(self.sts.variables)

        self.template_vars = []
        self.template_index = 0

        self.template_cnt_init_and_post = None
        self.template_cnt_trans = None

        self.add_template_vars()
        self.add_template_cnts()

    def add_template_vars(self) -> None:
        """Initialize template variables for interval bounds."""
        # Create lower and upper bound variables for each program variable
        bounds = []

        for i, var in enumerate(self.sts.variables):
            var_size = var.sort().size()

            # Lower bound for variable i
            lower_name = f"recur_lower_{i}_{str(var)}"
            lower = z3.BitVec(lower_name, var_size)
            bounds.append(lower)

            # Upper bound for variable i
            upper_name = f"recur_upper_{i}_{str(var)}"
            upper = z3.BitVec(upper_name, var_size)
            bounds.append(upper)

        self.template_vars.append(bounds)

    def add_template_cnts(self) -> None:
        """Add constraints ensuring lower bounds ≤ upper bounds."""
        constraints_init_post = []

        if self.template_vars and self.template_vars[0]:
            bounds = self.template_vars[0]

            # Ensure lower_i ≤ upper_i for each variable
            for i in range(0, len(bounds), 2):
                if i + 1 < len(bounds):
                    lower = bounds[i]
                    upper = bounds[i + 1]

                    if self.signedness == Signedness.UNSIGNED:
                        constraint = z3.ULE(lower, upper)
                    else:
                        constraint = lower <= upper

                    constraints_init_post.append(constraint)

        self.template_cnt_init_and_post = (
            big_and(constraints_init_post)
            if constraints_init_post
            else z3.BoolVal(True)
        )
        self.template_cnt_trans = z3.BoolVal(True)

    def _build_recurrence_expr(self, variables: List[z3.ExprRef]) -> z3.ExprRef:
        """
        Build interval recurrence set expression: ∧ᵢ (lᵢ ≤ xᵢ ≤ uᵢ)

        Args:
            variables: List of program variables

        Returns:
            Z3 expression representing the interval recurrence set
        """
        if not self.template_vars or not self.template_vars[0]:
            return z3.BoolVal(True)

        bounds = self.template_vars[0]
        constraints = []

        # Build interval constraints for each variable
        for i, var in enumerate(variables):
            if 2 * i + 1 < len(bounds):
                lower = bounds[2 * i]
                upper = bounds[2 * i + 1]

                # Ensure bounds have same bit-width as variable
                if lower.sort().size() != var.sort().size():
                    if lower.sort().size() < var.sort().size():
                        lower = z3.ZeroExt(
                            var.sort().size() - lower.sort().size(), lower
                        )
                        upper = z3.ZeroExt(
                            var.sort().size() - upper.sort().size(), upper
                        )
                    else:
                        lower = z3.Extract(var.sort().size() - 1, 0, lower)
                        upper = z3.Extract(var.sort().size() - 1, 0, upper)

                # Add interval constraint: lower ≤ var ≤ upper
                if self.signedness == Signedness.UNSIGNED:
                    lower_constraint = z3.ULE(lower, var)
                    upper_constraint = z3.ULE(var, upper)
                else:
                    lower_constraint = lower <= var
                    upper_constraint = var <= upper

                constraints.append(z3.And(lower_constraint, upper_constraint))

        return big_and(constraints) if constraints else z3.BoolVal(True)

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables: bool = False
    ) -> z3.ExprRef:
        """Build interval recurrence set expression from model."""
        variables = (
            self.sts.prime_variables if use_prime_variables else self.sts.variables
        )

        if not self.template_vars or not self.template_vars[0]:
            return z3.BoolVal(True)

        bounds = self.template_vars[0]
        constraints = []

        # Build interval constraints with evaluated bounds
        for i, var in enumerate(variables):
            if 2 * i + 1 < len(bounds):
                lower_val = model.eval(bounds[2 * i])
                upper_val = model.eval(bounds[2 * i + 1])

                # Ensure bounds have same bit-width as variable
                if lower_val.sort().size() != var.sort().size():
                    if lower_val.sort().size() < var.sort().size():
                        lower_val = z3.ZeroExt(
                            var.sort().size() - lower_val.sort().size(), lower_val
                        )
                        upper_val = z3.ZeroExt(
                            var.sort().size() - upper_val.sort().size(), upper_val
                        )
                    else:
                        lower_val = z3.Extract(var.sort().size() - 1, 0, lower_val)
                        upper_val = z3.Extract(var.sort().size() - 1, 0, upper_val)

                # Add interval constraint
                if self.signedness == Signedness.UNSIGNED:
                    lower_constraint = z3.ULE(lower_val, var)
                    upper_constraint = z3.ULE(var, upper_val)
                else:
                    lower_constraint = lower_val <= var
                    upper_constraint = var <= upper_val

                constraints.append(z3.And(lower_constraint, upper_constraint))

        return big_and(constraints) if constraints else z3.BoolVal(True)


class BitVecDisjunctiveRecurrenceTemplate(Template):
    """
    Disjunctive recurrence set template for bit-vector programs.

    Generates recurrence sets of the form:
    S(x) ≡ S₁(x) ∨ S₂(x) ∨ ... ∨ Sₖ(x)

    where each Sᵢ is a simpler recurrence set (e.g., interval or linear).
    """

    def __init__(self, sts: TransitionSystem, **kwargs):
        self.template_type = TemplateType.BV_INTERVAL

        if sts.signedness == "signed":
            self.signedness = Signedness.SIGNED
        elif sts.signedness == "unsigned":
            self.signedness = Signedness.UNSIGNED
        else:
            self.signedness = Signedness.UNSIGNED

        self.sts = sts
        self.arity = len(self.sts.variables)

        # Number of disjuncts
        self.num_disjuncts = kwargs.get("num_disjuncts", 2)
        # Type of base template for each disjunct
        self.base_template_type = kwargs.get(
            "base_template_type", "interval"
        )  # "interval" or "linear"

        self.template_vars = []
        self.template_index = 0

        self.template_cnt_init_and_post = None
        self.template_cnt_trans = None

        self.add_template_vars()
        self.add_template_cnts()

    def add_template_vars(self) -> None:
        """Initialize template variables for disjunctive recurrence set."""
        # Create template variables for each disjunct
        for disjunct_idx in range(self.num_disjuncts):
            if self.base_template_type == "interval":
                # Create interval bounds for this disjunct
                bounds = []
                for i, var in enumerate(self.sts.variables):
                    var_size = var.sort().size()

                    lower_name = f"disj_{disjunct_idx}_lower_{i}_{str(var)}"
                    lower = z3.BitVec(lower_name, var_size)
                    bounds.append(lower)

                    upper_name = f"disj_{disjunct_idx}_upper_{i}_{str(var)}"
                    upper = z3.BitVec(upper_name, var_size)
                    bounds.append(upper)

                self.template_vars.append(bounds)

            elif self.base_template_type == "linear":
                # Create linear coefficients for this disjunct
                coeffs = []
                for i, var in enumerate(self.sts.variables):
                    coeff_name = f"disj_{disjunct_idx}_coeff_{i}_{str(var)}"
                    coeff = z3.BitVec(coeff_name, var.sort().size())
                    coeffs.append(coeff)

                # Constant term
                if self.sts.variables:
                    const_size = self.sts.variables[0].sort().size()
                else:
                    const_size = 32
                const_name = f"disj_{disjunct_idx}_const"
                const = z3.BitVec(const_name, const_size)
                coeffs.append(const)

                self.template_vars.append(coeffs)

    def add_template_cnts(self) -> None:
        """Add constraints for disjunctive recurrence set."""
        constraints_init_post = []

        if self.base_template_type == "interval":
            # Ensure lower ≤ upper for each interval in each disjunct
            for disjunct_idx in range(self.num_disjuncts):
                if disjunct_idx < len(self.template_vars):
                    bounds = self.template_vars[disjunct_idx]

                    for i in range(0, len(bounds), 2):
                        if i + 1 < len(bounds):
                            lower = bounds[i]
                            upper = bounds[i + 1]

                            if self.signedness == Signedness.UNSIGNED:
                                constraint = z3.ULE(lower, upper)
                            else:
                                constraint = lower <= upper

                            constraints_init_post.append(constraint)

        self.template_cnt_init_and_post = (
            big_and(constraints_init_post)
            if constraints_init_post
            else z3.BoolVal(True)
        )
        self.template_cnt_trans = z3.BoolVal(True)

    def _build_recurrence_expr(self, variables: List[z3.ExprRef]) -> z3.ExprRef:
        """Build disjunctive recurrence set expression: S₁(x) ∨ S₂(x) ∨ ... ∨ Sₖ(x)"""
        disjuncts = []

        for disjunct_idx in range(self.num_disjuncts):
            if disjunct_idx < len(self.template_vars):
                if self.base_template_type == "interval":
                    disjunct_expr = self._build_interval_disjunct(
                        disjunct_idx, variables
                    )
                elif self.base_template_type == "linear":
                    disjunct_expr = self._build_linear_disjunct(disjunct_idx, variables)
                else:
                    disjunct_expr = z3.BoolVal(True)

                disjuncts.append(disjunct_expr)

        return big_or(disjuncts) if disjuncts else z3.BoolVal(True)

    def _build_interval_disjunct(
        self, disjunct_idx: int, variables: List[z3.ExprRef]
    ) -> z3.ExprRef:
        """Build interval constraint for a specific disjunct."""
        bounds = self.template_vars[disjunct_idx]
        constraints = []

        for i, var in enumerate(variables):
            if 2 * i + 1 < len(bounds):
                lower = bounds[2 * i]
                upper = bounds[2 * i + 1]

                # Handle bit-width compatibility
                if lower.sort().size() != var.sort().size():
                    if lower.sort().size() < var.sort().size():
                        lower = z3.ZeroExt(
                            var.sort().size() - lower.sort().size(), lower
                        )
                        upper = z3.ZeroExt(
                            var.sort().size() - upper.sort().size(), upper
                        )
                    else:
                        lower = z3.Extract(var.sort().size() - 1, 0, lower)
                        upper = z3.Extract(var.sort().size() - 1, 0, upper)

                if self.signedness == Signedness.UNSIGNED:
                    lower_constraint = z3.ULE(lower, var)
                    upper_constraint = z3.ULE(var, upper)
                else:
                    lower_constraint = lower <= var
                    upper_constraint = var <= upper

                constraints.append(z3.And(lower_constraint, upper_constraint))

        return big_and(constraints) if constraints else z3.BoolVal(True)

    def _build_linear_disjunct(
        self, disjunct_idx: int, variables: List[z3.ExprRef]
    ) -> z3.ExprRef:
        """Build linear constraint for a specific disjunct."""
        coeffs = self.template_vars[disjunct_idx]

        # Start with constant term
        expr = coeffs[-1]

        # Add coefficient * variable terms
        for i, var in enumerate(variables):
            if i < len(coeffs) - 1:
                coeff = coeffs[i]
                # Handle bit-width compatibility
                if coeff.sort().size() != var.sort().size():
                    if coeff.sort().size() < var.sort().size():
                        coeff = z3.ZeroExt(
                            var.sort().size() - coeff.sort().size(), coeff
                        )
                    else:
                        coeff = z3.Extract(var.sort().size() - 1, 0, coeff)

                expr = expr + coeff * var

        # Return constraint: linear expression ≥ 0
        if self.signedness == Signedness.UNSIGNED:
            return z3.UGE(expr, z3.BitVecVal(0, expr.sort().size()))
        else:
            return expr >= z3.BitVecVal(0, expr.sort().size())

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables: bool = False
    ) -> z3.ExprRef:
        """Build disjunctive recurrence set expression from model."""
        variables = (
            self.sts.prime_variables if use_prime_variables else self.sts.variables
        )
        disjuncts = []

        for disjunct_idx in range(self.num_disjuncts):
            if disjunct_idx < len(self.template_vars):
                if self.base_template_type == "interval":
                    disjunct_expr = self._build_evaluated_interval_disjunct(
                        disjunct_idx, variables, model
                    )
                elif self.base_template_type == "linear":
                    disjunct_expr = self._build_evaluated_linear_disjunct(
                        disjunct_idx, variables, model
                    )
                else:
                    disjunct_expr = z3.BoolVal(True)

                disjuncts.append(disjunct_expr)

        return big_or(disjuncts) if disjuncts else z3.BoolVal(True)

    def _build_evaluated_interval_disjunct(
        self, disjunct_idx: int, variables: List[z3.ExprRef], model: z3.ModelRef
    ) -> z3.ExprRef:
        """Build evaluated interval constraint for a specific disjunct."""
        bounds = self.template_vars[disjunct_idx]
        constraints = []

        for i, var in enumerate(variables):
            if 2 * i + 1 < len(bounds):
                lower_val = model.eval(bounds[2 * i])
                upper_val = model.eval(bounds[2 * i + 1])

                # Handle bit-width compatibility
                if lower_val.sort().size() != var.sort().size():
                    if lower_val.sort().size() < var.sort().size():
                        lower_val = z3.ZeroExt(
                            var.sort().size() - lower_val.sort().size(), lower_val
                        )
                        upper_val = z3.ZeroExt(
                            var.sort().size() - upper_val.sort().size(), upper_val
                        )
                    else:
                        lower_val = z3.Extract(var.sort().size() - 1, 0, lower_val)
                        upper_val = z3.Extract(var.sort().size() - 1, 0, upper_val)

                if self.signedness == Signedness.UNSIGNED:
                    lower_constraint = z3.ULE(lower_val, var)
                    upper_constraint = z3.ULE(var, upper_val)
                else:
                    lower_constraint = lower_val <= var
                    upper_constraint = var <= upper_val

                constraints.append(z3.And(lower_constraint, upper_constraint))

        return big_and(constraints) if constraints else z3.BoolVal(True)

    def _build_evaluated_linear_disjunct(
        self, disjunct_idx: int, variables: List[z3.ExprRef], model: z3.ModelRef
    ) -> z3.ExprRef:
        """Build evaluated linear constraint for a specific disjunct."""
        coeffs = self.template_vars[disjunct_idx]

        # Start with evaluated constant term
        expr = model.eval(coeffs[-1])

        # Add evaluated coefficient * variable terms
        for i, var in enumerate(variables):
            if i < len(coeffs) - 1:
                coeff_val = model.eval(coeffs[i])
                # Handle bit-width compatibility
                if coeff_val.sort().size() != var.sort().size():
                    if coeff_val.sort().size() < var.sort().size():
                        coeff_val = z3.ZeroExt(
                            var.sort().size() - coeff_val.sort().size(), coeff_val
                        )
                    else:
                        coeff_val = z3.Extract(var.sort().size() - 1, 0, coeff_val)

                expr = expr + coeff_val * var

        # Return constraint: linear expression ≥ 0
        if self.signedness == Signedness.UNSIGNED:
            return z3.UGE(expr, z3.BitVecVal(0, expr.sort().size()))
        else:
            return expr >= z3.BitVecVal(0, expr.sort().size())
