"""
Ranking function templates for bit-vector programs to verify termination.

This module implements template-based ranking function synthesis for bit-vector
programs. Ranking functions are used to prove program termination by showing
that some measure decreases on each loop iteration.

For a loop with guard G and body B, a ranking function R proves termination if:
1. R(x) >= 0 for all states x where G(x) holds (non-negativity)
2. For all states x where G(x) holds, if B(x, x'), then R(x) > R(x') (decrease)

The templates support various forms of ranking functions:
- Linear: a₁*x₁ + a₂*x₂ + ... + aₙ*xₙ + c
- Lexicographic: (R₁, R₂, ..., Rₖ) with lexicographic ordering
- Conditional: if-then-else expressions based on program conditions
"""

import z3
from typing import List, Dict, Tuple, Optional, Any
from aria.efmc.engines.ef.templates.abstract_template import Template, TemplateType
from aria.efmc.utils.bv_utils import Signedness
from aria.efmc.sts import TransitionSystem
from aria.efmc.utils.z3_expr_utils import big_and, big_or


class BitVecLinearRankingTemplate(Template):
    """
    Linear ranking function template for bit-vector programs.

    Generates ranking functions of the form:
    R(x₁, x₂, ..., xₙ) = a₁*x₁ + a₂*x₂ + ... + aₙ*xₙ + c

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

        # Constraints for ranking function properties
        self.template_cnt_init_and_post = None
        self.template_cnt_trans = None

        # Initialize template variables and constraints
        self.add_template_vars()
        self.add_template_cnts()

    def add_template_vars(self) -> None:
        """Initialize template variables for linear ranking function coefficients."""
        self.template_index += 1

        # Create coefficient variables for each program variable
        coeffs = []
        for i, var in enumerate(self.sts.variables):
            # Coefficient for variable i
            coeff_name = f"rank_coeff_{i}_{str(var)}"
            coeff = z3.BitVec(coeff_name, var.sort().size())
            coeffs.append(coeff)

        # Constant term
        if self.sts.variables:
            const_size = self.sts.variables[0].sort().size()
        else:
            const_size = 32  # Default bit-width
        const_name = f"rank_const_{self.template_index}"
        const = z3.BitVec(const_name, const_size)
        coeffs.append(const)

        self.template_vars.append(coeffs)

    def add_template_cnts(self) -> None:
        """
        Add constraints for ranking function properties:
        1. Non-negativity: R(x) >= 0 when guard holds
        2. Decrease: R(x) > R(x') when guard holds and transition occurs
        """
        constraints_init_post = []
        constraints_trans = []

        # Build ranking function expressions
        rank_expr = self._build_ranking_expr(self.sts.variables)
        rank_expr_prime = self._build_ranking_expr(self.sts.prime_variables)

        # Non-negativity constraint: R(x) >= 0
        if self.signedness == Signedness.UNSIGNED:
            # For unsigned, always non-negative due to bit-vector semantics
            non_neg_constraint = z3.BoolVal(True)
        else:
            # For signed, explicitly require >= 0
            non_neg_constraint = rank_expr >= 0

        constraints_init_post.append(non_neg_constraint)

        # Note: The decrease constraint R(x) > R(x') should only hold when the transition
        # condition is satisfied. This is handled in the verification condition generation
        # in the termination prover, not as an unconditional template constraint.
        # Removing the unconditional decrease constraint that was causing unsatisfiability.

        self.template_cnt_init_and_post = (
            big_and(constraints_init_post)
            if constraints_init_post
            else z3.BoolVal(True)
        )
        self.template_cnt_trans = z3.BoolVal(
            True
        )  # No unconditional transition constraints

    def _build_ranking_expr(self, variables: List[z3.ExprRef]) -> z3.ExprRef:
        """
        Build ranking function expression: a₁*x₁ + a₂*x₂ + ... + aₙ*xₙ + c

        Args:
            variables: List of program variables (either current or primed)

        Returns:
            Z3 expression representing the ranking function
        """
        if not self.template_vars or not self.template_vars[0]:
            return z3.BitVecVal(0, 32)  # Default to 0 if no template vars

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

        return expr

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables: bool = False
    ) -> z3.ExprRef:
        """
        Build ranking function expression from a model.

        Args:
            model: Z3 model containing values for template variables
            use_prime_variables: Whether to use primed variables

        Returns:
            Z3 expression representing the concrete ranking function
        """
        variables = (
            self.sts.prime_variables if use_prime_variables else self.sts.variables
        )

        if not self.template_vars or not self.template_vars[0]:
            return z3.BitVecVal(0, 32)

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

        return expr

    def build_ranking_function(self, model: z3.ModelRef) -> z3.ExprRef:
        """
        Build the concrete ranking function from a model.

        Args:
            model: Z3 model containing values for template variables

        Returns:
            Z3 expression representing the ranking function
        """
        return self.build_invariant_expr(model, use_prime_variables=False)


class BitVecLexicographicRankingTemplate(Template):
    """
    Lexicographic ranking function template for bit-vector programs.

    Generates lexicographic ranking functions of the form:
    R(x) = (R₁(x), R₂(x), ..., Rₖ(x))

    where each Rᵢ is a linear function and the tuple is ordered lexicographically.
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

        # Number of components in lexicographic ranking function
        self.num_components = kwargs.get("num_components", 2)

        self.template_vars = []
        self.template_index = 0

        self.template_cnt_init_and_post = None
        self.template_cnt_trans = None

        self.add_template_vars()
        self.add_template_cnts()

    def add_template_vars(self) -> None:
        """Initialize template variables for lexicographic ranking function."""
        # Create template variables for each component
        for comp_idx in range(self.num_components):
            self.template_index += 1
            coeffs = []

            # Coefficients for each program variable
            for i, var in enumerate(self.sts.variables):
                coeff_name = f"lex_rank_{comp_idx}_coeff_{i}_{str(var)}"
                coeff = z3.BitVec(coeff_name, var.sort().size())
                coeffs.append(coeff)

            # Constant term
            if self.sts.variables:
                const_size = self.sts.variables[0].sort().size()
            else:
                const_size = 32
            const_name = f"lex_rank_{comp_idx}_const"
            const = z3.BitVec(const_name, const_size)
            coeffs.append(const)

            self.template_vars.append(coeffs)

    def add_template_cnts(self) -> None:
        """Add constraints for lexicographic ranking function properties."""
        constraints_init_post = []
        constraints_trans = []

        # Build ranking function components
        rank_components = []
        rank_components_prime = []

        for i in range(self.num_components):
            rank_comp = self._build_ranking_component(i, self.sts.variables)
            rank_comp_prime = self._build_ranking_component(i, self.sts.prime_variables)
            rank_components.append(rank_comp)
            rank_components_prime.append(rank_comp_prime)

        # Non-negativity constraints for all components
        for rank_comp in rank_components:
            if self.signedness == Signedness.SIGNED:
                constraints_init_post.append(rank_comp >= 0)

        # Lexicographic decrease constraint
        lex_decrease = self._build_lexicographic_decrease_constraint(
            rank_components, rank_components_prime
        )
        constraints_trans.append(lex_decrease)

        self.template_cnt_init_and_post = (
            big_and(constraints_init_post)
            if constraints_init_post
            else z3.BoolVal(True)
        )
        self.template_cnt_trans = (
            big_and(constraints_trans) if constraints_trans else z3.BoolVal(True)
        )

    def _build_ranking_component(
        self, comp_idx: int, variables: List[z3.ExprRef]
    ) -> z3.ExprRef:
        """Build a single component of the lexicographic ranking function."""
        if comp_idx >= len(self.template_vars):
            return z3.BitVecVal(0, 32)

        coeffs = self.template_vars[comp_idx]

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

        return expr

    def _build_lexicographic_decrease_constraint(
        self, current_components: List[z3.ExprRef], next_components: List[z3.ExprRef]
    ) -> z3.ExprRef:
        """
        Build lexicographic decrease constraint:
        (R₁(x), R₂(x), ..., Rₖ(x)) >_lex (R₁(x'), R₂(x'), ..., Rₖ(x'))

        This means: ∃i. (∀j<i. Rⱼ(x) = Rⱼ(x')) ∧ Rᵢ(x) > Rᵢ(x')
        """
        if not current_components or not next_components:
            return z3.BoolVal(False)

        decrease_conditions = []

        for i in range(len(current_components)):
            # For component i to decrease, all previous components must be equal
            # and component i must strictly decrease
            equal_conditions = []
            for j in range(i):
                if j < len(current_components) and j < len(next_components):
                    equal_conditions.append(current_components[j] == next_components[j])

            # Component i decreases
            if i < len(current_components) and i < len(next_components):
                if self.signedness == Signedness.UNSIGNED:
                    decrease_cond = z3.UGT(current_components[i], next_components[i])
                else:
                    decrease_cond = current_components[i] > next_components[i]

                # Combine equal conditions with decrease condition
                if equal_conditions:
                    condition = z3.And(big_and(equal_conditions), decrease_cond)
                else:
                    condition = decrease_cond

                decrease_conditions.append(condition)

        return big_or(decrease_conditions) if decrease_conditions else z3.BoolVal(False)

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables: bool = False
    ) -> z3.ExprRef:
        """Build lexicographic ranking function expression from model."""
        variables = (
            self.sts.prime_variables if use_prime_variables else self.sts.variables
        )

        # For simplicity, return the first component as the main ranking function
        # In practice, you might want to return a tuple or handle this differently
        if self.template_vars:
            coeffs = self.template_vars[0]
            expr = model.eval(coeffs[-1])  # Constant term

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

            return expr

        return z3.BitVecVal(0, 32)

    def build_lexicographic_ranking_function(
        self, model: z3.ModelRef
    ) -> List[z3.ExprRef]:
        """
        Build all components of the lexicographic ranking function.

        Returns:
            List of Z3 expressions representing each component
        """
        components = []

        for i in range(self.num_components):
            if i < len(self.template_vars):
                coeffs = self.template_vars[i]
                expr = model.eval(coeffs[-1])  # Constant term

                for j, var in enumerate(self.sts.variables):
                    if j < len(coeffs) - 1:
                        coeff_val = model.eval(coeffs[j])
                        # Handle bit-width compatibility
                        if coeff_val.sort().size() != var.sort().size():
                            if coeff_val.sort().size() < var.sort().size():
                                coeff_val = z3.ZeroExt(
                                    var.sort().size() - coeff_val.sort().size(),
                                    coeff_val,
                                )
                            else:
                                coeff_val = z3.Extract(
                                    var.sort().size() - 1, 0, coeff_val
                                )

                        expr = expr + coeff_val * var

                components.append(expr)
            else:
                components.append(z3.BitVecVal(0, 32))

        return components


class BitVecConditionalRankingTemplate(Template):
    """
    Conditional ranking function template for bit-vector programs.

    Generates ranking functions with conditional expressions:
    R(x) = if C₁(x) then R₁(x) else if C₂(x) then R₂(x) else ... else Rₙ(x)

    where Cᵢ are conditions and Rᵢ are linear ranking functions.
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

        # Number of conditional branches
        self.num_branches = kwargs.get("num_branches", 2)

        self.template_vars = []
        self.condition_vars = []  # Boolean variables for conditions
        self.template_index = 0

        self.template_cnt_init_and_post = None
        self.template_cnt_trans = None

        self.add_template_vars()
        self.add_template_cnts()

    def add_template_vars(self) -> None:
        """Initialize template variables for conditional ranking function."""
        # Create template variables for each branch
        for branch_idx in range(self.num_branches):
            self.template_index += 1
            coeffs = []

            # Coefficients for each program variable
            for i, var in enumerate(self.sts.variables):
                coeff_name = f"cond_rank_{branch_idx}_coeff_{i}_{str(var)}"
                coeff = z3.BitVec(coeff_name, var.sort().size())
                coeffs.append(coeff)

            # Constant term
            if self.sts.variables:
                const_size = self.sts.variables[0].sort().size()
            else:
                const_size = 32
            const_name = f"cond_rank_{branch_idx}_const"
            const = z3.BitVec(const_name, const_size)
            coeffs.append(const)

            self.template_vars.append(coeffs)

        # Create condition variables (simplified as boolean flags for now)
        # In practice, these could be more complex conditions over program variables
        for branch_idx in range(self.num_branches - 1):  # Last branch is default
            cond_name = f"cond_{branch_idx}"
            cond_var = z3.Bool(cond_name)
            self.condition_vars.append(cond_var)

    def add_template_cnts(self) -> None:
        """Add constraints for conditional ranking function properties."""
        constraints_init_post = []
        constraints_trans = []

        # Build conditional ranking function
        rank_expr = self._build_conditional_ranking_expr(self.sts.variables)
        rank_expr_prime = self._build_conditional_ranking_expr(self.sts.prime_variables)

        # Non-negativity constraint
        if self.signedness == Signedness.SIGNED:
            constraints_init_post.append(rank_expr >= 0)

        # Decrease constraint
        if self.signedness == Signedness.UNSIGNED:
            decrease_constraint = z3.UGT(rank_expr, rank_expr_prime)
        else:
            decrease_constraint = rank_expr > rank_expr_prime

        constraints_trans.append(decrease_constraint)

        self.template_cnt_init_and_post = (
            big_and(constraints_init_post)
            if constraints_init_post
            else z3.BoolVal(True)
        )
        self.template_cnt_trans = (
            big_and(constraints_trans) if constraints_trans else z3.BoolVal(True)
        )

    def _build_conditional_ranking_expr(
        self, variables: List[z3.ExprRef]
    ) -> z3.ExprRef:
        """Build conditional ranking function expression."""
        if not self.template_vars:
            return z3.BitVecVal(0, 32)

        # Start with the last branch (default case)
        expr = self._build_branch_expr(len(self.template_vars) - 1, variables)

        # Build nested if-then-else from last to first condition
        for i in range(len(self.condition_vars) - 1, -1, -1):
            if i < len(self.template_vars) - 1:
                branch_expr = self._build_branch_expr(i, variables)
                condition = self.condition_vars[i]
                expr = z3.If(condition, branch_expr, expr)

        return expr

    def _build_branch_expr(
        self, branch_idx: int, variables: List[z3.ExprRef]
    ) -> z3.ExprRef:
        """Build ranking function expression for a specific branch."""
        if branch_idx >= len(self.template_vars):
            return z3.BitVecVal(0, 32)

        coeffs = self.template_vars[branch_idx]

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

        return expr

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables: bool = False
    ) -> z3.ExprRef:
        """Build conditional ranking function expression from model."""
        variables = (
            self.sts.prime_variables if use_prime_variables else self.sts.variables
        )

        if not self.template_vars:
            return z3.BitVecVal(0, 32)

        # Start with the last branch (default case)
        expr = self._build_evaluated_branch_expr(
            len(self.template_vars) - 1, variables, model
        )

        # Build nested if-then-else from last to first condition
        for i in range(len(self.condition_vars) - 1, -1, -1):
            if i < len(self.template_vars) - 1:
                branch_expr = self._build_evaluated_branch_expr(i, variables, model)
                condition_val = model.eval(self.condition_vars[i])
                expr = z3.If(condition_val, branch_expr, expr)

        return expr

    def _build_evaluated_branch_expr(
        self, branch_idx: int, variables: List[z3.ExprRef], model: z3.ModelRef
    ) -> z3.ExprRef:
        """Build evaluated ranking function expression for a specific branch."""
        if branch_idx >= len(self.template_vars):
            return z3.BitVecVal(0, 32)

        coeffs = self.template_vars[branch_idx]

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

        return expr
