"""Floating-point polyhedron template for QF_FP variables

This template provides polyhedral invariant inference for floating-point variables,
supporting linear constraints over floating-point arithmetic.
"""

import logging
from aria.efmc.engines.ef.templates.abstract_template import Template, TemplateType
from aria.efmc.sts import TransitionSystem
from aria.efmc.utils import big_and
import z3

logger = logging.getLogger(__name__)


class FPPolyhedronTemplate(Template):
    """Floating-point polyhedron domain template.

    For floating-point variables [x1, x2, ..., xn], we maintain constraints of the form:
        c0 + c1*x1 + c2*x2 + ... + cn*xn >= 0

    Each linear inequality is represented by coefficients c0, c1, ..., cn and an enable flag.
    """

    def __init__(self, sts: TransitionSystem, num_constraints: int = 2, **kwargs):
        if not sts.has_fp:
            raise ValueError("FPPolyhedronTemplate requires floating-point variables")

        self.template_type = TemplateType.FP_POLYHEDRON
        self.sts = sts
        self.arity = len(self.sts.variables)
        self.num_constraints = num_constraints  # Number of linear inequalities

        # Get floating-point sort from first variable
        self.fp_sort = self.sts.variables[0].sort()
        self.ebits = self.fp_sort.ebits()
        self.sbits = self.fp_sort.sbits()

        # Template variables: coefficients for each linear inequality
        self.template_vars = []
        self.template_index = 0

        self.add_template_vars()

        # Pre-computed constraints
        self.template_cnt_init_and_post = None
        self.template_cnt_trans = None
        self.add_template_cnts()

    def add_template_vars(self):
        """Add template variables for floating-point polyhedron.

        For each linear inequality i, we add:
        - c_i_0: constant coefficient (FP value)
        - c_i_j: coefficient for variable j (FP value)
        - enable_i: boolean flag to enable constraint i
        """
        for constraint_idx in range(self.num_constraints):
            self.template_index += 1

            # Constant coefficient
            coeffs = [z3.FP(f"c{self.template_index}_0", self.fp_sort)]

            # Variable coefficients
            for var_idx in range(self.arity):
                coeff = z3.FP(f"c{self.template_index}_{var_idx+1}", self.fp_sort)
                coeffs.append(coeff)

            # Enable flag for this constraint
            enable_flag = z3.Bool(f"enable_c{self.template_index}")
            coeffs.append(enable_flag)

            self.template_vars.append(coeffs)

    def add_template_cnts(self):
        """Add template constraints for initialization/post and transition."""
        cnts_init_post = []
        cnts_trans = []

        # For each linear inequality
        for constraint_idx in range(self.num_constraints):
            coeffs = self.template_vars[constraint_idx]
            constant = coeffs[0]
            var_coeffs = coeffs[1:-1]  # Variable coefficients
            enable_flag = coeffs[-1]  # Enable flag

            # Build linear expression for current state variables
            linear_expr_current = constant
            for var_idx, var in enumerate(self.sts.variables):
                coeff = var_coeffs[var_idx]
                linear_expr_current = z3.fpAdd(
                    z3.RNE(), linear_expr_current, z3.fpMul(z3.RNE(), coeff, var)
                )

            # Build linear expression for next state variables
            linear_expr_next = constant
            for var_idx, prime_var in enumerate(self.sts.prime_variables):
                coeff = var_coeffs[var_idx]
                linear_expr_next = z3.fpAdd(
                    z3.RNE(), linear_expr_next, z3.fpMul(z3.RNE(), coeff, prime_var)
                )

            # Create constraints: linear_expr >= 0 (when enabled)
            zero = z3.FPVal(0.0, self.fp_sort)

            constraint_current = z3.Implies(
                enable_flag, z3.fpGEQ(linear_expr_current, zero)
            )
            constraint_next = z3.Implies(enable_flag, z3.fpGEQ(linear_expr_next, zero))

            cnts_init_post.append(constraint_current)
            cnts_trans.append(constraint_next)

        self.template_cnt_init_and_post = big_and(cnts_init_post)
        self.template_cnt_trans = big_and(cnts_trans)

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables: bool
    ) -> z3.ExprRef:
        """Build invariant expression from model."""
        constraints = []
        variables = (
            self.sts.prime_variables if use_prime_variables else self.sts.variables
        )
        zero = z3.FPVal(0.0, self.fp_sort)

        for constraint_idx in range(self.num_constraints):
            coeffs = self.template_vars[constraint_idx]
            constant = coeffs[0]
            var_coeffs = coeffs[1:-1]
            enable_flag = coeffs[-1]

            # Check if this constraint is enabled in the model
            enable_val = model.eval(enable_flag, model_completion=True)
            if not z3.is_true(enable_val):
                continue

            # Build linear expression with model values
            constant_val = model.eval(constant, model_completion=True)
            linear_expr = constant_val

            for var_idx, var in enumerate(variables):
                coeff_val = model.eval(var_coeffs[var_idx], model_completion=True)
                term = z3.fpMul(z3.RNE(), coeff_val, var)
                linear_expr = z3.fpAdd(z3.RNE(), linear_expr, term)

            # Add constraint: linear_expr >= 0
            constraints.append(z3.fpGEQ(linear_expr, zero))

        return big_and(constraints) if constraints else z3.BoolVal(True)

    def build_invariant(self, model: z3.ModelRef) -> z3.ExprRef:
        """Build invariant using current state variables."""
        return self.build_invariant_expr(model, use_prime_variables=False)

    def get_additional_cnts_for_template_vars(self) -> z3.ExprRef:
        """Additional constraints for template variables."""
        constraints = []

        for constraint_idx in range(self.num_constraints):
            coeffs = self.template_vars[constraint_idx]
            constant = coeffs[0]
            var_coeffs = coeffs[1:-1]
            enable_flag = coeffs[-1]

            # When constraint is enabled, coefficients should be normal (not NaN/infinity)
            constraints.append(z3.Implies(enable_flag, z3.fpIsNormal(constant)))
            for coeff in var_coeffs:
                constraints.append(z3.Implies(enable_flag, z3.fpIsNormal(coeff)))

            # At least one coefficient should be non-zero when constraint is enabled
            # (to avoid trivial 0 >= 0 constraints)
            non_zero_coeffs = []
            zero = z3.FPVal(0.0, self.fp_sort)

            non_zero_coeffs.append(z3.Not(z3.fpEQ(constant, zero)))
            for coeff in var_coeffs:
                non_zero_coeffs.append(z3.Not(z3.fpEQ(coeff, zero)))

            at_least_one_nonzero = z3.Or(non_zero_coeffs)
            constraints.append(z3.Implies(enable_flag, at_least_one_nonzero))

        return big_and(constraints) if constraints else z3.BoolVal(True)
