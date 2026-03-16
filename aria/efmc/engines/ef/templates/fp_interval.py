"""Floating-point interval template for QF_FP variables

This template provides interval-based invariant inference for floating-point variables,
supporting both single and double precision floating-point arithmetic.
"""

import logging
from aria.efmc.engines.ef.templates.abstract_template import Template, TemplateType
from aria.efmc.sts import TransitionSystem
from aria.efmc.utils import big_and
import z3

logger = logging.getLogger(__name__)


class FPIntervalTemplate(Template):
    """Floating-point interval domain template.

    For each floating-point variable x, we maintain constraints of the form:
        x >= lower_bound  (or lower_bound is -infinity)
        x <= upper_bound  (or upper_bound is +infinity)

    Template variables represent the bounds and whether they are active.
    """

    def __init__(self, sts: TransitionSystem, **kwargs):
        if not sts.has_fp:
            raise ValueError("FPIntervalTemplate requires floating-point variables")

        self.template_type = TemplateType.FP_INTERVAL
        self.sts = sts
        self.arity = len(self.sts.variables)

        # Get floating-point sort from first variable
        self.fp_sort = self.sts.variables[0].sort()
        self.ebits = self.fp_sort.ebits()
        self.sbits = self.fp_sort.sbits()

        # Template variables: for each variable, we have lower/upper bounds and enable flags
        self.template_vars = []
        self.template_index = 0

        self.add_template_vars()

        # Pre-computed constraints
        self.template_cnt_init_and_post = None
        self.template_cnt_trans = None
        self.add_template_cnts()

    def add_template_vars(self):
        """Add template variables for floating-point intervals.

        For each variable x, we add:
        - lower_x: lower bound (FP value)
        - upper_x: upper bound (FP value)
        - enable_lower_x: boolean flag to enable lower bound
        - enable_upper_x: boolean flag to enable upper bound
        """
        for i, var in enumerate(self.sts.variables):
            var_name = str(var)

            # Create floating-point bounds with same sort as variables
            lower_bound = z3.FP(f"lower_{var_name}", self.fp_sort)
            upper_bound = z3.FP(f"upper_{var_name}", self.fp_sort)

            # Create boolean flags to enable/disable bounds
            enable_lower = z3.Bool(f"enable_lower_{var_name}")
            enable_upper = z3.Bool(f"enable_upper_{var_name}")

            self.template_vars.append(
                [lower_bound, upper_bound, enable_lower, enable_upper]
            )
            self.template_index += 1

    def add_template_cnts(self):
        """Add template constraints for initialization/post and transition."""
        cnts_init_post = []
        cnts_trans = []

        for i, var in enumerate(self.sts.variables):
            lower_bound, upper_bound, enable_lower, enable_upper = self.template_vars[i]

            # Constraints for current state variables
            lower_constraint = z3.Implies(enable_lower, z3.fpGEQ(var, lower_bound))
            upper_constraint = z3.Implies(enable_upper, z3.fpLEQ(var, upper_bound))
            cnts_init_post.extend([lower_constraint, upper_constraint])

            # Constraints for next state variables (primed)
            prime_var = self.sts.prime_variables[i]
            lower_constraint_prime = z3.Implies(
                enable_lower, z3.fpGEQ(prime_var, lower_bound)
            )
            upper_constraint_prime = z3.Implies(
                enable_upper, z3.fpLEQ(prime_var, upper_bound)
            )
            cnts_trans.extend([lower_constraint_prime, upper_constraint_prime])

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

        for i, var in enumerate(variables):
            lower_bound, upper_bound, enable_lower, enable_upper = self.template_vars[i]

            # Check if bounds are enabled in the model
            enable_lower_val = model.eval(enable_lower, model_completion=True)
            enable_upper_val = model.eval(enable_upper, model_completion=True)

            if z3.is_true(enable_lower_val):
                lower_val = model.eval(lower_bound, model_completion=True)
                constraints.append(z3.fpGEQ(var, lower_val))

            if z3.is_true(enable_upper_val):
                upper_val = model.eval(upper_bound, model_completion=True)
                constraints.append(z3.fpLEQ(var, upper_val))

        return big_and(constraints) if constraints else z3.BoolVal(True)

    def build_invariant(self, model: z3.ModelRef) -> z3.ExprRef:
        """Build invariant using current state variables."""
        return self.build_invariant_expr(model, use_prime_variables=False)

    def get_additional_cnts_for_template_vars(self) -> z3.ExprRef:
        """Additional constraints for template variables."""
        constraints = []

        for i in range(len(self.sts.variables)):
            lower_bound, upper_bound, enable_lower, enable_upper = self.template_vars[i]

            # If both bounds are enabled, lower must be <= upper
            both_enabled = z3.And(enable_lower, enable_upper)
            bounds_ordered = z3.fpLEQ(lower_bound, upper_bound)
            constraints.append(z3.Implies(both_enabled, bounds_ordered))

            # Bounds should be normal (not NaN or infinity for practical intervals)
            constraints.append(z3.Implies(enable_lower, z3.fpIsNormal(lower_bound)))
            constraints.append(z3.Implies(enable_upper, z3.fpIsNormal(upper_bound)))

        return big_and(constraints) if constraints else z3.BoolVal(True)
