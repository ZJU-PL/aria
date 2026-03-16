"""
FIXME: this file is not used yet (by LLM, to check)
Non-linear template over integer or real variables

E.g., polynomial inequalities, trigonometric inequalities, etc.
"""

import logging
import z3
from aria.efmc.sts import TransitionSystem
from aria.efmc.engines.ef.templates.abstract_template import TemplateType, Template

logger = logging.getLogger(__name__)


class PolynomialTemplate(Template):
    """
    Non-linear template over integer or real variables
    """

    def __init__(self, system: TransitionSystem, **kwargs):
        super().__init__(system, **kwargs)
        self.template_type = TemplateType.POLYNOMIAL
        self.sts = system
        self.degree = kwargs.get("degree", 2)

    def add_template_vars(self, **kwargs):
        """
        Add template variables to the template.
        """
        # Initialize template variables
        self.template_vars = []  # vector of vector
        self.template_index = 0  # number of templates

        # Check if the system uses real variables
        if self.sts.has_real:
            self.use_real = True
        else:
            self.use_real = False

        self.arity = len(self.sts.variables)

        # Number of polynomial terms in the template
        self.num_templates = kwargs.get("num_templates", 1)

        # Generate template variables for polynomial terms
        for i in range(self.num_templates):
            template = []
            # Constant term
            if self.use_real:
                template.append(z3.Real(f"p{i}_0"))
            else:
                template.append(z3.Int(f"p{i}_0"))

            # Linear terms
            for j, var in enumerate(self.sts.variables.values()):
                if self.use_real:
                    template.append(z3.Real(f"p{i}_{j + 1}"))
                else:
                    template.append(z3.Int(f"p{i}_{j + 1}"))

            # Higher degree terms (for degree > 1)
            if self.degree > 1:
                term_idx = self.arity + 1
                for d in range(2, self.degree + 1):
                    for combo in self._get_variable_combinations(d):
                        if self.use_real:
                            template.append(z3.Real(f"p{i}_{term_idx}"))
                        else:
                            template.append(z3.Int(f"p{i}_{term_idx}"))
                        term_idx += 1

            self.template_vars.append(template)
            self.template_index += 1

        # Pre-compute to reduce redundant calling
        self.template_cnt_init_and_post = None
        self.template_cnt_trans = None

    def add_template_cnts(self):
        """
        Add template constraints to the template.
        """
        # Initialize constraints
        cnts = []

        # Add constraints for each template
        for template in self.template_vars:
            # Create polynomial expression
            poly = z3.Sum(
                [
                    coeff
                    * z3.Product(
                        [var**exp for var, exp in zip(self.sts.variables, powers)]
                    )
                    for coeff, powers in zip(template[: self.arity], self.powers)
                ]
            )

            # Add inequality constraint
            cnts.append(poly >= 0)

        # Combine constraints
        self.template_cnt_init_and_post = z3.And(cnts)
        self.template_cnt_trans = z3.And(cnts)

    def build_invariant_expr(self, model: z3.ModelRef, use_prime_variables=False):
        """
        Build the invariant expression for the template.
        """
        # Initialize list to hold all polynomial constraints
        constraints = []

        # Get the appropriate set of variables based on whether we're using primed variables
        variables = (
            self.sts.prime_variables if use_prime_variables else self.sts.variables
        )

        # For each template (polynomial inequality)
        for template in self.template_vars:
            # Initialize polynomial terms
            terms = []

            # Add constant term
            constant_term = (
                z3.RealVal(model.eval(template[0]).as_decimal(10))
                if self.use_real
                else z3.IntVal(model.eval(template[0]).as_string())
            )
            terms.append(constant_term)

            # Add linear terms
            for j, var in enumerate(variables.values()):
                coeff = (
                    z3.RealVal(model.eval(template[j + 1]).as_decimal(10))
                    if self.use_real
                    else z3.IntVal(model.eval(template[j + 1]).as_string())
                )
                if coeff != 0:  # Skip zero coefficients
                    terms.append(coeff * var)

            # Add higher degree terms if degree > 1
            if self.degree > 1:
                term_idx = self.arity + 1
                for d in range(2, self.degree + 1):
                    for combo in self._get_variable_combinations(d):
                        coeff = (
                            z3.RealVal(model.eval(template[term_idx]).as_decimal(10))
                            if self.use_real
                            else z3.IntVal(model.eval(template[term_idx]).as_string())
                        )
                        if coeff != 0:  # Skip zero coefficients
                            # Create the monomial term
                            monomial = z3.Product(
                                [
                                    var**exp
                                    for var, exp in zip(variables.values(), combo)
                                ]
                            )
                            terms.append(coeff * monomial)
                        term_idx += 1

            # Sum all terms to form the polynomial
            poly = z3.Sum(terms)

            # Add the inequality constraint
            constraints.append(poly >= 0)

        # Combine all constraints with AND
        return z3.And(constraints) if constraints else z3.BoolVal(True)
