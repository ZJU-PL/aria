from typing import List, Optional

from aria.efmc.engines.ef.templates.abstract_template import *
from aria.utils.z3.bv import Signedness
from aria.efmc.sts import TransitionSystem
from aria.efmc.utils import big_and


class BitVecXorParityTemplate(Template):

    def __init__(self, sts: TransitionSystem, **kwargs):

        self.template_type = TemplateType.BV_XOR_PARITY

        signedness_val = self._init_signedness_from_sts(sts)
        if signedness_val is not None:
            self.signedness = signedness_val

        self.sts = sts
        self.bit_len = self.sts.variables[0].sort().size()
        self.arity = len(self.sts.variables)
        # list of names of variables constrained by template
        self.template_vars = []
        self.template_index = 0  # number of templates
        self.add_template_vars()
        # the constant variable
        self.constant = z3.BitVec("constant", self.sts.variables[0].sort().size())
        # pre compute to reduce redundant calling
        self.template_cnt_init_and_post = None
        self.template_cnt_trans = None

        self.add_template_cnts()

    def add_template_vars(self):
        """Add several groups of template variables"""
        # analysis init sts first, to denote the init values of each variables, so we can build the tamplate var
        for var in self.sts.variables:
            self.template_vars.append(z3.BitVec(str(var) + "_a", self.bit_len))

    def get_additional_cnts_for_template_vars(self):
        extra_res = []
        for i in self.arity:
            # a is 1 or 0
            extra_res.append(
                z3.Or(
                    self.template_vars[i] == z3.BitVecVal(0, self.bit_len),
                    self.template_vars[i] == z3.BitVecVal(1, self.bit_len),
                )
            )
        return big_and(extra_res)

    def add_template_cnts(self) -> None:
        """Add cnts for init and post assertions (a trick)"""
        cnts = []
        cnts_prime = []

        sum = z3.BitVecVal(0, self.bit_len)
        sum_prime = z3.BitVecVal(0, self.bit_len)
        for i in range(self.arity):
            var = self.sts.variables[i]
            var_prime = self.sts.prime_variables[i]
            # var need to be the one that we want template to constraint it
            sum = sum ^ (self.template_vars[i] * var)
            sum_prime = sum_prime ^ (self.template_vars[i] * var_prime)

        cnts.append(sum == self.constant)
        cnts_prime.append(sum_prime == self.constant)
        self.template_cnt_init_and_post = z3.simplify(big_and(cnts))
        self.template_cnt_trans = z3.simplify(big_and(cnts_prime))

    def build_invariant(self, model: z3.ModelRef) -> z3.ExprRef:
        """Build an invariant from a model.

        Args:
            model: Z3 model containing values for template variables

        Returns:
            Z3 expression representing the invariant
        """
        return self.build_invariant_expr(model)

    def build_invariant_expr(self, model: z3.ModelRef, use_prime_variables=False):
        """Build an invariant from a model (fixing the values of the template vars)"""
        constraints = []
        sum = z3.BitVecVal(0, self.bit_len)
        for i in range(self.arity):
            var = self.sts.variables[i]
            sum = sum ^ (model.eval(self.template_vars[i], True) * var)

        constraints.append(sum == model.eval(self.constant))
        return z3.And(constraints)


class DisjunctiveBitVecXorParityTemplate(Template):
    def __init__(self, sts: TransitionSystem, **kwargs):

        self.template_type = TemplateType.BV_DISJUNCTIVE_XOR_PARITY

        # TODO: infer the signedness of variables? (or design a domain that is
        #  signedness-irrelevant. Currently, we use unsigned by default
        signedness_val = self._init_signedness_from_sts(sts)
        if signedness_val is not None:
            self.signedness = signedness_val

        self.sts = sts
        self.arity = len(self.sts.variables)
        self.bit_len = self.sts.variables[0].sort().size()
        self.num_disjunctions = kwargs.get("num_disjunctions", 3)
        print(self.num_disjunctions)
        self.constants = [
            z3.BitVec("c_{}".format(i), self.bit_len)
            for i in range(self.num_disjunctions)
        ]

        self.template_vars: List[List[z3.ExprRef]] = []  # vector of vector
        self.template_index = 0  # number of templates

        self.add_template_vars()

        # pre compute to reduce redundant calling
        self.template_cnt_init_and_post: Optional[z3.ExprRef] = None
        self.template_cnt_trans: Optional[z3.ExprRef] = None
        self.add_template_cnts()

    def add_template_vars(self):
        for i in range(self.num_disjunctions):
            vars_for_dis: List[z3.ExprRef] = []
            """Add several groups of template variables,(m and p)"""
            for var in self.sts.variables:
                tvar = z3.BitVec("a{0}_{1}".format(i, str(var)), var.sort().size())
                vars_for_dis.append(tvar)
            self.template_vars.append(vars_for_dis)

    def get_additional_cnts_for_template_vars(self) -> z3.ExprRef:
        """This implementation does not need additional ones"""
        return z3.BoolVal(True)

    def add_template_cnts(self):
        cnt_init_and_post_dis: List[z3.ExprRef] = []
        cnt_trans_dis: List[z3.ExprRef] = []
        for i in range(self.num_disjunctions):
            # Invariant: len(self.template_vars) = self.num_disjunctions
            vars_for_ith_disjunct = self.template_vars[i]
            cnt_init_post: List[z3.ExprRef] = []  # For sts.variables
            cnt_trans: List[z3.ExprRef] = []  # For sts.prime_variables
            # init the sum for each dis
            sum = z3.BitVecVal(0, self.bit_len)
            sum_prime = z3.BitVecVal(0, self.bit_len)
            for j in range(self.arity):
                var = self.sts.variables[j]
                var_prime = self.sts.prime_variables[j]
                var_a = vars_for_ith_disjunct[j]
                sum = sum ^ (var_a * var)
                sum_prime = sum_prime ^ (var_a * var_prime)

            cnt_init_post = sum == self.constants[i]
            cnt_trans = sum_prime == self.constants[i]

            ith_cnt_init_post = z3.simplify(cnt_init_post)
            ith_cnt_trans = z3.simplify(cnt_trans)

            cnt_init_and_post_dis.append(ith_cnt_init_post)
            cnt_trans_dis.append(ith_cnt_trans)

        self.template_cnt_init_and_post = z3.Or(cnt_init_and_post_dis)
        self.template_cnt_trans = z3.Or(cnt_trans_dis)

    def get_additional_cnts_for_template_vars(self) -> z3.ExprRef:
        """This implementation does not need additional ones"""
        extra_res = []
        for j in range(self.num_disjunctions):
            for i in range(self.arity):
                # a is 1 or 0
                extra_res.append(
                    z3.Or(
                        self.template_vars[j][i] == z3.BitVecVal(0, self.bit_len),
                        self.template_vars[j][i] == z3.BitVecVal(1, self.bit_len),
                    )
                )
        return big_and(extra_res)

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables: bool = False
    ) -> z3.ExprRef:
        # TODO: check for correctness
        cnts_dis: List[z3.ExprRef] = []
        for i in range(self.num_disjunctions):
            # Invariant: len(self.template_vars) = self.num_disjunctions
            vars_for_ith_disjunct = self.template_vars[i]

            sum = z3.BitVecVal(0, self.bit_len)

            for j in range(self.arity):
                var = self._get_variable(j, use_prime_variables)

                tvar_a = vars_for_ith_disjunct[j]
                sum = sum ^ model[tvar_a] * var

            cnts_dis.append(sum == model[self.constants[i]])

        return z3.Or(cnts_dis)

    def build_invariant(self, model: z3.ModelRef) -> z3.ExprRef:
        """Build an invariant from a model.

        Args:
            model: Z3 model containing values for template variables

        Returns:
            Z3 expression representing the invariant
        """
        return self.build_invariant_expr(model)
