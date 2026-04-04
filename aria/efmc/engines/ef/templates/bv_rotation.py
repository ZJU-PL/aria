# equivalence rule R: rotl(x, k) == x (mod r)
# algorighm to veryfy R
# let s1, s2 be to bitvector(m bits)
# s = concat(s1, s2)
# if s2 is subvector of s
#      return true
# else
#       return false
# (O(m))
# template var -> X x0 + x0
# Inv template:  x and x' is substr of X
from typing import List, Optional

from aria.efmc.engines.ef.templates.abstract_template import *
from aria.utils.bv_utils import Signedness
from aria.efmc.sts import TransitionSystem
from aria.efmc.utils import big_and


class BitVecRotationTemplate(Template):

    def __init__(self, sts: TransitionSystem, **kwargs):

        self.template_type = TemplateType.BV_ROTATION

        signedness_val = self._init_signedness_from_sts(sts)
        if signedness_val is not None:
            self.signedness = signedness_val

        self.sts = sts
        self.arity = len(self.sts.variables)

        self.template_vars = []
        self.template_index = 0  # number of templates
        self.add_template_vars()
        # pre compute to reduce redundant calling
        self.template_cnt_init_and_post = None
        self.template_cnt_trans = None
        self.add_template_cnts()

    def add_template_vars(self):
        """Add several groups of template variables"""
        # analysis init sts first, to denote the init values of each variables, so we can build the tamplate var
        for var in self.sts.variables:
            self.template_vars.append(z3.BitVec(str(var) + "_k", var.sort().size()))

    def get_additional_cnts_for_template_vars(self):
        extra_res = []
        for i in range(self.arity):
            # when var_k is 0, it means even with no inv, we can verify the program
            extra_res.append(
                self.template_vars[i] <= self.template_vars[i].sort().size()
            )
            extra_res.append(self.template_vars[i] >= 0)
        return big_and(extra_res)

    def add_template_cnts(self) -> None:
        """Add cnts for init and post assertions (a trick)"""
        cnts = []
        cnts_prime = []
        for i in range(self.arity):
            var = self.sts.variables[i]
            var_prime = self.sts.prime_variables[i]
            # rotl(x, k) == x
            cnts.append(
                (var << self.template_vars[i])
                | var >> (var.sort().size() - self.template_vars[i])
                == var
            )
            cnts_prime.append(
                (var_prime << self.template_vars[i])
                | (var_prime >> ((var.sort().size() - self.template_vars[i])))
                == var_prime
            )

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
        for i in range(self.arity):
            var = self.sts.variables[i]
            constraints.append(
                (
                    (var << model.eval(self.template_vars[i], True))
                    | (
                        var
                        >> (var.sort().size() - model.eval(self.template_vars[i], True))
                    )
                )
                == var
            )
        return z3.And(constraints)


class DisjunctiveBitVecRotationTemplate(Template):
    def __init__(self, sts: TransitionSystem, **kwargs):

        self.template_type = TemplateType.BV_DISJUNCTIVE_ROTATION

        # TODO: infer the signedness of variables? (or design a domain that is
        #  signedness-irrelevant. Currently, we use unsigned by default
        signedness_val = self._init_signedness_from_sts(sts)
        if signedness_val is not None:
            self.signedness = signedness_val

        self.sts = sts
        self.arity = len(self.sts.variables)
        self.bit_len = self.sts.variables[0].sort().size()
        self.num_disjunctions = kwargs.get("num_disjunctions", 2)

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
                tvar = z3.BitVec("k{0}_{1}".format(i, str(var)), var.sort().size())
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
            for j in range(self.arity):
                var = self.sts.variables[j]
                var_prime = self.sts.prime_variables[j]
                var_k = vars_for_ith_disjunct[j]
                cnt_init_post.append(
                    (var << var_k) | var >> (self.bit_len - var_k) == var
                )
                cnt_trans.append(
                    (var_prime << var_k) | var_prime >> (self.bit_len - var_k)
                    == var_prime
                )

            # the ith cnt AND
            ith_cnt_init_post = z3.simplify(big_and(cnt_init_post))
            ith_cnt_trans = z3.simplify(big_and(cnt_trans))

            cnt_init_and_post_dis.append(ith_cnt_init_post)
            cnt_trans_dis.append(ith_cnt_trans)

        self.template_cnt_init_and_post = z3.Or(cnt_init_and_post_dis)
        self.template_cnt_trans = z3.Or(cnt_trans_dis)

    def get_additional_cnts_for_template_vars(self) -> z3.ExprRef:
        """This implementation does not need additional ones"""
        return z3.BoolVal(True)

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables: bool = False
    ) -> z3.ExprRef:
        # TODO: check for correctness
        cnts_dis: List[z3.ExprRef] = []
        for i in range(self.num_disjunctions):
            # Invariant: len(self.template_vars) = self.num_disjunctions
            vars_for_ith_disjunct = self.template_vars[i]
            cnts: List[z3.ExprRef] = []  # constraints for one disjunct

            sum = z3.BitVecVal(0, self.bit_len)

            for j in range(self.arity):
                var = self._get_variable(j, use_prime_variables)

                tvar_k = vars_for_ith_disjunct[j]
                cnts.append(
                    (var << model[tvar_k]) | var >> (self.bit_len - model[tvar_k])
                    == var
                )

            cnts_dis.append(z3.And(cnts))

        return z3.Or(cnts_dis)

    def build_invariant(self, model: z3.ModelRef) -> z3.ExprRef:
        """Build an invariant from a model.

        Args:
            model: Z3 model containing values for template variables

        Returns:
            Z3 expression representing the invariant
        """
        return self.build_invariant_expr(model)
