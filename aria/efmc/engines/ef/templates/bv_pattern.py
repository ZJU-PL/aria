"""pattern template over bit-vector variables"""

from typing import List, Optional
from aria.efmc.engines.ef.templates.abstract_template import *
from aria.utils.bv_utils import Signedness
from aria.efmc.sts import TransitionSystem
from aria.efmc.utils import big_and

"pattern : (m, p)"
"∀𝑖 ∈ {0, 1, . . . , w - 1} : 𝑚[i] = 1 =⇒ x [i] = p [i mod |p|]"
"extend m and p so that len(p) = len(m) = len(x)"
"here just set len(m) = len(p) to len(x)"
"add cnt :m & (x^p) = bv0"


class BitVecPatternTemplate(Template):
    def __init__(self, sts):
        self.template_type = TemplateType.BV_PATTERN

        signedness_val = self._init_signedness_from_sts(sts)
        if signedness_val is not None:
            self.signedness = signedness_val

        self.sts = sts
        # number of bits in vector
        self.bit_len = self.sts.variables[0].sort().size()
        self.zero = z3.BitVecVal(0, self.bit_len)
        self.arity = len(self.sts.variables)

        self.template_vars = []
        # number of templates
        self.template_index = 0
        self.add_template_vars()

        # pre compute to reduce redundant calling
        self.template_cnt_init_and_post: Optional[z3.ExprRef] = None
        self.template_cnt_trans: Optional[z3.ExprRef] = None

        self.add_template_cnts()

    def add_template_vars(self) -> None:
        """Add several groups of template variables,(m and p)"""
        for var in self.sts.variables:
            tvars = [
                z3.BitVec("m_{}".format(str(var)), var.sort().size()),
                z3.BitVec("p_{}".format(str(var)), var.sort().size()),
            ]
            self.template_vars.append(tvars)

    def add_template_cnts(self) -> None:
        """Add cnts for init and post assertions (a trick)"""
        cnts = []
        cnts_prime = []

        for i in range(self.arity):
            var = self.sts.variables[i]
            var_prime = self.sts.prime_variables[i]
            var_m = self.template_vars[i][0]  # m
            var_p = self.template_vars[i][1]  # p
            cnts.append((var ^ var_p) & var_m == self.zero)
            cnts_prime.append((var_prime ^ var_p) & var_m == self.zero)

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

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables=False
    ) -> z3.ExprRef:
        """Build an invariant from a model (fixing the values of the template vars)"""
        constraints = []
        for i in range(self.arity):
            if use_prime_variables:
                var = self.sts.prime_variables[i]
            else:
                var = self.sts.variables[i]
            m = self.template_vars[i][0]
            p = self.template_vars[i][1]
            # this module is sign-irrelative
            constraints.append((var ^ model[p]) & model[m] == self.zero)
        return z3.And(constraints)


class DisjunctiveBitVecPatternTemplate(Template):
    "the disjunctive model of BitVecPatternTemplate"

    def __init__(self, sts: TransitionSystem, **kwargs):

        self.template_type = TemplateType.BV_DISJUNCTIVE_PATTERN

        # TODO: infer the signedness of variables? (or design a domain that is
        #  signedness-irrelevant. Currently, we use unsigned by default
        signedness_val = self._init_signedness_from_sts(sts)
        if signedness_val is not None:
            self.signedness = signedness_val

        self.sts = sts
        self.arity = len(self.sts.variables)
        # number of bits in vector
        self.bit_len = self.sts.variables[0].sort().size()
        self.zero = z3.BitVecVal(0, self.bit_len)

        self.num_disjunctions = kwargs.get("num_disjunctions", 2)

        self.template_vars: List[List[List[z3.ExprRef]]] = []  # vector of vector
        self.template_index = 0  # number of templates

        self.add_template_vars()

        # pre compute to reduce redundant calling
        self.template_cnt_init_and_post: Optional[z3.ExprRef] = None
        self.template_cnt_trans: Optional[z3.ExprRef] = None
        self.add_template_cnts()

    def add_template_vars(self):
        for i in range(self.num_disjunctions):
            vars_for_dis: List[List[z3.ExprRef]] = []
            """Add several groups of template variables,(m and p)"""
            for var in self.sts.variables:
                tvars = [
                    z3.BitVec("d{0}_m{1}".format(i, str(var)), var.sort().size()),
                    z3.BitVec("d{0}_p{1}".format(i, str(var)), var.sort().size()),
                ]
                vars_for_dis.append(tvars)
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
            for j in range(self.arity):
                var = self.sts.variables[j]
                var_prime = self.sts.prime_variables[j]
                var_m = vars_for_ith_disjunct[j][0]  # m
                var_p = vars_for_ith_disjunct[j][1]  # p

                cnt_init_post.append((var ^ var_p) & var_m == self.zero)
                cnt_trans.append((var_prime ^ var_p) & var_m == self.zero)

            ith_cnt_init_post = z3.simplify(big_and(cnt_init_post))
            ith_cnt_trans = z3.simplify(big_and(cnt_trans))

            cnt_init_and_post_dis.append(ith_cnt_init_post)
            cnt_trans_dis.append(ith_cnt_trans)

        self.template_cnt_init_and_post = z3.Or(cnt_init_and_post_dis)
        self.template_cnt_trans = z3.Or(cnt_trans_dis)

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables: bool = False
    ) -> z3.ExprRef:
        # TODO: check for correctness
        cnts_dis: List[z3.ExprRef] = []
        for i in range(self.num_disjunctions):
            # Invariant: len(self.template_vars) = self.num_disjunctions
            vars_for_ith_disjunct = self.template_vars[i]
            cnts: List[z3.ExprRef] = []  # constraints for one disjunct
            for j in range(self.arity):
                if use_prime_variables:
                    var = self.sts.prime_variables[j]
                else:
                    var = self.sts.variables[j]
                tvar_m = vars_for_ith_disjunct[j][0]
                tvar_p = vars_for_ith_disjunct[j][1]
                cnts.append((var ^ model[tvar_p]) & model[tvar_m] == self.zero)

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
