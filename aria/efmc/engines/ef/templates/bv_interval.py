"""Interval template over bit-vector variables"""

from typing import List, Optional

from aria.efmc.engines.ef.templates.abstract_template import *
from aria.utils.z3.bv import Signedness
from aria.efmc.sts import TransitionSystem
from aria.efmc.utils import big_and


class BitVecIntervalTemplate(Template):

    def __init__(self, sts: TransitionSystem, **kwargs):

        self.template_type = TemplateType.BV_INTERVAL

        # TODO: infer the signedness of variables? (or design a domain that is
        #  signedness-irrelevant. Currently, we use unsigned by default
        signedness_val = self._init_signedness_from_sts(sts)
        if signedness_val is not None:
            self.signedness = signedness_val

        self.sts = sts
        self.arity = len(self.sts.variables)

        # self.obj_no_overflow = kwargs.get("no_overflow", False)
        # self.obj_no_underflow = kwargs.get("no_underflow", False)

        self.template_vars: List[List[z3.ExprRef]] = []  # vector of vector
        self.template_index = 0  # number of templates

        self.add_template_vars()

        # pre compute to reduce redundant calling
        self.template_cnt_init_and_post: Optional[z3.ExprRef] = None
        self.template_cnt_trans: Optional[z3.ExprRef] = None
        self.add_template_cnts()

    def add_template_vars(self) -> None:
        """Add several groups of template variables"""
        for var in self.sts.variables:
            tvars = [
                z3.BitVec("l_{}".format(str(var)), var.sort().size()),
                z3.BitVec("u_{}".format(str(var)), var.sort().size()),
            ]
            self.template_vars.append(tvars)

    def get_additional_cnts_for_template_vars(self) -> z3.ExprRef:
        """This implementation does not need additional ones"""
        return z3.BoolVal(True)

    def add_template_cnts(self) -> None:
        """Add cnts for init and post assertions (a trick)"""
        cnts: List[z3.ExprRef] = []
        cnts_prime: List[z3.ExprRef] = []
        for i in range(self.arity):
            var = self.sts.variables[i]
            var_prime = self.sts.prime_variables[i]
            var_l = self.template_vars[i][0]  # lower bound
            var_u = self.template_vars[i][1]  # upper bound

            if self.signedness == Signedness.UNSIGNED:
                cnts.append(z3.And(z3.UGE(var, var_l), z3.ULE(var, var_u)))
                cnts_prime.append(
                    z3.And(z3.UGE(var_prime, var_l), z3.ULE(var_prime, var_u))
                )
            else:
                cnts.append(z3.And(var >= var_l, var <= var_u))
                cnts_prime.append(z3.And(var_prime >= var_l, var_prime <= var_u))

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
        self, model: z3.ModelRef, use_prime_variables: bool = False
    ) -> z3.ExprRef:
        """Build an invariant from a model (fixing the values of the template vars)"""
        constraints: List[z3.ExprRef] = []
        for i in range(self.arity):
            if use_prime_variables:
                var = self.sts.prime_variables[i]
            else:
                var = self.sts.variables[i]
            tvar_l = self.template_vars[i][0]
            tvar_u = self.template_vars[i][1]
            if self.signedness == Signedness.UNSIGNED:
                constraints.append(
                    z3.And(z3.UGE(var, model[tvar_l]), z3.ULE(var, model[tvar_u]))
                )
            else:
                constraints.append(z3.And(var >= model[tvar_l], var <= model[tvar_u]))
        return z3.And(constraints)


class DisjunctiveBitVecIntervalTemplate(Template):
    """
    Disjunctive Interval domain
    """

    def __init__(self, sts: TransitionSystem, **kwargs):

        self.template_type = TemplateType.BV_DISJUNCTIVE_INTERVAL

        # TODO: infer the signedness of variables? (or design a domain that is
        #  signedness-irrelevant. Currently, we use unsigned by default
        signedness_val = self._init_signedness_from_sts(sts)
        if signedness_val is not None:
            self.signedness = signedness_val

        self.sts = sts
        self.arity = len(self.sts.variables)

        self.num_disjunctions = kwargs.get("num_disjunctions", 2)

        self.template_vars: List[List[List[z3.ExprRef]]] = []  # vector of vector
        self.template_index = 0  # number of templates

        self.add_template_vars()

        # pre compute to reduce redundant calling
        self.template_cnt_init_and_post: Optional[z3.ExprRef] = None
        self.template_cnt_trans: Optional[z3.ExprRef] = None
        self.add_template_cnts()

    def add_template_vars(self) -> None:
        """Add several groups of template variables"""
        for i in range(self.num_disjunctions):
            vars_for_dis: List[List[z3.ExprRef]] = []
            for var in self.sts.variables:
                tvars = [
                    z3.BitVec("d{0}_l_{1}".format(i, str(var)), var.sort().size()),
                    z3.BitVec("d{0}_u_{1}".format(i, str(var)), var.sort().size()),
                ]
                vars_for_dis.append(tvars)
            self.template_vars.append(vars_for_dis)

    def get_additional_cnts_for_template_vars(self) -> z3.ExprRef:
        """This implementation does not need additional ones"""
        return z3.BoolVal(True)

    def add_template_cnts(self) -> None:
        # FIXME: the following is from IntervalTemplate
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
                var_l = vars_for_ith_disjunct[j][0]  # lower bound
                var_u = vars_for_ith_disjunct[j][1]  # upper bound

                if self.signedness == Signedness.UNSIGNED:
                    cnt_init_post.append(z3.And(z3.UGE(var, var_l), z3.ULE(var, var_u)))
                    cnt_trans.append(
                        z3.And(z3.UGE(var_prime, var_l), z3.ULE(var_prime, var_u))
                    )
                else:
                    cnt_init_post.append(z3.And(var >= var_l, var <= var_u))
                    cnt_trans.append(z3.And(var_prime >= var_l, var_prime <= var_u))

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
                tvar_l = vars_for_ith_disjunct[j][0]
                tvar_u = vars_for_ith_disjunct[j][1]
                if self.signedness == Signedness.UNSIGNED:
                    cnts.append(
                        z3.And(z3.UGE(var, model[tvar_l]), z3.ULE(var, model[tvar_u]))
                    )
                else:
                    cnts.append(z3.And(var >= model[tvar_l], var <= model[tvar_u]))
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
