"""Bit-vector Zone Domain"""

import itertools
from typing import List, Optional

from aria.efmc.engines.ef.templates.abstract_template import *
from aria.utils.bv_utils import Signedness
from aria.efmc.sts import TransitionSystem
from aria.efmc.utils import get_variables, big_and, big_or


class BitVecZoneTemplate(Template):

    def __init__(self, sts: TransitionSystem, **kwargs):

        self.template_type = TemplateType.BV_ZONE

        # TODO: infer the signedness of variables? (or design a domain that is signedness-irrelevant
        signedness_val = self._init_signedness_from_sts(sts)
        if signedness_val is not None:
            self.signedness = signedness_val

        # prevent over/under flows in the template exprs, e.g., x - y
        self.obj_no_overflow = kwargs.get("no_overflow", False)
        self.obj_no_underflow = kwargs.get("no_underflow", False)

        self.sts = sts
        self.arity = len(self.sts.variables)
        assert self.arity >= 2
        assert len(self.sts.prime_variables) >= 2

        self.zones: List[z3.ExprRef] = []

        self.wrap_around_cnts_vars: List[z3.ExprRef] = (
            []
        )  # for preventing under/under flow in the tempalte exprs
        self.wrap_around_cnts_prime_vars: List[z3.ExprRef] = []
        signed = True if self.signedness == Signedness.SIGNED else False

        for x, y in list(itertools.combinations(self.sts.variables, 2)):
            self.zones.append(x - y)
            if self.obj_no_overflow:
                self.wrap_around_cnts_vars.append(z3.BVSubNoOverflow(x, y))
            if self.obj_no_underflow:
                self.wrap_around_cnts_vars.append(
                    z3.BVSubNoUnderflow(x, y, signed=signed)
                )

        self.prime_zones: List[z3.ExprRef] = []
        for px, py in list(itertools.combinations(self.sts.prime_variables, 2)):
            self.prime_zones.append(px - py)

            if self.obj_no_overflow:
                self.wrap_around_cnts_prime_vars.append(z3.BVSubNoOverflow(px, py))
            if self.obj_no_underflow:
                self.wrap_around_cnts_prime_vars.append(
                    z3.BVSubNoUnderflow(px, py, signed=signed)
                )

        self.template_vars: List[List[z3.ExprRef]] = []  # vector of vector
        self.template_index = 0  # number of templates

        self.add_template_vars()

        # pre_compute to reduce redundant calling
        self.template_cnt_init_and_post: Optional[z3.ExprRef] = None
        self.template_cnt_trans: Optional[z3.ExprRef] = None
        self.add_template_cnts()

    def add_template_vars(self) -> None:
        # follow interval, but use self.zones

        for i in range(len(self.zones)):
            term = self.zones[i]
            term_vars = get_variables(term)
            term_name = "{}{}".format(term_vars[0], term_vars[1])
            self.template_index += 1
            tvars = [
                z3.BitVec("{}_l".format(term_name), term.size()),
                z3.BitVec("{}_u".format(term_name), term.size()),
            ]
            self.template_vars.append(tvars)
        # raise NotImplementedError

    def get_additional_cnts_for_template_vars(self) -> z3.ExprRef:
        return z3.BoolVal(True)

    def add_template_cnts(self) -> None:
        # TODO: test correctness
        cnts: List[z3.ExprRef] = []
        cnts_prime: List[z3.ExprRef] = []

        # Add constraints for x - y, x - z, ...
        for i in range(len(self.zones)):
            term = self.zones[i]  # e.g., x - y
            term_prime = self.prime_zones[i]  # e.g., x' - y'
            template_vars_for_term = self.template_vars[i]
            term_l = template_vars_for_term[0]  # lower bound of the term
            term_u = template_vars_for_term[1]  # upper bound of the term

            if self.signedness == Signedness.UNSIGNED:
                cnts.append(z3.And(z3.ULE(term_l, term), z3.UGE(term_u, term)))
                cnts_prime.append(
                    z3.And(z3.ULE(term_l, term_prime), z3.UGE(term_u, term_prime))
                )
            else:
                cnts.append(z3.And(term_l <= term, term_u >= term))
                cnts_prime.append(z3.And(term_l <= term_prime, term_u >= term_prime))

        self.template_cnt_init_and_post = big_and(cnts)
        if len(self.wrap_around_cnts_vars) > 0:
            # print(self.wrap_around_cnts_vars)
            self.template_cnt_init_and_post = z3.And(
                self.template_cnt_init_and_post, big_and(self.wrap_around_cnts_vars)
            )

        self.template_cnt_trans = big_and(cnts_prime)
        if len(self.wrap_around_cnts_prime_vars) > 0:
            self.template_cnt_trans = z3.And(
                self.template_cnt_trans, big_and(self.wrap_around_cnts_prime_vars)
            )

        # raise NotImplementedError

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables: bool = False
    ) -> z3.ExprRef:
        """
        Build the inductive invariant corresponding to a model, which is obtained by
         solving the exists-forall formula encoding the verification condition
        """
        cnts: List[z3.ExprRef] = []
        for i in range(len(self.zones)):
            if use_prime_variables:
                term = self.prime_zones[i]
            else:
                term = self.zones[i]
            template_vars_for_term = self.template_vars[i]
            term_l = template_vars_for_term[0]  # lower bound of the term
            term_u = template_vars_for_term[1]  # upper bound of the term

            if self.signedness == Signedness.UNSIGNED:
                cnts.append(
                    z3.And(z3.ULE(model[term_l], term), z3.UGE(model[term_u], term))
                )
            else:
                cnts.append(z3.And(model[term_l] <= term, model[term_u] >= term))

        return big_and(cnts)


class DisjunctiveBitVecZoneTemplate(Template):

    def __init__(self, sts: TransitionSystem, **kwargs):
        self.template_type = TemplateType.BV_DISJUNCTIVE_ZONE

        # TODO: infer the signedness of variables? (or design a domain that is signedness-irrelevant
        signedness_val = self._init_signedness_from_sts(sts)
        if signedness_val is not None:
            self.signedness = signedness_val

        # prevent over/under flows in the template exprs, e.g., x - y
        self.obj_no_overflow = kwargs.get("no_overflow", False)
        self.obj_no_underflow = kwargs.get("no_underflow", False)

        self.sts = sts
        self.arity = len(self.sts.variables)

        self.wrap_around_cnts_vars: List[z3.ExprRef] = (
            []
        )  # for preventing under/under flow in the tempalte exprs
        self.wrap_around_cnts_prime_vars: List[z3.ExprRef] = []
        signed = True if self.signedness == Signedness.SIGNED else False

        self.zones: List[z3.ExprRef] = []  # x - y, x - z, y - z, ...
        for x, y in list(itertools.combinations(self.sts.variables, 2)):
            self.zones.append(x - y)
            if self.obj_no_overflow:
                self.wrap_around_cnts_vars.append(z3.BVSubNoOverflow(x, y))
            if self.obj_no_underflow:
                self.wrap_around_cnts_vars.append(
                    z3.BVSubNoUnderflow(x, y, signed=signed)
                )

        self.prime_zones: List[z3.ExprRef] = []
        for px, py in list(itertools.combinations(self.sts.prime_variables, 2)):
            self.prime_zones.append(px - py)
            if self.obj_no_overflow:
                self.wrap_around_cnts_prime_vars.append(z3.BVSubNoOverflow(px, py))
            if self.obj_no_underflow:
                self.wrap_around_cnts_prime_vars.append(
                    z3.BVSubNoUnderflow(px, py, signed=signed)
                )

        self.template_vars: List[List[List[z3.ExprRef]]] = []  # vector of vector
        self.template_index = 0  # number of templates

        self.num_disjunctions = kwargs.get("num_disjunctions", 2)

        self.add_template_vars()

        # pre-compute to reduce redundant calling
        self.template_cnt_init_and_post: Optional[z3.ExprRef] = None
        self.template_cnt_trans: Optional[z3.ExprRef] = None
        self.add_template_cnts()

    def add_template_vars(self) -> None:
        """
        Add several groups of template vars
        """
        for i in range(self.num_disjunctions):
            vars_for_dis: List[List[z3.ExprRef]] = []
            for j in range(len(self.zones)):
                term = self.zones[j]
                term_vars = get_variables(term)
                term_name = "{}{}".format(term_vars[0], term_vars[1])
                self.template_index += 1
                tvars = [
                    z3.BitVec("d{0}_{1}_l".format(i, term_name), term.size()),
                    z3.BitVec("d{0}_{1}_u".format(i, term_name), term.size()),
                ]
                vars_for_dis.append(tvars)

            self.template_vars.append(vars_for_dis)
        # print(self.template_vars)

    def get_additional_cnts_for_template_vars(self) -> z3.ExprRef:
        """
        This implementation does not need additional ones?
        """
        raise z3.BoolVal(True)

    def add_template_cnts(self) -> None:
        """
        Add cnts for init and post assertions (a trick)
        """
        cnt_init_and_post_dis: List[z3.ExprRef] = []
        cnt_trans_dis: List[z3.ExprRef] = []

        for i in range(self.num_disjunctions):
            # Invariant: len(self.template_vars) = self.num_disjunctions
            vars_for_ith_disjunct = self.template_vars[i]
            cnt_init_post: List[z3.ExprRef] = []  # For sts.variables
            cnt_trans: List[z3.ExprRef] = []  # For sts.prime_variables
            for j in range(len(self.zones)):
                term = self.zones[j]  # e.g., x - y
                term_prime = self.prime_zones[j]  # e.g., x' - y'
                template_vars_for_term = vars_for_ith_disjunct[j]
                term_l = template_vars_for_term[0]  # lower bound of the term
                term_u = template_vars_for_term[1]  # upper bound of the term

                if self.signedness == Signedness.UNSIGNED:
                    cnt_init_post.append(
                        z3.And(z3.ULE(term_l, term), z3.UGE(term_u, term))
                    )
                    cnt_trans.append(
                        z3.And(z3.ULE(term_l, term_prime), z3.UGE(term_u, term_prime))
                    )
                else:
                    cnt_init_post.append(z3.And(term_l <= term, term_u >= term))
                    cnt_trans.append(z3.And(term_l <= term_prime, term_u >= term_prime))

            ith_cnt_init_post = big_and(cnt_init_post)
            ith_cnt_trans = big_and(cnt_trans)

            # for preventing over/under flows in x - y,...
            if len(self.wrap_around_cnts_vars) > 0:
                ith_cnt_init_post = z3.And(
                    ith_cnt_init_post, big_and(self.wrap_around_cnts_vars)
                )
            if len(self.wrap_around_cnts_prime_vars) > 0:
                ith_cnt_trans = z3.And(
                    ith_cnt_trans, big_and(self.wrap_around_cnts_prime_vars)
                )

            cnt_init_and_post_dis.append(ith_cnt_init_post)
            cnt_trans_dis.append(ith_cnt_trans)

        self.template_cnt_init_and_post = big_or(cnt_init_and_post_dis)
        self.template_cnt_trans = big_or(cnt_trans_dis)
        # print(self.template_cnt_init_and_post)
        # print(self.template_cnt_trans)

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables: bool = False
    ) -> z3.ExprRef:
        """
        Build an invariant from a model (fixing the values of the template vars)
        """
        cnts_dis: List[z3.ExprRef] = []
        for i in range(self.num_disjunctions):
            # Invariant: len(self.template_vars) = self.num_disjunctions
            vars_for_ith_disjunct = self.template_vars[i]
            cnts: List[z3.ExprRef] = []  # constraints for one disjunct
            for j in range(len(self.zones)):
                term = self.prime_zones[j] if use_prime_variables else self.zones[j]
                template_vars_for_term = vars_for_ith_disjunct[j]
                term_l = template_vars_for_term[0]  # lower bound of the term
                term_u = template_vars_for_term[1]  # upper bound of the term
                if self.signedness == Signedness.UNSIGNED:
                    cnts.append(
                        z3.And(z3.ULE(model[term_l], term), z3.UGE(model[term_u], term))
                    )
                else:
                    cnts.append(z3.And(model[term_l] <= term, model[term_u] >= term))

            cnts_dis.append(big_and(cnts))
        return big_or(cnts_dis)
