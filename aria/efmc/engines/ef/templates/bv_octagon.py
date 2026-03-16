"""Octagon template over bit-vector variables"""

import itertools
from aria.efmc.engines.ef.templates.abstract_template import *
from aria.efmc.utils.bv_utils import Signedness
from aria.efmc.sts import TransitionSystem
from aria.efmc.utils import get_variables, big_and, big_or


class BitVecOctagonTemplate(Template):

    def __init__(self, sts: TransitionSystem, **kwargs):
        self.sts = sts
        self.template_type = TemplateType.BV_OCTAGON

        # TODO: infer the signedness of variables? (or design a domain that is signedness-irrelevant
        signedness_val = self._init_signedness_from_sts(sts)
        if signedness_val is not None:
            self.signedness = signedness_val

        #  prevent over/under flows in the template exprs, e.g., x - y, x + y
        self.obj_no_overflow = kwargs.get("no_overflow", False)
        self.obj_no_underflow = kwargs.get("no_underflow", False)

        self.sts = sts
        self.arity = len(self.sts.variables)
        assert self.arity >= 2
        assert len(self.sts.prime_variables) >= 2

        self.octagons = []
        self.wrap_around_cnts_vars = (
            []
        )  # for preventing under/under flow in the tempalte exprs
        self.wrap_around_cnts_prime_vars = []
        signed = True if self.signedness == Signedness.SIGNED else False

        for x, y in list(itertools.combinations(self.sts.variables, 2)):
            self.octagons.append(x - y)
            self.octagons.append(x + y)

            if self.obj_no_overflow:
                self.wrap_around_cnts_vars.append(z3.BVSubNoOverflow(x, y))
                self.wrap_around_cnts_vars.append(
                    z3.BVAddNoOverflow(x, y, signed=signed)
                )
            if self.obj_no_underflow:
                self.wrap_around_cnts_vars.append(
                    z3.BVSubNoUnderflow(x, y, signed=signed)
                )
                self.wrap_around_cnts_vars.append(z3.BVAddNoUnderflow(x, y))

        self.prime_octagons = []
        for px, py in list(itertools.combinations(self.sts.prime_variables, 2)):
            self.prime_octagons.append(px - py)
            self.prime_octagons.append(px + py)

            if self.obj_no_overflow:
                self.wrap_around_cnts_prime_vars.append(z3.BVSubNoOverflow(px, py))
                self.wrap_around_cnts_prime_vars.append(
                    z3.BVAddNoOverflow(px, py, signed=signed)
                )
            if self.obj_no_underflow:
                self.wrap_around_cnts_prime_vars.append(
                    z3.BVSubNoUnderflow(px, py, signed=signed)
                )
                self.wrap_around_cnts_prime_vars.append(z3.BVAddNoUnderflow(px, py))

        self.template_vars = []
        self.template_vars_for_vars = []  # aux variables for x, y, z, ...
        self.template_vars_for_terms = []  # aux variables for x - y, x + y, x - z, ...
        self.template_index = 0  # number of templates

        # create template variables, e.g., if we care about the interval of x,
        # then the template cnts are a <= x <= b, where a and b are template variables
        self.add_template_vars()

        # pre_compute to reduce redundant calling
        self.template_cnt_init_and_post = None
        self.template_cnt_trans = None
        self.add_template_cnts()

    def add_template_vars(self):
        """
        Add several groups of template vars
        """
        for var in self.sts.variables:
            tvars = [
                z3.BitVec("{}_l".format(str(var)), var.sort().size()),
                z3.BitVec("{}_u".format(str(var)), var.sort().size()),
            ]
            self.template_vars_for_vars.append(tvars)
            self.template_vars.append(tvars)

        # follow interval, but use self.octagon
        for i in range(len(self.octagons)):
            term = self.octagons[i]
            term_vars = get_variables(term)
            term_name = "{}{}".format(term_vars[0], term_vars[1])
            self.template_index += 1
            tvars = [
                z3.BitVec("{}_l".format(term_name), term.size()),
                z3.BitVec("{}_u".format(term_name), term.size()),
            ]
            self.template_vars_for_terms.append(tvars)
            self.template_vars.append(tvars)
        # raise NotImplementedError

    def get_additional_cnts_for_template_vars(self):
        """
        This implementation does not need additional ones?
        """
        return z3.BoolVal(True)

    def add_template_cnts(self):
        """
        Add cnts for init and post assertions (a trick)
        """
        cnts = []
        cnts_prime = []

        # Add constraints for x, y, z, ...
        for i in range(self.arity):
            var = self.sts.variables[i]
            var_prime = self.sts.prime_variables[i]
            var_l = self.template_vars_for_vars[i][0]  # lower bound
            var_u = self.template_vars_for_vars[i][1]  # upper bound
            if self.signedness == Signedness.UNSIGNED:
                cnts.append(z3.And(z3.UGE(var, var_l), z3.ULE(var, var_u)))
                cnts_prime.append(
                    z3.And(z3.UGE(var_prime, var_l), z3.ULE(var_prime, var_u))
                )
            else:
                cnts.append(z3.And(var >= var_l, var <= var_u))
                cnts_prime.append(z3.And(var_prime >= var_l, var_prime <= var_u))

        # Add constraints for x - y, x + y, x - z, ...
        for i in range(len(self.octagons)):
            term = self.octagons[i]  # e.g., x - y
            term_prime = self.prime_octagons[i]  # e.g., x' - y'
            template_vars_for_term = self.template_vars_for_terms[i]
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

    def build_invariant_expr(self, model: z3.ModelRef, use_prime_variables=False):
        """
        Build an invariant from a model (fixing the values of the template vars)
        """
        cnts = []

        # For building formulas in the forms of a <= x < = b, c <= y < = d, ...
        for i in range(self.arity):
            if use_prime_variables:
                var = self.sts.prime_variables[i]
            else:
                var = self.sts.variables[i]
            tvar_l = self.template_vars_for_vars[i][0]
            tvar_u = self.template_vars_for_vars[i][1]
            if self.signedness == Signedness.UNSIGNED:
                cnts.append(
                    z3.And(z3.UGE(var, model[tvar_l]), z3.ULE(var, model[tvar_u]))
                )
            else:
                cnts.append(z3.And(var >= model[tvar_l], var <= model[tvar_u]))

        # For building formulas in the forms of a <= x - y <= b, c <= x + y <= d,...
        for i in range(len(self.octagons)):
            if use_prime_variables:
                term = self.prime_octagons[i]
            else:
                term = self.octagons[i]
            template_vars_for_term = self.template_vars_for_terms[i]
            term_l = template_vars_for_term[0]  # lower bound of the term
            term_u = template_vars_for_term[1]  # upper bound of the term

            if self.signedness == Signedness.UNSIGNED:
                cnts.append(
                    z3.And(z3.ULE(model[term_l], term), z3.UGE(model[term_u], term))
                )
            else:
                cnts.append(z3.And(model[term_l] <= term, model[term_u] >= term))

        return big_and(cnts)

    def build_invariant(self, model: z3.ModelRef) -> z3.ExprRef:
        """Build an invariant from a model.

        Args:
            model: Z3 model containing values for template variables

        Returns:
            Z3 expression representing the invariant
        """
        return self.build_invariant_expr(model)


class DisjunctiveBitVecOctagonTemplate(Template):

    def __init__(self, sts: TransitionSystem, **kwargs):
        self.sts = sts
        self.template_type = TemplateType.BV_DISJUNCTIVE_OCTAGON

        # TODO: infer the signedness of variables? (or design a domain that is signedness-irrelevant
        signedness_val = self._init_signedness_from_sts(sts)
        if signedness_val is not None:
            self.signedness = signedness_val

        self.obj_no_overflow = kwargs.get("no_overflow", False)
        self.obj_no_underflow = kwargs.get("no_underflow", False)

        self.sts = sts
        self.arity = len(self.sts.variables)
        assert self.arity >= 2
        assert len(self.sts.prime_variables) >= 2

        self.wrap_around_cnts_vars = (
            []
        )  # for preventing under/under flow in the tempalte exprs
        self.wrap_around_cnts_prime_vars = []
        signed = True if self.signedness == Signedness.SIGNED else False

        self.octagons = []  # x - y, x + y, ...
        for x, y in list(itertools.combinations(self.sts.variables, 2)):
            self.octagons.append(x - y)
            self.octagons.append(x + y)

            if self.obj_no_overflow:
                self.wrap_around_cnts_vars.append(z3.BVSubNoOverflow(x, y))
                self.wrap_around_cnts_vars.append(
                    z3.BVAddNoOverflow(x, y, signed=signed)
                )
            if self.obj_no_underflow:
                self.wrap_around_cnts_vars.append(
                    z3.BVSubNoUnderflow(x, y, signed=signed)
                )
                self.wrap_around_cnts_vars.append(z3.BVAddNoUnderflow(x, y))

        self.prime_octagons = []  # # x' - y', x' + y', ...
        for px, py in list(itertools.combinations(self.sts.prime_variables, 2)):
            self.prime_octagons.append(px - py)
            self.prime_octagons.append(px + py)

            if self.obj_no_overflow:
                self.wrap_around_cnts_prime_vars.append(z3.BVSubNoOverflow(px, py))
                self.wrap_around_cnts_prime_vars.append(
                    z3.BVAddNoOverflow(px, py, signed=signed)
                )
            if self.obj_no_underflow:
                self.wrap_around_cnts_prime_vars.append(
                    z3.BVSubNoUnderflow(px, py, signed=signed)
                )
                self.wrap_around_cnts_prime_vars.append(z3.BVAddNoUnderflow(px, py))

        self.template_vars = []  # vector of vector
        self.template_vars_for_vars = []  # aux variables for x, y, z, ...
        self.template_vars_for_terms = []  # aux variables for x - y, x + y, x - z, ...

        self.template_index = 0  # number of templates

        self.num_disjunctions = kwargs.get("num_disjunctions", 2)

        self.add_template_vars()

        # pre-compute to reduce redundant calling
        self.template_cnt_init_and_post = None
        self.template_cnt_trans = None
        self.add_template_cnts()

    def add_template_vars(self):
        """
        Add several groups of template vars
        """
        for i in range(self.num_disjunctions):
            # crate self.num_disjunctions classes of variables

            #  aux variables for x, y, z ,...
            aux_vars = []
            for v in self.sts.variables:
                # print(v, v.sort().size())
                tvars = [
                    z3.BitVec("d{0}_{1}_l".format(i, str(v)), v.sort().size()),
                    z3.BitVec("d{0}_{1}_u".format(i, str(v)), v.sort().size()),
                ]
                aux_vars.append(tvars)
            self.template_vars_for_vars.append(aux_vars)
            self.template_vars.append(aux_vars)

            #  aux variables for x - y, x + y, ...
            aux_vars_for_terms = []
            for j in range(len(self.octagons)):
                term = self.octagons[j]
                term_vars = get_variables(term)
                term_name = "{}{}".format(term_vars[0], term_vars[1])
                self.template_index += 1
                tvars = [
                    z3.BitVec("d{0}_{1}_l".format(i, term_name), term.size()),
                    z3.BitVec("d{0}_{1}_u".format(i, term_name), term.size()),
                ]
                aux_vars_for_terms.append(tvars)
            self.template_vars_for_terms.append(aux_vars_for_terms)
            self.template_vars.append(aux_vars_for_terms)
        # raise NotImplementedError

    def get_additional_cnts_for_template_vars(self):
        """
        This implementation does not need additional ones?
        """
        raise z3.BoolVal(True)

    def add_template_cnts(self):
        """
        Add cnts for init and post assertions (a trick)
        """
        cnt_init_and_post_dis = []
        cnt_trans_dis = []

        for i in range(self.num_disjunctions):
            cnt_init_post = []  # For sts.variables
            cnt_trans = []  # For sts.prime_variables

            # and cnts for x, y, z,...
            for j in range(self.arity):
                var = self.sts.variables[j]  # e.g., x, y
                prime_var = self.sts.prime_variables[j]  # e.g., x!, y!
                template_vars_for_var = self.template_vars_for_vars[i][j]
                if self.signedness == Signedness.UNSIGNED:
                    cnt_init_post.append(
                        z3.And(
                            z3.UGE(var, template_vars_for_var[0]),
                            z3.ULE(var, template_vars_for_var[1]),
                        )
                    )
                    cnt_trans.append(
                        z3.And(
                            z3.UGE(prime_var, template_vars_for_var[0]),
                            z3.ULE(prime_var, template_vars_for_var[1]),
                        )
                    )
                else:
                    cnt_init_post.append(
                        z3.And(
                            var >= template_vars_for_var[0],
                            var <= template_vars_for_var[1],
                        )
                    )
                    cnt_trans.append(
                        z3.And(
                            prime_var >= template_vars_for_var[0],
                            prime_var <= template_vars_for_var[1],
                        )
                    )

            # and cnts for x - y, x + y, ...
            for j in range(len(self.octagons)):
                term = self.octagons[j]  # e.g., x - y
                term_prime = self.prime_octagons[j]  # e.g., x' - y'
                template_vars_for_term = self.template_vars_for_terms[i][j]
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

            # for preventing over/under flows in x - y, x + y, ...
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

    def build_invariant_expr(self, model: z3.ModelRef, use_prime_variables=False):
        """
        Build an invariant from a model (fixing the values of the template vars)
        """
        cnts_dis = []

        for i in range(self.num_disjunctions):
            cnts = []

            # Invariant: len(self.template_vars) = self.num_disjunctions
            aux_vars_for_vars_ith_disjunct = self.template_vars_for_vars[i]
            for j in range(self.arity):
                var = self._get_variable(j, use_prime_variables)
                tvar_l = aux_vars_for_vars_ith_disjunct[j][0]
                tvar_u = aux_vars_for_vars_ith_disjunct[j][1]
                if self.signedness == Signedness.UNSIGNED:
                    cnts.append(
                        z3.And(z3.UGE(var, model[tvar_l]), z3.ULE(var, model[tvar_u]))
                    )
                else:
                    cnts.append(z3.And(var >= model[tvar_l], var <= model[tvar_u]))

            aux_vars_for_terms_ith_disjunct = self.template_vars_for_terms[i]
            for j in range(len(self.octagons)):
                if use_prime_variables:
                    term = self.prime_octagons[j]
                else:
                    term = self.octagons[j]
                term_l = aux_vars_for_terms_ith_disjunct[j][
                    0
                ]  # lower bound of the term
                term_u = aux_vars_for_terms_ith_disjunct[j][
                    1
                ]  # upper bound of the term

                if self.signedness == Signedness.UNSIGNED:
                    cnts.append(
                        z3.And(z3.ULE(model[term_l], term), z3.UGE(model[term_u], term))
                    )
                else:
                    cnts.append(z3.And(model[term_l] <= term, model[term_u] >= term))

            cnts_dis.append(big_and(cnts))
        return big_or(cnts_dis)

    def build_invariant(self, model: z3.ModelRef) -> z3.ExprRef:
        """Build an invariant from a model.

        Args:
            model: Z3 model containing values for template variables

        Returns:
            Z3 expression representing the invariant
        """
        return self.build_invariant_expr(model)
