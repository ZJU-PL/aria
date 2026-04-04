"""Template polyhedral domain over bit-vectors"""

from aria.efmc.engines.ef.templates.abstract_template import *
from aria.utils.bv_utils import Signedness

# from typing import List
from aria.efmc.sts import TransitionSystem
from aria.efmc.utils import big_and, big_or


class BitVecPolyhedronTemplate(Template):
    """
    Bit-vec polyhedron
    """

    def __init__(self, sts: TransitionSystem, **kwargs):
        self.template_type = TemplateType.BV_POLYHEDRON

        # TODO: infer the signedness of variables? (or design a domain that is signedness-irrelevant
        signedness_val = self._init_signedness_from_sts(sts)
        if signedness_val is not None:
            self.signedness = signedness_val
        else:
            raise NotImplementedError

        self.obj_no_overflow = kwargs.get("no_overflow", False)
        self.obj_no_underflow = kwargs.get("no_underflow", False)

        self.sts = sts
        self.arity = len(self.sts.variables)

        self.template_vars = []  # vector of vector

        self.template_index = 0

        #  number of linear inequalities (NOTE: interval, zone, and octagon domains do not need this)
        #  thus, the following field is polyhedron-specific
        #  NOTE: this parameter means (p1 and p2...), but not disjunction
        self.num_templates = 1

        # FIXME: we assume all variables are bit-vectors, and they have the same size
        #  Otherwise, many operators (e.g., +, -) cannot be applied two variables directly.
        #  However, this is not a good idea for real-world cases
        self.bv_size = self.sts.variables[0].sort().size()

        self.add_template_vars()  # init self.template_vars

        # pre-compute to reduce redundant calling
        self.template_cnt_init_and_post = None
        self.template_cnt_trans = None
        self.add_template_cnts()  # init the above constraints

    def add_template_vars(self):
        """
        Initialize self.template_vars
        E.g., assume that self.sts.variables are [x, y], and self.num_templates = 2.
         Then, we will add two templates
            p1_0 + x*p1_1 + y*p1_2  >= 0
                     AND
            p2_0 + x*p2_1 + y*p2_2  >= 0
         NOTE: we always use self.num_templates = 1 by default.
        After this function, self.template_vars will be:
        [[p1_0, p1_1, p1_2], [p2_0, p2_1, p2_2]]
        """
        for _ in range(self.num_templates):
            self.template_index += 1
            # the following is for creating p1_0, p2_0, ...
            tvars = [z3.BitVec("p%d_%d" % (self.template_index, 0), self.bv_size)]
            # creating p1_1, p1_2, p2_1, p2_2, ...
            for i in range(1, self.arity + 1):
                tvars.append(
                    z3.BitVec("p%d_%d" % (self.template_index, i), self.bv_size)
                )
            self.template_vars.append(tvars)
        # print(self.template_vars)

    def add_template_cnts(self):
        """For initializing self.template_cnt_init_and_post and self.template_cnt_trans"""
        cnts_init_post = []  # For sts.variables
        cnts_trans = []  # For sts.prime_variables
        for i in range(self.template_index):  # num. of templates
            term_init_post = self._build_linear_term(
                self.template_vars[i], self.sts.variables
            )
            term_trans = self._build_linear_term(
                self.template_vars[i], self.sts.prime_variables
            )

            if self.signedness == Signedness.UNSIGNED:
                cnts_init_post.append(z3.UGE(term_init_post, 0))
                cnts_trans.append(z3.UGE(term_trans, 0))
            else:
                cnts_init_post.append(term_init_post >= 0)
                cnts_trans.append(term_trans >= 0)

        self.template_cnt_init_and_post = big_and(cnts_init_post)
        self.template_cnt_trans = big_and(cnts_trans)

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables=False
    ) -> z3.ExprRef:
        """
        Build an invariant from a model, i.e., fixing the values of the template vars
        :param model the model used for building expr
        :param use_prime_variables deciding using x, y or x!, y!
        """
        cnts = []
        for i in range(self.template_index):  # num. of templates
            term = model[self.template_vars[i][0]]
            for j in range(1, self.arity + 1):
                tvar = self.template_vars[i][j]
                # model[tvar] is the value of tvar in the model
                var = self._get_variable(j - 1, use_prime_variables)
                term = term + var * model[tvar]

            if self.signedness == Signedness.UNSIGNED:
                cnts.append(z3.UGE(term, 0))
            else:
                cnts.append(term >= 0)
        return big_and(cnts)


class DisjunctiveBitVecPolyhedronTemplate(Template):
    """
    TODO: There are several variants. Some references
    """

    def __init__(self, sts: TransitionSystem, **kwargs):
        self.template_type = TemplateType.BV_DISJUNCTIVE_POLYHEDRON

        signedness_val = self._init_signedness_from_sts(sts)
        if signedness_val is not None:
            self.signedness = signedness_val
        else:
            raise NotImplementedError

        self.num_disjunctions = kwargs.get("num_disjunctions", 2)

        self.obj_no_overflow = kwargs.get("no_overflow", False)
        self.obj_no_underflow = kwargs.get("no_underflow", False)

        self.sts = sts
        self.arity = len(self.sts.variables)

        self.template_vars = []  # vector of vector
        self.template_index = 0

        #  number of linear inequalities (NOTE: interval, zone, and octagon domains do not need this)
        #  thus, the following field is polyhedron-specific
        self.num_templates = 1

        # FIXME: we assume all variables are bit-vectors, and they have the same size
        #  Otherwise, many operators (e.g., +, -) cannot be applied two variables directly.
        #  However, this is not a good idea for real-world cases
        self.bv_size = self.sts.variables[0].sort().size()

        self.add_template_vars()  # init self.template_vars

        # pre-compute to reduce redundant calling
        self.template_cnt_init_and_post = None
        self.template_cnt_trans = None
        self.add_template_cnts()  # init the above constraints

    def add_template_vars(self):
        """
        Initialize self.template_vars
        E.g., assume that self.sts.variables are [x, y], and self.num_disjunctions = 2.
         Then, we will add two templates
            d1_0 + x*d1_1 + y*p1_2  >= 0
                   OR
            d2_0 + x*d2_1 + y*d2_2  >= 0
         NOTE: we always use self.num_templates = 1 by default. So, we do not consider
           the impact of self.num_templates here..
        After this function, self.template_vars will be:
        [[d1_0, d1_1, d1_2], [d2_0, d2_1, d2_2]]
        """
        for i in range(self.num_disjunctions):
            # the following is for creating d1_0, d2_0, ...
            vars_for_dis = [z3.BitVec("d%d_%d" % (i + 1, 0), self.bv_size)]
            # creating d1_1, d1_2, d2_1, d2_2,...
            for j in range(1, self.arity + 1):
                vars_for_dis.append(z3.BitVec("d%d_%d" % (i + 1, j), self.bv_size))
            self.template_vars.append(vars_for_dis)
        # print(self.template_vars)

    def add_template_cnts(self):
        """For initializing self.template_cnt_init_and_post and self.template_cnt_trans"""
        cnt_init_and_post_dis = []
        cnt_trans_dis = []

        for i in range(self.num_disjunctions):  # num. of disjunctions
            # d1_0 + x*d1_1 + y*p1_2  >= 0 OR
            # d2_0 + x*d2_1 + y*d2_2  >= 0
            term_init_post = self._build_linear_term(
                self.template_vars[i], self.sts.variables
            )
            term_trans = self._build_linear_term(
                self.template_vars[i], self.sts.prime_variables
            )

            if self.signedness == Signedness.UNSIGNED:
                cnt_init_and_post_dis.append(z3.UGE(term_init_post, 0))
                cnt_trans_dis.append(z3.UGE(term_trans, 0))
            else:
                cnt_init_and_post_dis.append(term_init_post >= 0)
                cnt_trans_dis.append(term_trans >= 0)

        self.template_cnt_init_and_post = big_or(cnt_init_and_post_dis)
        self.template_cnt_trans = big_or(cnt_trans_dis)

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables=False
    ) -> z3.ExprRef:
        """
        Build an invariant from a model, i.e., fixing the values of the template vars
        :param model the model used for building expr
        :param use_prime_variables deciding using x, y or x!, y!
        """
        cnts = []
        for i in range(self.num_disjunctions):  # num. of disjunctions
            term = model[self.template_vars[i][0]]
            for j in range(1, self.arity + 1):
                tvar = self.template_vars[i][j]
                # model[tvar] is the value of tvar in the model
                var = self._get_variable(j - 1, use_prime_variables)
                term = term + var * model[tvar]

            if self.signedness == Signedness.UNSIGNED:
                cnts.append(z3.UGE(term, 0))
            else:
                cnts.append(term >= 0)
        return big_or(cnts)
