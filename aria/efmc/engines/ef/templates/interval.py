"""Interval template over integer or real variables  (Version 1)

The encoding in this version is not elegant (but IntervalTemplateV2 is not complete, i.e., it will miss some states)
"""

import logging

from aria.efmc.engines.ef.templates.abstract_template import *
from aria.efmc.sts import TransitionSystem
from aria.efmc.utils import big_and

logger = logging.getLogger(__name__)


class IntervalTemplate(Template):
    """Interval domain
    Assume that self.sts.variables are [x, y], we will add four templates!!
        ix0 + x * ix1  >= 0  for lower (ix1 must be 1 or 0)
        ix2 + x * ix3  >= 0  for upper (ix3 must be -1 or 0)
        iy0 + y * iy1  >= 0  for lower
        iy2 + y * iy3  >= 0  for upper
    """

    def __init__(self, sts: TransitionSystem, **kwargs):

        self.template_type = TemplateType.INTERVAL

        if sts.has_real:
            self.use_real = True
        else:
            self.use_real = False

        self.sts = sts
        self.arity = len(self.sts.variables)

        self.template_vars = []  # vector of vector
        self.template_index = 0  # number of templates

        self.add_template_vars()

        # pre compute to reduce redundant calling
        self.template_cnt_init_and_post = None
        self.template_cnt_trans = None
        self.add_template_cnts()

    def add_template_vars(self):
        """Add several groups of template vars
        Assume that self.sts.variables are [x, y], we will add four templates!!
            ix0 + x * ix1  >= 0
            ix2 + x * ix3  >= 0

            iy0 + y * iy1  >= 0
            iy2 + y * iy3  >= 0
        After this function, self.template_vars will be:
        [[ix0, ix1, ix2, ix3], [iy0, iy1, iy2, iy3]]
        """
        for i in range(self.arity):
            var = self.sts.variables[i]
            self.template_index += 1
            if self.use_real:
                tvars = [
                    z3.Real("i{}0".format(var)),
                    z3.Real("i{}1".format(var)),
                    z3.Real("i{}2".format(var)),
                    z3.Real("i{}3".format(var)),
                ]
            else:
                tvars = [
                    z3.Int("i{}0".format(var)),
                    z3.Int("i{}1".format(var)),
                    z3.Int("i{}2".format(var)),
                    z3.Int("i{}3".format(var)),
                ]
            self.template_vars.append(tvars)

        # print(self.template_vars)

    def get_additional_cnts_for_template_vars(self):
        """FIXME: this encoding is not elegant (but IntervalTemplateV2 is not complete)
            i.e., it will miss some states
        Add constraints for restricting the template variables.
        (For interval, zone, etc. over integer/real variables?)

        For example, consider the two variables x and y
            ix0 + x * ix1  >= 0   for lower (ix1 must be 1 or 0)
            ix2 + x * ix3  >= 0   for upper (ix3 must be -1 or 0)
            iy0 + y * iy1  >= 0   for lower
            iy2 + y * iy3  >= 0   for upper
        self.template_vars are as follows:
        [[ix0, ix1, ix2, ix3], [iy0, iy1, iy2, iy3]]
        """
        # IMPORTANT: the following are additional cnts for interval
        cnts = []
        for tvars in self.template_vars:
            i0, i1, i2, i3 = tvars[0], tvars[1], tvars[2], tvars[3]
            # the second means "no lower bound"?
            cnts.append(z3.Or(i1 == 1, z3.And(i1 == 0, i0 == 0)))
            # the second means "no upper bound"?
            cnts.append(z3.Or(i3 == -1, z3.And(i3 == 0, i2 == 0)))
        return big_and(cnts)

    def add_template_cnts(self):
        """Consider two variables x and y
            ix0 + x * ix1  >= 0   for lower (ix1 must be 1 or 0)
            ix2 + x * ix3  >= 0   for upper (ix3 must be -1 or 0)

            iy0 + y * iy1  >= 0   for lower
            iy2 + y * iy3  >= 0   for upper
        self.template_vars are as follows:
        [[ix0, ix1, ix2, ix3], [iy0, iy1, iy2, iy3]]
        """
        cnts_init_post = []  # For sts.variables
        cnts_trans = []  # For sts.prime_variables
        for i in range(self.arity):
            var = self.sts.variables[i]  # x, y
            prime_var = self.sts.prime_variables[i]  # x!, y!
            template_vars_for_var = self.template_vars[i]

            i0, i1 = template_vars_for_var[0], template_vars_for_var[1]
            i2, i3 = template_vars_for_var[2], template_vars_for_var[3]

            cnts_init_post.append(i0 + var * i1 >= 0)
            cnts_init_post.append(i2 + var * i3 >= 0)

            cnts_trans.append(i0 + prime_var * i1 >= 0)
            cnts_trans.append(i2 + prime_var * i3 >= 0)

        self.template_cnt_init_and_post = big_and(cnts_init_post)
        self.template_cnt_trans = big_and(cnts_trans)

    def build_invariant_expr(self, model: z3.ModelRef, use_prime_variables=False):
        """
        Consider two variables x and y
            ix0 + x * ix1  >= 0   for lower (ix1 must be 1 or 0)
            ix2 + x * ix3  >= 0   for upper (ix3 must be -1 or 0)

            iy0 + y * iy1  >= 0   for lower
            iy2 + y * iy3  >= 0   for upper
        self.template_vars are as follows:
        [[ix0, ix1, ix2, ix3], [iy0, iy1, iy2, iy3]]
        """
        cnts = []
        for i in range(self.arity):
            if use_prime_variables:
                var = self.sts.prime_variables[i]
            else:
                var = self.sts.variables[i]

            template_vars_for_var = self.template_vars[i]
            i0, i1 = template_vars_for_var[0], template_vars_for_var[1]
            i2, i3 = template_vars_for_var[2], template_vars_for_var[3]

            if model.eval(i1, True).as_long() != 0:
                cnts.append(model.eval(i0, True) + var * model.eval(i1, True) >= 0)
            if model.eval(i3, True).as_long() != 0:
                cnts.append(model.eval(i2, True) + var * model.eval(i3, True) >= 0)

        # print("cnts for invariant: ", cnts)
        return big_and(cnts)


class DisjunctiveIntervalTemplate(Template):
    """
    Disjunctive Interval domain
    """

    def __init__(self, sts: TransitionSystem, **kwargs):

        self.template_type = TemplateType.DISJUNCTIVE_INTERVAL

        if sts.has_real:
            self.use_real = True
        else:
            self.use_real = False

        self.sts = sts
        self.arity = len(self.sts.variables)

        self.template_vars = []  # vector of vector
        self.template_index = 0  # number of templates

        self.num_disjunctions = kwargs.get("num_disjunctions", 2)

        self.add_template_vars()

        # pre compute to reduce redundant calling
        self.template_cnt_init_and_post = None
        self.template_cnt_trans = None
        self.add_template_cnts()

    def add_template_vars(self):
        for i in range(self.num_disjunctions):
            vars_for_dis = []
            for j in range(self.arity):
                var = self.sts.variables[j]
                self.template_index += 1
                if self.use_real:
                    tvars = [
                        z3.Real("d_{}i{}0".format(i, var)),
                        z3.Real("d_{}i{}1".format(i, var)),
                        z3.Real("d_{}i{}2".format(i, var)),
                        z3.Real("d_{}i{}3".format(i, var)),
                    ]
                else:
                    tvars = [
                        z3.Int("d_{}i{}0".format(i, var)),
                        z3.Int("d_{}i{}1".format(i, var)),
                        z3.Int("d_{}i{}2".format(i, var)),
                        z3.Int("d_{}i{}3".format(i, var)),
                    ]
                vars_for_dis.append(tvars)

            self.template_vars.append(vars_for_dis)
        # print(self.template_vars)

    def get_additional_cnts_for_template_vars(self):
        # IMPORTANT: the following are additional cnts for interval
        dis_cnts = []
        for vars_for_dis in self.template_vars:
            cnts = []
            for tvars in vars_for_dis:
                i0, i1, i2, i3 = tvars[0], tvars[1], tvars[2], tvars[3]
                # the second means "no lower bound"?
                cnts.append(z3.Or(i1 == 1, z3.And(i1 == 0, i0 == 0)))
                # the second means "no upper bound"?
                cnts.append(z3.Or(i3 == -1, z3.And(i3 == 0, i2 == 0)))
            dis_cnts.append(z3.And(cnts))
        return big_and(dis_cnts)

    def add_template_cnts(self):
        # FIXME: the following is from IntervalTemplate
        cnt_init_and_post_dis = []
        cnt_trans_dis = []
        for vars_for_dis in self.template_vars:
            # print("XXX", vars_for_dis)
            cnts_init_post = []  # For sts.variables
            cnts_trans = []  # For sts.prime_variables
            for i in range(self.arity):
                var = self.sts.variables[i]  # x, y
                prime_var = self.sts.prime_variables[i]  # x!, y!
                template_vars_for_var = vars_for_dis[i]  # TODO: is this correct?

                i0, i1 = template_vars_for_var[0], template_vars_for_var[1]
                i2, i3 = template_vars_for_var[2], template_vars_for_var[3]

                cnts_init_post.append(i0 + var * i1 >= 0)
                cnts_init_post.append(i2 + var * i3 >= 0)

                cnts_trans.append(i0 + prime_var * i1 >= 0)
                cnts_trans.append(i2 + prime_var * i3 >= 0)

            cnt_init_and_post_dis.append(z3.And(cnts_init_post))
            cnt_trans_dis.append(z3.And(cnts_trans))

        self.template_cnt_init_and_post = z3.Or(cnt_init_and_post_dis)
        self.template_cnt_trans = z3.Or(cnt_trans_dis)

    def build_invariant_expr(self, model: z3.ModelRef, use_prime_variables=False):
        # FIXME: the following is from IntervalTemplate
        cnts_dis = []
        # print("groups of template vars: ", len(self.template_vars))
        for vars_for_dis in self.template_vars:
            cnts = []
            for i in range(self.arity):
                var = self._get_variable(i, use_prime_variables)
                template_vars_for_var = vars_for_dis[i]
                i0, i1 = template_vars_for_var[0], template_vars_for_var[1]
                i2, i3 = template_vars_for_var[2], template_vars_for_var[3]

                if model[i1].as_long() != 0:
                    cnts.append(model[i0] + var * model[i1] >= 0)
                if model[i3].as_long() != 0:
                    cnts.append(model[i2] + var * model[i3] >= 0)
            cnts_dis.append(big_and(cnts))

        return z3.Or(cnts_dis)
