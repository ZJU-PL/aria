"""Octagon template over integer or real variables"""

import logging

from aria.efmc.engines.ef.templates.abstract_template import *
from aria.efmc.sts import TransitionSystem

logger = logging.getLogger(__name__)


class OctagonTemplate(Template):

    def __init__(self, sts: TransitionSystem, **kwargs):
        self.template_type = TemplateType.OCTAGON

        if sts.has_real:
            self.use_real = True
        else:
            self.use_real = False
        self.sts = sts

    def add_template_vars(self):
        raise NotImplementedError

    def add_template_cnts(self):
        raise NotImplementedError

    def build_invariant_expr(self, model: z3.ModelRef, use_prime_variables: bool):
        raise NotImplementedError


class DisjunctiveOctagonTemplate(Template):
    def __init__(self, sts: TransitionSystem, **kwargs):
        self.template_type = TemplateType.DISJUNCTIVE_OCTAGON

        if sts.has_real:
            self.use_real = True
        else:
            self.use_real = False
        self.sts = sts

    def add_template_vars(self):
        raise NotImplementedError

    def add_template_cnts(self):
        raise NotImplementedError

    def build_invariant_expr(self, model: z3.ModelRef, use_prime_variables: bool):
        raise NotImplementedError
