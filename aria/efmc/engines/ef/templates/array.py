"""
FIXME: this file is not used yet
"""

import logging
from typing import Optional, Any

from aria.efmc.engines.ef.templates.abstract_template import *
from aria.efmc.sts import TransitionSystem

logger = logging.getLogger(__name__)


class ArrayTemplate(Template):
    """
    Currently, we focus on ALIA.
    TODO: a major challenge is to infer "quantified invariants"
    E.g., https://www.cs.upc.edu/~albert/papers/vmcai2013.pdf
    """

    def __init__(self, system: TransitionSystem, **kwargs: Any) -> None:
        self.template_type = TemplateType.ARRAY
        self.sts = system
        # because it only deals with int/real?

        self.numeric_template: Optional[Template] = None

    def build_numeric_domain(self) -> None:
        return

    def add_template_vars(self) -> None:
        if self.numeric_template:
            self.numeric_template.add_teplate_vars()

    def add_template_cnts(self) -> None:
        raise NotImplementedError

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables: bool
    ) -> z3.ExprRef:
        raise NotImplementedError


class UFTemplate(Template):
    """
    Currently, we focus on UFLIA
    """

    def __init__(self, system: TransitionSystem, **kwargs: Any) -> None:
        self.template_type = TemplateType.ARRAY
        self.sts = system
        # TODO: we should be "directly" pass system to IntervalTemplate
        # because it only deals with int/real?

        self.numeric_template: Optional[Template] = None

    def build_numeric_domain(self) -> None:
        return

    def add_template_vars(self) -> None:
        if self.numeric_template:
            self.numeric_template.add_teplate_vars()

    def add_template_cnts(self) -> None:
        raise NotImplementedError

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables: bool
    ) -> z3.ExprRef:
        raise NotImplementedError
