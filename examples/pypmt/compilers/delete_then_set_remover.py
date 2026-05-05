"""This module defines the delete-then-set remover class."""

import unified_planning as up
import unified_planning.engines as engines
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import replace_action
from unified_planning.shortcuts import EffectKind

from unified_planning.model import (
    Problem,
    ProblemKind,
    Action,
)

from typing import Optional, Dict
from functools import partial

# TODO: check if this can be better integrated via the 
# add_engine: https://github.com/aiplan4eu/unified-planning/blob/master/unified_planning/engines/factory.py
class DeleteThenSetRemover(engines.engine.Engine, CompilerMixin):
    """
    Delete-then-set remover class: this class offers the capability to transform
    a :class:`~unified_planning.model.Problem` with delete-then-set effects into
    a `Problem` without them.  This capability is offered by the
    :meth:`~compile`
    method, that returns a :class:`~unified_planning.engines.CompilerResult` in
    which the :meth:`problem <unified_planning.engines.CompilerResult.problem>`
    field is the compiled Problem.
    
    This is done by following the FD approach outlined here:
    https://www.fast-downward.org/ForDevelopers/ProblemTransformations

    "Following PDDL's add-after-delete semantics, if an action adds and deletes a
    fact at the same time, the translator can remove the delete effect."

    HACK: This class does not conform to the UP API. Hopefully will get integrated in the future.
    TODO: check if this is sound, as effectively we do not remove any supported kind ...
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.GROUNDING)

    @property
    def name(self):
        return "dtsrm"

    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = ProblemKind(version=LATEST_PROBLEM_KIND_VERSION)
        supported_kind.set_problem_class("ACTION_BASED")
        supported_kind.set_typing("FLAT_TYPING")
        supported_kind.set_typing("HIERARCHICAL_TYPING")
        supported_kind.set_numbers("BOUNDED_TYPES")
        supported_kind.set_problem_type("SIMPLE_NUMERIC_PLANNING")
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
        supported_kind.set_fluents_type("INT_FLUENTS")
        supported_kind.set_fluents_type("REAL_FLUENTS")
        supported_kind.set_fluents_type("OBJECT_FLUENTS")
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITIES")
        supported_kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
        supported_kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        supported_kind.set_effects_kind("CONDITIONAL_EFFECTS")
        supported_kind.set_effects_kind("INCREASE_EFFECTS")
        supported_kind.set_effects_kind("DECREASE_EFFECTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_BOOLEAN_ASSIGNMENTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_NUMERIC_ASSIGNMENTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_OBJECT_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_BOOLEAN_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_NUMERIC_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_OBJECT_ASSIGNMENTS")
        supported_kind.set_effects_kind("FORALL_EFFECTS")
        supported_kind.set_simulated_entities("SIMULATED_EFFECTS")
        supported_kind.set_constraints_kind("STATE_INVARIANTS")
        supported_kind.set_constraints_kind("TRAJECTORY_CONSTRAINTS")
        supported_kind.set_quality_metrics("ACTIONS_COST")
        supported_kind.set_actions_cost_kind("STATIC_FLUENTS_IN_ACTIONS_COST")
        supported_kind.set_actions_cost_kind("FLUENTS_IN_ACTIONS_COST")
        supported_kind.set_quality_metrics("PLAN_LENGTH")
        supported_kind.set_quality_metrics("OVERSUBSCRIPTION")
        supported_kind.set_quality_metrics("MAKESPAN")
        supported_kind.set_quality_metrics("FINAL_VALUE")
        supported_kind.set_actions_cost_kind("INT_NUMBERS_IN_ACTIONS_COST")
        supported_kind.set_actions_cost_kind("REAL_NUMBERS_IN_ACTIONS_COST")
        supported_kind.set_oversubscription_kind("INT_NUMBERS_IN_OVERSUBSCRIPTION")
        supported_kind.set_oversubscription_kind("REAL_NUMBERS_IN_OVERSUBSCRIPTION")
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= DeleteThenSetRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return True #compilation_kind == CompilationKind.DELETE_THEN_SET_REMOVING # we do not support anything in particular, just cleaning up the problem

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind,
        compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        return problem_kind.clone() # we do not change the problem kind

    def _compile(
        self,
        problem: "up.model.AbstractProblem",
        compilation_kind: "up.engines.CompilationKind",
    ) -> CompilerResult:
        """
        Takes an instance of a :class:`~unified_planning.model.Problem` and the wanted :class:`~unified_planning.engines.CompilationKind`
        and returns a :class:`~unified_planning.engines.results.CompilerResult` where the :meth:`problem<unified_planning.engines.results.CompilerResult.problem>`
        field does not have state innvariants.

        :param problem: The instance of the
        :class:`~unified_planning.model.Problem` that must be returned without
        state innvariants.

        :param compilation_kind: The
        :class:`~unified_planning.engines.CompilationKind` that must be applied
        on the given problem; only
        :class:`~unified_planning.engines.CompilationKind.STATE_INVARIANTS_REMOVING`
        is supported by this compiler
        
        :return: The resulting :class:`~unified_planning.engines.results.CompilerResult` data structure.
        """
        assert isinstance(problem, Problem)
        env = problem.environment
        em = env.expression_manager

        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        new_problem.clear_actions()

        new_to_old: Dict[Action, Optional[Action]] = {}
        for a in problem.actions:
            fixed_action = self.remove_delete_then_set(a)
            new_problem.add_action(fixed_action)
            new_to_old[fixed_action] = a

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )

    def remove_delete_then_set(self, dirty_action: Action) -> Action:
        """!
        Removes delete-then-set effects from the list of effects.
        @param effects: list of effects
        @return list of effects without delete-then-set effects
        """

        def has_positive_effect(fluent, action) -> bool:
            """ Does the action has an effect that assigns the fluent to true? """
            for eff in action.effects:
                if eff.kind == EffectKind.ASSIGN and eff.fluent == fluent and eff.value.is_true():
                    return True
            return False

        clean_effects = []
        for eff in dirty_action.effects:
        # we avoid adding the effect if it is a delete effect and the action has also an add effect for the same fluent
            if eff.fluent.type.is_bool_type(): # only check boolean fluents
                if eff.kind == EffectKind.ASSIGN and \
                    eff.value.is_false() and \
                    has_positive_effect(eff.fluent, dirty_action):
                    pass
                else: 
                    clean_effects.append(eff)
            else: 
                clean_effects.append(eff)

        fixed_action = dirty_action.clone() # we copy the old action
        fixed_action.clear_effects()        # and remove all the effects
        for eff in clean_effects:           # now we copy over only the good effects
            if eff.kind == EffectKind.ASSIGN:
                fixed_action.add_effect(eff.fluent, eff.value, eff.condition, forall=eff.forall)
            if eff.kind == EffectKind.DECREASE:
                fixed_action.add_decrease_effect(eff.fluent, eff.value, eff.condition, forall=eff.forall)
            if eff.kind == EffectKind.INCREASE:
                fixed_action.add_increase_effect(eff.fluent, eff.value, eff.condition, forall=eff.forall)
        return fixed_action
