from functools import lru_cache
import z3
import time

from unified_planning.model.walkers.free_vars import FreeVarsExtractor

from enum import Enum, auto
from ..utilities import log
from .base import Modifier

import networkx as nx

class MutexSemantics(Enum):
    """Enum to define the mutex semantics for parallel execution."""
    FORALL = auto()  # All mutex constraints are added
    EXISTS = auto()  # Only mutexes in strongly connected components are added

class ParallelModifier(Modifier):
    """
    Parallel modifier, contains method to implement parallel execution semantics.
    """
    def __init__(self, semantics=MutexSemantics.FORALL, lazy=False):
        super().__init__("ParallelModifier")
        self.graph = nx.DiGraph()
        self.semantics = semantics
        self.lazy = lazy
        self.mutexes = set()

    def encode(self, encoder, actions) -> set:
        """!
        Computes mutually exclusive actions:
        Two actions (a1, a2) are mutex if:

            1- The effects of a1 can prevent the execution of a2
            - intersection pre_a1 and eff_a2 (or viceversa) is non-empty

            2- The effects of a1 and a2 interfere
            - intersection between eff_a1+ and eff_a2- (or viceversa) is non-empty
            - intersection between numeric effects is non-empty

        Note that condition 1 is non-symmetric, while condition 2 is.

        See,'Efficient SMT Encodings for the Petrobras Domain' Espasa et al. 
        Sec 3.3 - Parallel Plans

        @return mutex: list of tuples defining action mutexes
        """

        @lru_cache(maxsize=None)
        def get_add_del_effects(action):
            """!
            Returns a tuple (add, del) of lists of effects of the action
            """
            effects_fluents = [effect for effect in action.effects if effect.value.type.is_bool_type()]

            add_effects = set([eff.fluent for eff in effects_fluents if eff.value.is_true()])
            del_effects = set([eff.fluent for eff in effects_fluents if eff.value.is_false()])

            return (add_effects, del_effects)

        @lru_cache(maxsize=None)
        def get_numeric_effects(action):
            """!
            Returns a set of numeric effects of the action
            """
            return set([effect.fluent 
                        for effect in action.effects if 
                        effect.value.type.is_int_type() or effect.value.type.is_real_type()])

        @lru_cache(maxsize=None)
        def get_preconditions(action):
            """!
            Returns a set of preconditions of the action
            """
            preconditions = set()
            nameextractor = FreeVarsExtractor()
            for pre in action.preconditions:
                for fluent in nameextractor.get(pre):
                    preconditions.add(fluent)
            for eff in action.effects:
                # we have to consider any variables in the rhs of the effect (as it might be a linear expression)
                for fluent in nameextractor.get(eff.value):
                    preconditions.add(fluent)
                # we also have to consider the conditional effects
                if eff.is_conditional():
                    for fluent in nameextractor.get(eff.condition):
                        preconditions.add(fluent)
            return preconditions

        # we avoid computing some of those twice on the following double for loop
        start_time = time.time()
        data_actions = {}
        for act in encoder.task.actions:
            add_a, del_a = get_add_del_effects(act)
            num_a = get_numeric_effects(act)
            pre_a = get_preconditions(act)
            data_actions[act] = (add_a, del_a, num_a, pre_a)
        end_time = time.time()
        log(f'indexing actions for mutex computation took {end_time-start_time:.2f}s', 2)

        # main body of the function
        start_time = time.time()
        actions = encoder.task.actions

        def add_edge(action1, action2):
            self.graph.add_edge(action1.name, action2.name)

        for action in actions:
            self.graph.add_node(action.name)

        # Iterate over actions to identify mutex pairs
        for i, action_1 in enumerate(actions):
            add_a1, del_a1, num_1, pre_1 = data_actions[action_1]
            for action_2 in actions[i + 1:]:
                add_a2, del_a2, num_2, pre_2 = data_actions[action_2]

                # Condition 1: can a1 prohibit the execution of a2 or vice-versa?
                if len(pre_2.intersection(set.union(*[add_a1, del_a1, num_1]))) > 0:
                    add_edge(action_1, action_2)
                if len(pre_1.intersection(set.union(*[add_a2, del_a2, num_2]))) > 0:
                    add_edge(action_2, action_1)

                ## Condition 2: Do the effects of a1 and a2 interfere?
                if len(add_a1.intersection(del_a2)) > 0 or \
                        len(add_a2.intersection(del_a1)) > 0 or \
                        len(num_1.intersection(num_2)) > 0:
                    add_edge(action_1, action_2)
                    add_edge(action_2, action_1)

        def generate_for_all():
            for edge in self.graph.edges():
                a1, a2 = edge
                a1 = encoder.get_action_var(a1, 0)
                a2 = encoder.get_action_var(a2, 0)
                m1 = z3.Not(z3.And(a1, a2))
                m2 = z3.Not(z3.And(a2, a1))
                if m1 not in self.mutexes and m2 not in self.mutexes:
                    self.mutexes.add(m1)

        def generate_exists():
            components = nx.strongly_connected_components(self.graph)
            for c in components:
                if len(c) > 1:
                    numbers = {a: i for i, a in enumerate(c)}
                    subgraph = self.graph.subgraph(c)
                    for edge in subgraph.edges():
                        act_1, act_2 = edge
                        a1 = encoder.get_action_var(act_1, 0)
                        a2 = encoder.get_action_var(act_2, 0)
                        if numbers[act_1] <= numbers[act_2]:
                            m1 = z3.Not(z3.And(a1, a2))
                            m2 = z3.Not(z3.And(a2, a1))
                            if m1 not in self.mutexes and m2 not in self.mutexes:
                                self.mutexes.add(m1)

        if not self.lazy:
            if self.semantics == MutexSemantics.FORALL:
                generate_for_all()
            else:
                generate_exists()
        end_time = time.time()
        log(f'computed {len(self.mutexes)} mutexes, took {end_time-start_time:.2f}s', 2)
        return self.mutexes
