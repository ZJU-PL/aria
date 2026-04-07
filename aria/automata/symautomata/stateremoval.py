"""
This module transforms a pyfst DFA to regular expressions
using the State Removal Method
"""

from sys import argv
from operator import attrgetter
from aria.automata.symautomata.alphabet import createalphabet

try:
    from aria.automata.symautomata.dfa import DFA as DFAClass
except ImportError:
    from aria.automata.symautomata.pythondfa import PythonDFA as DFAClass


class StateRemoval:
    """Transforms a pyfst DFA to regular expressions"""

    def __init__(self, input_fst_a, alphabet=None):
        """
        Args:
            input_fst_a (DFA): The DFA states
            alphabet (list): the input alphabet
        Returns:
            None
        """
        if alphabet is None:
            alphabet = createalphabet()
        self.alphabet = alphabet
        self.mma = DFAClass(self.alphabet)
        self.mma.init_from_acceptor(input_fst_a)

        self.l_transitions = {}
        self.epsilon = ""
        self.empty = None

    def star(self, input_string):
        """
        Kleene star operation
        Args:
            input_string (str): The string that the kleene star will be made
        Returns:
            str: The applied Kleene star operation on the input string
        """
        if input_string != self.epsilon and input_string != self.empty:
            return "(" + input_string + ")*"
        else:
            return ""

    def _state_removal_init(self):
        """State Removal Operation Initialization"""
        # First, we remove all multi-edges:
        for state_i in self.mma.states:
            for state_j in self.mma.states:
                if state_i.stateid == state_j.stateid:
                    self.l_transitions[state_i.stateid, state_j.stateid] = self.epsilon
                else:
                    self.l_transitions[state_i.stateid, state_j.stateid] = self.empty

                for arc in state_i.arcs:
                    if arc.nextstate == state_j.stateid:
                        if (
                            self.l_transitions[state_i.stateid, state_j.stateid]
                            != self.empty
                        ):
                            self.l_transitions[
                                state_i.stateid, state_j.stateid
                            ] += self.mma.isyms.find(arc.ilabel)
                        else:
                            self.l_transitions[state_i.stateid, state_j.stateid] = (
                                self.mma.isyms.find(arc.ilabel)
                            )

    def _state_removal_remove(self, k):
        """
        State Removal Remove operation
        l_transitions[i,j] += l_transitions[i,k] . star(l_transitions[k,k]) . l_transitions[k,j]
        Args:
            k (int): The node that will be removed
        Returns:
            None
        """
        previous = dict(self.l_transitions)

        for state_i in self.mma.states:
            for state_j in self.mma.states:
                l_ij = previous[state_i.stateid, state_j.stateid]
                l_ik = previous[state_i.stateid, k]
                l_kj = previous[k, state_j.stateid]

                if l_ik == self.empty or l_kj == self.empty:
                    via_k = self.empty
                else:
                    via_k = l_ik + self.star(previous[k, k]) + l_kj

                if l_ij == self.empty:
                    self.l_transitions[state_i.stateid, state_j.stateid] = via_k
                elif via_k != self.empty:
                    self.l_transitions[state_i.stateid, state_j.stateid] += via_k

    def _state_removal_solve(self):
        """The State Removal Operation"""
        initial = sorted(self.mma.states, key=attrgetter("initial"), reverse=True)[
            0
        ].stateid
        for state_k in self.mma.states:
            if state_k.final:
                continue
            if state_k.stateid == initial:
                continue
            self._state_removal_remove(state_k.stateid)

        return self.l_transitions

    def get_regex(self):
        """Regular Expression Generation"""
        self._state_removal_init()
        self._state_removal_solve()
        initial = sorted(self.mma.states, key=attrgetter("initial"), reverse=True)[
            0
        ].stateid
        final_states = [state.stateid for state in self.mma.states if state.final]
        regexes = []
        for final_state in final_states:
            regex = self.l_transitions.get((initial, final_state), self.empty)
            if regex != self.empty:
                regexes.append(regex)
        if not regexes:
            return self.epsilon
        return "|".join(regexes)


def main():
    """Testing function for DFA _Brzozowski Operation"""
    from aria.automata.symautomata.flex2fst import Flexparser

    if len(argv) < 2:
        targetfile = "target.y"
    else:
        targetfile = argv[1]
    print("Parsing ruleset: " + targetfile, end=" ")
    flex_a = Flexparser()
    mma = flex_a.yyparse(targetfile)
    print("OK")
    print("Perform minimization on initial automaton:", end=" ")
    mma.minimize()
    print("OK")
    print("Perform StateRemoval on minimal automaton:", end=" ")
    state_removal = StateRemoval(mma)
    mma_regex = state_removal.get_regex()
    print(mma_regex)


if __name__ == "__main__":
    main()
