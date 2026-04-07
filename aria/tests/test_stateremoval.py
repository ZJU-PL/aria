import unittest

from aria.automata.symautomata.pythondfa import PythonDFA
from aria.automata.symautomata.stateremoval import StateRemoval


def make_dfa(alphabet, transitions, initial, finals):
    dfa = PythonDFA(alphabet)
    for src, dst, label in transitions:
        dfa.add_arc(src, dst, label)
    dfa[initial].initial = True
    for final in finals:
        dfa[final].final = True
    dfa.yy_accept = [1 if state.final else 0 for state in dfa.states]
    return dfa


class TestStateRemoval(unittest.TestCase):
    def test_star_handles_epsilon_and_nonempty_inputs(self):
        dfa = make_dfa(["a"], [(0, 1, "a")], initial=0, finals={1})
        state_removal = StateRemoval(dfa, ["a"])

        self.assertEqual(state_removal.star("ab"), "(ab)*")
        self.assertEqual(state_removal.star(state_removal.epsilon), "")
        self.assertEqual(state_removal.star(state_removal.empty), "")

    def test_state_removal_init_collects_parallel_transitions(self):
        dfa = make_dfa(
            ["a", "b"],
            [(0, 1, "a"), (0, 1, "b")],
            initial=0,
            finals={1},
        )
        state_removal = StateRemoval(dfa, ["a", "b"])

        state_removal._state_removal_init()

        self.assertEqual(state_removal.l_transitions[(0, 0)], "")
        self.assertEqual(state_removal.l_transitions[(0, 1)], "ab")
        self.assertIsNone(state_removal.l_transitions[(1, 0)])

    def test_get_regex_eliminates_nonfinal_intermediate_states(self):
        dfa = make_dfa(
            ["a", "b"],
            [(0, 1, "a"), (1, 2, "b")],
            initial=0,
            finals={2},
        )

        self.assertEqual(StateRemoval(dfa, ["a", "b"]).get_regex(), "ab")


if __name__ == "__main__":
    unittest.main()
