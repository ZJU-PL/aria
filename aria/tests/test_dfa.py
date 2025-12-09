import unittest
from typing import Set, Dict, Tuple
from aria.automata.fa import DFA


class TestDFA(unittest.TestCase):
    def setUp(self):
        # Simple DFA accepting strings ending with 'b'
        self.states = {'q0', 'q1'}
        self.alphabet = {'a', 'b'}
        self.transitions = {
            ('q0', 'a'): 'q0',
            ('q0', 'b'): 'q1',
            ('q1', 'a'): 'q0',
            ('q1', 'b'): 'q1'
        }
        self.initial_state = 'q0'
        self.accepting_states = {'q1'}
        self.dfa = DFA(self.states, self.alphabet, self.transitions,
                       self.initial_state, self.accepting_states)

    def test_init_valid(self):
        """Test valid DFA initialization"""
        self.assertEqual(self.dfa.states, self.states)
        self.assertEqual(self.dfa.alphabet, self.alphabet)
        self.assertEqual(self.dfa.transitions, self.transitions)
        self.assertEqual(self.dfa.initial_state, self.initial_state)
        self.assertEqual(self.dfa.accepting_states, self.accepting_states)

    def test_init_empty_states(self):
        """Test DFA initialization with empty states"""
        with self.assertRaises(ValueError):
            DFA(set(), self.alphabet, self.transitions,
                self.initial_state, self.accepting_states)

    def test_init_invalid_initial_state(self):
        """Test DFA initialization with invalid initial state"""
        with self.assertRaises(ValueError):
            DFA(self.states, self.alphabet, self.transitions,
                'invalid', self.accepting_states)

    def test_init_invalid_accepting_states(self):
        """Test DFA initialization with invalid accepting states"""
        with self.assertRaises(ValueError):
            DFA(self.states, self.alphabet, self.transitions,
                self.initial_state, {'invalid'})

    def test_init_invalid_transitions(self):
        """Test DFA initialization with invalid transitions"""
        invalid_transitions = {
            ('q0', 'a'): 'invalid',
            ('q0', 'b'): 'q1'
        }
        with self.assertRaises(ValueError):
            DFA(self.states, self.alphabet, invalid_transitions,
                self.initial_state, self.accepting_states)

    def test_accepts_valid_strings(self):
        """Test DFA accepts valid strings"""
        self.assertTrue(self.dfa.accepts('b'))
        self.assertTrue(self.dfa.accepts('ab'))
        self.assertTrue(self.dfa.accepts('aab'))
        self.assertTrue(self.dfa.accepts('abb'))

    def test_rejects_invalid_strings(self):
        """Test DFA rejects invalid strings"""
        self.assertFalse(self.dfa.accepts(''))
        self.assertFalse(self.dfa.accepts('a'))
        self.assertFalse(self.dfa.accepts('aa'))
        self.assertFalse(self.dfa.accepts('ba'))

    def test_rejects_invalid_symbols(self):
        """Test DFA rejects strings with invalid symbols"""
        self.assertFalse(self.dfa.accepts('c'))
        self.assertFalse(self.dfa.accepts('abc'))

    def test_minimize_equivalent_states(self):
        """Test DFA minimization with equivalent states"""
        # DFA with equivalent states q1 and q2
        states = {'q0', 'q1', 'q2'}
        transitions = {
            ('q0', 'a'): 'q1',
            ('q0', 'b'): 'q2',
            ('q1', 'a'): 'q1',
            ('q1', 'b'): 'q1',
            ('q2', 'a'): 'q2',
            ('q2', 'b'): 'q2'
        }
        dfa = DFA(states, self.alphabet, transitions, 'q0', {'q1', 'q2'})
        min_dfa = dfa.minimize()

        # Should merge q1 and q2 into single state
        self.assertEqual(len(min_dfa.states), 2)
        self.assertTrue(min_dfa.accepts('a'))
        self.assertTrue(min_dfa.accepts('b'))
        self.assertTrue(min_dfa.accepts('aa'))
        self.assertTrue(min_dfa.accepts('ab'))


if __name__ == '__main__':
    unittest.main()
