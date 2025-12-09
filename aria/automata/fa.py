"""
Manipulation of Finite Automata (FA)

- DFA
- NFA
"""
from typing import Set, Dict, Tuple
from collections import defaultdict


class DFA:
    """Deterministic Finite Automaton"""

    def __init__(self, states: Set[str], alphabet: Set[str],
                 transitions: Dict[Tuple[str, str], str],
                 initial_state: str, accepting_states: Set[str]):
        """Initialize DFA with validation
        
        Args:
            states: Set of states
            alphabet: Input alphabet
            transitions: Transition function as (state, symbol) -> state
            initial_state: Initial state
            accepting_states: Set of accepting states
            
        Raises:
            ValueError: If any validation fails
        """
        # Validate non-empty states
        if not states:
            raise ValueError("States set cannot be empty")

        # Validate initial state exists
        if initial_state not in states:
            raise ValueError(f"Initial state '{initial_state}' not in states set")

        # Validate accepting states exist
        if not accepting_states.issubset(states):
            invalid = accepting_states - states
            raise ValueError(f"Accepting states {invalid} not in states set")

        # Validate transitions
        for (state, symbol), next_state in transitions.items():
            if state not in states:
                raise ValueError(f"Transition from invalid state '{state}'")
            if symbol not in alphabet:
                raise ValueError(f"Transition on invalid symbol '{symbol}'")
            if next_state not in states:
                raise ValueError(f"Transition to invalid state '{next_state}'")

        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.initial_state = initial_state
        self.accepting_states = accepting_states

    def accepts(self, word: str) -> bool:
        """Check if the DFA accepts a word"""
        current = self.initial_state
        for symbol in word:
            if symbol not in self.alphabet:
                return False
            if (current, symbol) not in self.transitions:
                return False
            current = self.transitions[(current, symbol)]
        return current in self.accepting_states

    def minimize(self) -> 'DFA':
        """Minimize DFA using Hopcroft's algorithm"""
        # Initial partition: accepting and non-accepting states
        partition = [self.accepting_states, self.states - self.accepting_states]
        partition = [s for s in partition if s]  # Remove empty sets

        while True:
            new_partition = []
            for group in partition:
                splits = defaultdict(set)
                for state in group:
                    signature = tuple(
                        next(i for i, g in enumerate(partition)
                             if self.transitions.get((state, a)) in g)
                        for a in sorted(self.alphabet)
                    )
                    splits[signature].add(state)
                new_partition.extend(splits.values())

            if len(new_partition) == len(partition):
                break
            partition = new_partition

        # Build minimized DFA
        state_map = {}
        for i, group in enumerate(partition):
            for state in group:
                state_map[state] = f"q{i}"

        new_states = set(state_map.values())
        new_transitions = {}
        for (state, symbol), next_state in self.transitions.items():
            new_transitions[(state_map[state], symbol)] = state_map[next_state]

        new_accepting = {state_map[s] for s in self.accepting_states}
        new_initial = state_map[self.initial_state]

        return DFA(new_states, self.alphabet, new_transitions, new_initial, new_accepting)


class NFA:
    """Nondeterministic Finite Automaton"""

    def __init__(self, states: Set[str], alphabet: Set[str],
                 transitions: Dict[Tuple[str, str], Set[str]],
                 initial_states: Set[str], accepting_states: Set[str]):
        """Initialize NFA with validation
        
        Args:
            states: Set of states
            alphabet: Input alphabet
            transitions: Transition function as (state, symbol) -> set of states
            initial_states: Set of initial states
            accepting_states: Set of accepting states
            
        Raises:
            ValueError: If any validation fails
        """
        # Validate non-empty states
        if not states:
            raise ValueError("States set cannot be empty")

        # Validate initial states exist
        if not initial_states.issubset(states):
            invalid = initial_states - states
            raise ValueError(f"Initial states {invalid} not in states set")

        # Validate accepting states exist
        if not accepting_states.issubset(states):
            invalid = accepting_states - states
            raise ValueError(f"Accepting states {invalid} not in states set")

        # Validate transitions
        for (state, symbol), next_states in transitions.items():
            if state not in states:
                raise ValueError(f"Transition from invalid state '{state}'")
            if symbol not in alphabet and symbol != "":  # Allow epsilon transitions
                raise ValueError(f"Transition on invalid symbol '{symbol}'")
            if not next_states.issubset(states):
                invalid = next_states - states
                raise ValueError(f"Transition to invalid states {invalid}")

        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.initial_states = initial_states
        self.accepting_states = accepting_states

    def accepts(self, word: str) -> bool:
        """Check if the NFA accepts a word"""
        current_states = self.initial_states

        for symbol in word:
            if symbol not in self.alphabet:
                return False

            next_states = set()
            for state in current_states:
                if (state, symbol) in self.transitions:
                    next_states.update(self.transitions[(state, symbol)])
                if (state, "") in self.transitions:  # Follow epsilon transitions
                    next_states.update(self.transitions[(state, "")])

            if not next_states:
                return False
            current_states = next_states

        # Follow final epsilon transitions
        final_states = current_states.copy()
        for state in current_states:
            if (state, "") in self.transitions:
                final_states.update(self.transitions[(state, "")])

        return bool(final_states & self.accepting_states)

    def to_dfa(self) -> DFA:
        """Convert NFA to equivalent DFA using subset construction"""
        dfa_states = set()
        dfa_transitions = {}
        work_list = [frozenset(self.initial_states)]
        dfa_states.add(work_list[0])

        while work_list:
            current_states = work_list.pop()

            for symbol in self.alphabet:
                next_states = set()
                for state in current_states:
                    if (state, symbol) in self.transitions:
                        next_states.update(self.transitions[(state, symbol)])
                    if (state, "") in self.transitions:  # Follow epsilon transitions
                        eps_states = self.transitions[(state, "")]
                        next_states.update(eps_states)
                        for eps_state in eps_states:
                            if (eps_state, symbol) in self.transitions:
                                next_states.update(self.transitions[(eps_state, symbol)])

                next_states_frozen = frozenset(next_states)
                if next_states and next_states_frozen not in dfa_states:
                    dfa_states.add(next_states_frozen)
                    work_list.append(next_states_frozen)

                dfa_transitions[(current_states, symbol)] = next_states_frozen

        # Convert frozenset states to strings
        state_map = {states: f"q{i}" for i, states in enumerate(dfa_states)}

        new_transitions = {
            (state_map[s], a): state_map[ns]
            for (s, a), ns in dfa_transitions.items()
        }

        new_accepting = {
            state_map[s] for s in dfa_states
            if s & self.accepting_states
        }

        return DFA(
            set(state_map.values()),
            self.alphabet,
            new_transitions,
            state_map[frozenset(self.initial_states)],
            new_accepting
        )
