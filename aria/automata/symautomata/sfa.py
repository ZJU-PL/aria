from __future__ import annotations

from collections import deque
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import z3


class Predicate:
    def is_sat(
        self,
        symbol: Optional[object] = None,
        universe: Optional[object] = None,
    ) -> bool:
        raise NotImplementedError("is_sat method not implemented")

    def conjunction(self, other: "Predicate") -> "Predicate":
        raise NotImplementedError("conjunction method not implemented")

    def disjunction(self, other: "Predicate") -> "Predicate":
        raise NotImplementedError("disjunction method not implemented")

    def negate(self) -> "Predicate":
        raise NotImplementedError("negate method not implemented")

    def is_equivalent(
        self,
        other: "Predicate",
        universe: Optional[object] = None,
    ) -> bool:
        raise NotImplementedError("is_equivalent method not implemented")

    def get_witness(self, universe: Optional[object] = None) -> object:
        raise NotImplementedError("get_witness method not implemented")

    def set_universe(self, universe: Iterable[object]) -> None:
        del universe

    def top(self) -> "Predicate":
        return self.disjunction(self.negate())

    def __and__(self, other: "Predicate") -> "Predicate":
        return self.conjunction(other)

    def __or__(self, other: "Predicate") -> "Predicate":
        return self.disjunction(other)

    def __invert__(self) -> "Predicate":
        return self.negate()

    def __iter__(self) -> Iterator[object]:
        raise TypeError("Predicate is not enumerable")


class SetPredicate(Predicate):
    def __init__(
        self,
        initializer: Iterable[object],
        universe: Optional[Iterable[object]] = None,
    ):
        self.pset: Set[object] = set(initializer)
        self.universe: Optional[Set[object]] = (
            set(universe) if universe is not None else None
        )

    def is_sat(
        self,
        symbol: Optional[object] = None,
        universe: Optional[object] = None,
    ) -> bool:
        del universe
        if symbol is None:
            return bool(self.pset)
        return symbol in self.pset

    def conjunction(self, other: Predicate) -> Predicate:
        if not isinstance(other, SetPredicate):
            raise TypeError("SetPredicate can only combine with SetPredicate")
        universe = self._merged_universe(other)
        return SetPredicate(self.pset & other.pset, universe)

    def disjunction(self, other: Predicate) -> Predicate:
        if not isinstance(other, SetPredicate):
            raise TypeError("SetPredicate can only combine with SetPredicate")
        universe = self._merged_universe(other)
        return SetPredicate(self.pset | other.pset, universe)

    def negate(self) -> Predicate:
        if self.universe is None:
            raise ValueError("SetPredicate negation requires a universe")
        return SetPredicate(self.universe - self.pset, self.universe)

    def is_equivalent(
        self,
        other: Predicate,
        universe: Optional[object] = None,
    ) -> bool:
        del universe
        if not isinstance(other, SetPredicate):
            return False
        return self.pset == other.pset

    def get_witness(self, universe: Optional[object] = None) -> object:
        del universe
        if not self.pset:
            raise ValueError("Unsatisfiable predicate has no witness")
        return next(iter(self.pset))

    def set_universe(self, universe: Iterable[object]) -> None:
        self.universe = set(universe)

    def _merged_universe(self, other: "SetPredicate") -> Optional[Set[object]]:
        if self.universe is None:
            return other.universe
        if other.universe is None:
            return self.universe
        return self.universe | other.universe

    def __iter__(self) -> Iterator[object]:
        return iter(self.pset)


class Z3Predicate(Predicate):
    def __init__(
        self,
        symbol: Any,
        formula: Any,
    ):
        self.symbol = symbol
        self.formula = z3.simplify(formula)

    @classmethod
    def bitvec(
        cls,
        width: int,
        formula: Optional[Any] = None,
        name: str = "sym",
    ) -> "Z3Predicate":
        symbol = z3.BitVec(name, width)
        if formula is None:
            formula = z3.BoolVal(True)
        return cls(symbol, formula)

    def is_sat(
        self,
        symbol: Optional[object] = None,
        universe: Optional[object] = None,
    ) -> bool:
        solver = z3.Solver()
        solver.add(self._semantic_formula(universe))
        if symbol is not None:
            solver.add(self.symbol == self._to_symbol_value(symbol))
        return solver.check() == z3.sat

    def conjunction(self, other: Predicate) -> Predicate:
        aligned_formula = self._aligned_formula(other)
        return Z3Predicate(self.symbol, z3.And(self.formula, aligned_formula))

    def disjunction(self, other: Predicate) -> Predicate:
        aligned_formula = self._aligned_formula(other)
        return Z3Predicate(self.symbol, z3.Or(self.formula, aligned_formula))

    def negate(self) -> Predicate:
        return Z3Predicate(self.symbol, z3.Not(self.formula))

    def is_equivalent(
        self,
        other: Predicate,
        universe: Optional[object] = None,
    ) -> bool:
        aligned_formula = self._aligned_formula(other)
        solver = z3.Solver()
        active_universe = universe if universe is not None else z3.BoolVal(True)
        solver.add(
            z3.Xor(
                self._semantic_formula(active_universe),
                z3.And(active_universe, aligned_formula),
            )
        )
        return solver.check() == z3.unsat

    def get_witness(self, universe: Optional[object] = None) -> object:
        solver = z3.Solver()
        solver.add(self._semantic_formula(universe))
        if solver.check() != z3.sat:
            raise ValueError("Unsatisfiable predicate has no witness")
        model = solver.model()
        return self._from_model_value(model.eval(self.symbol, model_completion=True))

    def top(self) -> Predicate:
        return Z3Predicate(self.symbol, z3.BoolVal(True))

    def _aligned_formula(self, other: Predicate) -> Any:
        if not isinstance(other, Z3Predicate):
            raise TypeError("Z3Predicate can only combine with Z3Predicate")
        if self.symbol.sort() != other.symbol.sort():
            raise TypeError("Z3Predicate sorts must match")
        substitution = [(other.symbol, self.symbol)]
        return z3.simplify(z3.substitute(other.formula, *substitution))

    def _to_symbol_value(self, symbol: object) -> Any:
        sort = self.symbol.sort()
        if z3.is_bv_sort(sort):
            bv_size = sort.size()
            if isinstance(symbol, str):
                if len(symbol) != 1:
                    raise ValueError("Only single-character strings map to bit-vectors")
                return z3.BitVecVal(ord(symbol), bv_size)
            return z3.BitVecVal(self._coerce_int(symbol), bv_size)
        if sort.kind() == z3.Z3_INT_SORT:
            return z3.IntVal(self._coerce_int(symbol))
        if sort.kind() == z3.Z3_BOOL_SORT:
            return z3.BoolVal(bool(symbol))
        raise TypeError("Unsupported Z3 sort for symbolic predicates")

    def _from_model_value(self, value: Any) -> object:
        if z3.is_bv_value(value):
            return value.as_long()
        if z3.is_int_value(value):
            return value.as_long()
        if z3.is_true(value):
            return True
        if z3.is_false(value):
            return False
        return value

    def _coerce_int(self, symbol: object) -> int:
        if isinstance(symbol, bool):
            return int(symbol)
        if isinstance(symbol, int):
            return symbol
        raise TypeError("Symbol must be an integer or single-character string")

    def _semantic_formula(self, universe: Optional[object] = None) -> Any:
        if universe is None:
            return self.formula
        return z3.And(universe, self.formula)


class SFAState:
    def __init__(self, sid: Optional[int] = None):
        self.final = False
        self.initial = False
        self.state_id = sid
        self.arcs: List[SFAArc] = []

    def __iter__(self) -> Iterator["SFAArc"]:
        return iter(self.arcs)


class SFAArc:
    def __init__(
        self,
        src_state_id: int,
        dst_state_id: int,
        guard_p: Predicate,
        term: Optional[object] = None,
    ):
        self.src_state = src_state_id
        self.dst_state = dst_state_id
        self.guard = guard_p
        self.term = term


class SFA:
    def __init__(
        self,
        alphabet: Optional[Sequence[object]] = None,
        predicate_factory: Optional[Callable[[], Predicate]] = None,
        symbolic_symbol: Optional[Any] = None,
        symbolic_universe: Optional[Any] = None,
    ):
        self.states: List[SFAState] = []
        self.arcs: List[SFAArc] = []
        self.alphabet = list(alphabet) if alphabet is not None else None
        self.predicate_factory = predicate_factory
        self.symbolic_symbol = symbolic_symbol
        self.symbolic_universe = (
            z3.simplify(symbolic_universe) if symbolic_universe is not None else None
        )

    def __getitem__(self, index: int) -> SFAState:
        return self.states[index]

    def add_state(self, initial: bool = False, final: bool = False) -> int:
        sid = len(self.states)
        state = SFAState(sid)
        if sid == 0 and not any(existing.initial for existing in self.states):
            state.initial = True
        if initial:
            for existing in self.states:
                existing.initial = False
            state.initial = True
        state.final = final
        self.states.append(state)
        return sid

    @classmethod
    def symbolic(
        cls,
        symbolic_symbol: Any,
        symbolic_universe: Any,
        alphabet: Optional[Sequence[object]] = None,
        predicate_factory: Optional[Callable[[], Predicate]] = None,
    ) -> "SFA":
        return cls(
            alphabet=alphabet,
            predicate_factory=predicate_factory,
            symbolic_symbol=symbolic_symbol,
            symbolic_universe=symbolic_universe,
        )

    @classmethod
    def symbolic_bitvec(
        cls,
        width: int,
        alphabet: Optional[Sequence[object]] = None,
        predicate_factory: Optional[Callable[[], Predicate]] = None,
        name: str = "sym",
        symbolic_universe: Optional[Any] = None,
    ) -> "SFA":
        symbolic_symbol = z3.BitVec(name, width)
        if symbolic_universe is None:
            symbolic_universe = z3.BoolVal(True)
        return cls.symbolic(
            symbolic_symbol=symbolic_symbol,
            symbolic_universe=symbolic_universe,
            alphabet=alphabet,
            predicate_factory=predicate_factory,
        )

    @classmethod
    def symbolic_int(
        cls,
        alphabet: Optional[Sequence[object]] = None,
        predicate_factory: Optional[Callable[[], Predicate]] = None,
        name: str = "sym",
        symbolic_universe: Optional[Any] = None,
    ) -> "SFA":
        symbolic_symbol = z3.Int(name)
        if symbolic_universe is None:
            symbolic_universe = z3.BoolVal(True)
        return cls.symbolic(
            symbolic_symbol=symbolic_symbol,
            symbolic_universe=symbolic_universe,
            alphabet=alphabet,
            predicate_factory=predicate_factory,
        )

    @classmethod
    def symbolic_bool(
        cls,
        alphabet: Optional[Sequence[object]] = None,
        predicate_factory: Optional[Callable[[], Predicate]] = None,
        name: str = "sym",
        symbolic_universe: Optional[Any] = None,
    ) -> "SFA":
        symbolic_symbol = z3.Bool(name)
        if symbolic_universe is None:
            symbolic_universe = z3.BoolVal(True)
        return cls.symbolic(
            symbolic_symbol=symbolic_symbol,
            symbolic_universe=symbolic_universe,
            alphabet=alphabet,
            predicate_factory=predicate_factory,
        )

    @classmethod
    def from_acceptor(
        cls,
        acceptor: object,
        alphabet: Optional[Sequence[object]] = None,
    ) -> "SFA":
        derived_alphabet = alphabet
        if derived_alphabet is None:
            derived_alphabet = getattr(acceptor, "alphabet", None)
        result = cls(derived_alphabet)
        result.init_from_acceptor(acceptor)
        return result

    def init_from_acceptor(self, acceptor: object) -> None:
        acceptor_obj: Any = acceptor
        derived_alphabet = getattr(acceptor, "alphabet", None)
        if self.alphabet is None and derived_alphabet is not None:
            self.alphabet = list(derived_alphabet)
        self.states = []
        self.arcs = []

        for state in getattr(acceptor_obj, "states", []):
            self.add_state(initial=bool(state.initial), final=bool(state.final))

        if not self.states and hasattr(acceptor_obj, "states"):
            return

        for state in getattr(acceptor_obj, "states", []):
            for arc in state.arcs:
                symbol = acceptor_obj.isyms.find(arc.ilabel)
                self.add_arc(
                    state.stateid,
                    arc.nextstate,
                    SetPredicate([symbol], self.alphabet),
                )

    def set_final(self, state_id: int, final: bool = True) -> None:
        self._ensure_state(state_id)
        self.states[state_id].final = final

    def add_arc(
        self,
        src: int,
        dst: int,
        guard: object,
        term: Optional[object] = None,
    ) -> None:
        self._ensure_state(max(src, dst))
        predicate = self._coerce_guard(guard)
        arc = SFAArc(src, dst, predicate, term)
        self.states[src].arcs.append(arc)
        self.arcs.append(arc)

    def consume_input(self, inp: Sequence[object]) -> bool:
        current_states = set(self.initial_states())
        for symbol in self._iter_input(inp):
            next_states: Set[int] = set()
            for state_id in current_states:
                for arc in self.states[state_id].arcs:
                    if self._guard_is_sat(arc.guard, symbol):
                        next_states.add(arc.dst_state)
            if not next_states:
                return False
            current_states = next_states
        return any(self.states[state_id].final for state_id in current_states)

    def accepts(self, inp: Sequence[object]) -> bool:
        return self.consume_input(inp)

    def is_empty(self) -> bool:
        return self.get_witness() is None

    def get_witness(self) -> Optional[List[object]]:
        if not self.states:
            return None
        queue: Deque[Tuple[int, List[object]]] = deque(
            (state_id, []) for state_id in self.initial_states()
        )
        visited = set(self.initial_states())
        while queue:
            state_id, word = queue.popleft()
            state = self.states[state_id]
            if state.final:
                return word
            for arc in state.arcs:
                if not self._guard_is_sat(arc.guard):
                    continue
                if arc.dst_state in visited:
                    continue
                visited.add(arc.dst_state)
                queue.append(
                    (
                        arc.dst_state,
                        word + [self._guard_get_witness(arc.guard)],
                    )
                )
        return None

    def intersection(self, other: "SFA") -> "SFA":
        return self._product(other, lambda left, right: left and right)

    def union(self, other: "SFA") -> "SFA":
        return self._product(other, lambda left, right: left or right, complete=True)

    def difference(self, other: "SFA") -> "SFA":
        return self._product(other, lambda left, right: left and not right, complete=True)

    def symmetric_difference(self, other: "SFA") -> "SFA":
        return self._product(other, lambda left, right: left != right, complete=True)

    def complement(self) -> "SFA":
        deterministic = self.determinize().complete()
        result = deterministic.copy()
        for state in result.states:
            state.final = not state.final
        return result

    def is_equivalent(self, other: "SFA") -> bool:
        return self.symmetric_difference(other).is_empty()

    def determinize(self) -> "SFA":
        if not self.states:
            result = SFA(
                self.alphabet,
                self.predicate_factory,
                self.symbolic_symbol,
                self.symbolic_universe,
            )
            result.add_state(initial=True, final=False)
            return result
        result = SFA(
            self.alphabet,
            self.predicate_factory,
            self.symbolic_symbol,
            self.symbolic_universe,
        )
        initial_subset = frozenset(self.initial_states())
        subset_to_state: Dict[frozenset[int], int] = {}
        queue: Deque[frozenset[int]] = deque([initial_subset])

        initial_state_id = result.add_state(initial=True)
        result.states[initial_state_id].final = any(
            self.states[state_id].final for state_id in initial_subset
        )
        subset_to_state[initial_subset] = initial_state_id

        while queue:
            current_subset = queue.popleft()
            current_id = subset_to_state[current_subset]
            outgoing = [
                arc for state_id in current_subset for arc in self.states[state_id].arcs
            ]
            transitions = self._partition_transitions(outgoing)
            for guard, destination_subset in transitions:
                if destination_subset not in subset_to_state:
                    subset_to_state[destination_subset] = result.add_state()
                    result.states[subset_to_state[destination_subset]].final = any(
                        self.states[state_id].final for state_id in destination_subset
                    )
                    queue.append(destination_subset)
                result.add_arc(current_id, subset_to_state[destination_subset], guard)
        return result

    def complete(self) -> "SFA":
        result = self.determinize().copy()
        sink_state_id: Optional[int] = None
        state_ids = list(range(len(result.states)))
        template = result._predicate_template() if result.states else None
        for state_id in state_ids:
            state = result.states[state_id]
            if not state.arcs:
                if template is None:
                    raise ValueError(
                        "Completion requires a finite alphabet or predicate factory"
                    )
                if sink_state_id is None:
                    sink_state_id = result.add_state(final=False)
                result.add_arc(state_id, sink_state_id, template.top())
                continue
            covered = state.arcs[0].guard
            for arc in state.arcs[1:]:
                covered = covered.disjunction(arc.guard)
            residual = covered.negate()
            if result._guard_is_sat(residual):
                if sink_state_id is None:
                    sink_state_id = result.add_state(final=False)
                result.add_arc(state_id, sink_state_id, residual)

        if sink_state_id is not None:
            assert template is not None
            result.add_arc(sink_state_id, sink_state_id, template.top())
        return result

    def concretize(self):
        if self.alphabet is None:
            raise ValueError("Concretization requires a finite alphabet")
        from aria.automata.symautomata import dfa as dfa_module
        from aria.automata.symautomata.pythondfa import PythonDFA

        deterministic = self.determinize()
        dfa_class = getattr(dfa_module, "DFA", PythonDFA)
        dfa = dfa_class(list(self.alphabet))
        while len(deterministic.states) > len(dfa.states):
            dfa.add_state()
        if deterministic.states:
            dfa[0].initial = True
        for state in deterministic.states:
            dfa[state.state_id].final = state.final
            for arc in state.arcs:
                for symbol in self.alphabet:
                        if self._guard_is_sat(arc.guard, symbol):
                            dfa.add_arc(arc.src_state, arc.dst_state, symbol)
        if hasattr(dfa, "yy_accept"):
            dfa.yy_accept = [1 if state.final else 0 for state in dfa.states]
        return dfa

    def to_regex(self) -> Optional[str]:
        from aria.automata.symautomata.regex import Regex

        return Regex(self.concretize()).get_regex()

    def copy(self) -> "SFA":
        result = SFA(
            self.alphabet,
            self.predicate_factory,
            self.symbolic_symbol,
            self.symbolic_universe,
        )
        for state in self.states:
            result.add_state(initial=state.initial, final=state.final)
        for arc in self.arcs:
            result.add_arc(arc.src_state, arc.dst_state, arc.guard, arc.term)
        return result

    def save(self, txt_fst_filename: str) -> None:
        if self.alphabet is None:
            raise ValueError("Saving requires a finite alphabet")
        deterministic = self.determinize()
        with open(txt_fst_filename, "w", encoding="utf-8") as handle:
            for state in deterministic.states:
                for arc in state.arcs:
                    for symbol in self.alphabet:
                        if deterministic._guard_is_sat(arc.guard, symbol):
                            handle.write(
                                f"{arc.src_state}\t{arc.dst_state}\t{symbol}\t{symbol}\n"
                            )
                if state.final:
                    handle.write(f"{state.state_id}\n")

    def load(self, txt_fst_filename: str) -> None:
        if self.alphabet is None:
            raise ValueError("Loading requires a finite alphabet")
        self.states = []
        self.arcs = []
        initial_state_set = False
        with open(txt_fst_filename, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 1:
                    self.set_final(int(parts[0]), True)
                    continue
                src = int(parts[0])
                dst = int(parts[1])
                symbol = self._load_symbol(parts[2])
                self.add_arc(src, dst, SetPredicate([symbol], self.alphabet))
                if not initial_state_set:
                    self.states[src].initial = True
                    initial_state_set = True

    def initial_states(self) -> List[int]:
        initial = [state.state_id for state in self.states if state.initial]
        if initial:
            return [state_id for state_id in initial if state_id is not None]
        if self.states:
            return [0]
        return []

    def _ensure_state(self, state_id: int) -> None:
        while state_id >= len(self.states):
            self.add_state()

    def _coerce_guard(self, guard: object) -> Predicate:
        if isinstance(guard, Predicate):
            if isinstance(guard, SetPredicate) and guard.universe is None and self.alphabet:
                guard.set_universe(self.alphabet)
            if isinstance(guard, Z3Predicate):
                self._register_symbolic_guard(guard)
                return self._align_guard_to_automaton(guard)
            return guard
        if self.alphabet is not None:
            return SetPredicate([guard], self.alphabet)
        return SetPredicate([guard])

    def _iter_input(self, inp: Sequence[object]) -> List[object]:
        if isinstance(inp, str):
            return list(inp)
        return list(inp)

    def _partition_transitions(
        self, outgoing: List[SFAArc]
    ) -> List[Tuple[Predicate, frozenset[int]]]:
        if not outgoing:
            return []
        regions: List[Tuple[Predicate, frozenset[int]]] = [
            (outgoing[0].guard.top(), frozenset())
        ]
        for arc in outgoing:
            next_regions: List[Tuple[Predicate, frozenset[int]]] = []
            for region_guard, destination_states in regions:
                enabled = region_guard.conjunction(arc.guard)
                if self._guard_is_sat(enabled):
                    next_regions.append(
                        (enabled, destination_states | frozenset([arc.dst_state]))
                    )
                disabled = region_guard.conjunction(arc.guard.negate())
                if self._guard_is_sat(disabled):
                    next_regions.append((disabled, destination_states))
            regions = next_regions
        grouped: Dict[frozenset[int], Predicate] = {}
        for predicate, frozen_destination in regions:
            if not frozen_destination:
                continue
            if frozen_destination in grouped:
                grouped[frozen_destination] = grouped[frozen_destination].disjunction(
                    predicate
                )
            else:
                grouped[frozen_destination] = predicate
        return [(guard, dst) for dst, guard in grouped.items()]

    def _product(
        self,
        other: "SFA",
        accept_method: Callable[[bool, bool], bool],
        complete: bool = False,
    ) -> "SFA":
        left = self.determinize()
        right = other.determinize()
        symbolic_symbol, symbolic_universe = left._merged_symbolic_context(right)
        if complete:
            left = left.complete()
            right = right.complete()

        result = SFA(
            self.alphabet if self.alphabet is not None else other.alphabet,
            self.predicate_factory
            if self.predicate_factory is not None
            else other.predicate_factory,
            symbolic_symbol,
            symbolic_universe,
        )
        initial_pair = (left.initial_states()[0], right.initial_states()[0])
        pair_to_state: Dict[Tuple[int, int], int] = {}
        queue: Deque[Tuple[int, int]] = deque([initial_pair])

        initial_state_id = result.add_state(initial=True)
        result.states[initial_state_id].final = accept_method(
            left.states[initial_pair[0]].final,
            right.states[initial_pair[1]].final,
        )
        pair_to_state[initial_pair] = initial_state_id

        while queue:
            current_pair = queue.popleft()
            current_id = pair_to_state[current_pair]
            grouped: Dict[Tuple[int, int], Predicate] = {}
            for left_arc in left.states[current_pair[0]].arcs:
                for right_arc in right.states[current_pair[1]].arcs:
                    guard = left_arc.guard.conjunction(right_arc.guard)
                    if not result._guard_is_sat(guard):
                        continue
                    dst_pair = (left_arc.dst_state, right_arc.dst_state)
                    if dst_pair in grouped:
                        grouped[dst_pair] = grouped[dst_pair].disjunction(guard)
                    else:
                        grouped[dst_pair] = guard
            for dst_pair, guard in grouped.items():
                if dst_pair not in pair_to_state:
                    pair_to_state[dst_pair] = result.add_state()
                    result.states[pair_to_state[dst_pair]].final = accept_method(
                        left.states[dst_pair[0]].final,
                        right.states[dst_pair[1]].final,
                    )
                    queue.append(dst_pair)
                result.add_arc(current_id, pair_to_state[dst_pair], guard)
        return result

    def _predicate_template(self) -> Predicate:
        for state in self.states:
            for arc in state.arcs:
                return arc.guard
        if self.predicate_factory is not None:
            predicate = self.predicate_factory()
            if isinstance(predicate, Z3Predicate):
                self._register_symbolic_guard(predicate)
                return self._align_guard_to_automaton(predicate)
            return predicate
        if self.alphabet is not None:
            return SetPredicate([], self.alphabet)
        raise ValueError("Cannot infer predicate universe from an empty automaton")

    def _guard_is_sat(
        self, guard: Predicate, symbol: Optional[object] = None
    ) -> bool:
        if isinstance(guard, Z3Predicate):
            return guard.is_sat(symbol, self._guard_universe(guard))
        return guard.is_sat(symbol)

    def _guard_get_witness(self, guard: Predicate) -> object:
        if isinstance(guard, Z3Predicate):
            return guard.get_witness(self._guard_universe(guard))
        return guard.get_witness()

    def _guard_universe(self, guard: Z3Predicate) -> Optional[Any]:
        if self.symbolic_universe is None:
            return None
        if self.symbolic_symbol is None or guard.symbol.eq(self.symbolic_symbol):
            return self.symbolic_universe
        return z3.simplify(
            z3.substitute(self.symbolic_universe, (self.symbolic_symbol, guard.symbol))
        )

    def _register_symbolic_guard(self, guard: Z3Predicate) -> None:
        if self.symbolic_symbol is None:
            self.symbolic_symbol = guard.symbol

    def _align_guard_to_automaton(self, guard: Z3Predicate) -> Z3Predicate:
        if self.symbolic_symbol is None or guard.symbol.eq(self.symbolic_symbol):
            return guard
        if self.symbolic_symbol.sort() != guard.symbol.sort():
            raise TypeError("Z3Predicate sorts must match the automaton symbol sort")
        return Z3Predicate(
            self.symbolic_symbol,
            z3.substitute(guard.formula, (guard.symbol, self.symbolic_symbol)),
        )

    def _merged_symbolic_context(
        self, other: "SFA"
    ) -> Tuple[Optional[Any], Optional[Any]]:
        left_symbol = self.symbolic_symbol
        right_symbol = other.symbolic_symbol
        left_universe = self.symbolic_universe
        right_universe = other.symbolic_universe

        if left_symbol is None and left_universe is not None:
            raise ValueError("Symbolic universe requires a symbolic symbol")
        if right_symbol is None and right_universe is not None:
            raise ValueError("Symbolic universe requires a symbolic symbol")

        if left_symbol is None:
            return right_symbol, right_universe
        if right_symbol is None:
            return left_symbol, left_universe
        if left_symbol.sort() != right_symbol.sort():
            raise TypeError("Symbolic automata must use the same symbol sort")
        if left_universe is None or right_universe is None:
            return left_symbol, left_universe if left_universe is not None else right_universe

        aligned_right_universe = z3.simplify(
            z3.substitute(right_universe, (right_symbol, left_symbol))
        )
        probe = Z3Predicate(left_symbol, left_universe)
        if not probe.is_equivalent(Z3Predicate(left_symbol, aligned_right_universe)):
            raise ValueError("Symbolic automata must share the same universe")
        return left_symbol, left_universe

    def _load_symbol(self, token: str) -> object:
        if self.alphabet is None:
            return token
        if token in self.alphabet:
            return token
        for caster in (int, float):
            try:
                candidate = caster(token)
            except ValueError:
                continue
            if candidate in self.alphabet:
                return candidate
        return token


def main() -> None:
    symbol = z3.BitVec("sym", 8)
    sfa = SFA(
        predicate_factory=lambda: Z3Predicate(symbol, z3.BoolVal(False)),
        symbolic_symbol=symbol,
        symbolic_universe=z3.ULE(symbol, z3.BitVecVal(255, 8)),
    )
    sfa.add_arc(0, 1, Z3Predicate(symbol, z3.ULT(symbol, z3.BitVecVal(98, 8))))
    sfa.set_final(1, True)
    print(sfa.accepts([ord("a")]))


if __name__ == "__main__":
    main()
