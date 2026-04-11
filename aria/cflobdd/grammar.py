"""Grammar-based CFL reachability over finite relations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from aria.cflobdd.fixpoint import least_fixed_point
from aria.cflobdd.relation import Relation, Witness


@dataclass(frozen=True)
class GrammarProduction:
    head: str
    body: Tuple[str, ...]


@dataclass(frozen=True)
class GrammarSolution:
    start_symbol: str
    relations: Mapping[str, Relation]
    iterations: int
    converged: bool

    def relation(self, symbol: str) -> Relation:
        return self.relations[symbol]

    def contains(self, symbol: str, src: int, dst: int) -> bool:
        return self.relations[symbol].contains({"src": src, "dst": dst})

    def witness(self, symbol: str, src: int, dst: int) -> Optional[Witness]:
        return self.relations[symbol].witness({"src": src, "dst": dst})


class GrammarReachability:
    def __init__(
        self,
        start_symbol: str,
        productions: Sequence[GrammarProduction],
        relation_variables: Tuple[str, str] = ("src", "dst"),
    ) -> None:
        self.start_symbol = start_symbol
        self.productions = tuple(productions)
        self.relation_variables = relation_variables
        self.nonterminals = tuple(
            sorted({production.head for production in self.productions})
        )

    def _empty_solution(self) -> Dict[str, Relation]:
        return {
            symbol: Relation.empty(self.relation_variables, name=symbol)
            for symbol in self.nonterminals
        }

    def _wrap_rule(
        self, relation: Relation, symbol: str, production: GrammarProduction
    ) -> Relation:
        return relation.with_wrapped_witness(
            "grammar",
            payload={"symbol": symbol, "production": list(production.body)},
            name=symbol,
        )

    def _compose_binary(self, left: Relation, right: Relation, name: str) -> Relation:
        return left.binary_compose(right, name=name)

    def _lookup_symbol(
        self,
        symbol: str,
        terminals: Mapping[str, Relation],
        current_map: Mapping[str, Relation],
    ) -> Relation:
        if symbol in terminals:
            return terminals[symbol]
        return current_map[symbol]

    def _derive_sequence(
        self,
        symbols: Sequence[str],
        terminals: Mapping[str, Relation],
        current_map: Mapping[str, Relation],
        nodes: Tuple[int, ...],
        head: str,
    ) -> Relation:
        if len(symbols) == 0:
            return Relation.identity(
                nodes,
                source=self.relation_variables[0],
                target=self.relation_variables[1],
                name=head,
            )

        current = self._lookup_symbol(symbols[0], terminals, current_map)
        for symbol in symbols[1:]:
            current = self._compose_binary(
                current, self._lookup_symbol(symbol, terminals, current_map), head
            )
        return current

    def solve(
        self,
        terminals: Mapping[str, Relation],
        nodes: Iterable[int],
        max_iterations: int = 100,
    ) -> GrammarSolution:
        nodes_tuple = tuple(nodes)
        relation_variables = self.relation_variables

        seed = self._empty_solution()

        def step_map(current_map: Mapping[str, Relation]) -> Dict[str, Relation]:
            next_map = {symbol: relation for symbol, relation in current_map.items()}

            for production in self.productions:
                current = next_map[production.head]

                derived = self._derive_sequence(
                    production.body,
                    terminals,
                    current_map,
                    nodes_tuple,
                    production.head,
                )

                wrapped = self._wrap_rule(derived, production.head, production)
                next_map[production.head] = current.union(wrapped, name=production.head)

            return next_map

        seed_relation = _SolutionRelation(seed, self.nonterminals)

        def step_solution(current: _SolutionRelation) -> _SolutionRelation:
            return _SolutionRelation(step_map(current.relations), self.nonterminals)

        result = least_fixed_point(seed_relation, step_solution, max_iterations=max_iterations)
        return GrammarSolution(
            start_symbol=self.start_symbol,
            relations=result.relation.relations,
            iterations=result.iterations,
            converged=result.converged,
        )


@dataclass(frozen=True)
class _SolutionRelation:
    relations: Mapping[str, Relation]
    order: Tuple[str, ...]

    @property
    def variables(self) -> Tuple[str, ...]:
        return self.order

    @property
    def facts(self) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
        return tuple(tuple(self.relations[symbol].tuples()) for symbol in self.order)
