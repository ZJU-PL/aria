"""Structural analysis helpers for Boolean CNF formulas."""

from dataclasses import dataclass
from typing import Dict, Iterable, Set

from pysat.formula import CNF


@dataclass(frozen=True)
class CNFAnalysis:
    """Compact structural summary of a CNF formula."""

    num_vars: int
    num_clauses: int
    min_clause_size: int
    max_clause_size: int
    mean_clause_size: float
    unit_clauses: int
    binary_clauses: int
    horn_clauses: int
    positive_literals: int
    negative_literals: int
    primal_edges: int
    primal_components: int

    @property
    def unit_fraction(self) -> float:
        return 0.0 if self.num_clauses == 0 else self.unit_clauses / self.num_clauses

    @property
    def binary_fraction(self) -> float:
        return 0.0 if self.num_clauses == 0 else self.binary_clauses / self.num_clauses

    @property
    def horn_fraction(self) -> float:
        return 0.0 if self.num_clauses == 0 else self.horn_clauses / self.num_clauses

    @property
    def literal_balance(self) -> float:
        total = self.positive_literals + self.negative_literals
        if total == 0:
            return 0.0
        return (self.positive_literals - self.negative_literals) / total


def _connected_components(graph: Dict[int, Set[int]]) -> int:
    remaining = set(graph)
    components = 0

    while remaining:
        start = remaining.pop()
        stack = [start]
        components += 1
        while stack:
            node = stack.pop()
            for neighbor in graph[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)

    return components


def _primal_graph(clauses: Iterable[Iterable[int]]) -> Dict[int, Set[int]]:
    graph: Dict[int, Set[int]] = {}
    for clause in clauses:
        variables = sorted({abs(literal) for literal in clause})
        for variable in variables:
            graph.setdefault(variable, set())
        for index, left in enumerate(variables):
            for right in variables[index + 1 :]:
                graph[left].add(right)
                graph[right].add(left)
    return graph


def analyze_cnf(formula: CNF) -> CNFAnalysis:
    """Compute a structural summary for a PySAT CNF formula."""

    clauses = list(formula.clauses)
    sizes = [len(clause) for clause in clauses]
    positive_literals = sum(1 for clause in clauses for literal in clause if literal > 0)
    negative_literals = sum(1 for clause in clauses for literal in clause if literal < 0)
    horn_clauses = sum(
        1 for clause in clauses if sum(1 for literal in clause if literal > 0) <= 1
    )
    graph = _primal_graph(clauses)

    return CNFAnalysis(
        num_vars=max([0] + [abs(literal) for clause in clauses for literal in clause]),
        num_clauses=len(clauses),
        min_clause_size=min(sizes, default=0),
        max_clause_size=max(sizes, default=0),
        mean_clause_size=0.0 if not sizes else sum(sizes) / len(sizes),
        unit_clauses=sum(1 for size in sizes if size == 1),
        binary_clauses=sum(1 for size in sizes if size == 2),
        horn_clauses=horn_clauses,
        positive_literals=positive_literals,
        negative_literals=negative_literals,
        primal_edges=sum(len(neighbors) for neighbors in graph.values()) // 2,
        primal_components=0 if not graph else _connected_components(graph),
    )
