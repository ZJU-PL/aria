"""Structural analysis helpers for prenex CNF QBF instances."""

from dataclasses import dataclass
from typing import Dict, List

from aria.bool.qbf.model import QDIMACSInstance


@dataclass(frozen=True)
class QBFAnalysis:
    """Compact structural summary for a QDIMACS instance."""

    num_vars: int
    num_clauses: int
    quantifier_blocks: int
    alternation_depth: int
    universal_vars: int
    existential_vars: int
    free_vars: int
    prefix_pattern: str
    min_clause_size: int
    max_clause_size: int
    mean_clause_size: float
    mean_level_span: float
    max_level_span: int

    @property
    def is_closed(self) -> bool:
        return self.free_vars == 0


def _variable_levels(instance: QDIMACSInstance) -> Dict[int, int]:
    levels: Dict[int, int] = {}
    for level, (_, variables) in enumerate(instance.parsed_prefix, start=1):
        for variable in variables:
            levels[variable] = level
    return levels


def analyze_qdimacs(instance: QDIMACSInstance) -> QBFAnalysis:
    """Compute a structural summary for a typed QDIMACS instance."""

    quantified_levels = _variable_levels(instance)
    clause_sizes = [len(clause) for clause in instance.clauses]
    clause_spans: List[int] = []

    for clause in instance.clauses:
        if not clause:
            clause_spans.append(0)
            continue
        levels = [quantified_levels.get(abs(literal), 0) for literal in clause]
        clause_spans.append(max(levels) - min(levels))

    universal_vars = sum(
        len(variables) for kind, variables in instance.parsed_prefix if kind == "a"
    )
    existential_vars = sum(
        len(variables) for kind, variables in instance.parsed_prefix if kind == "e"
    )
    quantified_vars = {variable for _, variables in instance.parsed_prefix for variable in variables}
    free_vars = len([variable for variable in instance.all_vars if variable not in quantified_vars])

    return QBFAnalysis(
        num_vars=instance.num_vars,
        num_clauses=instance.num_clauses,
        quantifier_blocks=len(instance.parsed_prefix),
        alternation_depth=len(instance.parsed_prefix),
        universal_vars=universal_vars,
        existential_vars=existential_vars,
        free_vars=free_vars,
        prefix_pattern="".join(kind for kind, _ in instance.parsed_prefix),
        min_clause_size=min(clause_sizes, default=0),
        max_clause_size=max(clause_sizes, default=0),
        mean_clause_size=0.0
        if not clause_sizes
        else sum(clause_sizes) / len(clause_sizes),
        mean_level_span=0.0
        if not clause_spans
        else sum(clause_spans) / len(clause_spans),
        max_level_span=max(clause_spans, default=0),
    )
