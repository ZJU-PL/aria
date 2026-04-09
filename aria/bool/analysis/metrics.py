"""Structural reports for CNF and prenex QBF formulas."""

from collections import Counter
from statistics import mean
from typing import Dict, List, Sequence, Union

from pysat.formula import CNF

from aria.bool.qbf import QCIRInstance, QDIMACSInstance


def clause_length_histogram(clauses: Sequence[Sequence[int]]) -> Dict[int, int]:
    """Return a histogram keyed by clause length."""

    histogram = Counter(len(clause) for clause in clauses)
    return dict(sorted(histogram.items()))


def literal_balance(clauses: Sequence[Sequence[int]]) -> Dict[str, int]:
    """Count positive and negative literal occurrences."""

    positive = 0
    negative = 0
    for clause in clauses:
        for literal in clause:
            if literal > 0:
                positive += 1
            else:
                negative += 1
    return {
        "positive_literals": positive,
        "negative_literals": negative,
        "total_literals": positive + negative,
    }


def variable_clause_degrees(clauses: Sequence[Sequence[int]]) -> Dict[int, int]:
    """Count how many clauses each variable appears in."""

    degrees: Counter[int] = Counter()
    for clause in clauses:
        for variable in {abs(literal) for literal in clause}:
            degrees[variable] += 1
    return dict(sorted(degrees.items()))


def cnf_analysis_report(formula: CNF) -> Dict[str, object]:
    """Return a compact structural summary for a CNF."""

    clauses = formula.clauses
    clause_count = len(clauses)
    variable_ids = sorted({abs(literal) for clause in clauses for literal in clause})
    degrees = variable_clause_degrees(clauses)
    average_clause_length = (
        mean(len(clause) for clause in clauses) if clauses else 0.0
    )
    return {
        "num_variables": len(variable_ids),
        "num_clauses": clause_count,
        "max_variable": max(variable_ids, default=0),
        "average_clause_length": average_clause_length,
        "clause_length_histogram": clause_length_histogram(clauses),
        "literal_balance": literal_balance(clauses),
        "variable_clause_degrees": degrees,
        "max_clause_degree": max(degrees.values(), default=0),
    }


def quantifier_prefix_report(
    instance: Union[QDIMACSInstance, QCIRInstance],
) -> Dict[str, object]:
    """Summarize the quantifier prefix of a parsed QBF instance."""

    prefix = instance.parsed_prefix
    block_sizes: List[int] = [len(variables) for _, variables in prefix]
    alternations = sum(
        1 for left, right in zip(prefix, prefix[1:]) if left[0] != right[0]
    )
    return {
        "num_blocks": len(prefix),
        "alternations": alternations,
        "block_sizes": block_sizes,
        "forall_variables": sum(len(vs) for kind, vs in prefix if kind == "a"),
        "exists_variables": sum(len(vs) for kind, vs in prefix if kind == "e"),
        "prefix_pattern": "".join(kind for kind, _ in prefix),
    }
