# coding: utf-8
"""Data models for the Dissolve algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from aria.utils.types import SolverResult


# ------------------------------ Data classes ------------------------------ #


@dataclass
class DilemmaTriple:
    """Represents a dilemma triple (v, a, b) where v is the variable and a, b are values."""

    variable: int
    value_a: int  # 0 or 1
    value_b: int  # 0 or 1

    def __post_init__(self) -> None:
        if self.value_a not in [0, 1] or self.value_b not in [0, 1]:
            raise ValueError("Values must be 0 or 1")


@dataclass
class DilemmaQuery:
    """Represents a dilemma-based SAT query with assumptions and dilemma information."""

    assumptions: List[int]  # Variable assignments (positive/negative literals)
    dilemma_triple: Optional[DilemmaTriple] = None
    round_id: int = 0
    query_id: int = 0


@dataclass
class WorkerResult:
    """Result from a worker process."""

    status: SolverResult
    learnt_clauses: List[List[int]]
    decision_literals: List[int]
    polarities: Dict[int, int]  # variable -> polarity (0 or 1)
    model_or_core: Optional[List[int]] = None
    dilemma_info: Optional[DilemmaTriple] = None


@dataclass
class DissolveResult:
    """Result from running the Dissolve algorithm."""

    result: SolverResult
    model: Optional[List[int]] = None
    unsat_core: Optional[List[int]] = None
    rounds: int = 0
    runtime_sec: float = 0.0


@dataclass
class DissolveConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for the Dissolve algorithm."""

    k_split_vars: int = 5
    per_query_conflict_budget: int = 20000
    max_rounds: Optional[int] = None
    num_workers: Optional[int] = None  # default: os.cpu_count()
    solver_name: str = "cd"  # cadical in PySAT
    seed: Optional[int] = None
    # clause sharing parameters
    max_shared_per_round: int = 50000
    important_bucket_size: int = 100
    store_small_clauses_threshold: int = 2
    # run strategy parameters
    budget_strategy: str = "constant"  # constant | luby
    budget_unit: int = 10_000  # conflict budget unit for the budget strategy
    distribution_strategy: str = "dilemma"  # dilemma | portfolio | hybrid
    kprime_for_decision_votes: int = 5
