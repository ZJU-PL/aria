"""
Weighted model counting over propositional CNF formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from pysat.formula import CNF
from pysat.solvers import Solver

from aria.bool.knowledge_compiler.dtree import Dtree_Compiler
from aria.bool.knowledge_compiler.dnnf import DNF_Node, DNNF_Compiler
from aria.prob.results import InferenceResult

from .._helpers import normalize_literal_sequence
from .base import LiteralWeights, WMCBackend, WMCOptions


def _validate_weights(
    weights: LiteralWeights, variables: Iterable[int], strict_complements: bool
) -> None:
    for lit, weight in weights.items():
        if lit == 0:
            raise ValueError("Literal 0 is not valid in a weight map")
        if weight < 0.0 or weight > 1.0:
            raise ValueError(
                "Weight of literal {} must be in [0,1], got {}".format(lit, weight)
            )

    if not strict_complements:
        return

    for var in variables:
        pos = weights.get(var)
        neg = weights.get(-var)
        if pos is None or neg is None:
            continue
        total = float(pos) + float(neg)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "Complementary weights for variable {} must sum to 1.0, got {}".format(
                    var, total
                )
            )


def _ensure_complement_weights(
    weights: LiteralWeights, variables: Iterable[int]
) -> LiteralWeights:
    completed = dict(weights)
    for var in variables:
        pos = completed.get(var)
        neg = completed.get(-var)
        if pos is None and neg is None:
            completed[var] = 0.5
            completed[-var] = 0.5
        elif pos is None:
            completed[var] = 1.0 - float(neg)
        elif neg is None:
            completed[-var] = 1.0 - float(pos)
    return completed


def _variables_of_cnf(cnf: CNF) -> List[int]:
    if cnf.nv:
        return list(range(1, cnf.nv + 1))
    variables = set()
    for clause in cnf.clauses:
        for lit in clause:
            variables.add(abs(lit))
    return sorted(variables)


def _compile_dnnf(cnf: CNF, variables: Sequence[int]) -> Optional[DNF_Node]:
    if len(cnf.clauses) == 0:
        return None
    dtree = Dtree_Compiler(cnf.clauses).el2dt(list(variables))
    compiler = DNNF_Compiler(dtree)
    return compiler.compile()


def _lift_missing_atoms(
    parent_atoms: Sequence[int],
    child_atoms: Sequence[int],
    weights: LiteralWeights,
    assignment: Dict[int, bool],
) -> float:
    factor = 1.0
    missing = set(parent_atoms).difference(set(child_atoms))
    for var in missing:
        if var in assignment:
            factor *= float(weights[var if assignment[var] else -var])
        else:
            factor *= float(weights[var]) + float(weights[-var])
    return factor


def _wmc_on_dnnf(
    root: DNF_Node, weights: LiteralWeights, forced_literals: Optional[Sequence[int]] = None
) -> float:
    memo = {}
    assignment = {}
    for lit in normalize_literal_sequence(forced_literals):
        assignment[abs(lit)] = lit > 0

    def eval_node(node: DNF_Node) -> float:
        if node.explore_id is not None and node.explore_id in memo:
            return memo[node.explore_id]

        if node.type == "L":
            if isinstance(node.literal, bool):
                value = 1.0 if node.literal else 0.0
            else:
                literal = int(node.literal)
                var = abs(literal)
                if var in assignment:
                    value = (
                        float(weights[literal])
                        if assignment[var] == (literal > 0)
                        else 0.0
                    )
                else:
                    value = float(weights[literal])
        elif node.type == "A":
            value = eval_node(node.left_child) * eval_node(node.right_child)
        elif node.type == "O":
            left_value = eval_node(node.left_child)
            right_value = eval_node(node.right_child)
            left_value *= _lift_missing_atoms(
                node.atoms, node.left_child.atoms, weights, assignment
            )
            right_value *= _lift_missing_atoms(
                node.atoms, node.right_child.atoms, weights, assignment
            )
            value = left_value + right_value
        else:
            raise RuntimeError("Unknown DNNF node type")

        if node.explore_id is not None:
            memo[node.explore_id] = value
        return value

    root.reset()
    root.count_node(0)
    return eval_node(root)


def _wmc_by_enumeration(
    cnf: CNF,
    weights: LiteralWeights,
    model_limit: Optional[int],
    forced_literals: Optional[Sequence[int]] = None,
) -> float:
    total = 0.0
    with Solver(bootstrap_with=cnf) as solver:
        for lit in normalize_literal_sequence(forced_literals):
            solver.add_clause([int(lit)])

        count = 0
        while solver.solve():
            model = solver.get_model()
            probability = 1.0
            for lit in model:
                if abs(lit) > cnf.nv:
                    continue
                probability *= float(weights.get(lit, 0.5))
            total += probability
            count += 1
            if model_limit is not None and count >= model_limit:
                break
            solver.add_clause([-lit for lit in model if abs(lit) <= cnf.nv])
    return total


@dataclass
class CompiledWMC:
    """Compiled exact WMC object for repeated evidence queries."""

    cnf: CNF
    weights: LiteralWeights
    root: Optional[DNF_Node]
    variables: List[int]
    tautology: bool = False
    backend: str = "wmc-dnnf"

    def _validate_literals(self, literals: Sequence[int]) -> None:
        known_variables = set(self.variables)
        unknown = sorted({abs(lit) for lit in literals if abs(lit) not in known_variables})
        if unknown:
            raise ValueError(
                "Evidence/query mentions variables not present in the compiled CNF: {}".format(
                    unknown
                )
            )

    def count(self, evidence: Optional[Sequence[int]] = None) -> float:
        normalized = normalize_literal_sequence(evidence)
        self._validate_literals(normalized)
        if self.root is None:
            return 1.0 if self.tautology else 0.0
        return _wmc_on_dnnf(self.root, self.weights, normalized)

    def infer(self, evidence: Optional[Sequence[int]] = None) -> InferenceResult:
        return InferenceResult(
            value=self.count(evidence=evidence),
            exact=True,
            backend=self.backend,
            stats={
                "num_variables": len(self.variables),
                "num_clauses": len(self.cnf.clauses),
                "evidence_literals": len(normalize_literal_sequence(evidence)),
            },
        )

    def probability(
        self,
        query: Optional[Sequence[int]] = None,
        evidence: Optional[Sequence[int]] = None,
    ) -> InferenceResult:
        normalized_evidence = normalize_literal_sequence(evidence)
        normalized_query = normalize_literal_sequence(query)
        self._validate_literals(normalized_evidence)
        self._validate_literals(normalized_query)

        denominator = self.count(normalized_evidence)
        if denominator == 0.0:
            raise ValueError("Evidence has zero probability under the compiled model")

        numerator = self.count(normalized_evidence + normalized_query)
        return InferenceResult(
            value=numerator / denominator,
            exact=True,
            backend=self.backend,
            stats={
                "numerator": numerator,
                "denominator": denominator,
                "query_literals": list(normalized_query),
                "evidence_literals": list(normalized_evidence),
                "num_variables": len(self.variables),
            },
        )


def compile_wmc(
    cnf: CNF, weights: LiteralWeights, options: Optional[WMCOptions] = None
) -> CompiledWMC:
    opts = options or WMCOptions()
    if opts.backend != WMCBackend.DNNF:
        raise ValueError("CompiledWMC currently supports only the DNNF backend")

    variables = _variables_of_cnf(cnf)
    _validate_weights(weights, variables, opts.strict_complements)
    completed_weights = _ensure_complement_weights(weights, variables)
    root = _compile_dnnf(cnf, variables)
    return CompiledWMC(
        cnf=cnf,
        weights=completed_weights,
        root=root,
        variables=variables,
        tautology=len(cnf.clauses) == 0,
        backend="wmc-dnnf",
    )


def wmc_count(
    cnf: CNF, weights: LiteralWeights, options: Optional[WMCOptions] = None
) -> float:
    """
    Compute the weighted model count of a propositional CNF.
    """

    opts = options or WMCOptions()
    variables = _variables_of_cnf(cnf)
    _validate_weights(weights, variables, opts.strict_complements)
    completed_weights = _ensure_complement_weights(weights, variables)

    if opts.backend == WMCBackend.DNNF:
        compiled = compile_wmc(cnf, completed_weights, opts)
        return compiled.count()
    if opts.backend == WMCBackend.ENUMERATION:
        return _wmc_by_enumeration(cnf, completed_weights, opts.model_limit)
    raise ValueError("Unsupported WMC backend: {}".format(opts.backend))
