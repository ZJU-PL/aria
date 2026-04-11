"""Finite relation operators for CFL reachability workflows.

This module provides an explicit relation algebra layer and an optional bridge
to the existing ``CFLOBVDD`` kernel for symbolic Boolean predicates over finite
variable blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

from aria.cflobdd.cflobvdd import CFLOBVDD

Witness = Dict[str, Any]
Fact = Tuple[int, ...]


@dataclass(frozen=True)
class SymbolicEncoding:
    variables: Tuple[str, ...]
    bytes_per_variable: int = 1
    swap_level: int = 0
    fork_level: int = 0
    reorder: bool = False

    @property
    def block_count(self) -> int:
        return len(self.variables) * self.bytes_per_variable

    @property
    def level(self) -> int:
        return max(0, ceil(log2(max(1, self.block_count))))

    def block_index(self, variable: str, byte_offset: int = 0) -> int:
        if variable not in self.variables:
            raise ValueError(f"unknown encoded variable: {variable}")
        return self.variables.index(variable) * self.bytes_per_variable + byte_offset


@dataclass(frozen=True)
class SymbolicRelation:
    encoding: SymbolicEncoding
    predicate: CFLOBVDD


@dataclass(frozen=True)
class Relation:
    """A finite relation together with lightweight witness provenance."""

    variables: Tuple[str, ...]
    facts: FrozenSet[Fact]
    provenance: Mapping[Fact, Witness]
    name: Optional[str] = None
    symbolic: Optional[SymbolicRelation] = None

    def __post_init__(self) -> None:
        arity = len(self.variables)
        for fact in self.facts:
            if len(fact) != arity:
                raise ValueError(
                    f"fact {fact} has arity {len(fact)} but expected {arity}"
                )

    @classmethod
    def empty(
        cls,
        variables: Iterable[str],
        name: Optional[str] = None,
        symbolic: bool = True,
    ) -> "Relation":
        normalized_variables = tuple(variables)
        symbolic_relation = (
            cls._build_symbolic_relation(normalized_variables, frozenset(), name)
            if symbolic
            else None
        )
        return cls(normalized_variables, frozenset(), {}, name=name, symbolic=symbolic_relation)

    @classmethod
    def from_tuples(
        cls,
        variables: Iterable[str],
        tuples: Iterable[Iterable[int]],
        provenance: Optional[Mapping[Fact, Witness]] = None,
        name: Optional[str] = None,
        symbolic: bool = True,
    ) -> "Relation":
        normalized_variables = tuple(variables)
        facts = frozenset(tuple(value for value in fact) for fact in tuples)
        normalized_provenance: Dict[Fact, Witness] = {}

        if provenance is None:
            for fact in facts:
                normalized_provenance[fact] = {
                    "kind": "fact",
                    "tuple": cls._tuple_to_dict_static(normalized_variables, fact),
                    "relation": name,
                }
        else:
            for fact in facts:
                if fact in provenance:
                    normalized_provenance[fact] = dict(provenance[fact])

        symbolic_relation = (
            cls._build_symbolic_relation(normalized_variables, facts, name)
            if symbolic
            else None
        )

        return cls(
            normalized_variables,
            facts,
            normalized_provenance,
            name=name,
            symbolic=symbolic_relation,
        )

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[Tuple[int, int]],
        source: str = "src",
        target: str = "dst",
        label: Optional[str] = None,
        name: Optional[str] = None,
        symbolic: bool = True,
    ) -> "Relation":
        facts = frozenset(tuple(edge) for edge in edges)
        provenance: Dict[Fact, Witness] = {}
        for edge in facts:
            provenance[edge] = {
                "kind": "edge",
                "edge": edge,
                "label": label,
                "tuple": {source: edge[0], target: edge[1]},
                "relation": name,
            }
        symbolic_relation = (
            cls._build_symbolic_relation((source, target), facts, name)
            if symbolic
            else None
        )
        return cls((source, target), facts, provenance, name=name, symbolic=symbolic_relation)

    @classmethod
    def from_labeled_edges(
        cls,
        edges: Iterable[Tuple[int, int, int]],
        source: str = "src",
        label: str = "label",
        target: str = "dst",
        name: Optional[str] = None,
        symbolic: bool = True,
    ) -> "Relation":
        facts = frozenset(tuple(edge) for edge in edges)
        provenance: Dict[Fact, Witness] = {}
        for edge in facts:
            provenance[edge] = {
                "kind": "labeled_edge",
                "edge": edge,
                "tuple": {source: edge[0], label: edge[1], target: edge[2]},
                "relation": name,
            }
        variables = (source, label, target)
        symbolic_relation = (
            cls._build_symbolic_relation(variables, facts, name) if symbolic else None
        )
        return cls(variables, facts, provenance, name=name, symbolic=symbolic_relation)

    @classmethod
    def identity(
        cls,
        nodes: Iterable[int],
        source: str = "src",
        target: str = "dst",
        name: Optional[str] = None,
        symbolic: bool = True,
    ) -> "Relation":
        tuples = [(node, node) for node in nodes]
        relation_name = name if name is not None else "identity"
        provenance: Dict[Fact, Witness] = {}
        for node in tuples:
            provenance[node] = {
                "kind": "identity",
                "tuple": {source: node[0], target: node[1]},
                "relation": relation_name,
            }
        symbolic_relation = (
            cls._build_symbolic_relation((source, target), frozenset(tuples), relation_name)
            if symbolic
            else None
        )
        return cls(
            (source, target),
            frozenset(tuples),
            provenance,
            name=relation_name,
            symbolic=symbolic_relation,
        )

    @staticmethod
    def _tuple_to_dict_static(variables: Tuple[str, ...], fact: Fact) -> Dict[str, int]:
        return {variable: fact[index] for index, variable in enumerate(variables)}

    @classmethod
    def _build_symbolic_relation(
        cls,
        variables: Tuple[str, ...],
        facts: FrozenSet[Fact],
        name: Optional[str],
    ) -> Optional[SymbolicRelation]:
        if not cls._is_symbolically_encodable(facts):
            return None

        encoding = SymbolicEncoding(variables)
        predicate = cls._predicate_for_facts(encoding, facts)
        return SymbolicRelation(encoding=encoding, predicate=predicate)

    @staticmethod
    def _is_symbolically_encodable(facts: Iterable[Fact]) -> bool:
        for fact in facts:
            for value in fact:
                if not isinstance(value, int) or value < 0 or value >= 256:
                    return False
        return True

    @staticmethod
    def _false_predicate(encoding: SymbolicEncoding) -> CFLOBVDD:
        return CFLOBVDD.false(encoding.level, encoding.swap_level, encoding.fork_level)

    @staticmethod
    def _true_predicate(encoding: SymbolicEncoding) -> CFLOBVDD:
        return CFLOBVDD.true(encoding.level, encoding.swap_level, encoding.fork_level)

    @staticmethod
    def _and_predicates(left: CFLOBVDD, right: CFLOBVDD) -> CFLOBVDD:
        return left.binary_apply_and_reduce(
            right, lambda x, y: bool(x and y), number_of_output_bits=1
        )

    @staticmethod
    def _or_predicates(left: CFLOBVDD, right: CFLOBVDD) -> CFLOBVDD:
        return left.binary_apply_and_reduce(
            right, lambda x, y: bool(x or y), number_of_output_bits=1
        )

    @staticmethod
    def _not_predicate(predicate: CFLOBVDD) -> CFLOBVDD:
        return predicate.unary_apply_and_reduce(lambda x: not x, number_of_output_bits=1)

    @classmethod
    def _byte_equals(
        cls, encoding: SymbolicEncoding, variable: str, value: int
    ) -> CFLOBVDD:
        if value < 0 or value >= 256:
            raise ValueError(f"symbolic byte values must be in [0, 255], got {value}")

        projection = CFLOBVDD.byte_projection(
            encoding.level,
            encoding.swap_level,
            encoding.fork_level,
            encoding.block_count,
            encoding.block_index(variable),
            encoding.reorder,
        )
        constant = CFLOBVDD.byte_constant(
            encoding.level,
            encoding.swap_level,
            encoding.fork_level,
            encoding.block_count,
            value,
        )
        return projection.binary_apply_and_reduce(
            constant, lambda x, y: x == y, number_of_output_bits=1
        )

    @classmethod
    def _predicate_for_fact(
        cls, encoding: SymbolicEncoding, fact: Fact
    ) -> CFLOBVDD:
        predicate = cls._true_predicate(encoding)
        for variable, value in zip(encoding.variables, fact):
            predicate = cls._and_predicates(predicate, cls._byte_equals(encoding, variable, value))
        return predicate

    @classmethod
    def _predicate_for_facts(
        cls, encoding: SymbolicEncoding, facts: FrozenSet[Fact]
    ) -> CFLOBVDD:
        predicate = cls._false_predicate(encoding)
        for fact in sorted(facts):
            predicate = cls._or_predicates(predicate, cls._predicate_for_fact(encoding, fact))
        return predicate

    def _tuple_to_dict(self, fact: Fact) -> Dict[str, int]:
        return self._tuple_to_dict_static(self.variables, fact)

    def _check_compatible(self, other: "Relation") -> None:
        if self.variables != other.variables:
            raise ValueError(
                f"relation variables differ: {self.variables} != {other.variables}"
            )

    def block_encoding(self) -> Optional[SymbolicEncoding]:
        return self.symbolic.encoding if self.symbolic is not None else None

    def tuples(self) -> List[Fact]:
        return sorted(self.facts)

    def as_dicts(self) -> List[Dict[str, int]]:
        return [self._tuple_to_dict(fact) for fact in sorted(self.facts)]

    def contains(self, values: Mapping[str, int]) -> bool:
        fact = tuple(values[variable] for variable in self.variables)
        return fact in self.facts

    def witness(self, values: Mapping[str, int]) -> Optional[Witness]:
        fact = tuple(values[variable] for variable in self.variables)
        witness = self.provenance.get(fact)
        return dict(witness) if witness is not None else None

    def symbolic_solutions(self) -> Optional[int]:
        if self.symbolic is None:
            return None
        return self.symbolic.predicate.number_of_solutions(True)

    def _combine_symbolic(
        self,
        other: "Relation",
        op: str,
        facts: FrozenSet[Fact],
        name: Optional[str],
    ) -> Optional[SymbolicRelation]:
        if (
            self.symbolic is None
            or other.symbolic is None
            or self.symbolic.encoding != other.symbolic.encoding
        ):
            return self._build_symbolic_relation(self.variables, facts, name)

        if op == "or":
            predicate = self._or_predicates(
                self.symbolic.predicate, other.symbolic.predicate
            )
        elif op == "and":
            predicate = self._and_predicates(
                self.symbolic.predicate, other.symbolic.predicate
            )
        elif op == "diff":
            predicate = self._and_predicates(
                self.symbolic.predicate,
                self._not_predicate(other.symbolic.predicate),
            )
        else:
            raise ValueError(f"unsupported symbolic combination op: {op}")

        return SymbolicRelation(self.symbolic.encoding, predicate)

    def union(self, other: "Relation", name: Optional[str] = None) -> "Relation":
        self._check_compatible(other)
        facts = self.facts | other.facts
        provenance: Dict[Fact, Witness] = {}
        for fact in sorted(facts):
            if fact in self.provenance:
                provenance[fact] = dict(self.provenance[fact])
            elif fact in other.provenance:
                provenance[fact] = dict(other.provenance[fact])
        symbolic_relation = self._combine_symbolic(other, "or", facts, name)
        return Relation(
            self.variables,
            frozenset(facts),
            provenance,
            name=name,
            symbolic=symbolic_relation,
        )

    def intersection(self, other: "Relation", name: Optional[str] = None) -> "Relation":
        self._check_compatible(other)
        facts = self.facts & other.facts
        provenance: Dict[Fact, Witness] = {}
        for fact in sorted(facts):
            if fact in self.provenance:
                provenance[fact] = dict(self.provenance[fact])
            elif fact in other.provenance:
                provenance[fact] = dict(other.provenance[fact])
        symbolic_relation = self._combine_symbolic(other, "and", facts, name)
        return Relation(
            self.variables,
            frozenset(facts),
            provenance,
            name=name,
            symbolic=symbolic_relation,
        )

    def difference(self, other: "Relation", name: Optional[str] = None) -> "Relation":
        self._check_compatible(other)
        facts = self.facts - other.facts
        provenance = {
            fact: dict(self.provenance[fact])
            for fact in sorted(facts)
            if fact in self.provenance
        }
        symbolic_relation = self._combine_symbolic(other, "diff", facts, name)
        return Relation(
            self.variables,
            frozenset(facts),
            provenance,
            name=name,
            symbolic=symbolic_relation,
        )

    def rename(
        self, variable_map: Mapping[str, str], name: Optional[str] = None
    ) -> "Relation":
        new_variables = tuple(variable_map.get(variable, variable) for variable in self.variables)
        if len(set(new_variables)) != len(new_variables):
            raise ValueError(f"rename introduces duplicate variables: {new_variables}")

        provenance: Dict[Fact, Witness] = {}
        for fact in sorted(self.facts):
            witness = dict(self.provenance.get(fact, {}))
            provenance[fact] = {
                "kind": "rename",
                "mapping": dict(variable_map),
                "tuple": self._tuple_to_dict_static(new_variables, fact),
                "child": witness,
            }

        symbolic_relation = None
        if self.symbolic is not None:
            symbolic_relation = SymbolicRelation(
                encoding=SymbolicEncoding(
                    new_variables,
                    bytes_per_variable=self.symbolic.encoding.bytes_per_variable,
                    swap_level=self.symbolic.encoding.swap_level,
                    fork_level=self.symbolic.encoding.fork_level,
                    reorder=self.symbolic.encoding.reorder,
                ),
                predicate=self.symbolic.predicate,
            )

        return Relation(
            new_variables,
            self.facts,
            provenance,
            name=name,
            symbolic=symbolic_relation,
        )

    def restrict(
        self, assignments: Mapping[str, int], name: Optional[str] = None
    ) -> "Relation":
        unknown = [variable for variable in assignments if variable not in self.variables]
        if unknown:
            raise ValueError(f"unknown variables in restriction: {unknown}")

        restricted_facts = []
        provenance: Dict[Fact, Witness] = {}
        for fact in sorted(self.facts):
            fact_dict = self._tuple_to_dict(fact)
            if all(fact_dict[variable] == value for variable, value in assignments.items()):
                restricted_facts.append(fact)
                provenance[fact] = {
                    "kind": "restrict",
                    "assignments": dict(assignments),
                    "tuple": fact_dict,
                    "child": dict(self.provenance.get(fact, {})),
                }
        facts = frozenset(restricted_facts)
        symbolic_relation = self._build_symbolic_relation(self.variables, facts, name)
        return Relation(self.variables, facts, provenance, name=name, symbolic=symbolic_relation)

    def project(
        self, variables_to_keep: Iterable[str], name: Optional[str] = None
    ) -> "Relation":
        keep = tuple(variables_to_keep)
        if len(set(keep)) != len(keep):
            raise ValueError(f"projection contains duplicate variables: {keep}")
        missing = [variable for variable in keep if variable not in self.variables]
        if missing:
            raise ValueError(f"unknown projection variables: {missing}")

        keep_indices = [self.variables.index(variable) for variable in keep]
        eliminated = [variable for variable in self.variables if variable not in keep]

        facts = []
        provenance: Dict[Fact, Witness] = {}
        for fact in sorted(self.facts):
            new_fact = tuple(fact[index] for index in keep_indices)
            if new_fact not in provenance:
                fact_dict = self._tuple_to_dict(fact)
                provenance[new_fact] = {
                    "kind": "project",
                    "keep": list(keep),
                    "eliminated": {variable: fact_dict[variable] for variable in eliminated},
                    "tuple": self._tuple_to_dict_static(keep, new_fact),
                    "child": dict(self.provenance.get(fact, {})),
                }
                facts.append(new_fact)
        frozen_facts = frozenset(facts)
        symbolic_relation = self._build_symbolic_relation(keep, frozen_facts, name)
        return Relation(keep, frozen_facts, provenance, name=name, symbolic=symbolic_relation)

    def exists(
        self, variables_to_eliminate: Iterable[str], name: Optional[str] = None
    ) -> "Relation":
        eliminate = set(variables_to_eliminate)
        keep = [variable for variable in self.variables if variable not in eliminate]
        return self.project(keep, name=name)

    def join(self, other: "Relation", shared: Iterable[str], name: Optional[str] = None) -> "Relation":
        shared_variables = tuple(shared)
        for variable in shared_variables:
            if variable not in self.variables or variable not in other.variables:
                raise ValueError(f"shared variable {variable!r} not present in both relations")

        result_variables = self.variables + tuple(
            variable for variable in other.variables if variable not in self.variables
        )

        left_shared_indices = [self.variables.index(variable) for variable in shared_variables]
        right_shared_indices = [other.variables.index(variable) for variable in shared_variables]
        right_extra_variables = [
            variable for variable in other.variables if variable not in self.variables
        ]
        right_extra_indices = [other.variables.index(variable) for variable in right_extra_variables]

        facts = []
        provenance: Dict[Fact, Witness] = {}

        for left_fact in sorted(self.facts):
            left_shared_values = tuple(left_fact[index] for index in left_shared_indices)
            left_dict = self._tuple_to_dict(left_fact)
            for right_fact in sorted(other.facts):
                right_shared_values = tuple(
                    right_fact[index] for index in right_shared_indices
                )
                if left_shared_values != right_shared_values:
                    continue

                result_dict = dict(left_dict)
                for variable, index in zip(right_extra_variables, right_extra_indices):
                    result_dict[variable] = right_fact[index]
                result_fact = tuple(result_dict[variable] for variable in result_variables)

                if result_fact not in provenance:
                    shared_assignment = {
                        variable: left_shared_values[index]
                        for index, variable in enumerate(shared_variables)
                    }
                    provenance[result_fact] = {
                        "kind": "join",
                        "shared": list(shared_variables),
                        "tuple": result_dict,
                        "left": dict(self.provenance.get(left_fact, {})),
                        "right": dict(other.provenance.get(right_fact, {})),
                        "matched": shared_assignment,
                    }
                    facts.append(result_fact)

        frozen_facts = frozenset(facts)
        symbolic_relation = self._build_symbolic_relation(result_variables, frozen_facts, name)
        return Relation(
            result_variables,
            frozen_facts,
            provenance,
            name=name,
            symbolic=symbolic_relation,
        )

    def quantified_compose(
        self,
        other: "Relation",
        shared: Iterable[str],
        eliminate: Optional[Iterable[str]] = None,
        keep: Optional[Iterable[str]] = None,
        name: Optional[str] = None,
    ) -> "Relation":
        joined = self.join(other, shared, name=name)
        eliminate_set = set(eliminate if eliminate is not None else shared)

        if keep is None:
            keep_variables = tuple(
                variable for variable in joined.variables if variable not in eliminate_set
            )
        else:
            keep_variables = tuple(keep)

        projected = joined.project(keep_variables, name=name)
        provenance: Dict[Fact, Witness] = {}
        for fact in sorted(projected.facts):
            child = dict(projected.provenance[fact])
            provenance[fact] = {
                "kind": "quantified_compose",
                "shared": list(shared),
                "eliminate": sorted(eliminate_set),
                "tuple": projected._tuple_to_dict(fact),
                "child": child,
            }
        return Relation(
            projected.variables,
            projected.facts,
            provenance,
            name=name,
            symbolic=projected.symbolic,
        )

    def compose(
        self,
        other: "Relation",
        shared: Iterable[str],
        name: Optional[str] = None,
    ) -> "Relation":
        return self.quantified_compose(other, shared=shared, eliminate=shared, name=name)

    def binary_compose(
        self,
        other: "Relation",
        left: str = "src",
        middle: str = "mid",
        right: str = "dst",
        name: Optional[str] = None,
    ) -> "Relation":
        if len(self.variables) != 2 or len(other.variables) != 2:
            raise ValueError("binary_compose requires binary relations")

        left_relation = self.rename({self.variables[0]: left, self.variables[1]: middle})
        right_relation = other.rename(
            {other.variables[0]: middle, other.variables[1]: right}
        )
        return left_relation.quantified_compose(
            right_relation,
            shared=(middle,),
            eliminate=(middle,),
            keep=(left, right),
            name=name,
        )

    def with_wrapped_witness(
        self,
        kind: str,
        payload: Optional[Mapping[str, Any]] = None,
        name: Optional[str] = None,
    ) -> "Relation":
        data = dict(payload or {})
        provenance: Dict[Fact, Witness] = {}
        for fact in sorted(self.facts):
            provenance[fact] = {
                "kind": kind,
                "tuple": self._tuple_to_dict(fact),
                **data,
                "child": dict(self.provenance.get(fact, {})),
            }
        return Relation(
            self.variables,
            self.facts,
            provenance,
            name=name,
            symbolic=self.symbolic,
        )
