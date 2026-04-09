"""Typed data structures shared by Boolean QBF utilities."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _dedupe_preserve_order(values: Sequence[int]) -> List[int]:
    deduped: List[int] = []
    seen: Set[int] = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _merge_adjacent_blocks(blocks: Iterable["QuantifierBlock"]) -> List["QuantifierBlock"]:
    merged: List[QuantifierBlock] = []
    for block in blocks:
        if merged and merged[-1].kind == block.kind:
            merged[-1] = QuantifierBlock(
                block.kind,
                _dedupe_preserve_order(merged[-1].variables + list(block.variables)),
            )
        else:
            merged.append(QuantifierBlock(block.kind, list(block.variables)))
    return merged


def _normalize_clause(clause: Sequence[int]) -> Optional[List[int]]:
    normalized: List[int] = []
    seen: Set[int] = set()
    for literal in clause:
        literal = int(literal)
        if literal == 0:
            raise ValueError("literal 0 is reserved as a clause terminator")
        if -literal in seen:
            return None
        if literal not in seen:
            normalized.append(literal)
            seen.add(literal)
    return normalized


@dataclass
class QuantifierBlock:
    """One alternating block in a prenex QBF prefix."""

    kind: str
    variables: List[int]

    def __post_init__(self) -> None:
        if self.kind not in {"a", "e", "f"}:
            raise ValueError(f"unsupported quantifier kind: {self.kind}")
        self.variables = [int(variable) for variable in self.variables]
        if any(variable <= 0 for variable in self.variables):
            raise ValueError("quantified variables must be positive integers")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("duplicate variables within one quantifier block")


@dataclass
class QDIMACSInstance:
    """Parsed QDIMACS formula."""

    num_vars: int
    num_clauses: int
    prefix: List[QuantifierBlock]
    clauses: List[List[int]]
    comments: List[str] = field(default_factory=list)

    @property
    def preamble(self) -> List[str]:
        return ["p", "cnf", str(self.num_vars), str(self.num_clauses)]

    @property
    def parsed_prefix(self) -> List[Tuple[str, List[int]]]:
        return [(block.kind, list(block.variables)) for block in self.prefix]

    @property
    def clause_lines(self) -> List[str]:
        return [
            (" ".join(str(literal) for literal in clause) + " 0").strip()
            for clause in self.clauses
        ]

    @property
    def all_vars(self) -> List[int]:
        variables = set()
        for block in self.prefix:
            variables.update(block.variables)
        for clause in self.clauses:
            variables.update(abs(literal) for literal in clause)
        return sorted(variables)

    def quantified_variables(self) -> Set[int]:
        return {variable for block in self.prefix for variable in block.variables}

    def free_variables(self) -> List[int]:
        quantified = self.quantified_variables()
        return [variable for variable in self.all_vars if variable not in quantified]

    def normalized(
        self,
        promote_free_vars: bool = True,
        drop_tautologies: bool = True,
        deduplicate_clauses: bool = True,
    ) -> "QDIMACSInstance":
        normalized_clauses: List[List[int]] = []
        seen_clauses: Set[Tuple[int, ...]] = set()
        for clause in self.clauses:
            normalized_clause = _normalize_clause(clause)
            if normalized_clause is None:
                if drop_tautologies:
                    continue
                raise ValueError("tautological clauses must be normalized away")
            clause_key = tuple(normalized_clause)
            if deduplicate_clauses and clause_key in seen_clauses:
                continue
            normalized_clauses.append(normalized_clause)
            seen_clauses.add(clause_key)

        observed_vars = {
            abs(literal) for clause in normalized_clauses for literal in clause
        }
        normalized_prefix = _merge_adjacent_blocks(
            QuantifierBlock(block.kind, list(block.variables))
            for block in self.prefix
        )
        if promote_free_vars:
            free_vars = sorted(observed_vars - self.quantified_variables())
            if free_vars:
                normalized_prefix = _merge_adjacent_blocks(
                    [QuantifierBlock("e", free_vars)] + normalized_prefix
                )
        active_vars = observed_vars.union(
            variable
            for block in normalized_prefix
            for variable in block.variables
            if variable in observed_vars
        )
        trimmed_prefix: List[QuantifierBlock] = []
        for block in normalized_prefix:
            kept = [variable for variable in block.variables if variable in active_vars]
            if kept:
                trimmed_prefix.append(QuantifierBlock(block.kind, kept))

        return QDIMACSInstance(
            num_vars=max(active_vars, default=0),
            num_clauses=len(normalized_clauses),
            prefix=trimmed_prefix,
            clauses=normalized_clauses,
            comments=list(self.comments),
        )

    def validate(self) -> None:
        if self.num_vars < 0:
            raise ValueError("QDIMACS variable count must be non-negative")
        if self.num_clauses != len(self.clauses):
            raise ValueError(
                f"QDIMACS clause count mismatch: header says {self.num_clauses}, "
                f"got {len(self.clauses)}"
            )

        quantified: Set[int] = set()
        for block in self.prefix:
            if block.kind not in {"a", "e"}:
                raise ValueError("QDIMACS only supports existential/universal blocks")
            for variable in block.variables:
                if variable in quantified:
                    raise ValueError(f"variable {variable} quantified multiple times")
                quantified.add(variable)

        max_variable = 0
        for clause in self.clauses:
            for literal in clause:
                literal = int(literal)
                if literal == 0:
                    raise ValueError("literal 0 cannot appear inside a clause")
                max_variable = max(max_variable, abs(literal))
        for block in self.prefix:
            for variable in block.variables:
                max_variable = max(max_variable, variable)

        if max_variable > self.num_vars:
            raise ValueError(
                f"QDIMACS header declares {self.num_vars} variables, "
                f"but formula references {max_variable}"
            )

    def to_qdimacs(self) -> str:
        lines = [f"p cnf {self.num_vars} {self.num_clauses}"]
        for block in self.prefix:
            lines.append(
                f"{block.kind} {' '.join(str(variable) for variable in block.variables)} 0"
            )
        for clause in self.clauses:
            lines.append(" ".join(str(literal) for literal in clause) + " 0")
        return "\n".join(lines) + "\n"


@dataclass
class QCIRGate:
    """One QCIR gate definition."""

    kind: str
    gate_id: int
    inputs: List[int]

    def __post_init__(self) -> None:
        if self.kind not in {"and", "or", "xor", "ite"}:
            raise ValueError(f"unsupported gate kind: {self.kind}")
        self.gate_id = int(self.gate_id)
        if self.gate_id <= 0:
            raise ValueError("QCIR gate ids must be positive integers")
        self.inputs = [int(input_ref) for input_ref in self.inputs]
        if any(input_ref == 0 for input_ref in self.inputs):
            raise ValueError("QCIR gate inputs must be non-zero references")
        if self.kind == "ite" and len(self.inputs) != 3:
            raise ValueError("QCIR ite gates require exactly three inputs")


@dataclass
class QCIRInstance:
    """Parsed QCIR formula."""

    prefix: List[QuantifierBlock]
    gates: List[QCIRGate]
    output_gate: int
    comments: List[str] = field(default_factory=list)

    @property
    def parsed_prefix(self) -> List[Tuple[str, List[int]]]:
        return [(block.kind, list(block.variables)) for block in self.prefix]

    @property
    def parsed_gates(self) -> List[Tuple[str, str, List[str]]]:
        return [
            (gate.kind, str(gate.gate_id), [str(input_ref) for input_ref in gate.inputs])
            for gate in self.gates
        ]

    @property
    def all_gates(self) -> List[int]:
        gate_ids = {gate.gate_id for gate in self.gates}
        for block in self.prefix:
            gate_ids.update(block.variables)
        return sorted(gate_ids)

    def gate_map(self) -> Dict[int, QCIRGate]:
        return {gate.gate_id: gate for gate in self.gates}

    def quantified_variables(self) -> Set[int]:
        return {
            variable
            for block in self.prefix
            if block.kind in {"a", "e"}
            for variable in block.variables
        }

    def free_variables(self) -> Set[int]:
        return {
            variable
            for block in self.prefix
            if block.kind == "f"
            for variable in block.variables
        }

    def leaf_variables(self) -> Set[int]:
        gate_ids = {gate.gate_id for gate in self.gates}
        leaves = {
            abs(input_ref)
            for gate in self.gates
            for input_ref in gate.inputs
            if abs(input_ref) not in gate_ids
        }
        if self.output_gate not in gate_ids:
            leaves.add(self.output_gate)
        return leaves

    def normalized(self, infer_free_variables: bool = True) -> "QCIRInstance":
        normalized_prefix = _merge_adjacent_blocks(
            QuantifierBlock(block.kind, list(block.variables))
            for block in self.prefix
        )
        free_vars: List[int] = []
        quantifier_blocks: List[QuantifierBlock] = []
        for block in normalized_prefix:
            if block.kind == "f":
                free_vars.extend(block.variables)
            else:
                quantifier_blocks.append(block)
        prefix_blocks: List[QuantifierBlock] = []
        if free_vars:
            prefix_blocks.append(QuantifierBlock("f", _dedupe_preserve_order(free_vars)))
        prefix_blocks.extend(_merge_adjacent_blocks(quantifier_blocks))

        if infer_free_variables:
            declared = {
                variable for block in prefix_blocks for variable in block.variables
            }
            inferred_free = sorted(self.leaf_variables() - declared)
            if inferred_free:
                if prefix_blocks and prefix_blocks[0].kind == "f":
                    prefix_blocks[0] = QuantifierBlock(
                        "f", prefix_blocks[0].variables + inferred_free
                    )
                else:
                    prefix_blocks = [QuantifierBlock("f", inferred_free)] + prefix_blocks

        return QCIRInstance(
            prefix=prefix_blocks,
            gates=[QCIRGate(gate.kind, gate.gate_id, list(gate.inputs)) for gate in self.gates],
            output_gate=self.output_gate,
            comments=list(self.comments),
        )

    def validate(self) -> None:
        declared_prefix_vars: Set[int] = set()
        for block in self.prefix:
            for variable in block.variables:
                if variable in declared_prefix_vars:
                    raise ValueError(f"variable {variable} declared multiple times in QCIR")
                declared_prefix_vars.add(variable)

        gate_ids: Set[int] = set()
        for gate in self.gates:
            if gate.gate_id in gate_ids:
                raise ValueError(f"duplicate QCIR gate id: {gate.gate_id}")
            if gate.gate_id in declared_prefix_vars:
                raise ValueError(
                    f"gate id {gate.gate_id} conflicts with a quantified/free variable"
                )
            gate_ids.add(gate.gate_id)

        for gate in self.gates:
            for input_ref in gate.inputs:
                atom = abs(input_ref)
                if atom not in gate_ids and atom not in declared_prefix_vars:
                    if any(block.kind == "f" for block in self.prefix):
                        raise ValueError(f"undefined QCIR atom reference: {input_ref}")

        if self.output_gate not in gate_ids and self.output_gate not in declared_prefix_vars:
            raise ValueError(f"QCIR output {self.output_gate} is not defined")

        visiting: Set[int] = set()
        visited: Set[int] = set()
        gate_map = self.gate_map()

        def visit(gate_id: int) -> None:
            if gate_id in visited:
                return
            if gate_id in visiting:
                raise ValueError("QCIR gate graph must be acyclic")
            visiting.add(gate_id)
            for input_ref in gate_map[gate_id].inputs:
                input_gate = abs(input_ref)
                if input_gate in gate_map:
                    visit(input_gate)
            visiting.remove(gate_id)
            visited.add(gate_id)

        for gate_id in gate_map:
            visit(gate_id)

    def to_qcir(self) -> str:
        lines = ["#QCIR-G14"]
        for block in self.prefix:
            if block.kind == "a":
                keyword = "forall"
            elif block.kind == "e":
                keyword = "exists"
            else:
                keyword = "free"
            payload = ",".join(str(variable) for variable in block.variables)
            lines.append(f"{keyword}({payload})")
        lines.append(f"output({self.output_gate})")
        for gate in self.gates:
            payload = ",".join(str(input_ref) for input_ref in gate.inputs)
            lines.append(f"{gate.gate_id} = {gate.kind}({payload})")
        return "\n".join(lines) + "\n"
