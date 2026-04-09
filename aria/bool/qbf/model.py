"""Typed data structures shared by Boolean QBF utilities."""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


@dataclass
class QuantifierBlock:
    """One alternating block in a prenex QBF prefix."""

    kind: str
    variables: List[int]

    def __post_init__(self) -> None:
        if self.kind not in {"a", "e"}:
            raise ValueError(f"unsupported quantifier kind: {self.kind}")
        self.variables = [int(variable) for variable in self.variables]


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
        if self.kind not in {"and", "or"}:
            raise ValueError(f"unsupported gate kind: {self.kind}")
        self.gate_id = int(self.gate_id)
        self.inputs = [int(input_ref) for input_ref in self.inputs]


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

    def to_qcir(self) -> str:
        lines = ["#QCIR-G14"]
        for block in self.prefix:
            keyword = "forall" if block.kind == "a" else "exists"
            payload = ",".join(str(variable) for variable in block.variables)
            lines.append(f"{keyword}({payload})")
        lines.append(f"output({self.output_gate})")
        for gate in self.gates:
            payload = ",".join(str(input_ref) for input_ref in gate.inputs)
            lines.append(f"{gate.gate_id} = {gate.kind}({payload})")
        return "\n".join(lines) + "\n"
