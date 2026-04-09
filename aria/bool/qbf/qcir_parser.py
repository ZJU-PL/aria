"""QCIR parser and compatibility wrapper."""

from pathlib import Path
from typing import List, Sequence

from .model import QCIRGate, QCIRInstance, QuantifierBlock


def _parse_qcir_int_list(payload: str) -> List[int]:
    payload = payload.strip()
    if not payload:
        return []
    return [int(token.strip()) for token in payload.split(",") if token.strip()]


def parse_qcir_string(content: str) -> QCIRInstance:
    """Parse a QCIR string into a typed instance."""

    comments: List[str] = []
    prefix: List[QuantifierBlock] = []
    gates: List[QCIRGate] = []
    output_gate = 0

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append(line)
            continue
        compact = line.replace(" ", "")
        if compact.startswith("exists(") or compact.startswith("forall("):
            is_exists = compact.startswith("exists(")
            kind = "e" if is_exists else "a"
            payload = compact[7:-1] if is_exists else compact[7:-1]
            prefix.append(QuantifierBlock(kind, _parse_qcir_int_list(payload)))
            continue
        if compact.startswith("output("):
            output_gate = int(compact[len("output(") : -1])
            continue
        gate_id_str, rhs = compact.split("=", maxsplit=1)
        if rhs.startswith("and("):
            gates.append(QCIRGate("and", int(gate_id_str), _parse_qcir_int_list(rhs[4:-1])))
        elif rhs.startswith("or("):
            gates.append(QCIRGate("or", int(gate_id_str), _parse_qcir_int_list(rhs[3:-1])))
        else:
            raise ValueError(f"unsupported QCIR gate expression: {line}")

    if output_gate == 0:
        raise ValueError("QCIR file is missing an output(...) declaration")

    merged_prefix: List[QuantifierBlock] = []
    for block in prefix:
        if merged_prefix and merged_prefix[-1].kind == block.kind:
            merged_prefix[-1].variables.extend(block.variables)
        else:
            merged_prefix.append(QuantifierBlock(block.kind, list(block.variables)))

    return QCIRInstance(
        prefix=merged_prefix,
        gates=gates,
        output_gate=output_gate,
        comments=comments,
    )


def parse_qcir_file(path: str) -> QCIRInstance:
    """Parse a QCIR file."""

    return parse_qcir_string(Path(path).read_text(encoding="utf-8"))


class PaserQCIR:
    """Backward-compatible QCIR parser wrapper."""

    def __init__(self, input_qbf: str):
        self.input_file = input_qbf
        self.instance = parse_qcir_file(input_qbf)
        self.parsed_prefix = self.instance.parsed_prefix
        self.parsed_gates = self.instance.parsed_gates
        self.all_gates = self.instance.all_gates
        self.output_gate = self.instance.output_gate

    def flip_and_assume(
        self, k: int, assum: Sequence[int], assertion: Sequence[Sequence[int]]
    ) -> str:
        """Flip the first ``k`` quantifier blocks and conjoin assumptions/assertions."""

        rewritten_prefix: List[QuantifierBlock] = []
        flipped_vars: List[int] = []
        for index, (kind, variables) in enumerate(self.parsed_prefix):
            if index < k:
                flipped_vars.extend(variables)
                continue
            rewritten_prefix.append(QuantifierBlock(kind, variables))
        if flipped_vars:
            rewritten_prefix.append(QuantifierBlock("e", flipped_vars))

        next_gate = max(self.all_gates + [self.output_gate])
        assertion_gate_ids: List[int] = []
        extra_gates = list(self.instance.gates)
        for clause in assertion:
            next_gate += 1
            extra_gates.append(QCIRGate("or", next_gate, list(clause)))
            assertion_gate_ids.append(next_gate)

        next_gate += 1
        new_output_gate = next_gate
        output_inputs = list(assum) + [self.output_gate] + assertion_gate_ids
        extra_gates.append(QCIRGate("and", new_output_gate, output_inputs))

        flipped = QCIRInstance(
            prefix=rewritten_prefix,
            gates=extra_gates,
            output_gate=new_output_gate,
            comments=self.instance.comments,
        )
        return flipped.to_qcir()


QCIRParser = PaserQCIR
