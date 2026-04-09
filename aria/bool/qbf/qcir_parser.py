"""QCIR parser and compatibility wrapper."""

from pathlib import Path
from typing import List, Sequence

from .model import QCIRGate, QCIRInstance, QuantifierBlock


def _parse_qcir_int_list(payload: str) -> List[int]:
    payload = payload.strip()
    if not payload:
        return []
    return [int(token.strip()) for token in payload.split(",") if token.strip()]


def _parse_qcir_gate_definition(line: str) -> QCIRGate:
    compact = line.replace(" ", "")
    gate_id_str, rhs = compact.split("=", maxsplit=1)
    open_paren = rhs.find("(")
    if open_paren < 0 or not rhs.endswith(")"):
        raise ValueError(f"malformed QCIR gate expression: {line}")
    kind = rhs[:open_paren]
    payload = rhs[open_paren + 1 : -1]
    return QCIRGate(kind, int(gate_id_str), _parse_qcir_int_list(payload))


def _build_qcir_instance(
    comments: Sequence[str],
    prefix: Sequence[QuantifierBlock],
    gates: Sequence[QCIRGate],
    output_gate: int,
    normalize: bool,
) -> QCIRInstance:
    instance = QCIRInstance(
        prefix=[QuantifierBlock(block.kind, list(block.variables)) for block in prefix],
        gates=[QCIRGate(gate.kind, gate.gate_id, list(gate.inputs)) for gate in gates],
        output_gate=output_gate,
        comments=list(comments),
    )
    if normalize:
        instance = instance.normalized()
    instance.validate()
    return instance


def parse_qcir_string(content: str, normalize: bool = True) -> QCIRInstance:
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
        if (
            compact.startswith("exists(")
            or compact.startswith("forall(")
            or compact.startswith("free(")
        ):
            if compact.startswith("exists("):
                kind = "e"
                payload = compact[len("exists(") : -1]
            elif compact.startswith("forall("):
                kind = "a"
                payload = compact[len("forall(") : -1]
            else:
                kind = "f"
                payload = compact[len("free(") : -1]
            prefix.append(QuantifierBlock(kind, _parse_qcir_int_list(payload)))
            continue
        if compact.startswith("output("):
            output_gate = int(compact[len("output(") : -1])
            continue
        gates.append(_parse_qcir_gate_definition(line))

    if output_gate == 0:
        raise ValueError("QCIR file is missing an output(...) declaration")

    return _build_qcir_instance(
        comments=comments,
        prefix=prefix,
        gates=gates,
        output_gate=output_gate,
        normalize=normalize,
    )


def parse_qcir_file(path: str, normalize: bool = True) -> QCIRInstance:
    """Parse a QCIR file."""

    return parse_qcir_string(Path(path).read_text(encoding="utf-8"), normalize=normalize)


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
