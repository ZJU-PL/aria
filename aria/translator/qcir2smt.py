"""Translate QCIR formulas to SMT-LIB2."""

import sys
from typing import Dict, Iterable, List, Optional, Set

from aria.bool.qbf import QCIRInstance, parse_qcir_file


def _normalize_output_reference(output_gate: int, gate_ids: Set[int]) -> int:
    if output_gate == 0:
        raise ValueError("QCIR output 0 is invalid")
    if output_gate > 0 or abs(output_gate) in gate_ids:
        return output_gate

    # Allow direct negated-variable outputs even though the parser stores the raw int.
    return output_gate


def _xor_expression(parts: List[str]) -> str:
    if not parts:
        return "false"
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"(xor {parts[0]} {parts[1]})"
    parity = parts[0]
    for part in parts[1:]:
        parity = f"(xor {parity} {part})"
    return parity


def _gate_expression(
    gate_id: int,
    gate_map: Dict[int, tuple],
    cache: Dict[int, str],
    visiting: Optional[Set[int]] = None,
) -> str:
    if gate_id in cache:
        return cache[gate_id]
    if visiting is None:
        visiting = set()
    if gate_id in visiting:
        raise ValueError("QCIR gate graph must be acyclic")
    if gate_id not in gate_map:
        raise ValueError(f"undefined QCIR gate reference: {gate_id}")

    visiting.add(gate_id)
    gate_type, refs = gate_map[gate_id]
    parts = [_reference_expression(ref, gate_map, cache, visiting) for ref in refs]

    if gate_type == "and":
        expr = "true" if not parts else f"(and {' '.join(parts)})"
    elif gate_type == "or":
        expr = "false" if not parts else f"(or {' '.join(parts)})"
    elif gate_type == "xor":
        expr = _xor_expression(parts)
    elif gate_type == "ite":
        if len(parts) != 3:
            raise ValueError("QCIR ite gate requires exactly three operands")
        expr = f"(ite {parts[0]} {parts[1]} {parts[2]})"
    else:
        raise ValueError(f"unsupported QCIR gate type: {gate_type}")

    visiting.remove(gate_id)
    cache[gate_id] = expr
    return expr


def _reference_expression(
    ref: int,
    gate_map: Dict[int, tuple],
    cache: Dict[int, str],
    visiting: Optional[Set[int]] = None,
) -> str:
    ref_value = int(ref)
    if ref_value < 0:
        return f"(not {_reference_expression(-ref_value, gate_map, cache, visiting)})"
    if ref_value in gate_map:
        return _gate_expression(ref_value, gate_map, cache, visiting)
    return f"q{ref_value}"


def qcir_to_smt2_string(input_path: str) -> str:
    parsed = parse_qcir_file(input_path)
    return instance_to_smt2_string(parsed)


def instance_to_smt2_string(parsed: QCIRInstance) -> str:
    parsed.validate()
    gate_map = {
        gate.gate_id: (gate.kind, list(gate.inputs))
        for gate in parsed.gates
    }
    gate_ids = set(gate_map)
    output_ref = _normalize_output_reference(parsed.output_gate, gate_ids)
    expr = _reference_expression(output_ref, gate_map, {})

    quantified_vars = {
        variable
        for kind, variables in parsed.parsed_prefix
        if kind in {"a", "e"}
        for variable in variables
    }
    free_vars = sorted(parsed.free_variables() | (parsed.leaf_variables() - quantified_vars))

    for quantifier_type, variables in reversed(parsed.parsed_prefix):
        if quantifier_type == "f" or not variables:
            continue
        quantifier = "forall" if quantifier_type == "a" else "exists"
        declarations = " ".join(f"(q{variable} Bool)" for variable in variables)
        expr = f"({quantifier} ({declarations}) {expr})"

    lines = ["(set-logic UFBV)"]
    for variable in free_vars:
        lines.append(f"(declare-const q{variable} Bool)")
    lines.append(f"(assert {expr})")
    lines.append("(check-sat)")
    return "\n".join(lines) + "\n"


def convert_qcir_to_smt2(input_path: str, output_path: str) -> str:
    output = qcir_to_smt2_string(input_path)
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(output)
    return output_path


def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print(f"usage {argv[0]} INPUT_QCIR OUTPUT_SMT2")
        return 1
    convert_qcir_to_smt2(argv[1], argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
