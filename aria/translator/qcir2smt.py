import sys

from aria.bool.qbf import parse_qcir_file


def _gate_expression(gate_id, gate_map, cache):
    if gate_id in cache:
        return cache[gate_id]

    gate_type, refs = gate_map[gate_id]
    parts = [_reference_expression(ref, gate_map, cache) for ref in refs]

    if gate_type == "and":
        expr = "true" if not parts else f"(and {' '.join(parts)})"
    elif gate_type == "or":
        expr = "false" if not parts else f"(or {' '.join(parts)})"
    elif gate_type == "xor":
        expr = "false" if not parts else f"(xor {' '.join(parts)})"
    elif gate_type == "ite":
        if len(parts) != 3:
            raise ValueError("QCIR ite gate requires exactly three operands")
        expr = f"(ite {parts[0]} {parts[1]} {parts[2]})"
    else:
        raise ValueError(f"unsupported QCIR gate type: {gate_type}")

    cache[gate_id] = expr
    return expr


def _reference_expression(ref, gate_map, cache):
    ref_value = int(ref)
    if ref_value < 0:
        return f"(not {_reference_expression(-ref_value, gate_map, cache)})"
    if ref_value in gate_map:
        return _gate_expression(ref_value, gate_map, cache)
    return f"q{ref_value}"


def convert_qcir_to_smt2(input_path, output_path):
    parsed = parse_qcir_file(input_path)
    gate_map = {
        gate.gate_id: (gate.kind, [str(ref) for ref in gate.inputs])
        for gate in parsed.gates
    }
    quantified_vars = {
        variable
        for kind, variables in parsed.parsed_prefix
        if kind in {"a", "e"}
        for variable in variables
    }
    free_vars = sorted(parsed.free_variables() | (parsed.leaf_variables() - quantified_vars))

    expr = _reference_expression(parsed.output_gate, gate_map, {})
    for quantifier_type, variables in reversed(parsed.parsed_prefix):
        if quantifier_type == "f":
            continue
        if not variables:
            continue
        quantifier = "forall" if quantifier_type == "a" else "exists"
        declarations = " ".join(f"(q{variable} Bool)" for variable in variables)
        expr = f"({quantifier} ({declarations}) {expr})"

    lines = []
    for variable in free_vars:
        lines.append(f"(declare-const q{variable} Bool)")
    lines.append(f"(assert {expr})")
    lines.append("(check-sat)")

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")

    return output_path


def main(argv):
    if len(argv) < 3:
        print(f"usage {argv[0]} INPUT_QCIR OUTPUT_SMT2")
        return 1
    convert_qcir_to_smt2(argv[1], argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
