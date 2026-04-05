import sys


def _parse_wcnf(input_path):
    header = None
    clauses = []

    with open(input_path, "r", encoding="utf-8") as input_file:
        for raw_line in input_file:
            line = raw_line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                header = line.split()
                continue
            clauses.append(line.split())

    if header is None or len(header) != 5 or header[1] != "wcnf":
        raise ValueError("Invalid WCNF header")

    num_vars = int(header[2])
    top = int(header[4])
    return num_vars, top, clauses


def _clause_expr(literals):
    if not literals:
        return "false"
    parts = []
    for literal in literals:
        literal_value = int(literal)
        variable = f"v_{abs(literal_value)}"
        parts.append(variable if literal_value > 0 else f"(not {variable})")
    if len(parts) == 1:
        return parts[0]
    return f"(or {' '.join(parts)})"


def convert_wcnf_to_smt2(input_path, output_path):
    num_vars, top, clauses = _parse_wcnf(input_path)

    lines = ["(set-logic QF_UF)"]
    for index in range(1, num_vars + 1):
        lines.append(f"(declare-const v_{index} Bool)")

    for clause_tokens in clauses:
        weight = int(clause_tokens[0])
        literal_tokens = clause_tokens[1:]
        if not literal_tokens or literal_tokens[-1] != "0":
            raise ValueError("Invalid WCNF clause terminator")
        expr = _clause_expr(literal_tokens[:-1])
        if weight == top:
            lines.append(f"(assert {expr})")
        elif weight < top:
            lines.append(f"(assert-soft {expr} :weight {weight})")
        else:
            raise ValueError("Clause weight exceeds top weight")

    lines.append("(check-sat)")
    lines.append("(get-model)")

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")

    return output_path


def main(argv):
    if len(argv) < 3:
        print(f"usage {argv[0]} INPUT_WCNF OUTPUT_SMT2")
        return 1
    convert_wcnf_to_smt2(argv[1], argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
