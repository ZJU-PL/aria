"""
Converting QDIMACS format to smtlib
This one is a bit tricky, as it uses bit-vector variables to "compactly" encode several Booleans.
"""

import os
import sys
from typing import List, NoReturn

from aria.translator.parsing import parse_qdimacs_file


def error(msg: str) -> NoReturn:
    """Print an error message and exit."""
    sys.stderr.write(f"{sys.argv[0]} : {msg}.{os.linesep}")
    sys.exit(1)


def spacesplit(string: str) -> List[str]:
    """Split a string by spaces and filter out empty strings."""
    return list(filter(None, string.split(" ")))


def tointlist(lst: List[str]) -> List[int]:
    """
    Converts a list of strings to a list of integers, and checks that it's 0-terminated.

    Args:
        lst (List[str]): The list to convert.

    Returns:
        List[int]: The list with strings converted to integers and 0 removed.

    Raises:
        ValueError: If the list is not a 0-terminated list of integers.
    """
    try:
        ns = [int(x) for x in lst]
        if not ns[-1] == 0:
            error("expected 0-terminated number list")
        return ns[:-1]

    except (ValueError, IndexError):
        error(f"expected number list (got: {lst})")


def _parse_clause_string(clause: str) -> List[int]:
    clause_values = [int(token) for token in clause.split()]
    if not clause_values or clause_values[-1] != 0:
        raise ValueError(f"expected 0-terminated number list (got: {clause})")
    return clause_values[:-1]


def qdimacs_to_smt2_string(filename: str) -> str:
    parsed = parse_qdimacs_file(filename)

    if len(parsed.preamble) != 4 or parsed.preamble[0] != "p":
        raise ValueError("unexpected problem description")
    if parsed.preamble[1] != "cnf":
        raise ValueError(f"unexpected problem format ('{parsed.preamble[1]}', not cnf?)")

    varcount = int(parsed.preamble[2])
    clausecount = int(parsed.preamble[3])

    lines = [
        f"; QBF variable count : {varcount}",
        f"; QBF clause count   : {clausecount}",
        "",
        "(set-logic UFBV)",
        "(assert",
    ]

    mapping = {}
    for level, (qtype, variables) in enumerate(parsed.parsed_prefix, start=1):
        quant = "forall" if qtype == "a" else "exists"
        lines.append(f"  ({quant} ((vec{level} (_ BitVec {len(variables)})))")
        for index, variable in enumerate(variables):
            if variable in mapping:
                raise ValueError(f"variable {variable} bound multiple times")
            mapping[variable] = (level, index)

    lines.append("    (and")
    for clause in parsed.clauses:
        if not clause.strip():
            continue
        clause_literals = _parse_clause_string(clause)
        if not clause_literals:
            lines.append("      false")
            continue
        parts = []
        for literal in clause_literals:
            variable = abs(literal)
            level, index = mapping[variable]
            bit_val = 1 if literal > 0 else 0
            parts.append(f"(= ((_ extract {index} {index}) vec{level}) #b{bit_val})")
        lines.append(f"      (or {' '.join(parts)})")

    lines.append("    )")
    lines.append(f"  {')' * len(parsed.parsed_prefix)}")
    lines.append(")")
    lines.append("")
    lines.append("(check-sat)")
    return "\n".join(lines) + "\n"


def convert_qdimacs_to_smt2(filename: str, output_path: str) -> str:
    output = qdimacs_to_smt2_string(filename)
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(output)
    return output_path


def parse(filename):
    """
    Parses a QDIMACS file and outputs its equivalent in SMT-LIB2 format, using UFBV logic.
    """
    sys.stdout.write(qdimacs_to_smt2_string(filename))
    return 0


def main(argv):
    """Main function for command-line execution."""
    if len(argv) < 2:
        error("expected file argument")

    parse(argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
