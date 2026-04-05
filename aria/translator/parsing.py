from typing import List, Tuple

from aria.bool.features.parse_cnf import parse_cnf_string as parse_external_cnf_string
from aria.bool.qbf.qdimacs_parser import PaserQDIMACS


def _normalize_nonempty_lines(content: str) -> str:
    return "\n".join(line.strip() for line in content.splitlines() if line.strip())


def parse_dimacs_content(content: str) -> Tuple[int, int, List[List[int]]]:
    normalized = _normalize_nonempty_lines(content)
    clauses, num_clauses, num_vars = parse_external_cnf_string(normalized)
    return num_vars, num_clauses, clauses


def parse_dimacs_file(path: str) -> Tuple[int, int, List[List[int]]]:
    with open(path, "r", encoding="utf-8") as input_file:
        return parse_dimacs_content(input_file.read())


def parse_qdimacs_file(path: str) -> PaserQDIMACS:
    return PaserQDIMACS(path)
