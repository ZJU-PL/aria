from typing import List, Tuple

import pytest
import z3


def parse_dimacs_string(dimacs_str: str) -> Tuple[int, int, List[List[int]]]:
    """
    Parse a DIMACS CNF string and return the number of variables, clauses, and the clauses.

    :param dimacs_str: String containing DIMACS CNF format
    :return: Tuple of (num_variables, num_clauses, clauses)
    """
    clauses = []
    num_vars = 0
    num_clauses = 0

    for line in dimacs_str.splitlines():
        line = line.strip()
        if not line or line.startswith('c'):  # Skip empty lines and comments
            continue

        if line.startswith('p cnf'):
            # Parse problem line
            _, _, vars, cls = line.split()
            num_vars = int(vars)
            num_clauses = int(cls)
            continue

        # Parse clause line
        clause = [int(x) for x in line.split()[:-1]]  # Ignore trailing 0
        clauses.append(clause)

    return num_vars, num_clauses, clauses


def parse_dimacs(filename: str) -> Tuple[int, int, List[List[int]]]:
    """
    Parse a DIMACS CNF file and return the number of variables, clauses, and the clauses.

    :param filename: Path to the DIMACS CNF file
    :return: Tuple of (num_variables, num_clauses, clauses)
    """
    clauses = []
    num_vars = 0
    num_clauses = 0

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):  # Skip empty lines and comments
                continue

            if line.startswith('p cnf'):
                # Parse problem line
                _, _, vars, cls = line.split()
                num_vars = int(vars)
                num_clauses = int(cls)
                continue

            # Parse clause line
            clause = [int(x) for x in line.split()[:-1]]  # Ignore trailing 0
            clauses.append(clause)

    return num_vars, num_clauses, clauses


def dimacs_to_z3(filename: str) -> z3.BoolRef:
    """
    Read a DIMACS CNF file and convert it directly to Z3 expression.

    :param filename: Path to the DIMACS CNF file
    :return: Z3 expression representing the CNF formula
    """
    _, _, clauses = parse_dimacs(filename)
    return int_clauses_to_z3(clauses)


def dimacs_string_to_z3(dimacs_str: str) -> z3.BoolRef:
    """
    Convert a DIMACS CNF string directly to Z3 expression.

    :param dimacs_str: String containing DIMACS CNF format
    :return: Z3 expression representing the CNF formula
    """
    _, _, clauses = parse_dimacs_string(dimacs_str)
    return int_clauses_to_z3(clauses)


def int_clauses_to_z3(clauses: List[List[int]]) -> z3.BoolRef:
    """
    Convert a list of integer clauses to Z3 expression.
    The function returns the conjunction (AND) of all clauses in the input.
    Each integer represents an atomic proposition.

    :param clauses: List[List[int]] representing the clauses of a CNF
    :return: Z3 expression
    """
    z3_clauses = []
    vars = {}
    for clause in clauses:
        conds = []
        for lit in clause:
            a = abs(lit)
            if a in vars:
                b = vars[a]
            else:
                b = z3.Bool(f"k!{a}")
                vars[a] = b
            b = z3.Not(b) if lit < 0 else b
            conds.append(b)
        z3_clauses.append(z3.Or(*conds))
    return z3.And(*z3_clauses)


# Test cases
def test_parse_dimacs_string_simple():
    dimacs_str = """c Simple test case
p cnf 2 2
1 2 0
-1 -2 0"""
    num_vars, num_clauses, clauses = parse_dimacs_string(dimacs_str)
    assert num_vars == 2
    assert num_clauses == 2
    assert clauses == [[1, 2], [-1, -2]]


def test_parse_dimacs_string_with_comments():
    dimacs_str = """c This is a comment
c Another comment
p cnf 3 3
1 2 3 0
-1 2 0
-2 -3 0"""
    num_vars, num_clauses, clauses = parse_dimacs_string(dimacs_str)
    assert num_vars == 3
    assert num_clauses == 3
    assert clauses == [[1, 2, 3], [-1, 2], [-2, -3]]


def test_dimacs_string_to_z3_simple():
    dimacs_str = """p cnf 2 2
1 2 0
-1 -2 0"""
    expr = dimacs_string_to_z3(dimacs_str)
    s = z3.Solver()
    s.add(expr)
    assert s.check() == z3.sat  # Formula should be satisfiable


def test_dimacs_string_to_z3_unsat():
    dimacs_str = """p cnf 1 2
1 0
-1 0"""
    expr = dimacs_string_to_z3(dimacs_str)
    s = z3.Solver()
    s.add(expr)
    assert s.check() == z3.unsat  # Formula should be unsatisfiable


def test_empty_clause():
    dimacs_str = """p cnf 1 1
0"""
    num_vars, num_clauses, clauses = parse_dimacs_string(dimacs_str)
    assert num_vars == 1
    assert num_clauses == 1
    assert clauses == [[]]


def test_single_literal_clause():
    dimacs_str = """p cnf 1 1
1 0"""
    expr = dimacs_string_to_z3(dimacs_str)
    s = z3.Solver()
    s.add(expr)
    assert s.check() == z3.sat
    m = s.model()
    assert z3.is_true(m[z3.Bool("k!1")])


if __name__ == "__main__":
    pytest.main([__file__])
