"""
Yet another parser
"""
from typing import List, Tuple
from pysat.formula import CNF


def parse_cnf_file(cnf_path: str) -> Tuple[List[List[int]], int, int]:
    """
    Parse number of variables, number of clauses and the clauses from a standard .cnf file
    :param cnf_path:
    :return: clauses, number of clauses, and number of variables
    """
    with open(cnf_path) as f:
        clauses_list = []
        c = 0
        v = 0
        for line in f:
            if line[0] == 'c':
                continue
            if line[0] == 'p':
                sizes = line.split(" ")
                v = int(sizes[2])
                c = int(sizes[3])
            else:
                # all following lines should represent a clause, so literals separated by spaces, with a 0 at the end,
                # denoting the end of the line.
                clauses_list.append([int(x) for x in line.split(" ")[:-1]])
    return clauses_list, c, v


def parse_cnf_string(cnf_str: str) -> Tuple[List[List[int]], int, int]:
    clauses_list = []
    c = 0
    v = 0
    for line in cnf_str.split("\n"):
        if line[0] == 'c':
            continue
        if line[0] == 'p':
            sizes = line.split(" ")
            v = int(sizes[2])
            c = int(sizes[3])
        else:
            clauses_list.append([int(x) for x in line.split(" ")[:-1]])
    return clauses_list, c, v


def parse_cnf_numeric_clauses(clauses: List[List[int]]) -> Tuple[List[List[int]], int, int]:
    clauses_list = []
    c = 0
    v = 0
    for clause in clauses:
        clauses_list.append([int(x) for x in clause])
    return clauses_list, c, v


def parse_pysat_cnf(cnf: CNF) -> Tuple[List[List[int]], int, int]:
    clauses_list = []
    c = 0
    v = 0
    for clause in cnf.clauses:
        clauses_list.append([int(x) for x in clause])
    return clauses_list, c, v
