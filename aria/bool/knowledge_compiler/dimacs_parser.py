"""
Yet another DIMAC parser:  read CNF file and save in a list
Reference: https://github.com/marcmelis/dpll-sat/blob/master/solvers/original_dpll.py
"""

import time
from typing import List, Tuple


def parse_cnf_string(cnf: str, verbose: bool) -> Tuple[List[List[int]], int]:
    """
    Parse CNF formula from string.

    Args:
        cnf: String containing CNF formula in DIMACS format
        verbose: Whether to print parsing statistics

    Returns:
        Tuple of (clauses, number_of_variables)
    """
    initial_time = time.time()
    clauses: List[List[int]] = []
    nvars: int = 0

    if verbose:
        print("=====================[ Problem Statistics ]=====================")
        print("|                                                              |")

    for line in cnf.split("\n"):
        print(line)
        if line.startswith("c"):
            continue
        if line.startswith("p"):
            nvars, nclauses = line.split()[2:4]
            if verbose:
                print(
                    f"|   Nb of variables:      {nvars:10s}                           |"
                )
                print(
                    f"|   Nb of clauses:        {nclauses:10s}                           |"
                )
            continue
        clause = [int(x) for x in line[:-2].split()]
        if len(clause) > 0:
            clauses.append(clause)

    end_time = time.time()
    if verbose:
        print(
            f"|   Parse time:      {end_time - initial_time:10.4f}s"
            f"                               |"
        )
        print("|                                                              |")

    return clauses, int(nvars)


def parse(filename: str, verbose: bool) -> Tuple[List[List[int]], int]:
    """
    Parse CNF formula from file.

    Args:
        filename: Path to CNF file in DIMACS format
        verbose: Whether to print parsing statistics

    Returns:
        Tuple of (clauses, number_of_variables)
    """
    initial_time = time.time()
    clauses: List[List[int]] = []
    nvars: int = 0

    if verbose:
        print("=====================[ Problem Statistics ]=====================")
        print("|                                                              |")
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("c"):
                continue
            if line.startswith("p"):
                nvars, nclauses = line.split()[2:4]
                if verbose:
                    print(
                        f"|   Nb of variables:      {nvars:10s}                           |"
                    )
                    print(
                        f"|   Nb of clauses:        {nclauses:10s}                           |"
                    )
                continue
            clause = [int(x) for x in line[:-2].split()]
            if len(clause) > 0:
                clauses.append(clause)

    end_time = time.time()
    if verbose:
        print(
            f"|   Parse time:      {end_time - initial_time:10.4f}s"
            f"                               |"
        )
        print("|                                                              |")

    return clauses, int(nvars)
