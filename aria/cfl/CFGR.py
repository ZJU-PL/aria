#!/usr/bin/env pypy3
"""
CFL-Reachability Analysis with Quantum Optimization Estimation

This script performs CFL-reachability analysis on input graphs using context-free
grammars and estimates the potential quantum computing speedup using Grover's
search algorithm. It compares classical and quantum iteration counts for
performance analysis.

The script reads a graph file (DOT format) and a grammar file (EBNF format),
then runs the CFL-reachability algorithm while tracking both classical and
quantum iteration counts.

Usage:
    python3 CFGR.py <graph_file> [grammar_file] [data_structure] [mode]

    graph_file: Path to the input graph file (DOT format)
    grammar_file: Path to the grammar file (default: demo/VM_Grammar.txt)
    data_structure: Graph data structure to use (default: Matrix)
    mode: Solving mode (default: Cubic)

Example:
    python3 CFGR.py demo/100KB.dot demo/VM_Grammar.txt Matrix Cubic

Author: aria team
"""

import sys
from typing import List

from aria.cfl.graph import Graph
from aria.cfl.grammar import Grammar
from aria.cfl.cfl_solver import CFLSolver

def main(argv: List[str]) -> None:
    """
    Main function for CFL-reachability analysis with quantum optimization estimation.

    This function sets up the analysis pipeline by reading command line arguments,
    initializing the graph and grammar objects, and running the CFL-reachability
    solver. It outputs performance statistics including classical and quantum
    iteration counts.

    Args:
        argv (List[str]): Command line arguments. The first argument should be
                         the path to the input graph file.

    The function uses default values for optional parameters:
    - Grammar file: demo/VM_Grammar.txt
    - Data structure: Matrix
    - Mode: Cubic
    """
    arg: List[str] = ['demo/200KB.dot','demo/VM_Grammar.txt','Matrix','Cubic']
    arg[0] = sys.argv[1]
    print('CFL start processing', arg[0])
    graph = Graph(arg[0],arg[2])
    grammar = Grammar(arg[1])
    solver = CFLSolver(arg[3])
    solver.solve(graph, grammar)
    print('complete', arg[0])


if __name__ == '__main__':
    main(sys.argv)
