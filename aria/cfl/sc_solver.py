"""
Set Constraint (SC) Reduction Solver for CFL-Reachability Analysis

This module implements a solver for Set Constraint (SC) reduction problems
using Dynamic Transitive Closure (DTC) algorithms with quantum computing
optimizations. SC-reduction is an alternative approach to CFL-reachability
that converts the problem into a set constraint satisfaction problem.

The SC-reduction approach is particularly useful for certain types of
program analysis problems and can provide different performance characteristics
compared to direct CFL-reachability algorithms.

Key Features:
- DTC-based SC-reduction algorithm
- Quantum iteration estimation using Grover's search
- Conversion from CFL-reachability to set constraints
- Worklist-based analysis for efficient computation
- Support for complex constraint structures

Author: aria team
Adapted from: "Dynamic Transitive Closure-Based Static Analysis through the Lens of Quantum Search"
"""

from math import pi, sqrt, floor, ceil, log
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Any, Union, Tuple, Optional, Set


class SCSolver:
    """
    A solver for Set Constraint (SC) reduction problems in CFL-reachability analysis.

    This class implements the DTC-based SC-reduction algorithm with quantum
    computing optimizations. It converts CFL-reachability problems into set
    constraint satisfaction problems and solves them using worklist-based
    analysis with quantum iteration estimation.

    The SC-reduction approach is particularly useful for program analysis
    applications where the constraint structure can be exploited for
    more efficient computation.

    Attributes:
        mode (str): The solving mode, currently supports "Cubic" for cubic-time algorithm.
        constraint (Dict[str, Dict[str, Set[str]]]): The set constraint structure.
        set_variable (Set[str]): Set of variables in the constraint system.
        nnn (int): Number of constraint variables.

    Example:
        >>> solver = SCSolver("Cubic")
        >>> solver.solve(graph, grammar)
    """

    def __init__(self, mode: str) -> None:
        self.mode: str = mode
        self.constraint: Dict[str, Dict[str, Set[str]]] = {}
        self.set_variable: Set[str] = set()
        self.nnn: int = 0

    # function to estimate the number of quantum iterations
    def estimate(self, sol: int, cand: int) -> int:
        res = floor(sqrt(cand))
        if sol == 0:
            return res
        it = sol
        while it > 0:
            # according to https://doi.org/10.1137/050644719,
            # one target is returned after at most 0.9\sqrt{N/M} queries,
            # where N is the search space size and M is the number of targets.
            res += floor(0.9*sqrt(cand/it))
            it -= 1
        return res

    def solve(self, graph: Any, grammar: Any) -> None:
        if self.mode == "Cubic":
            self.__cubic_solve(graph, grammar)

    # convert the CFL-reachability to SC-reduction
    # according to https://doi.org/10.1109/LICS.1997.614960
    def convert(self, graph: Any, grammar: Any) -> None:
        constraint = defaultdict(lambda: defaultdict(set))
        set_variable: Set[str] = set()
        # process graph
        for v in graph.ds_structure.vertices:
            index = str(graph.ds_structure.edge_indices[v])
            constraint['con0']['X'+index].add('X'+index+','+'node'+index)
            set_variable.add('X'+index)
        for p, elist in graph.ds_structure.symbol_pair.items():
            for e in elist:
                index1 = str(graph.ds_structure.edge_indices[e[0]])
                index2 = str(graph.ds_structure.edge_indices[e[1]])
                constraint['con1']['X'+index1].add('X'+index1+','+p+','+'X'+index2)
        for left,v in grammar.items():
            for right in v:
                if len(right) == 2:
                    for i in graph.ds_structure.edge_indices.values():
                        constraint['pro']['X'+str(i)].add('Rchd'+right[0]+str(1)+str(i)+','+right[0]+str(1)+','+'X'+str(i))
                        constraint['pro']['Rchd'+right[0]+str(1)+str(i)].add('Dst'+left+str(i)+','+right[1]+str(1)+','+'Rchd'+right[0]+str(1)+str(i))
                        constraint['con1']['X'+str(i)].add('X'+str(i)+','+left+','+'Dst'+left+str(i))
                        set_variable.add('Rchd'+right[0]+str(1)+str(i))
                        set_variable.add('Dst'+left+str(i))
                elif len(right) == 1:
                    if right[0] == 'ε':
                        for i in graph.ds_structure.edge_indices.values():
                            constraint['con1']['X'+str(i)].add('X'+str(i)+','+left+','+'X'+str(i))
                    else:
                        for i in graph.ds_structure.edge_indices.values():
                            constraint['con1']['X'+str(i)].add('X'+str(i)+','+left+','+'Dst'+left+str(i))
                            constraint['pro']['X'+str(i)].add('Dst'+left+str(i)+','+right[0]+str(1)+','+'X'+str(i))
                            set_variable.add('Dst'+left+str(i))
                else:
                    raise Exception('grammar error')
        # save the constraint info
        self.constraint = constraint
        self.set_variable = set_variable
        self.nnn = len(constraint['con1'])

    # DTC-based SC-reduction
    def __cubic_solve(self, graph: Any, grammar: Any) -> None:
        # convert the graph to constraints
        # using https://doi.org/10.1109/LICS.1997.614960
        self.convert(graph,grammar)
        constraint = deepcopy(self.constraint)

        # step 1
        W: Set[Tuple[str, str, str]] = set()
        for k in constraint['con0']:
            for i in constraint['con0'][k]:
                left, right = i.split(',')
                W.add(('con', left, right))
        # step 2
        ground: Dict[str, int] = dict()
        for i in self.set_variable:
            ground[i] = 0
        # number of classical iteraitons
        whole_iteration: int = 0
        # number of quantum iterations
        gs_iteration: int = 0

        # start the worklist analysis of SC-reduction
        while len(W) > 0:
            a = W.pop()
            if a[0] == 'con':
                if len(a) < 4:
                    X = a[1]
                    iteration: int = 0
                    num_of_sol: int = 0
                    for k in constraint['con1']:
                        iteration += 1
                        for i in constraint['con1'][k]:
                            if i.split(',')[0] == X:
                                if ground[i.split(',')[2]] == 0:
                                    ground[i.split(',')[2]] = 1
                                    W.add(('con', i.split(',')[0], i.split(',')[2]))
                                    num_of_sol += 1
                    gs_iteration += self.estimate(num_of_sol, iteration)
                    whole_iteration += iteration
                else:
                    X = a[1]
                    Y = a[2]
                    Z = a[3]
                    iteration = 0
                    num_of_sol = 0
                    for k in constraint['pro']:
                        iteration += 1
                        for i in constraint['pro'][k]:
                            if i.split(',')[0] == X and i.split(',')[2] == Y:
                                if ground[i.split(',')[2]] == 1:
                                    if ground[Z] == 0:
                                        ground[Z] = 1
                                        W.add(('con', X, Z))
                                        num_of_sol += 1
                    gs_iteration += self.estimate(num_of_sol, iteration)
                    whole_iteration += iteration
            else:
                X = a[1]
                Y = a[2]
                iteration = 0
                num_of_sol = 0
                for k in constraint['pro']:
                    iteration += 1
                    for i in constraint['pro'][k]:
                        if i.split(',')[0] == X and i.split(',')[2] == Y:
                            if ground[i.split(',')[2]] == 1:
                                if ground[Y] == 0:
                                    ground[Y] = 1
                                    W.add(('con', X, Y))
                                    num_of_sol += 1
                gs_iteration += self.estimate(num_of_sol, iteration)
                whole_iteration += iteration
        # print estimation result
        print('the number of classical iterations: ', whole_iteration)
        print('the number of quantum iterations: ', gs_iteration * 3)
