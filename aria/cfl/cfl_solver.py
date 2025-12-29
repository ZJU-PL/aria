"""
Context-Free Language (CFL) Reachability Solver

This module implements a solver for CFL-reachability problems using Dynamic
Transitive Closure (DTC) algorithms with quantum computing optimizations.
The solver can analyze whether certain paths exist in a graph that satisfy
a given context-free grammar.

The implementation includes:
- DTC-based CFL-reachability algorithm
- Quantum iteration estimation using Grover's search
- Worklist-based analysis for efficient computation
- Support for various grammar production rules

Author: aria team
Adapted from: "Dynamic Transitive Closure-Based Static Analysis through the Lens of Quantum Search"
"""

from math import sqrt, floor, log
from typing import List, Dict, Any, Union, Tuple


class CFLSolver:
    """
    A solver for Context-Free Language (CFL) reachability problems.

    This class implements the DTC-based CFL-reachability algorithm with quantum
    computing optimizations. It can determine whether certain paths exist in a
    graph that satisfy a given context-free grammar.

    Attributes:
        mode (str): The solving mode, currently supports "Cubic" for cubic-time algorithm.

    Example:
        >>> solver = CFLSolver("Cubic")
        >>> solver.solve(graph, grammar)
    """

    def __init__(self, mode: str) -> None:
        """
        Initialize the CFL solver with the specified mode.

        Args:
            mode (str): The solving mode. Currently only "Cubic" is supported,
                       which uses a cubic-time DTC-based algorithm.

        Raises:
            ValueError: If mode is not supported.
        """
        self.mode: str = mode

    def estimate(self, sol: int, cand: int, nnn: int) -> int:
        """
        Estimate the number of quantum iterations needed for Grover's search.

        This method estimates the number of quantum iterations required to find
        all solutions using Grover's search algorithm. The estimation is based
        on the theoretical analysis from the paper referenced in the comments.

        Args:
            sol (int): Number of solutions (targets) to find.
            cand (int): Number of candidates in the search space.
            nnn (int): Total number of vertices in the graph (used for scaling).

        Returns:
            int: Estimated number of quantum iterations needed.

        Note:
            The estimation is based on the theoretical result that one target
            is returned after at most 0.9√(N/M) queries, where N is the search
            space size and M is the number of targets.
        """
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
        return floor(res * log(nnn))

    def solve(self, graph: Any, grammar: Any) -> None:
        """
        Solve the CFL-reachability problem for the given graph and grammar.

        This method is the main entry point for solving CFL-reachability problems.
        It delegates to the appropriate solving algorithm based on the mode.

        Args:
            graph: The input graph containing vertices and edges with labels.
            grammar: The context-free grammar defining the reachability rules.

        Raises:
            NotImplementedError: If the specified mode is not implemented.
        """
        if self.mode == "Cubic":
            self.__cubic_solve(graph, grammar)

    def __cubic_solve(self, graph: Any, grammar: Any) -> None:
        """
        Solve CFL-reachability using the cubic-time DTC-based algorithm.

        This method implements the Dynamic Transitive Closure (DTC) algorithm
        for CFL-reachability. It uses a worklist-based approach to iteratively
        compute reachability relations and estimates quantum iterations for
        potential speedup using Grover's search.

        The algorithm processes grammar productions of the form:
        - X → Y (unary productions)
        - X → YZ (binary productions)

        Args:
            graph: The input graph with labeled edges.
            grammar: The context-free grammar with production rules.

        Note:
            This method prints the number of classical and quantum iterations
            to stdout for performance analysis.
        """
        # each edge in worklist stand by [(edge, node, node)]
        # number of classical iterations
        whole_iteration: int = 0
        # number of quantum iterations
        gs_iteration: int = 0
        print('graph size: ',len(graph.ds_structure.vertices))
        nnn = len(graph.ds_structure.vertices)
        Worklist: List[List[Union[str, Any]]] = graph.output_edge()
        for nullable_variable in grammar.epsilon:
            for node in graph.get_vertice():
                graph.add_edge(node, node, nullable_variable)
                Worklist.append([nullable_variable,node,node])
        # worklist analysis of DTC-based CFL-reachability
        while Worklist != []:
            selected_edge = Worklist.pop()
            for X, right_list in grammar.items():
                # X: key: variable right_list : list of all right handside of production
                for right in right_list:
                    # X = Y
                    if len(right) == 1 and right[0] == selected_edge[0]:
                        Y = right[0]
                        for pair in graph.symbol_pair_l(Y):
                            # O(n) for graph.symbol_pair_l return list of node pair
                            if not graph.new_check_edge(pair[0],pair[1],X):
                                # O(m) m stand for len(varibale, terminal)
                                graph.add_edge(pair[0],pair[1],X)
                                Worklist.append([X,pair[0],pair[1]])
            # codes that lead to cubic bottleneck
            for X, right in grammar.items():
                for right_symbols in right:
                    if len(right_symbols) == 2 and right_symbols[0] == selected_edge[0]:
                        Y = right_symbols[0]
                        Z = right_symbols[1]
                        if Z in graph.symbol_pair():
                            num_of_sol: int = 0
                            iteration: int = 0
                            for pair in graph.symbol_pair_l(Z):
                                iteration+=1
                                j = selected_edge[2]
                                i = selected_edge[1]
                                k = pair[1]
                                if pair[0] == selected_edge[2]:
                                    if not graph.new_check_edge(i,k,X):
                                        graph.add_edge(i,k,X)
                                        Worklist.append([X,i,k])
                                        num_of_sol+=1
                            # update number of classical and quantum iterations
                            gs_iteration+=self.estimate(num_of_sol,iteration,nnn)
                            whole_iteration+=iteration
            for X, right in grammar.items():
                for right_symbols in right:
                    if len(right_symbols) == 2 and right_symbols[1] == selected_edge[0]:
                        Y = right_symbols[1]
                        Z = right_symbols[0]
                        if Z in graph.symbol_pair():
                            num_of_sol: int = 0
                            iteration: int = 0
                            for pair in graph.symbol_pair_l(Z):
                                iteration+=1
                                j = selected_edge[2]
                                i = selected_edge[1]
                                k = pair[0]
                                if pair[1] == i:
                                    if not graph.new_check_edge(k,j,X):
                                        graph.add_edge(k,j,X)
                                        Worklist.append([X,k,j])
                                        num_of_sol+=1
                            # update number of classical and quantum iterations
                            gs_iteration+=self.estimate(num_of_sol,iteration,nnn)
                            whole_iteration+=iteration
        # print estimation result
        print('the number of classical iterations: ', whole_iteration)
        print('the number of quantum iterations: ', gs_iteration * 3)
