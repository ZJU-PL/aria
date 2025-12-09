
"""
Graph Representation for CFL-Reachability Analysis

This module provides a unified graph interface for CFL-reachability problems.
It supports multiple underlying data structures (Matrix, PAG_Matrix) and
provides a consistent API for graph operations needed in CFL-reachability
algorithms.

The Graph class acts as a wrapper around different graph implementations,
allowing the CFL-reachability algorithms to work with various graph
representations without modification.

Key Features:
- Unified interface for different graph data structures
- Support for labeled edges and vertices
- Efficient edge lookup and iteration
- DOT file output for visualization
- Symbol-pair tracking for CFL analysis

Author: aria team
"""

from typing import List, Dict, Any, Union, Tuple, Optional, Set
from aria.cfl.matrix import Matrix
from aria.cfl.pag_matrix import PAG_Matrix


class Graph:
    """
    A unified graph interface for CFL-reachability analysis.

    This class provides a consistent API for graph operations regardless of
    the underlying data structure implementation. It supports multiple graph
    representations and provides methods for adding vertices/edges, checking
    connectivity, and iterating over graph elements.

    Attributes:
        source_file (str): Path to the input graph file.
        ds_mode (str): The data structure mode ("Matrix" or "PAG_Matrix").
        ds_structure: The underlying graph data structure implementation.

    Example:
        >>> graph = Graph("input.dot", "Matrix")
        >>> graph.add_vertex("node1")
        >>> graph.add_edge("node1", "node2", "label")
        >>> edges = graph.output_edge()
    """

    def __init__(self, source_file: str, ds_mode: str) -> None:
        self.source_file: str = source_file
        self.ds_mode: str = ds_mode
        if ds_mode == "Matrix":
            self.ds_structure = Matrix(source_file)
        elif ds_mode == "PAG_Matrix":
            self.ds_structure = PAG_Matrix(source_file)
        else:
            raise Exception("This is not a valide ds_mode, ds_mode including Matrix")

    def add_vertex(self, vertex: Any) -> bool:
        """
        Add a vertex to the graph.

        Args:
            vertex: The vertex to add (can be a Vertex object or vertex name).

        Returns:
            bool: True if the vertex was added successfully, False if it already exists.
        """
        return self.ds_structure.add_vertex(vertex)

    def add_edge(self, u: Any, v: Any, label: str) -> bool:
        """
        Add a labeled edge between two vertices.

        Args:
            u: Source vertex (can be a Vertex object or vertex name).
            v: Target vertex (can be a Vertex object or vertex name).
            label: The label for the edge.

        Returns:
            bool: True if the edge was added successfully, False otherwise.

        Raises:
            Exception: If either vertex doesn't exist in the graph.
        """
        return self.ds_structure.add_edge(u, v, label)

    def output_edge(self) -> List[List[Union[str, Any]]]:
        """
        Get all edges in the graph as a list of [label, source, target] triplets.

        Returns:
            List[List[Union[str, Any]]]: List of edge triplets where each triplet
                                       contains [label, source_vertex, target_vertex].
        """
        return self.ds_structure.output_edge()

    def check_edge(self, u: Any, v: Any, lable: str) -> bool:
        """
        Check if an edge exists between two vertices with a specific label.

        Args:
            u: Source vertex.
            v: Target vertex.
            lable: The edge label to check for.

        Returns:
            bool: True if the edge exists, False otherwise.
        """
        return self.ds_structure.check_edge(u, v, lable)

    def new_check_edge(self, u: Any, v: Any, lable: str) -> bool:
        """
        Check if an edge exists between two vertices with a specific label (new implementation).

        This method provides an alternative implementation for edge checking that
        may be more efficient for certain use cases.

        Args:
            u: Source vertex.
            v: Target vertex.
            lable: The edge label to check for.

        Returns:
            bool: True if the edge exists, False otherwise.
        """
        return self.ds_structure.new_check_edge(u, v, lable)

    def output_set(self) -> List[Tuple[Any, Any]]:
        return self.ds_structure.output_set()

    def get_vertice(self) -> List[str]:
        return list(self.ds_structure.vertices.keys())

    def symbol_pair_l(self, label: str) -> List[Tuple[Any, Any]]:
        return self.ds_structure.symbol_pair[label]

    def symbol_pair(self) -> Dict[str, List[Tuple[Any, Any]]]:
        return self.ds_structure.symbol_pair

    def dump_dot(self) -> None:
        return self.ds_structure.dump_dot()

    def dump_dot1(self) -> None:
        return self.ds_structure.dump_dot1()
