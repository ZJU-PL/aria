"""
Matrix-based Graph Representation for CFL-Reachability Analysis

This module implements a matrix-based representation of graphs specifically
optimized for CFL-reachability problems. It provides efficient storage and
lookup of labeled edges using adjacency matrices and symbol-pair tracking.

The Matrix class uses a 2D list structure where each cell contains a list
of edge labels, allowing multiple edges between the same pair of vertices
with different labels. This is essential for CFL-reachability analysis where
different grammar symbols can create different types of connections.

Key Features:
- Efficient edge storage using adjacency matrix
- Support for multiple edges between same vertex pair
- Symbol-pair tracking for CFL analysis
- DOT file parsing and generation
- Fast edge existence checking

Author: aria team
"""

import re
from typing import Dict, List, Any, Union, Tuple, Optional


class Vertex:
    """
    A simple vertex representation for graph nodes.

    This class represents a single vertex in the graph with a unique name.
    It's used as a lightweight wrapper around vertex names for type safety
    and potential future extensions.

    Attributes:
        name (str): The unique identifier/name of the vertex.
    """

    def __init__(self, n: str) -> None:
        """
        Initialize a vertex with the given name.

        Args:
            n (str): The name/identifier for this vertex.
        """
        self.name: str = n

class Matrix:
    """
    A matrix-based graph representation for CFL-reachability analysis.

    This class implements a graph using an adjacency matrix where each cell
    contains a list of edge labels. This allows multiple labeled edges between
    the same pair of vertices, which is essential for CFL-reachability analysis.

    The matrix is automatically populated by reading from a graph file (DOT or text format).
    It maintains symbol-pair mappings for efficient CFL analysis and provides
    methods for adding vertices/edges and checking connectivity.

    Attributes:
        source_file (str): Path to the input graph file.
        vertices (Dict[str, Vertex]): Mapping from vertex names to Vertex objects.
        edges (List[List[List[str]]]): 2D matrix where edges[i][j] contains list of labels.
        edge_indices (Dict[str, int]): Mapping from vertex names to matrix indices.
        symbol_pair (Dict[str, List[Tuple[str, str]]]): Mapping from labels to vertex pairs.

    Example:
        >>> matrix = Matrix("graph.dot")
        >>> matrix.add_vertex("node1")
        >>> matrix.add_edge("node1", "node2", "label")
        >>> has_edge = matrix.check_edge("node1", "node2", "label")
    """

    def __init__(self, source_file: str) -> None:
        self.source_file: str = source_file
        self.vertices: Dict[str, Vertex] = {}
        self.edges: List[List[List[str]]] = []
        self.edge_indices: Dict[str, int] = {}
        self.symbol_pair: Dict[str, List[Tuple[str, str]]] = {}
        self.__read_graph()

    def __read_graph(self) -> None:
        suffix = self.source_file.split('.')[-1]
        if suffix == 'txt':
            with open(self.source_file,'r') as f:
                for line in f.readlines():
                    node1_name, node2_name, edge = line.strip().split(',')
                    node1_name = node1_name.strip()
                    node2_name = node2_name.strip()
                    edge = edge.strip()
                    if node1_name not in self.vertices:
                        node1 = Vertex(node1_name)
                        self.add_vertex(node1)
                    if node2_name not in self.vertices:
                        node2 = Vertex(node2_name)
                        self.add_vertex(node2)
                    self.add_edge(node1_name,node2_name,edge)
            return
        else:
            return self.__read_dot_file()

    def __read_dot_file(self) -> None:
        edge_pattern = re.compile(r'(\w+)\s*->\s*(\w+)\s*\[.*color=(.*)\]')
        node_pattern = re.compile(r'(\w+)')
        with open(self.source_file, 'r') as f:
            lines = f.readlines()
            line_1 = lines[0]
            line_2 = lines[-1]
            line_1 = line_1.split('{')[1]
            line_2 = line_2.split('}')[0]
            lines[0] = line_1
            lines[-1] = line_2
            for line in lines:
                if ('=' in line and "[" in line) or ("=" not in line and "[" not in line):
                    if "->" in line:
                        match = edge_pattern.search(line)
                        if match is not None:
                            node_1, node_2, label = match.group(1), match.group(2), match.group(3)
                            node_i1 = Vertex(node_1)
                            node_i2 = Vertex(node_2)
                            self.add_vertex(node_i1)
                            self.add_vertex(node_i2)
                            if label == "red":
                                label = 'd'
                                self.add_edge(node_2,node_1,label+'bar')
                            elif label == "black" or label == "purple":
                                label = 'a'
                                self.add_edge(node_2,node_1,label+'bar')
                            self.add_edge(node_1,node_2,label)
                    else:
                        match = node_pattern.search(line)
                        if match is not None:
                            node = match.group(1)
                            node_i = Vertex(node)
                            self.add_vertex(node_i)
        return

    def add_vertex(self, vertex: Vertex) -> bool:
        """
        Add a vertex to the matrix-based graph representation.

        This method adds a new vertex to the graph and updates the adjacency
        matrix structure accordingly. It also updates the edge indices mapping
        and symbol-pair tracking.

        Args:
            vertex (Vertex): The vertex object to add to the graph.

        Returns:
            bool: True if the vertex was added successfully, False if it already exists.
        """
        if isinstance(vertex, Vertex) and vertex.name not in self.vertices:
            self.vertices[vertex.name] = vertex
            for row in self.edges:
                row.append([])
            self.edges.append([])
            for _ in range(len(self.edges)):
                self.edges[-1].append([])
            self.edge_indices[vertex.name] = len(self.edge_indices)
            return True
        else:
            return False

    def add_edge(self, u: str, v: str, label: str) -> bool:
        """
        Add a labeled edge between two vertices in the matrix representation.

        This method adds an edge with the specified label between vertices u and v.
        It updates both the adjacency matrix and the symbol-pair tracking structure.

        Args:
            u (str): Source vertex name.
            v (str): Target vertex name.
            label (str): The label for the edge.

        Returns:
            bool: True if the edge was added successfully.

        Raises:
            Exception: If either vertex doesn't exist in the graph.
        """
        if u in self.vertices and v in self.vertices:
            self.edges[self.edge_indices[u]][self.edge_indices[v]].append(label)
            if label in self.symbol_pair.keys():
                if self.new_check_edge(u, v, label):
                    self.symbol_pair[label].append((u,v))
            else:
                self.symbol_pair[label] = []
                self.symbol_pair[label].append((u,v))
            return True
        elif u not in self.vertices and v in self.vertices:
            raise Exception(f'Node {u} is not in the graph')
        elif v not in self.vertices and u in self.vertices:
            raise Exception(f'Node {v} is not in the graph')
        else:
            raise Exception(f'Node {u} and Node {v} are not in the graph')

    def output_edge(self) -> List[List[Union[str, str]]]:
        output_list: List[List[Union[str, str]]] = []
        for edge, pair_list in self.symbol_pair.items():
            for pair in pair_list:
                output_list.append([edge,pair[0],pair[1]])
        return output_list

    def check_edge(self, u: str, v: str, lable: str) -> bool:
        if lable not in self.symbol_pair:
            return False
        if (u,v) in self.symbol_pair[lable]:
            return True
        return False

    def new_check_edge(self, u: Union[Vertex, str], v: Union[Vertex, str], lable: str) -> bool:
        if isinstance(u,Vertex) and isinstance(v,Vertex):
            u_index = self.edge_indices[u.name]
            v_index = self.edge_indices[v.name]
        else:
            u_index = self.edge_indices[u]
            v_index = self.edge_indices[v]
        if lable in self.edges[u_index][v_index]:
            return True
        else:
            return False

    def output_set(self) -> List[Tuple[str, str]]:
        return self.symbol_pair["M"]

    def dump_dot(self) -> None:
        with open('generated_file/dump_dot.dot','w') as f:
            f.write('digraph CFG{\n')
            for node in self.vertices:
                f.write(f'\tn{node};\n')
            for symbol in self.symbol_pair:
                if symbol in 'MVaabarddbar':
                    for pair in self.symbol_pair[symbol]:
                        f.write(f'\tn{pair[0]}->n{pair[1]}[label="{symbol}"]\n')
            f.write('}')
