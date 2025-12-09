"""
Tests for the OCaml-parity fixpoint analyzer (analyze function).
"""

import unittest
from typing import Callable, Dict, List, Set, Tuple

from aria.srk.fixpoint import analyze


class SelfLoopGraph:
    """Single-vertex graph with a self-loop."""

    def __init__(self, vertex: str = 'X') -> None:
        self.vertex = vertex

    # For loop forest construction
    def iter_vertex(self, f: Callable[[str], None]) -> None:
        f(self.vertex)

    def iter_succ(self, f: Callable[[str], None], v: str) -> None:
        if v == self.vertex:
            f(self.vertex)

    # For fixpoint propagation
    def fold_pred_edges(self, f: Callable[[Tuple[str, str], int], int], v: str, acc: int) -> int:
        if v == self.vertex:
            return f((self.vertex, self.vertex), acc)
        return acc

    def edge_src(self, e: Tuple[str, str]) -> str:
        return e[0]


class SaturatingIntDomain:
    """A simple increasing domain on integers with saturation at 5."""

    def join(self, a: int, b: int) -> int:
        return max(a, b)

    def widening(self, a: int, b: int) -> int:
        # For this bounded domain, widening can be the same as join
        return max(a, b)

    def equal(self, a: int, b: int) -> bool:
        return a == b

    def transform(self, edge: Tuple[str, str], data: int) -> int:
        # Monotone step that saturates at 5
        return min(data + 1, 5)


class TestAnalyze(unittest.TestCase):
    def test_self_loop_saturating(self) -> None:
        graph = SelfLoopGraph('X')
        domain = SaturatingIntDomain()

        def init(_: str) -> int:
            return 0

        annot = analyze(graph, init, domain, delay=0)
        self.assertEqual(annot('X'), 5)


if __name__ == '__main__':
    unittest.main()
