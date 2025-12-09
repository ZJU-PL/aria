"""
Fixpoint analysis (Bourdoncle-style chaotic iteration with widenings),
mirroring src/fixpoint.ml.
"""

from __future__ import annotations
from typing import Protocol, TypeVar, Callable, Any, Dict, List, Union

from .loop import compute_loop_nesting_forest, Loop as SRKLoop

V = TypeVar('V')
E = TypeVar('E')
T = TypeVar('T')


class GraphProtocol(Protocol[V, E]):
    """Graph interface required by the analyzer.

    Must support iteration over vertices/successors (for loop-nest construction)
    and folding over predecessor edges (for dataflow propagation).
    """

    def iter_vertex(self, f: Callable[[V], None]) -> None: ...
    def iter_succ(self, f: Callable[[V], None], v: V) -> None: ...

    def fold_pred_edges(self, f: Callable[[E, T], T], v: V, acc: T) -> T: ...
    def edge_src(self, e: E) -> V: ...


class DomainProtocol(Protocol[T, E]):
    """Abstract domain interface matching src/fixpoint.ml's Domain signature."""

    def join(self, a: T, b: T) -> T: ...
    def widening(self, a: T, b: T) -> T: ...
    def equal(self, a: T, b: T) -> bool: ...
    def transform(self, edge: E, data: T) -> T: ...


def analyze(
    graph: GraphProtocol[V, E],
    init: Callable[[V], T],
    domain: DomainProtocol[T, E],
    delay: int = 0,
) -> Callable[[V], T]:
    """Compute a fixpoint over the given graph using Bourdoncle's strategy.

    Returns a function mapping vertices to their final annotations.
    """
    loop_forest: List[Union[V, SRKLoop]] = compute_loop_nesting_forest(graph)
    annotation: Dict[V, T] = {}

    def get_annotation(vertex: V) -> T:
        try:
            return annotation[vertex]
        except KeyError:
            return init(vertex)

    def set_annotation(vertex: V, data: T) -> None:
        annotation[vertex] = data

    def next_annotation(v: V) -> T:
        def folder(e: E, flow_in: T) -> T:
            return domain.join(
                flow_in,
                domain.transform(e, get_annotation(graph.edge_src(e)))
            )

        return graph.fold_pred_edges(folder, v, get_annotation(v))

    def fix(item: Union[V, SRKLoop]) -> None:
        if isinstance(item, SRKLoop):
            header: V = item.header

            def fix_loop(local_delay: int) -> None:
                old = get_annotation(header)
                nxt = next_annotation(header)
                new = nxt if local_delay > 0 else domain.widening(old, nxt)
                set_annotation(header, new)
                for child in item.children:
                    fix(child)
                if not domain.equal(old, new):
                    fix_loop(local_delay - 1)

            fix_loop(delay)
        else:
            v: V = item
            set_annotation(v, next_annotation(v))

    for elem in loop_forest:
        fix(elem)

    return get_annotation
