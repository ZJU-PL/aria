from toolz import first, groupby
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from aria.unification.core import reify, unify
from aria.unification.utils import _toposort, freeze
from aria.unification.variable import isvar, Var


class Dispatcher(object):
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.funcs: Dict[Tuple, Callable] = dict()
        self.ordering: List[Tuple] = []

    def add(self, signature: Tuple, func: Callable) -> None:
        self.funcs[freeze(signature)] = func
        self.ordering = ordering(self.funcs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        func, s = self.resolve(args)
        return func(*args, **kwargs)

    def resolve(self, args: Tuple) -> Tuple[Callable, Dict]:
        n = len(args)
        frozen_args = freeze(args)
        for signature in self.ordering:
            if len(signature) != n:
                continue
            s = unify(frozen_args, signature)
            if s is not False:
                result = self.funcs[signature]
                return result, s
        raise NotImplementedError(
            f"No match found. \nKnown matches: {self.ordering} \nInput: {args}"
        )

    def register(self, *signature: Any) -> Callable[[Callable], "Dispatcher"]:
        def _(func: Callable) -> "Dispatcher":
            self.add(signature, func)
            return self

        return _


class VarDispatcher(Dispatcher):
    """A dispatcher that calls functions with variable names.

    >>> d = VarDispatcher('d')
    >>> x = var('x')

    >>> @d.register('inc', x)
    ... def f(x):
    ...     return x + 1

    >>> @d.register('double', x)
    ... def f(x):
    ...     return x * 2

    >>> d('inc', 10)
    11

    >>> d('double', 10)
    20

    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        func, s = self.resolve(args)
        d = dict((k.token, v) for k, v in s.items())
        return func(**d)


global_namespace: Dict[str, Dispatcher] = dict()


def match(*signature: Any, **kwargs: Any) -> Callable[[Callable], Dispatcher]:
    namespace = kwargs.get("namespace", global_namespace)
    dispatcher = kwargs.get("Dispatcher", Dispatcher)

    def _(func: Callable) -> Dispatcher:
        name = func.__name__

        if name not in namespace:
            namespace[name] = dispatcher(name)
        d = namespace[name]

        d.add(signature, func)

        return d

    return _


def supercedes(a: Any, b: Any) -> bool:
    """Check if ``a`` is a more specific match than ``b``."""
    if isvar(b) and not isvar(a):
        return True
    s = unify(a, b)
    if s is False:
        return False
    s = dict((k, v) for k, v in s.items() if not isvar(k) or not isvar(v))
    if reify(a, s) == a:
        return True
    if reify(b, s) == b:
        return False


def edge(a: Any, b: Any, tie_breaker: Callable = hash) -> bool:
    """Check A before B.

    Tie broken by tie_breaker, defaults to ``hash``
    """
    if supercedes(a, b):
        if supercedes(b, a):
            return tie_breaker(a) > tie_breaker(b)
        else:
            return True
    return False


def ordering(signatures: List[Tuple]) -> List[Tuple]:
    """Check a sane ordering of signatures, first to last.

    Topological sort of edges as given by ``edge`` and ``supercedes``
    """
    signatures = list(map(tuple, signatures))
    edges = [(a, b) for a in signatures for b in signatures if edge(a, b)]
    edges = groupby(first, edges)
    for s in signatures:
        if s not in edges:
            edges[s] = []
    edges = dict((k, [b for a, b in v]) for k, v in edges.items())
    return _toposort(edges)
