import weakref
from abc import ABCMeta
from contextlib import contextmanager, suppress
from typing import Any, Hashable, Iterator, List, Optional, Set, Union

_global_logic_variables: Set[Any] = set()
_glv = _global_logic_variables


class LVarType(ABCMeta):
    def __instancecheck__(self, o: Any) -> bool:
        with suppress(TypeError):
            return issubclass(type(o), (Var, LVarType)) or o in _glv


class Var(metaclass=LVarType):
    """A logic variable type.

    Fresh logic variables will unify with anything:

        >>> unify(var(), 1)
        {~_1: 1}
        >>> unify(var(), [2])
        {~_2: [2]}
        >>> unify(var(), var())
        {~_3: ~_4}

    """

    __slots__ = ("token", "__weakref__")
    _refs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
    _id: int = 1

    def __new__(cls, token: Optional[Hashable] = None, prefix: str = "") -> "Var":
        """Construct a new logic variable.

        Parameters
        ----------
        token: Hashable (optional)
            A unique identifier for the logic variable.
        prefix: str (optional)
            A prefix to use when token isn't specified and the internal count
            value is used.  Useful as a means of identifying
            "non-globally"-scoped logic variables from their `str`/`repr`
            output.
        """
        if token is None:
            token = f"{prefix}_{Var._id}"
            cls._id += 1

        obj = cls._refs.get(token, None)

        if obj is None:
            obj = object.__new__(cls)
            obj.token = token
            cls._refs[token] = obj

        return obj

    def __str__(self) -> str:
        return f"~{self.token}"

    __repr__ = __str__

    def __eq__(self, other: Any) -> Union[bool, type(NotImplemented)]:
        if type(self) == type(other):
            return self.token == other.token
        return NotImplemented

    def __hash__(self) -> int:
        return hash((type(self), self.token))


var = Var


def vars(n: int, **kwargs: Any) -> List[Var]:
    """Create n-many fresh logic variables."""
    return [var(**kwargs) for i in range(n)]


def isvar(o: Any) -> bool:
    return isinstance(o, Var)


@contextmanager
def variables(*variables: Any) -> Iterator[None]:
    """Create a context manager within which arbitrary objects can be logic variables.

    >>> with variables(1):
    ...     print(isvar(1))
    True

    >>> print(isvar(1))
    False

    Normal approach

    >>> from unification import unify
    >>> x = var('x')
    >>> unify(x, 1)
    {~x: 1}

    Context Manager approach
    >>> with variables('x'):
    ...     print(unify('x', 1))
    {'x': 1}
    """
    old_global_logic_variables = _global_logic_variables.copy()
    _global_logic_variables.update(set(variables))
    try:
        yield
    finally:
        _global_logic_variables.clear()
        _global_logic_variables.update(old_global_logic_variables)
