"""Unification engine."""

from collections import OrderedDict, deque
from collections.abc import Generator, Iterator, Mapping, Set
from copy import copy
from functools import partial
from operator import length_hint
from typing import Any, Callable, Optional, Set as TypeSet, Union

from aria.unification.dispatch import dispatch
from aria.unification.utils import transitive_get as walk
from aria.unification.variable import Var, isvar

# An object used to tell the reifier that the next yield constructs the reified
# object from its constituent refications (if any).
construction_sentinel = object()


@dispatch(Mapping, object, object)
def assoc(s: Mapping, u: Any, v: Any) -> Mapping:
    """Add an entry to a `Mapping` and return it."""
    if hasattr(s, "copy"):
        s = s.copy()
    else:
        s = copy(s)  # pragma: no cover
    s[u] = v
    return s


def stream_eval(z: Union[Generator, Any], res_filter: Optional[Callable] = None) -> Any:
    r"""Evaluate a stream of `_reify`/`_unify` results.

    This implementation consists of a deque that simulates an evaluation stack
    of `_reify`/`_unify`-produced generators.  We're able to overcome
    `RecursionError`\s this way.
    """

    if not isinstance(z, Generator):
        return z

    stack: deque = deque()
    z_args: Any = None
    z_out: Any = None
    stack.append(z)

    while stack:
        z = stack[-1]
        try:
            z_out = z.send(z_args)

            if res_filter:
                _ = res_filter(z, z_out)

            if isinstance(z_out, Generator):
                stack.append(z_out)
                z_args = None
            else:
                z_args = z_out

        except StopIteration:
            _ = stack.pop()

    return z_out


class UngroundLVarException(Exception):
    """An exception signaling that an unground variable was found."""


@dispatch(object, Mapping)
def _reify(o: Any, s: Mapping) -> Any:  # noqa: ARG001
    """Reify a non-variable object."""
    return o


@_reify.register(Var, Mapping)
def _reify_var(o: Var, s: Mapping) -> Generator[Any, None, Any]:
    """Reify a variable."""
    o_w = walk(o, s)

    if o_w is o:
        yield o_w
    else:
        yield _reify(o_w, s)


def _reify_iterable_ctor(ctor: Callable, t: Any, s: Mapping) -> Generator[Any, None, Any]:
    """Create a generator that yields `_reify` generators.

    The yielded generators need to be evaluated by the caller and the fully
    reified results "sent" back to this generator so that it can finish
    constructing reified iterable.

    This approach allows us "collapse" nested `_reify` calls by pushing nested
    calls up the stack.
    """
    res: list = []

    if isinstance(t, Mapping):
        t = t.items()

    for y in t:
        r = _reify(y, s)
        if isinstance(r, Generator):
            r = yield r
        res.append(r)

    yield construction_sentinel

    yield ctor(res)


for seq, ctor in (
    (tuple, tuple),
    (list, list),
    (Iterator, iter),
    (set, set),
    (frozenset, frozenset),
):
    _reify.add((seq, Mapping), partial(_reify_iterable_ctor, ctor))


for seq in (dict, OrderedDict):
    _reify.add((seq, Mapping), partial(_reify_iterable_ctor, seq))


@_reify.register(slice, Mapping)
def _reify_slice(o: slice, s: Mapping) -> Generator[slice, None, slice]:
    start = yield _reify(o.start, s)
    stop = yield _reify(o.stop, s)
    step = yield _reify(o.step, s)

    yield construction_sentinel

    yield slice(start, stop, step)


@dispatch(object, Mapping)
def reify(e: Any, s: Mapping) -> Any:
    """Replace logic variables in a term, `e`, with their substitutions in `s`.

    >>> x, y = var(), var()
    >>> e = (1, x, (3, y))
    >>> s = {x: 2, y: 4}
    >>> reify(e, s)
    (1, 2, (3, 4))

    >>> e = {1: x, 3: (y, 5)}
    >>> reify(e, s)
    {1: 2, 3: (4, 5)}
    """

    if len(s) == 0:
        return e

    return stream_eval(_reify(e, s))


@dispatch(object, object, Mapping)
def _unify(u: Any, v: Any, s: Mapping) -> Union[Mapping, bool]:
    return s if u == v else False


@_unify.register(Var, (Var, object), Mapping)
def _unify_var_object(
    u: Var, v: Union[Var, Any], s: Mapping
) -> Generator[Union[Mapping, bool], None, None]:
    """Unify a variable with another variable or object."""
    u_w = walk(u, s)

    if isvar(v):
        v_w = walk(v, s)
    else:
        v_w = v

    if u_w == v_w:
        yield s
    elif isvar(u_w):
        yield assoc(s, u_w, v_w)
    elif isvar(v_w):
        yield assoc(s, v_w, u_w)
    else:
        yield _unify(u_w, v_w, s)


_unify.add((object, Var, Mapping), _unify_var_object)


def _unify_iterable(u: Any, v: Any, s: Mapping) -> Generator[Union[Mapping, bool], None, None]:
    """Unify two iterable objects."""
    len_u = length_hint(u, -1)
    len_v = length_hint(v, -1)

    if len_u != len_v:
        yield False
        return

    for uu, vv in zip(u, v):
        s = yield _unify(uu, vv, s)
        if s is False:
            return
    yield s


for seq in (tuple, list, Iterator):
    _unify.add((seq, seq, Mapping), _unify_iterable)


@_unify.register(Set, Set, Mapping)
def _unify_set(u: Set, v: Set, s: Mapping) -> Generator[Union[Mapping, bool], None, None]:
    """Unify two sets."""
    i = u & v
    u = u - i
    v = v - i
    yield _unify(iter(u), iter(v), s)


@_unify.register(Mapping, Mapping, Mapping)
def _unify_mapping(
    u: Mapping, v: Mapping, s: Mapping
) -> Generator[Union[Mapping, bool], None, None]:
    """Unify two mapping objects."""
    if len(u) != len(v):
        yield False
        return

    for key, uval in u.items():
        if key not in v:
            yield False
            return

        s = yield _unify(uval, v[key], s)

        if s is False:
            return
    yield s


@_unify.register(slice, slice, Mapping)
def _unify_slice(u: slice, v: slice, s: Mapping) -> Generator[Union[Mapping, bool], None, None]:
    s = yield _unify(u.start, v.start, s)
    if s is False:
        return
    s = yield _unify(u.stop, v.stop, s)
    if s is False:
        return
    s = yield _unify(u.step, v.step, s)


@dispatch(object, object, Mapping)
def unify(u: Any, v: Any, s: Mapping) -> Union[Mapping, bool]:
    """Find substitution so that ``u == v`` while satisfying `s`.

    >>> x = var('x')
    >>> unify((1, x), (1, 2), {})
    {~x: 2}
    """
    if u is v:
        return s

    return stream_eval(_unify(u, v, s))


@unify.register(object, object)
def unify_no_map(u: Any, v: Any) -> Union[Mapping, bool]:
    """Unify two objects without an initial mapping."""
    return unify(u, v, {})


def unground_lvars(u: Any, s: Mapping) -> TypeSet[Var]:
    """Return the unground logic variables from a term and state."""

    lvars: TypeSet[Var] = set()

    def lvar_filter(z: Generator, r: Any) -> None:
        nonlocal lvars

        if isvar(r):
            lvars.add(r)

        if r is construction_sentinel:
            z.close()

            # Remove this generator from the stack.
            raise StopIteration()

    z = _reify(u, s)
    stream_eval(z, lvar_filter)

    return lvars


def isground(u: Any, s: Mapping) -> bool:
    """Determine whether or not `u` contains an unground logic variable under mappings `s`."""

    def lvar_filter(z: Generator, r: Any) -> None:

        if isvar(r):
            raise UngroundLVarException()
        if r is construction_sentinel:
            z.close()

            # Remove this generator from the stack.
            raise StopIteration()

    try:
        z = _reify(u, s)
        stream_eval(z, lvar_filter)
    except UngroundLVarException:
        return False

    return True


def debug_unify(u: Any, v: Any, s: Mapping) -> Union[Mapping, bool]:  # pragma: no cover
    """Stop in the debugger when unify fails.

    You can inspect the generator-based stack by looking through the
    generator frames in the `stack` variable in `stream_eval`:

        (Pdb) up
        > .../unification/unification/core.py(39)stream_eval()
        -> _ = res_filter(z, z_out)
        (Pdb) stack[-2].gi_frame.f_locals
        {'u': <set_iterator at 0x7f5ee32414c8>,
        'v': <set_iterator at 0x7f5ee3241510>,
        's': {},
        'len_u': 2,
        'len_v': 2,
        'uu': ('debit', ~amount),
        'vv': ('name', 'Bob')}
    """

    def _filter(z: Generator, r: Any) -> None:  # noqa: ARG001
        if r is False:
            import pdb  # noqa: T100

            pdb.set_trace()  # noqa: T100

    z = _unify(u, v, s)
    return stream_eval(z, _filter)
