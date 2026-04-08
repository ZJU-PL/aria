"""Pythonic builder API for the vendored pyDatalog runtime.

This layer keeps the existing engine and DSL intact, but offers a higher-level
workflow closer to Z3's Python APIs: declare relations, assert facts/rules, and
issue queries through explicit Python objects.
"""

from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from . import pyDatalog
from . import pyEngine
from . import pyParser


def _unwrap(value):
    if isinstance(value, DatalogExpression):
        return value._inner
    if isinstance(value, Goal):
        return value._inner
    return value


class DatalogAPIError(Exception):
    """Base exception for the Pythonic datalog API."""


class UndefinedPredicateError(DatalogAPIError):
    """Raised when a query targets a predicate with no registered definition."""

    def __init__(self, predicate_name: str):
        super(UndefinedPredicateError, self).__init__(
            "Predicate %r is not registered in the current Program. "
            "Declare it with Program.relation(...) before querying." % predicate_name
        )
        self.predicate_name = predicate_name


class Goal(object):
    """A queryable boolean goal."""

    def __init__(self, inner):
        self._inner = inner

    def __and__(self, other: "Goal") -> "Goal":
        return Goal(self._inner & _require_goal(other)._inner)

    def __invert__(self) -> "Goal":
        return Goal(~self._inner)

    def ask(self):
        return self._inner.ask()

    def variables(self) -> List["Variable"]:
        variables = []
        if hasattr(self._inner, "_variables"):
            for term in self._inner._variables().values():
                variables.append(Variable(term._pyD_name, term))
        return variables

    def predicate_name(self) -> Optional[str]:
        return getattr(self._inner, "predicate_name", None)


class DatalogExpression(object):
    """Base wrapper for pyDatalog expressions."""

    def __init__(self, inner):
        self._inner = inner

    def __eq__(self, other) -> Goal:  # type: ignore[override]
        return Goal(self._inner == _unwrap(other))

    def __ne__(self, other) -> Goal:  # type: ignore[override]
        return Goal(self._inner != _unwrap(other))

    def __lt__(self, other) -> Goal:
        return Goal(self._inner < _unwrap(other))

    def __le__(self, other) -> Goal:
        return Goal(self._inner <= _unwrap(other))

    def __gt__(self, other) -> Goal:
        return Goal(self._inner > _unwrap(other))

    def __ge__(self, other) -> Goal:
        return Goal(self._inner >= _unwrap(other))

    def __add__(self, other):
        return DatalogExpression(self._inner + _unwrap(other))

    def __radd__(self, other):
        return DatalogExpression(_unwrap(other) + self._inner)

    def __sub__(self, other):
        return DatalogExpression(self._inner - _unwrap(other))

    def __rsub__(self, other):
        return DatalogExpression(_unwrap(other) - self._inner)

    def __mul__(self, other):
        return DatalogExpression(self._inner * _unwrap(other))

    def __rmul__(self, other):
        return DatalogExpression(_unwrap(other) * self._inner)

    def __floordiv__(self, other):
        return DatalogExpression(self._inner // _unwrap(other))

    def __truediv__(self, other):
        return DatalogExpression(self._inner / _unwrap(other))

    def __getitem__(self, item):
        return DatalogExpression(self._inner[_unwrap(item)])

    def in_(self, values) -> Goal:
        return Goal(self._inner.in_(values))

    def not_in_(self, values) -> Goal:
        return Goal(self._inner.not_in_(values))

    def __str__(self) -> str:
        return str(self._inner)


class Variable(DatalogExpression):
    """A named logic variable."""

    def __init__(self, name: str, inner=None):
        self.name = name
        super(Variable, self).__init__(inner or pyParser.Term(name))

    def values(self) -> List[object]:
        data = getattr(self._inner, "_data", [])
        if data is True:
            return []
        return list(data)

    def __repr__(self) -> str:
        return "Var(%s)" % self.name


class Atom(Goal):
    """A relation application."""

    def __init__(self, relation: "Relation", args: Sequence[object], inner):
        super(Atom, self).__init__(inner)
        self.relation = relation
        self.args = tuple(args)

    def __repr__(self) -> str:
        return "%s(%s)" % (
            self.relation.name,
            ", ".join(repr(arg) for arg in self.args),
        )


class Relation(object):
    """A declared predicate/relation."""

    def __init__(self, name: str, arity: int):
        if arity <= 0:
            raise ValueError("Relation arity must be positive.")
        self.name = name
        self.arity = arity

    def __call__(self, *args) -> Atom:
        if len(args) != self.arity:
            raise ValueError(
                "Relation %s expects %d arguments, got %d."
                % (self.name, self.arity, len(args))
            )
        inner = pyParser.Literal.make(
            self.name, tuple(_unwrap(arg) for arg in args), {}
        )
        return Atom(self, args, inner)

    def __repr__(self) -> str:
        return "Relation(%s/%d)" % (self.name, self.arity)


class Rule(object):
    """A declared rule handle."""

    def __init__(self, head: Atom, body: Goal):
        self.head = head
        self.body = body

    def __repr__(self) -> str:
        return "Rule(%r <= %r)" % (self.head, self.body)


class RuleBuilder(object):
    """Fluent builder for `Program.rule(...).when(...)`."""

    def __init__(self, program: "Program", head: Atom):
        self.program = program
        self.head = head

    def when(self, *goals: Goal) -> Rule:
        if not goals:
            raise ValueError("A rule body cannot be empty.")
        body = _require_goal(goals[0])
        for goal in goals[1:]:
            body = body & _require_goal(goal)
        self.head._inner <= body._inner
        return self.program._record_rule(Rule(self.head, body))


class QueryResult(object):
    """Materialized query result."""

    def __init__(self, raw_result, variables: Sequence[Variable]):
        self.raw_result = raw_result
        self.variables = tuple(variables)

    def succeeded(self) -> bool:
        return self.raw_result is True or bool(self.raw_result)

    def rows(self) -> List[Tuple[object, ...]]:
        if self.raw_result is True or self.raw_result is None:
            return []
        return [tuple(row) for row in self.raw_result]

    def scalar_rows(self) -> List[object]:
        return [row[0] for row in self.rows()]

    def named_rows(self) -> List[Dict[str, object]]:
        names = [variable.name for variable in self.variables]
        return [dict(zip(names, row)) for row in self.rows()]

    def first(self) -> Optional[Tuple[object, ...]]:
        rows = self.rows()
        return rows[0] if rows else None

    def first_value(self) -> Optional[object]:
        row = self.first()
        return row[0] if row else None

    def one(self) -> Tuple[object, ...]:
        rows = self.rows()
        if len(rows) != 1:
            raise DatalogAPIError(
                "Expected exactly one result row, got %d." % len(rows)
            )
        return rows[0]

    def one_value(self) -> object:
        row = self.one()
        if len(row) != 1:
            raise DatalogAPIError(
                "Expected exactly one column, got %d." % len(row)
            )
        return row[0]

    def __bool__(self) -> bool:
        return self.succeeded()

    def __iter__(self) -> Iterator[Tuple[object, ...]]:
        return iter(self.rows())

    def __len__(self) -> int:
        return len(self.rows())

    def __repr__(self) -> str:
        return "QueryResult(%r)" % (
            self.rows() if self.raw_result is not True else True
        )


class Program(object):
    """A small fixedpoint-like wrapper over the vendored pyDatalog engine.

    The engine remains thread-global in practice, so multiple `Program`
    instances share state inside the same caller thread. Use one `Program` per
    thread, or a dedicated worker thread/process per program if true isolation
    is required.
    """

    def __init__(self, reset: bool = True):
        self._relations = {}
        if reset:
            self.clear()

    def clear(self) -> None:
        pyDatalog.clear()
        for relation in self._relations.values():
            self._register_relation(relation)

    def relation(self, name: str, arity: int) -> Relation:
        key = (name, arity)
        relation = self._relations.get(key)
        if relation is None:
            relation = Relation(name, arity)
            self._relations[key] = relation
            self._register_relation(relation)
        return relation

    def relations(self, *specs: Union[str, Tuple[str, int]]) -> Tuple[Relation, ...]:
        result = []
        for spec in specs:
            if isinstance(spec, tuple):
                name, arity = spec
            else:
                name, arity = spec, 2
            result.append(self.relation(name, arity))
        return tuple(result)

    def var(self, name: str) -> Variable:
        return Variable(name)

    def vars(self, names: Union[str, Iterable[str]]) -> Tuple[Variable, ...]:
        if isinstance(names, str):
            tokens = names.replace(",", " ").split()
        else:
            tokens = list(names)
        return tuple(self.var(name) for name in tokens)

    def fact(self, atom: Atom) -> Atom:
        atom = _require_atom(atom)
        +atom._inner
        return atom

    def retract(self, atom: Atom) -> Atom:
        atom = _require_atom(atom)
        -atom._inner
        return atom

    def rule(self, head: Atom) -> RuleBuilder:
        return RuleBuilder(self, _require_atom(head))

    def query(self, goal: Goal) -> QueryResult:
        goal = _require_goal(goal)
        try:
            raw = goal.ask()
        except AttributeError as exc:
            if "Predicate without definition" not in str(exc):
                raise
            predicate_name = goal.predicate_name() or "<unknown>"
            raise UndefinedPredicateError(predicate_name)
        return QueryResult(raw, goal.variables())

    def _record_rule(self, rule: Rule) -> Rule:
        return rule

    def _register_relation(self, relation: Relation) -> None:
        pyEngine.insert(pyEngine.Pred(relation.name, relation.arity))


def vars_(names: Union[str, Iterable[str]]) -> Tuple[Variable, ...]:
    """Convenience constructor that mirrors `Program.vars(...)`."""

    if isinstance(names, str):
        tokens = names.replace(",", " ").split()
    else:
        tokens = list(names)
    return tuple(Variable(name) for name in tokens)


def _require_goal(goal: Goal) -> Goal:
    if not isinstance(goal, Goal):
        raise TypeError("Expected a Goal, got %r." % (goal,))
    return goal


def _require_atom(atom: Atom) -> Atom:
    if not isinstance(atom, Atom):
        raise TypeError("Expected an Atom, got %r." % (atom,))
    return atom
