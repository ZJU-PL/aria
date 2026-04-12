"""Pythonic builder API for the vendored pyDatalog runtime.

This layer keeps the existing engine and DSL intact, but offers a higher-level
workflow closer to Z3's Python APIs: declare relations, assert facts/rules, and
issue queries through explicit Python objects.
"""

import ast
from pathlib import Path
import re
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from . import Aggregate
from . import pyDatalog
from . import pyEngine
from . import pyParser
from . import util


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


class DatalogParseError(DatalogAPIError):
    """Raised when textual datalog source cannot be parsed safely."""

    def __init__(
        self,
        message: str,
        *,
        line: Optional[int] = None,
        column: Optional[int] = None,
        source_line: Optional[str] = None,
        source_name: str = "<string>",
    ):
        details = [message]
        location = []
        if line is not None:
            location.append("line %d" % line)
        if column is not None:
            location.append("column %d" % column)
        if source_name:
            location.append("in %s" % source_name)
        if location:
            details.append("(%s)" % ", ".join(location))
        if source_line:
            details.append("\n%s" % source_line)
            if column is not None and column > 0:
                details.append("\n%s^" % (" " * (column - 1)))
        super(DatalogParseError, self).__init__("".join(details))
        self.line = line
        self.column = column
        self.source_line = source_line
        self.source_name = source_name


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

    def __eq__(self, other) -> Goal:  # type: ignore[override]
        goal = self._inner == _unwrap(other)
        if isinstance(self._inner, pyParser.Function):
            return Assignment(goal)
        return Goal(goal)


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


class Assignment(Goal):
    """A function-style head assignment such as `f[X] == value`."""

    def __init__(self, inner):
        super(Assignment, self).__init__(inner)


class AggregateBuilder(object):
    """Namespace of aggregate constructors for the Pythonic API."""

    @staticmethod
    def count(value) -> Aggregate.Len:
        return Aggregate.Len(_unwrap(value))

    @staticmethod
    def sum(
        value,
        *,
        for_each: Union[object, Sequence[object]],
    ) -> Aggregate.Sum:
        return Aggregate.Sum(
            _unwrap(value),
            for_each=_unwrap_many(for_each),
        )

    @staticmethod
    def min(
        value,
        *,
        order_by: Union[object, Sequence[object]],
    ) -> Aggregate.Min:
        return Aggregate.Min(
            _unwrap(value),
            order_by=_unwrap_many(order_by),
        )

    @staticmethod
    def max(
        value,
        *,
        order_by: Union[object, Sequence[object]],
    ) -> Aggregate.Max:
        return Aggregate.Max(
            _unwrap(value),
            order_by=_unwrap_many(order_by),
        )

    @staticmethod
    def tuple(
        value,
        *,
        order_by: Union[object, Sequence[object]],
    ) -> Aggregate.Tuple:
        return Aggregate.Tuple(
            _unwrap(value),
            order_by=_unwrap_many(order_by),
        )

    @staticmethod
    def concat(
        value,
        *,
        order_by: Union[object, Sequence[object]],
        sep: str,
    ) -> Aggregate.Concat:
        return Aggregate.Concat(
            _unwrap(value),
            order_by=_unwrap_many(order_by),
            sep=sep,
        )

    @staticmethod
    def rank(
        *,
        group_by: Union[object, Sequence[object]] = (),
        order_by: Union[object, Sequence[object]],
    ) -> Aggregate.Rank:
        return Aggregate.Rank(
            None,
            group_by=_unwrap_many(group_by),
            order_by=_unwrap_many(order_by),
        )

    @staticmethod
    def running_sum(
        value,
        *,
        group_by: Union[object, Sequence[object]] = (),
        order_by: Union[object, Sequence[object]],
    ) -> Aggregate.Running_sum:
        return Aggregate.Running_sum(
            _unwrap(value),
            group_by=_unwrap_many(group_by),
            order_by=_unwrap_many(order_by),
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


class Function(object):
    """A declared function-style predicate."""

    def __init__(self, name: str, arity: int):
        if arity <= 0:
            raise ValueError("Function arity must be positive.")
        self.name = name
        self.arity = arity

    def __call__(self, *args) -> DatalogExpression:
        if len(args) != self.arity:
            raise ValueError(
                "Function %s expects %d arguments, got %d."
                % (self.name, self.arity, len(args))
            )
        return DatalogExpression(
            pyParser.Function(self.name, tuple(_unwrap(arg) for arg in args))
        )

    def __repr__(self) -> str:
        return "Function(%s/%d)" % (self.name, self.arity)


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


class FunctionRuleBuilder(object):
    """Fluent builder for `Program.define(...).when(...)`."""

    def __init__(self, program: "Program", head: Assignment):
        self.program = program
        self.head = head

    def when(self, *goals: Goal) -> Rule:
        if not goals:
            +self.head._inner
            return self.program._record_rule(Rule(self.head, Goal(True)))
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
        self._functions = {}
        self.agg = AggregateBuilder()
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

    def function(self, name: str, arity: int) -> Function:
        key = (name, arity)
        function = self._functions.get(key)
        if function is None:
            function = Function(name, arity)
            self._functions[key] = function
        return function

    def functions(self, *specs: Union[str, Tuple[str, int]]) -> Tuple[Function, ...]:
        result = []
        for spec in specs:
            if isinstance(spec, tuple):
                name, arity = spec
            else:
                name, arity = spec, 1
            result.append(self.function(name, arity))
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

    def facts(self, *atoms: Atom) -> Tuple[Atom, ...]:
        """Assert multiple facts in order."""

        return tuple(self.fact(atom) for atom in atoms)

    def retract(self, atom: Atom) -> Atom:
        atom = _require_atom(atom)
        -atom._inner
        return atom

    def retract_all(self, *atoms: Atom) -> Tuple[Atom, ...]:
        """Retract multiple facts in order."""

        return tuple(self.retract(atom) for atom in atoms)

    def rule(self, head: Atom) -> RuleBuilder:
        return RuleBuilder(self, _require_atom(head))

    def rule_of(self, head: Atom, *goals: Goal) -> Rule:
        """Define a rule in one call."""

        return self.rule(head).when(*goals)

    def define(self, head: Goal) -> FunctionRuleBuilder:
        return FunctionRuleBuilder(self, _require_assignment(head))

    def define_of(self, head: Goal, *goals: Goal) -> Rule:
        """Define a function-style rule in one call."""

        return self.define(head).when(*goals)

    def load(self, code: str) -> None:
        """Load pyDatalog rules/facts from a string into this Program."""

        source = _normalize_source(code)
        for name, arity in _collect_relation_specs(source, source_name="load"):
            self.relation(name, arity)
        try:
            pyDatalog.load(source)
        except util.DatalogError as exc:
            raise _translate_datalog_error(exc, source, source_name="load")
        except SyntaxError as exc:
            raise _translate_syntax_error(exc, source, source_name="load")

    def load_file(self, path: Union[str, Path]) -> Path:
        """Load pyDatalog rules/facts from a file into this Program."""

        source_path = Path(path)
        source = source_path.read_text(encoding="utf-8")
        normalized = _normalize_source(source)
        for name, arity in _collect_relation_specs(
            normalized, source_name=str(source_path)
        ):
            self.relation(name, arity)
        try:
            pyDatalog.load(normalized)
        except util.DatalogError as exc:
            raise _translate_datalog_error(
                exc,
                normalized,
                source_name=str(source_path),
            )
        except SyntaxError as exc:
            raise _translate_syntax_error(
                exc,
                normalized,
                source_name=str(source_path),
            )
        return source_path

    def ask(self, code: str) -> QueryResult:
        """Run a textual query and return a materialized QueryResult."""

        source = _normalize_source(code)
        try:
            raw = pyDatalog.ask(source)
        except util.DatalogError as exc:
            raise _translate_datalog_error(exc, source, source_name="ask")
        except SyntaxError as exc:
            raise _translate_syntax_error(exc, source, source_name="ask")
        if raw is None:
            raise UndefinedPredicateError(_first_query_predicate_name(source))
        variables = [Variable(name) for name in _collect_query_variables(source)]
        return QueryResult(raw.answers if raw is not True else True, variables)

    def exists(self, goal: Goal) -> bool:
        """Return whether a query has at least one answer."""

        return bool(self.query(goal))

    def rows(self, goal: Goal) -> List[Tuple[object, ...]]:
        """Return materialized rows for a goal."""

        return self.query(goal).rows()

    def scalar_rows(self, goal: Goal) -> List[object]:
        """Return flattened rows for a single-column goal."""

        return self.query(goal).scalar_rows()

    def first(self, goal: Goal) -> Optional[Tuple[object, ...]]:
        """Return the first row for a goal, if any."""

        return self.query(goal).first()

    def first_value(self, goal: Goal) -> Optional[object]:
        """Return the first scalar value for a goal, if any."""

        return self.query(goal).first_value()

    def one(self, goal: Goal) -> Tuple[object, ...]:
        """Return the unique row for a goal."""

        return self.query(goal).one()

    def one_value(self, goal: Goal) -> object:
        """Return the unique scalar value for a goal."""

        return self.query(goal).one_value()

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


def _require_assignment(goal: Goal) -> Assignment:
    if not isinstance(goal, Assignment):
        raise TypeError("Expected a function assignment, got %r." % (goal,))
    return goal


_IGNORED_CALL_NAMES = {
    "len",
    "len_",
    "sum",
    "sum_",
    "min",
    "min_",
    "max",
    "max_",
    "tuple_",
    "concat_",
    "rank_",
    "running_sum_",
    "range",
    "range_",
    "format_",
    "mean_",
    "linear_regression_",
}


def _collect_relation_specs(
    code: str,
    source_name: str = "load",
) -> List[Tuple[str, int]]:
    try:
        tree = ast.parse(code, source_name, "exec")
    except SyntaxError as exc:
        raise _translate_syntax_error(exc, code, source_name=source_name)
    collector = _RelationSpecCollector()
    collector.visit(tree)
    return sorted(collector.relations.items())


class _RelationSpecCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.relations = {}

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name and name not in _IGNORED_CALL_NAMES:
            arity = len(node.args) + len(node.keywords)
            previous = self.relations.get(name)
            if previous is None or arity > previous:
                self.relations[name] = arity
        self.generic_visit(node)


def _call_name(node) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        if parent is None:
            return None
        return "%s.%s" % (parent, node.attr)
    return None


def _normalize_source(code: str) -> str:
    lines = code.splitlines()
    pattern = re.compile(r"^\s*")
    spaces = ""
    for line in lines:
        spaces = pattern.match(line).group()
        if spaces and line != spaces:
            break
    if not spaces:
        normalized = "\n".join(lines)
    else:
        normalized = "\n".join(re.sub("^" + spaces, "", line) for line in lines)
    return normalized.lstrip("\n")


def _collect_query_variables(code: str) -> List[str]:
    try:
        tree = ast.parse(code, "datalog_query", "eval")
    except SyntaxError as exc:
        raise _translate_syntax_error(exc, code, source_name="datalog_query")
    collector = _QueryVariableCollector()
    collector.visit(tree.body)
    return collector.variables


def _first_query_predicate_name(code: str) -> str:
    try:
        tree = ast.parse(code, "datalog_query", "eval")
    except SyntaxError:
        return "<unknown>"
    predicate = _find_first_call_name(tree.body)
    return predicate or "<unknown>"


class _QueryVariableCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.variables = []
        self._seen = set()

    def visit_Name(self, node: ast.Name) -> None:
        if node.id[:1].isupper() or node.id.startswith("_"):
            if node.id not in self._seen:
                self._seen.add(node.id)
                self.variables.append(node.id)


def _find_first_call_name(node) -> Optional[str]:
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    for child in ast.iter_child_nodes(node):
        name = _find_first_call_name(child)
        if name is not None:
            return name
    return None


def _unwrap_many(values: Union[object, Sequence[object]]) -> Tuple[object, ...]:
    if isinstance(values, (list, tuple)):
        return tuple(_unwrap(value) for value in values)
    return (_unwrap(values),)


def _translate_datalog_error(
    exc: util.DatalogError,
    source: str,
    *,
    source_name: str,
) -> DatalogParseError:
    line = getattr(exc, "lineno", None)
    source_line = _source_line(source, line)
    return DatalogParseError(
        str(getattr(exc, "message", None) or exc.value),
        line=line,
        column=1 if line is not None else None,
        source_line=source_line,
        source_name=source_name,
    )


def _translate_syntax_error(
    exc: SyntaxError,
    source: str,
    *,
    source_name: str,
) -> DatalogParseError:
    return DatalogParseError(
        exc.msg,
        line=exc.lineno,
        column=exc.offset,
        source_line=_source_line(source, exc.lineno),
        source_name=source_name,
    )


def _source_line(source: str, line: Optional[int]) -> Optional[str]:
    if line is None or line <= 0:
        return None
    lines = source.splitlines()
    if line > len(lines):
        return None
    return lines[line - 1]
