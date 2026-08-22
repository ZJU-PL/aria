"""Semantics-preserving source instrumentation for native concolic tracing."""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .runtime import RUNTIME_HOOKS


class InstrumentationError(RuntimeError):
    """Raised when a callable cannot be safely instrumented."""


_BINARY_OPERATORS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
}
_UNARY_OPERATORS = {ast.Not: "not", ast.UAdd: "+", ast.USub: "-", ast.Invert: "~"}
_COMPARISON_OPERATORS = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.In: "in",
    ast.NotIn: "not in",
    ast.Is: "is",
    ast.IsNot: "is not",
}


def expression_descriptor(node: ast.AST) -> Tuple[Any, ...]:
    """Convert an AST expression to a data-only symbolic descriptor."""
    if isinstance(node, ast.Constant):
        return ("const", node.value)
    if isinstance(node, ast.Name):
        return ("name", node.id)
    if isinstance(node, ast.Attribute):
        fallback = ("attribute", expression_descriptor(node.value), node.attr)
        attribute_id = getattr(node, "_aria_attribute_id", None)
        if attribute_id is not None:
            return ("attribute_result", attribute_id, fallback)
        return fallback
    if isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPERATORS.get(type(node.op))
        if operation is not None:
            return ("unary", operation, expression_descriptor(node.operand))
    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS.get(type(node.op))
        if operation is not None:
            return (
                "binary",
                operation,
                expression_descriptor(node.left),
                expression_descriptor(node.right),
            )
    if isinstance(node, ast.BoolOp):
        operation = "and" if isinstance(node.op, ast.And) else "or"
        return (
            "bool",
            operation,
            tuple(expression_descriptor(value) for value in node.values),
        )
    if isinstance(node, ast.Compare):
        operations = tuple(
            _COMPARISON_OPERATORS.get(type(op), type(op).__name__) for op in node.ops
        )
        return (
            "compare",
            expression_descriptor(node.left),
            operations,
            tuple(expression_descriptor(value) for value in node.comparators),
        )
    if isinstance(node, ast.IfExp):
        return (
            "ifexp",
            expression_descriptor(node.test),
            expression_descriptor(node.body),
            expression_descriptor(node.orelse),
        )
    if isinstance(node, ast.Tuple):
        return ("tuple", tuple(expression_descriptor(item) for item in node.elts))
    if isinstance(node, ast.List):
        return ("list", tuple(expression_descriptor(item) for item in node.elts))
    if isinstance(node, ast.Dict) and all(key is not None for key in node.keys):
        return (
            "dict",
            tuple(
                (expression_descriptor(key), expression_descriptor(value))
                for key, value in zip(node.keys, node.values)
            ),
        )
    if isinstance(node, ast.Call):
        fallback: Tuple[Any, ...]
        if isinstance(node.func, ast.Name) and not node.keywords:
            fallback = (
                "call",
                node.func.id,
                tuple(expression_descriptor(arg) for arg in node.args),
            )
        elif isinstance(node.func, ast.Attribute) and not node.keywords:
            fallback = (
                "method",
                expression_descriptor(node.func.value),
                node.func.attr,
                tuple(expression_descriptor(arg) for arg in node.args),
            )
        else:
            fallback = (
                "opaque",
                "function call with dynamic target or keyword arguments",
            )
        call_id = getattr(node, "_aria_call_id", None)
        if call_id is not None:
            return ("call_result", call_id, fallback)
        return fallback
    if isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Slice):
            return (
                "slice",
                expression_descriptor(node.value),
                expression_descriptor(node.slice.lower) if node.slice.lower else None,
                expression_descriptor(node.slice.upper) if node.slice.upper else None,
                expression_descriptor(node.slice.step) if node.slice.step else None,
            )
        return (
            "subscript",
            expression_descriptor(node.value),
            expression_descriptor(node.slice),
        )
    return ("opaque", f"expression node {type(node).__name__}")


def _literal(value: Any) -> ast.expr:
    if isinstance(value, tuple):
        return ast.Tuple(elts=[_literal(item) for item in value], ctx=ast.Load())
    if isinstance(value, list):
        return ast.List(elts=[_literal(item) for item in value], ctx=ast.Load())
    if isinstance(value, dict):
        return ast.Dict(
            keys=[_literal(key) for key in value],
            values=[_literal(item) for item in value.values()],
        )
    return ast.Constant(value=value)


@dataclass(frozen=True)
class InstrumentedCallable:
    """A transformed callable and its static instrumentation metadata."""

    original: Any
    callable: Any
    filename: str
    function_id: str
    branch_locations: Dict[str, Tuple[Any, ...]]
    namespace: Dict[str, Any]
    global_names: Tuple[str, ...]
    closure_cells: Dict[str, Any]

    def refresh_environment(self) -> None:
        """Refresh mutable globals and closure reads before concrete execution."""
        target = (
            self.original.__func__ if inspect.ismethod(self.original) else self.original
        )
        for name in self.global_names:
            if name in target.__globals__:
                self.namespace[name] = target.__globals__[name]
        for name, cell in self.closure_cells.items():
            try:
                self.namespace[name] = cell.cell_contents
            except ValueError:
                self.namespace.pop(name, None)


class _FunctionTransformer(ast.NodeTransformer):
    def __init__(
        self,
        filename: str,
        function_id: str,
        source_line_offset: int,
        runtime_name: str,
        environment_names: Sequence[str],
    ) -> None:
        self.filename = filename
        self.function_id = function_id
        self.source_line_offset = source_line_offset
        self.runtime_name = runtime_name
        self.environment_names = tuple(environment_names)
        self.branch_locations: Dict[str, Tuple[Any, ...]] = {}
        self._function_depth = 0
        self._temp_counter = 0
        self._target_name: Optional[str] = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._visit_function(node)

    def _visit_function(self, node: Any) -> ast.AST:
        if self._function_depth:
            return node
        self._function_depth += 1
        self._target_name = node.name
        self._annotate_call_ids(node)
        node.decorator_list = []
        self._strip_annotations(node)
        node.body = [self.visit(statement) for statement in node.body]
        node.body = _flatten_statements(node.body)
        parameter_specs = _parameter_specs(node.args)
        parameter_names = [name for name, _ in parameter_specs]
        environment_names = [
            name for name in self.environment_names if name not in parameter_names
        ]
        shadow_names = [*parameter_names, *environment_names]
        enter_call = ast.Call(
            func=ast.Attribute(
                ast.Name(self.runtime_name, ast.Load()), "enter", ast.Load()
            ),
            args=[
                ast.Constant(self.function_id),
                ast.Dict(
                    keys=[ast.Constant(name) for name in shadow_names],
                    values=[ast.Name(name, ast.Load()) for name in shadow_names],
                ),
                _literal(parameter_specs),
            ],
            keywords=[],
        )
        marker_name = (
            f"__aria_frame_{hashlib.sha1(self.function_id.encode()).hexdigest()[:10]}"
        )
        enter = ast.Assign([ast.Name(marker_name, ast.Store())], enter_call)
        leave = ast.Expr(
            ast.Call(
                func=ast.Attribute(
                    ast.Name(self.runtime_name, ast.Load()), "leave", ast.Load()
                ),
                args=[ast.Name(marker_name, ast.Load())],
                keywords=[],
            )
        )
        node.body = [
            enter,
            ast.Try(body=node.body, handlers=[], orelse=[], finalbody=[leave]),
        ]
        self._function_depth -= 1
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Bridge safe calls into instrumented callees and model fallbacks."""
        if isinstance(node.func, ast.Name) and node.func.id in {
            "super",
            "locals",
            "globals",
            "vars",
            "eval",
            "exec",
        }:
            return self.generic_visit(node)
        positional_descriptors = tuple(
            (
                ("star", expression_descriptor(argument.value))
                if isinstance(argument, ast.Starred)
                else ("arg", expression_descriptor(argument))
            )
            for argument in node.args
        )
        keyword_descriptors = tuple(
            (keyword.arg, expression_descriptor(keyword.value))
            for keyword in node.keywords
        )
        call_id = getattr(node, "_aria_call_id")
        return ast.copy_location(
            ast.Call(
                ast.Attribute(
                    ast.Name(self.runtime_name, ast.Load()),
                    "interprocedural_call",
                    ast.Load(),
                ),
                [
                    ast.Constant(call_id),
                    self.visit(node.func),
                    ast.Tuple(
                        [self.visit(argument) for argument in node.args], ast.Load()
                    ),
                    ast.Dict(
                        [
                            (
                                ast.Constant(keyword.arg)
                                if keyword.arg is not None
                                else None
                            )
                            for keyword in node.keywords
                        ],
                        [self.visit(keyword.value) for keyword in node.keywords],
                    ),
                    _literal(positional_descriptors),
                    _literal(keyword_descriptors),
                ],
                [],
            ),
            node,
        )

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if not isinstance(node.ctx, ast.Load):
            return self.generic_visit(node)
        attribute_id = getattr(node, "_aria_attribute_id", None)
        if attribute_id is None:
            return self.generic_visit(node)
        owner_descriptor = expression_descriptor(node.value)
        return ast.copy_location(
            ast.Call(
                ast.Attribute(
                    ast.Name(self.runtime_name, ast.Load()),
                    "getattr",
                    ast.Load(),
                ),
                [
                    ast.Constant(attribute_id),
                    self.visit(node.value),
                    ast.Constant(node.attr),
                    _literal(owner_descriptor),
                ],
                [],
            ),
            node,
        )

    def visit_Return(self, node: ast.Return) -> ast.AST:
        if node.value is None:
            return node
        descriptor = expression_descriptor(node.value)
        concrete_value = self.visit(node.value)
        node.value = ast.Call(
            ast.Attribute(
                ast.Name(self.runtime_name, ast.Load()),
                "returning",
                ast.Load(),
            ),
            [concrete_value, _literal(descriptor)],
            [],
        )
        return node

    def _annotate_call_ids(self, node: ast.AST) -> None:
        for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
            stable = (
                f"{self.function_id}:call:"
                f"{self.source_line_offset + call.lineno}:{call.col_offset}"
            )
            setattr(
                call,
                "_aria_call_id",
                hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16],
            )
        for attribute in (
            item
            for item in ast.walk(node)
            if isinstance(item, ast.Attribute) and isinstance(item.ctx, ast.Load)
        ):
            stable = (
                f"{self.function_id}:attribute:"
                f"{self.source_line_offset + attribute.lineno}:{attribute.col_offset}"
            )
            setattr(
                attribute,
                "_aria_attribute_id",
                hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16],
            )

    def _strip_annotations(self, node: Any) -> None:
        node.returns = None
        all_args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        if node.args.vararg:
            all_args.append(node.args.vararg)
        if node.args.kwarg:
            all_args.append(node.args.kwarg)
        for argument in all_args:
            argument.annotation = None
        node.args.defaults = [ast.Constant(None) for _ in node.args.defaults]
        node.args.kw_defaults = [
            None if default is None else ast.Constant(None)
            for default in node.args.kw_defaults
        ]

    def visit_If(self, node: ast.If) -> ast.AST:
        node.test = self._branch(node.test, "if")
        node.body = _flatten_statements(
            [self.visit(statement) for statement in node.body]
        )
        node.orelse = _flatten_statements(
            [self.visit(statement) for statement in node.orelse]
        )
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        node.test = self._branch(node.test, "while")
        node.body = _flatten_statements(
            [self.visit(statement) for statement in node.body]
        )
        node.orelse = _flatten_statements(
            [self.visit(statement) for statement in node.orelse]
        )
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        original_iter = node.iter
        node.body = _flatten_statements(
            [self.visit(statement) for statement in node.body]
        )
        node.orelse = _flatten_statements(
            [self.visit(statement) for statement in node.orelse]
        )
        if (
            not isinstance(node.target, ast.Name)
            or not isinstance(original_iter, ast.Call)
            or not isinstance(original_iter.func, ast.Name)
            or original_iter.func.id != "range"
            or original_iter.keywords
            or not 1 <= len(original_iter.args) <= 3
        ):
            node.iter = self.visit(original_iter)
            if not isinstance(node.target, ast.Name):
                return node
            location = (
                self.filename,
                self.source_line_offset + node.lineno,
                node.col_offset,
                self.source_line_offset + getattr(node, "end_lineno", node.lineno),
                getattr(node, "end_col_offset", node.col_offset),
            )
            stable = f"{self.function_id}:for-iter:{location[1]}:{location[2]}"
            branch_id = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]
            self.branch_locations[branch_id] = location
            node.iter = ast.Call(
                ast.Attribute(
                    ast.Name(self.runtime_name, ast.Load()),
                    "iterate",
                    ast.Load(),
                ),
                [
                    ast.Constant(branch_id),
                    node.iter,
                    _literal(expression_descriptor(original_iter)),
                    _literal(location),
                    ast.Constant(node.target.id),
                ],
                [],
            )
            return node
        location = (
            self.filename,
            self.source_line_offset + node.lineno,
            node.col_offset,
            self.source_line_offset + getattr(node, "end_lineno", node.lineno),
            getattr(node, "end_col_offset", node.col_offset),
        )
        stable = f"{self.function_id}:for-range:{location[1]}:{location[2]}"
        branch_id = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]
        self.branch_locations[branch_id] = location
        descriptors = tuple(
            expression_descriptor(argument) for argument in original_iter.args
        )
        concrete_range = self.visit(original_iter)
        node.iter = ast.Call(
            ast.Attribute(
                ast.Name(self.runtime_name, ast.Load()), "range_iter", ast.Load()
            ),
            [
                ast.Constant(branch_id),
                concrete_range,
                _literal(descriptors),
                _literal(location),
                ast.Constant(node.target.id),
            ],
            [],
        )
        return node

    def visit_Assert(self, node: ast.Assert) -> ast.AST:
        node.test = self._branch(node.test, "assert")
        if node.msg is not None:
            node.msg = self.visit(node.msg)
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        original_test = node.test
        node.test = self._branch(original_test, "ifexp")
        node.body = self.visit(node.body)
        node.orelse = self.visit(node.orelse)
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and all(isinstance(item, ast.Name) for item in node.targets[0].elts)
        ):
            names = tuple(item.id for item in node.targets[0].elts)
            descriptor = expression_descriptor(node.value)
            node.value = ast.Call(
                ast.Attribute(
                    ast.Name(self.runtime_name, ast.Load()),
                    "unpack_assign",
                    ast.Load(),
                ),
                [_literal(names), self.visit(node.value), _literal(descriptor)],
                [],
            )
            return node
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute):
            target = node.targets[0]
            value_descriptor = expression_descriptor(node.value)
            self._temp_counter += 1
            suffix = self._temp_counter
            value_name = f"__aria_setattr_value_{suffix}"
            owner_name = f"__aria_setattr_owner_{suffix}"
            statements = [
                ast.Assign(
                    [ast.Name(value_name, ast.Store())],
                    self.visit(node.value),
                ),
                ast.Assign(
                    [ast.Name(owner_name, ast.Store())],
                    self.visit(target.value),
                ),
                ast.Expr(
                    ast.Call(
                        ast.Attribute(
                            ast.Name(self.runtime_name, ast.Load()),
                            "setattr",
                            ast.Load(),
                        ),
                        [
                            ast.Name(owner_name, ast.Load()),
                            ast.Constant(target.attr),
                            ast.Name(value_name, ast.Load()),
                            _literal(value_descriptor),
                        ],
                        [],
                    )
                ),
            ]
            return [ast.copy_location(statement, node) for statement in statements]
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and not isinstance(node.targets[0].slice, ast.Slice)
        ):
            target = node.targets[0]
            value_descriptor = expression_descriptor(node.value)
            self._temp_counter += 1
            suffix = self._temp_counter
            value_name = f"__aria_setitem_value_{suffix}"
            owner_name = f"__aria_setitem_owner_{suffix}"
            key_name = f"__aria_setitem_key_{suffix}"
            statements = [
                ast.Assign(
                    [ast.Name(value_name, ast.Store())],
                    self.visit(node.value),
                ),
                ast.Assign(
                    [ast.Name(owner_name, ast.Store())],
                    self.visit(target.value),
                ),
                ast.Assign(
                    [ast.Name(key_name, ast.Store())],
                    self.visit(target.slice),
                ),
                ast.Expr(
                    ast.Call(
                        ast.Attribute(
                            ast.Name(self.runtime_name, ast.Load()),
                            "setitem",
                            ast.Load(),
                        ),
                        [
                            ast.Name(owner_name, ast.Load()),
                            ast.Name(key_name, ast.Load()),
                            ast.Name(value_name, ast.Load()),
                            _literal(value_descriptor),
                        ],
                        [],
                    )
                ),
            ]
            return [ast.copy_location(statement, node) for statement in statements]
        descriptor = expression_descriptor(node.value)
        node.value = self.visit(node.value)
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if len(names) != len(node.targets):
            return node
        method = "assign" if len(names) == 1 else "assign_many"
        first_arg: ast.expr = (
            ast.Constant(names[0]) if len(names) == 1 else _literal(tuple(names))
        )
        node.value = ast.Call(
            func=ast.Attribute(
                ast.Name(self.runtime_name, ast.Load()), method, ast.Load()
            ),
            args=[first_arg, node.value, _literal(descriptor)],
            keywords=[],
        )
        return node

    def visit_Delete(self, node: ast.Delete) -> Any:
        if (
            len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Subscript)
            or isinstance(node.targets[0].slice, ast.Slice)
        ):
            return self.generic_visit(node)
        target = node.targets[0]
        return ast.copy_location(
            ast.Expr(
                ast.Call(
                    ast.Attribute(
                        ast.Name(self.runtime_name, ast.Load()),
                        "delitem",
                        ast.Load(),
                    ),
                    [self.visit(target.value), self.visit(target.slice)],
                    [],
                )
            ),
            node,
        )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        if node.value is None or not isinstance(node.target, ast.Name):
            return node
        node.annotation = ast.Name("object", ast.Load())
        original = node.value
        node.value = ast.Call(
            func=ast.Attribute(
                ast.Name(self.runtime_name, ast.Load()), "assign", ast.Load()
            ),
            args=[
                ast.Constant(node.target.id),
                self.visit(original),
                _literal(expression_descriptor(original)),
            ],
            keywords=[],
        )
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        if isinstance(node.target, ast.Attribute):
            operation = _BINARY_OPERATORS.get(type(node.op))
            if operation is None:
                return self.generic_visit(node)
            descriptor = (
                "binary",
                operation,
                (
                    "attribute",
                    expression_descriptor(node.target.value),
                    node.target.attr,
                ),
                expression_descriptor(node.value),
            )
            return ast.copy_location(
                ast.Expr(
                    ast.Call(
                        ast.Attribute(
                            ast.Name(self.runtime_name, ast.Load()),
                            "augattr",
                            ast.Load(),
                        ),
                        [
                            self.visit(node.target.value),
                            ast.Constant(node.target.attr),
                            self.visit(node.value),
                            ast.Constant(operation),
                            _literal(descriptor),
                        ],
                        [],
                    )
                ),
                node,
            )
        if not isinstance(node.target, ast.Name):
            return self.generic_visit(node)
        operation = _BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            return node
        name = node.target.id
        combined_descriptor = (
            "binary",
            operation,
            ("name", name),
            expression_descriptor(node.value),
        )
        self._temp_counter += 1
        temp_name = f"__aria_symbolic_temp_{self._temp_counter}"
        prepare = ast.Assign(
            [ast.Name(temp_name, ast.Store())],
            ast.Call(
                ast.Attribute(
                    ast.Name(self.runtime_name, ast.Load()), "prepare", ast.Load()
                ),
                [_literal(combined_descriptor)],
                [],
            ),
        )
        updated = ast.AugAssign(node.target, node.op, self.visit(node.value))
        commit = ast.Expr(
            ast.Call(
                ast.Attribute(
                    ast.Name(self.runtime_name, ast.Load()), "commit", ast.Load()
                ),
                [
                    ast.Constant(name),
                    ast.Name(name, ast.Load()),
                    ast.Name(temp_name, ast.Load()),
                ],
                [],
            )
        )
        return [prepare, updated, commit]

    def visit_NamedExpr(self, node: ast.NamedExpr) -> ast.AST:
        if not isinstance(node.target, ast.Name):
            return self.generic_visit(node)
        original = node.value
        node.value = ast.Call(
            ast.Attribute(
                ast.Name(self.runtime_name, ast.Load()), "assign", ast.Load()
            ),
            [
                ast.Constant(node.target.id),
                self.visit(original),
                _literal(expression_descriptor(original)),
            ],
            [],
        )
        return node

    def _branch(self, test: ast.expr, kind: str) -> ast.expr:
        location = (
            self.filename,
            self.source_line_offset + test.lineno,
            test.col_offset,
            self.source_line_offset + getattr(test, "end_lineno", test.lineno),
            getattr(test, "end_col_offset", test.col_offset),
        )
        stable = f"{self.function_id}:{kind}:{location[1]}:{location[2]}"
        branch_id = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]
        self.branch_locations[branch_id] = location
        descriptor = expression_descriptor(test)
        concrete_test = self.visit(test)
        return ast.copy_location(
            ast.Call(
                func=ast.Attribute(
                    ast.Name(self.runtime_name, ast.Load()), "branch", ast.Load()
                ),
                args=[
                    ast.Constant(branch_id),
                    concrete_test,
                    _literal(descriptor),
                    _literal(location),
                ],
                keywords=[],
            ),
            test,
        )


def instrument_callable(target: Any) -> InstrumentedCallable:
    """Compile a callable with native concolic runtime hooks."""
    original_target = target
    bound_instance = None
    if inspect.ismethod(target) and target.__self__ is not None:
        bound_instance = target.__self__
        target = target.__func__
    if not inspect.isfunction(target):
        raise InstrumentationError(
            "native instrumentation requires a Python function or bound method"
        )
    try:
        source_lines, starting_line = inspect.getsourcelines(target)
        filename = inspect.getsourcefile(target) or inspect.getfile(target)
    except (OSError, TypeError) as exc:
        raise InstrumentationError(f"source is unavailable for {target!r}") from exc
    source = textwrap.dedent("".join(source_lines))
    module = ast.parse(source, filename=filename)
    function_nodes = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not function_nodes:
        raise InstrumentationError(
            f"could not locate the function definition for {target.__qualname__}"
        )
    function_node = function_nodes[0]
    if any(
        isinstance(node, (ast.Nonlocal, ast.Global)) for node in ast.walk(function_node)
    ):
        raise InstrumentationError(
            "functions containing nonlocal/global declarations are not safely "
            "instrumentable"
        )

    function_id = f"{target.__module__}.{target.__qualname__}"
    runtime_name = (
        f"__aria_concolic_runtime_{hashlib.sha1(function_id.encode()).hexdigest()[:12]}"
    )
    closure = inspect.getclosurevars(target)
    environment_values = {
        **closure.globals,
        **closure.nonlocals,
        **closure.builtins,
    }
    primitive_environment = {
        name
        for name, value in environment_values.items()
        if type(value)
        in {bool, int, float, str, bytes, bytearray, list, tuple, dict, set}
        or callable(value)
        or isinstance(value, type)
    }
    refreshable_environment = {
        name
        for name, value in environment_values.items()
        if type(value)
        in {bool, int, float, str, bytes, bytearray, list, tuple, dict, set}
    }
    transformer = _FunctionTransformer(
        filename,
        function_id,
        starting_line - 1,
        runtime_name,
        sorted(primitive_environment),
    )
    transformed = transformer.visit(module)
    ast.fix_missing_locations(transformed)
    if starting_line > 1:
        ast.increment_lineno(transformed, starting_line - 1)

    namespace = dict(target.__globals__)
    namespace.update(closure.globals)
    namespace.update(closure.nonlocals)
    namespace[runtime_name] = RUNTIME_HOOKS
    local_namespace: Dict[str, Any] = {}
    code = compile(transformed, filename, "exec")
    exec(code, namespace, local_namespace)
    namespace.update(local_namespace)
    compiled = local_namespace[function_node.name]
    compiled.__defaults__ = target.__defaults__
    compiled.__kwdefaults__ = target.__kwdefaults__
    compiled.__annotations__ = dict(target.__annotations__)
    compiled.__dict__.update(target.__dict__)
    compiled.__name__ = target.__name__
    compiled.__qualname__ = target.__qualname__
    compiled.__module__ = target.__module__
    compiled.__doc__ = target.__doc__
    if bound_instance is not None:
        compiled = types.MethodType(compiled, bound_instance)
    closure_cells = {
        name: cell
        for name, cell in zip(target.__code__.co_freevars, target.__closure__ or ())
        if name in primitive_environment
    }
    return InstrumentedCallable(
        original=original_target,
        callable=compiled,
        filename=filename,
        function_id=function_id,
        branch_locations=transformer.branch_locations,
        namespace=namespace,
        global_names=tuple(sorted(set(closure.globals) & refreshable_environment)),
        closure_cells=closure_cells,
    )


def _parameter_specs(arguments: ast.arguments) -> Tuple[Tuple[str, str], ...]:
    specs = [(arg.arg, "positional") for arg in arguments.posonlyargs]
    specs.extend((arg.arg, "positional") for arg in arguments.args)
    if arguments.vararg:
        specs.append((arguments.vararg.arg, "var_positional"))
    specs.extend((arg.arg, "keyword_only") for arg in arguments.kwonlyargs)
    if arguments.kwarg:
        specs.append((arguments.kwarg.arg, "var_keyword"))
    return tuple(specs)


def _flatten_statements(statements: Sequence[Any]) -> List[ast.stmt]:
    flattened: List[ast.stmt] = []
    for statement in statements:
        if isinstance(statement, list):
            flattened.extend(statement)
        else:
            flattened.append(statement)
    return flattened
