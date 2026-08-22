"""Typed symbolic semantics used by the native AST runtime."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

import z3

from .models import DiagnosticCode, StructuredValue, SymbolicTerm
from .model_registry import DEFAULT_MODEL_REGISTRY, SymbolicModelRegistry

Descriptor = Tuple[Any, ...]
DiagnosticSink = Callable[[DiagnosticCode, str], None]
TruthOracle = Callable[[SymbolicTerm], Optional[bool]]
CallResultOracle = Callable[[str], Optional[SymbolicTerm]]


def concrete_to_z3(value: Any) -> Optional[z3.ExprRef]:
    """Convert supported concrete Python values to exact Z3 values."""
    if type(value) is bool:
        return z3.BoolVal(value)
    if type(value) is int:
        return z3.IntVal(value)
    if type(value) is str:
        return exact_string_val(value)
    if type(value) is bytes:
        return exact_string_val(value.decode("latin-1"))
    if type(value) is float:
        return z3.FPVal(value, z3.Float64())
    return None


def exact_string_val(value: str) -> z3.SeqRef:
    """Create a Z3 string without treating literal backslash-u text as an escape."""
    if not value:
        return z3.StringVal("")
    if "\\u{" not in value:
        return z3.StringVal(value)
    return z3.Concat(*[z3.StringVal(character) for character in value])


def decode_z3_string_escapes(value: str) -> str:
    """Decode the escape notation emitted by z3py model rendering."""
    return re.sub(
        r"\\u\{([0-9a-fA-F]+)\}",
        lambda match: chr(int(match.group(1), 16)),
        value,
    )


def symbolic_input(name: str, value: Any) -> Optional[z3.ExprRef]:
    """Create an exact symbolic input matching a supported Python value."""
    if type(value) is bool:
        return z3.Bool(name)
    if type(value) is int:
        return z3.Int(name)
    if type(value) is str:
        return z3.String(name)
    if type(value) is bytes:
        return z3.String(name)
    if type(value) is float:
        return z3.FP(name, z3.Float64())
    return None


def truthy(term: SymbolicTerm) -> SymbolicTerm:
    """Apply Python truth-value semantics to a supported symbolic term."""
    expr = term.expression
    if expr is None:
        return term
    if isinstance(expr, StructuredValue):
        return SymbolicTerm(z3.BoolVal(bool(expr.children)), term.guards)
    if z3.is_bool(expr):
        return term
    if z3.is_int(expr) or z3.is_real(expr):
        return SymbolicTerm(expr != 0, term.guards)
    if z3.is_string(expr):
        return SymbolicTerm(z3.Length(expr) != 0, term.guards)
    if z3.is_fp(expr):
        return SymbolicTerm(z3.Not(z3.fpIsZero(expr)), term.guards)
    return SymbolicTerm(
        None, term.guards, f"truthiness for sort {expr.sort()} is unsupported"
    )


def merge_guards(*terms: SymbolicTerm) -> Tuple[z3.BoolRef, ...]:
    seen: set[str] = set()
    merged = []
    for term in terms:
        for guard in term.guards:
            key = guard.sexpr()
            if key not in seen:
                seen.add(key)
                merged.append(guard)
    return tuple(merged)


@dataclass
class SymbolicEvaluator:
    """Evaluate serializable expression descriptors over a shadow store."""

    shadow: Mapping[str, SymbolicTerm]
    diagnose: DiagnosticSink
    truth_oracle: Optional[TruthOracle] = None
    call_result_oracle: Optional[CallResultOracle] = None
    registry: SymbolicModelRegistry = field(
        default_factory=lambda: DEFAULT_MODEL_REGISTRY
    )

    def evaluate(self, descriptor: Descriptor) -> SymbolicTerm:
        try:
            return self._evaluate(descriptor)
        except (z3.Z3Exception, TypeError, ValueError) as exc:
            message = f"symbolic evaluation failed: {exc}"
            self.diagnose(DiagnosticCode.UNSUPPORTED_OPERATION, message)
            return SymbolicTerm(None, reason=message)

    def _evaluate(self, descriptor: Descriptor) -> SymbolicTerm:
        kind = descriptor[0]
        if kind == "const":
            expr = concrete_to_z3(descriptor[1])
            if expr is None:
                return self._unsupported(f"constant {descriptor[1]!r}")
            return SymbolicTerm(expr)

        if kind == "name":
            name = descriptor[1]
            term = self.shadow.get(name)
            if term is None:
                self.diagnose(
                    DiagnosticCode.MISSING_SYMBOLIC_VALUE,
                    f"no symbolic value is available for {name!r}",
                )
                return SymbolicTerm(None, reason=f"missing symbolic value for {name}")
            return term

        if kind == "attribute":
            owner = self._evaluate(descriptor[1])
            if not isinstance(owner.expression, StructuredValue):
                return self._unsupported(f"attribute {descriptor[2]!r}", owner.guards)
            child = owner.expression.children.get(descriptor[2])
            if child is None:
                return self._unsupported(f"attribute {descriptor[2]!r}", owner.guards)
            return SymbolicTerm(
                child.expression,
                (*owner.guards, *child.guards),
                child.reason,
            )

        if kind == "call_result":
            call_id, fallback = descriptor[1], descriptor[2]
            if self.call_result_oracle is not None:
                result = self.call_result_oracle(call_id)
                if result is not None:
                    return result
            return self._evaluate(fallback)

        if kind == "attribute_result":
            attribute_id, fallback = descriptor[1], descriptor[2]
            if self.call_result_oracle is not None:
                result = self.call_result_oracle(attribute_id)
                if result is not None:
                    return result
            return self._evaluate(fallback)

        if kind == "unary":
            operation, child_desc = descriptor[1], descriptor[2]
            child = self._evaluate(child_desc)
            if child.expression is None:
                return child
            if operation == "not":
                condition = truthy(child)
                if condition.expression is None:
                    return condition
                return SymbolicTerm(z3.Not(condition.expression), condition.guards)
            if operation == "+" and (
                z3.is_int(child.expression) or z3.is_real(child.expression)
            ):
                return child
            if operation == "-" and (
                z3.is_int(child.expression) or z3.is_real(child.expression)
            ):
                return SymbolicTerm(-child.expression, child.guards)
            if operation == "-" and z3.is_fp(child.expression):
                return SymbolicTerm(-child.expression, child.guards)
            return self._unsupported(f"unary operation {operation!r}", child.guards)

        if kind == "binary":
            return self._binary(
                descriptor[1],
                self._evaluate(descriptor[2]),
                self._evaluate(descriptor[3]),
            )

        if kind == "bool":
            operation = descriptor[1]
            children = []
            expressions = []
            for item in descriptor[2]:
                child = truthy(self._evaluate(item))
                children.append(child)
                if child.expression is None:
                    return self._unsupported(
                        f"Boolean operation {operation!r}", merge_guards(*children)
                    )
                expressions.append(child.expression)
                concrete_truth = (
                    self.truth_oracle(child) if self.truth_oracle is not None else None
                )
                if operation == "and" and concrete_truth is False:
                    break
                if operation == "or" and concrete_truth is True:
                    break
            result = z3.And(*expressions) if operation == "and" else z3.Or(*expressions)
            return SymbolicTerm(result, merge_guards(*children))

        if kind == "compare":
            left = self._evaluate(descriptor[1])
            operations: Sequence[str] = descriptor[2]
            right_descriptors: Sequence[Descriptor] = descriptor[3]
            comparisons = []
            terms = [left]
            current = left
            for operation, right_descriptor in zip(operations, right_descriptors):
                right = self._evaluate(right_descriptor)
                terms.append(right)
                comparison = self._comparison(operation, current, right)
                if comparison.expression is None:
                    return comparison
                comparisons.append(comparison.expression)
                current = right
            return SymbolicTerm(z3.And(*comparisons), merge_guards(*terms))

        if kind == "ifexp":
            condition = truthy(self._evaluate(descriptor[1]))
            body = self._evaluate(descriptor[2])
            otherwise = self._evaluate(descriptor[3])
            if any(term.expression is None for term in (condition, body, otherwise)):
                return self._unsupported(
                    "conditional expression", merge_guards(condition, body, otherwise)
                )
            if body.expression.sort() != otherwise.expression.sort():
                return self._unsupported(
                    "conditional expression with unlike branch types"
                )
            return SymbolicTerm(
                z3.If(condition.expression, body.expression, otherwise.expression),
                merge_guards(condition, body, otherwise),
            )

        if kind in {"tuple", "list"}:
            children = {
                index: self._evaluate(item) for index, item in enumerate(descriptor[1])
            }
            return SymbolicTerm(StructuredValue(kind, children))

        if kind == "dict":
            children = {}
            for key_descriptor, value_descriptor in descriptor[1]:
                key_term = self._evaluate(key_descriptor)
                if key_term.expression is None:
                    return self._unsupported("dictionary literal key")
                key = _z3_literal_to_python(key_term.expression)
                children[key] = self._evaluate(value_descriptor)
            return SymbolicTerm(StructuredValue("dict", children))

        if kind == "call":
            return self._call(
                descriptor[1], [self._evaluate(arg) for arg in descriptor[2]]
            )

        if kind == "method":
            owner = self._evaluate(descriptor[1])
            args = [self._evaluate(arg) for arg in descriptor[3]]
            return self._method(owner, descriptor[2], args)

        if kind == "subscript":
            owner = self._evaluate(descriptor[1])
            index = self._evaluate(descriptor[2])
            if owner.expression is None or index.expression is None:
                return self._unsupported("subscript", merge_guards(owner, index))
            if isinstance(owner.expression, StructuredValue):
                return self._structured_subscript(
                    owner,
                    index,
                    descriptor[2],
                )
            if z3.is_string(owner.expression) and z3.is_int(index.expression):
                length = z3.Length(owner.expression)
                normalized_index = z3.If(
                    index.expression >= 0,
                    index.expression,
                    length + index.expression,
                )
                guard = z3.And(
                    normalized_index >= 0,
                    normalized_index < length,
                )
                return SymbolicTerm(
                    z3.SubString(owner.expression, normalized_index, 1),
                    (*merge_guards(owner, index), guard),
                )
            return self._unsupported(
                "subscript for these operand types", merge_guards(owner, index)
            )

        if kind == "slice":
            owner = self._evaluate(descriptor[1])
            lower = self._evaluate(descriptor[2]) if descriptor[2] else None
            upper = self._evaluate(descriptor[3]) if descriptor[3] else None
            step = self._evaluate(descriptor[4]) if descriptor[4] else None
            if owner.expression is None:
                return self._unsupported("slice", owner.guards)
            if step is not None:
                if (
                    not z3.is_int_value(step.expression)
                    or step.expression.as_long() != 1
                ):
                    return self._unsupported("slice step", merge_guards(owner, step))
            if z3.is_string(owner.expression):
                length = z3.Length(owner.expression)
                start = _normalize_slice_bound(
                    lower.expression if lower else z3.IntVal(0), length
                )
                stop = _normalize_slice_bound(
                    upper.expression if upper else length, length
                )
                terms = [owner]
                if lower:
                    terms.append(lower)
                if upper:
                    terms.append(upper)
                return SymbolicTerm(
                    z3.SubString(
                        owner.expression, start, z3.If(stop >= start, stop - start, 0)
                    ),
                    merge_guards(*terms),
                )
            if isinstance(
                owner.expression, StructuredValue
            ) and owner.expression.kind in {
                "list",
                "tuple",
            }:
                if lower is not None and not z3.is_int_value(lower.expression):
                    return self._unsupported("symbolic structured slice lower bound")
                if upper is not None and not z3.is_int_value(upper.expression):
                    return self._unsupported("symbolic structured slice upper bound")
                length = len(owner.expression.children)
                lower_value = lower.expression.as_long() if lower else None
                upper_value = upper.expression.as_long() if upper else None
                start, stop, _ = slice(lower_value, upper_value, 1).indices(length)
                selected = [
                    owner.expression.children[index] for index in range(start, stop)
                ]
                return SymbolicTerm(
                    StructuredValue(
                        owner.expression.kind,
                        {index: term for index, term in enumerate(selected)},
                    ),
                    owner.guards,
                )
            return self._unsupported("slice", owner.guards)

        if kind == "opaque":
            return self._unsupported(str(descriptor[1]))

        return self._unsupported(f"descriptor kind {kind!r}")

    def _binary(
        self, operation: str, left: SymbolicTerm, right: SymbolicTerm
    ) -> SymbolicTerm:
        if left.expression is None or right.expression is None:
            return self._unsupported(
                f"binary operation {operation!r}", merge_guards(left, right)
            )
        lhs, rhs = left.expression, right.expression
        guards = merge_guards(left, right)
        if (
            operation == "+"
            and isinstance(lhs, StructuredValue)
            and isinstance(rhs, StructuredValue)
            and lhs.kind in {"list", "tuple"}
            and lhs.kind == rhs.kind
        ):
            left_values = [lhs.children[index] for index in sorted(lhs.children)]
            right_values = [rhs.children[index] for index in sorted(rhs.children)]
            return SymbolicTerm(
                StructuredValue(
                    lhs.kind,
                    {
                        index: term
                        for index, term in enumerate((*left_values, *right_values))
                    },
                ),
                guards,
            )
        numeric = (z3.is_int(lhs) or z3.is_real(lhs)) and (
            z3.is_int(rhs) or z3.is_real(rhs)
        )
        if z3.is_fp(lhs) and z3.is_fp(rhs) and lhs.sort() == rhs.sort():
            operations = {
                "+": z3.fpAdd,
                "-": z3.fpSub,
                "*": z3.fpMul,
                "/": z3.fpDiv,
            }
            function = operations.get(operation)
            if function is not None:
                extra_guards = (z3.Not(z3.fpIsZero(rhs)),) if operation == "/" else ()
                return SymbolicTerm(
                    function(z3.RNE(), lhs, rhs),
                    (*guards, *extra_guards),
                )
        if operation == "+":
            if numeric:
                return SymbolicTerm(lhs + rhs, guards)
            if z3.is_string(lhs) and z3.is_string(rhs):
                return SymbolicTerm(z3.Concat(lhs, rhs), guards)
        if operation == "-" and numeric:
            return SymbolicTerm(lhs - rhs, guards)
        if operation == "*" and numeric:
            return SymbolicTerm(lhs * rhs, guards)
        if operation in {"//", "%"} and z3.is_int(lhs) and z3.is_int(rhs):
            nonzero = rhs != 0
            quotient = z3.ToInt(z3.ToReal(lhs) / z3.ToReal(rhs))
            expression = quotient if operation == "//" else lhs - rhs * quotient
            return SymbolicTerm(expression, (*guards, nonzero))
        return self._unsupported(
            f"binary operation {operation!r} for {lhs.sort()} and {rhs.sort()}", guards
        )

    def _comparison(
        self, operation: str, left: SymbolicTerm, right: SymbolicTerm
    ) -> SymbolicTerm:
        if left.expression is None or right.expression is None:
            return self._unsupported(
                f"comparison {operation!r}", merge_guards(left, right)
            )
        lhs, rhs = left.expression, right.expression
        guards = merge_guards(left, right)
        if operation in {"==", "!="}:
            if isinstance(lhs, StructuredValue) or isinstance(rhs, StructuredValue):
                equality = self._structured_equality(lhs, rhs)
                if equality is None:
                    return self._unsupported("structured equality", guards)
                return SymbolicTerm(
                    equality if operation == "==" else z3.Not(equality),
                    guards,
                )
            if z3.is_fp(lhs) and z3.is_fp(rhs) and lhs.sort() == rhs.sort():
                equality = z3.fpEQ(lhs, rhs)
                return SymbolicTerm(
                    equality if operation == "==" else z3.Not(equality), guards
                )
            if lhs.sort() != rhs.sort():
                result = z3.BoolVal(operation == "!=")
            else:
                result = lhs == rhs if operation == "==" else lhs != rhs
            return SymbolicTerm(result, guards)
        if operation in {"<", "<=", ">", ">="}:
            if lhs.sort() != rhs.sort():
                return self._unsupported(
                    f"ordered comparison of {lhs.sort()} and {rhs.sort()}", guards
                )
            if z3.is_fp(lhs):
                operators = {
                    "<": z3.fpLT,
                    "<=": z3.fpLEQ,
                    ">": z3.fpGT,
                    ">=": z3.fpGEQ,
                }
                return SymbolicTerm(operators[operation](lhs, rhs), guards)
            operators = {
                "<": lhs < rhs,
                "<=": lhs <= rhs,
                ">": lhs > rhs,
                ">=": lhs >= rhs,
            }
            return SymbolicTerm(operators[operation], guards)
        if operation in {"in", "not in"} and z3.is_string(lhs) and z3.is_string(rhs):
            contained = z3.Contains(rhs, lhs)
            return SymbolicTerm(
                contained if operation == "in" else z3.Not(contained), guards
            )
        if operation in {"in", "not in"} and isinstance(rhs, StructuredValue):
            terms = [left, right]
            if rhs.kind == "dict":
                values = [SymbolicTerm(concrete_to_z3(key)) for key in rhs.children]
            else:
                values = list(rhs.children.values())
            comparisons = []
            for value in values:
                terms.append(value)
                if value.expression is None or value.expression.sort() != lhs.sort():
                    continue
                comparisons.append(lhs == value.expression)
            if not comparisons:
                return self._unsupported("structured membership", merge_guards(*terms))
            contained = z3.Or(*comparisons)
            return SymbolicTerm(
                contained if operation == "in" else z3.Not(contained),
                merge_guards(*terms),
            )
        return self._unsupported(f"comparison operation {operation!r}", guards)

    def _call(self, name: str, args: Sequence[SymbolicTerm]) -> SymbolicTerm:
        model = self.registry.function_model(name)
        if model is not None:
            result = model(args)
            if result is not None:
                return result
        return self._unsupported(f"call to {name}()", merge_guards(*args))

    def _method(
        self, owner: SymbolicTerm, name: str, args: Sequence[SymbolicTerm]
    ) -> SymbolicTerm:
        all_terms = (owner, *args)
        model = self.registry.method_model(name)
        if model is not None:
            result = model(owner, args)
            if result is not None:
                return result
        return self._unsupported(f"method {name}()", merge_guards(*all_terms))

    def _structured_subscript(
        self,
        owner: SymbolicTerm,
        index: SymbolicTerm,
        index_descriptor: Descriptor,
    ) -> SymbolicTerm:
        structured = owner.expression
        assert isinstance(structured, StructuredValue)
        guards = merge_guards(owner, index)
        if index_descriptor[0] == "const":
            key = index_descriptor[1]
            if structured.kind in {"list", "tuple", "bytearray"} and isinstance(
                key, int
            ):
                key = key if key >= 0 else len(structured.children) + key
            child = structured.children.get(key)
            if child is None:
                return self._unsupported("structured index outside fixed shape", guards)
            return SymbolicTerm(
                child.expression,
                (*guards, *child.guards),
                child.reason,
            )

        choices = []
        lookup_expression = index.expression
        if structured.kind in {"list", "tuple", "bytearray"} and z3.is_int(
            index.expression
        ):
            lookup_expression = z3.If(
                index.expression >= 0,
                index.expression,
                len(structured.children) + index.expression,
            )
        for key, child in structured.children.items():
            key_expr = concrete_to_z3(key)
            if key_expr is None or child.expression is None:
                return self._unsupported("dynamic structured index", guards)
            if isinstance(child.expression, StructuredValue):
                return self._unsupported("dynamic index returning a container", guards)
            if key_expr.sort() != lookup_expression.sort():
                continue
            choices.append((key_expr, child))
        if not choices:
            return self._unsupported("dynamic structured index", guards)
        result = choices[-1][1].expression
        assert result is not None
        for key_expr, child in reversed(choices[:-1]):
            result = z3.If(lookup_expression == key_expr, child.expression, result)
        defined = z3.Or(*[lookup_expression == key for key, _ in choices])
        child_guards = tuple(guard for _, child in choices for guard in child.guards)
        return SymbolicTerm(result, (*guards, *child_guards, defined))

    def _structured_equality(self, lhs: Any, rhs: Any) -> Optional[z3.BoolRef]:
        if not isinstance(lhs, StructuredValue) or not isinstance(rhs, StructuredValue):
            return z3.BoolVal(False)
        if lhs.kind != rhs.kind or set(lhs.children) != set(rhs.children):
            return z3.BoolVal(False)
        equalities = []
        for key in lhs.children:
            left = lhs.children[key].expression
            right = rhs.children[key].expression
            if isinstance(left, StructuredValue) or isinstance(right, StructuredValue):
                nested = self._structured_equality(left, right)
                if nested is None:
                    return None
                equalities.append(nested)
            elif left is None or right is None or left.sort() != right.sort():
                return None
            else:
                equalities.append(left == right)
        return z3.And(*equalities)

    def _unsupported(
        self, subject: str, guards: Iterable[z3.BoolRef] = ()
    ) -> SymbolicTerm:
        message = f"unsupported symbolic {subject}"
        self.diagnose(DiagnosticCode.UNSUPPORTED_EXPRESSION, message)
        return SymbolicTerm(None, tuple(guards), message)


def _z3_literal_to_python(expression: z3.ExprRef) -> Any:
    if z3.is_string_value(expression):
        return expression.as_string()
    if z3.is_int_value(expression):
        return expression.as_long()
    if z3.is_true(expression):
        return True
    if z3.is_false(expression):
        return False
    return str(expression)


def _normalize_slice_bound(bound: z3.ExprRef, length: z3.ArithRef) -> z3.ArithRef:
    normalized = z3.If(bound < 0, length + bound, bound)
    return z3.If(normalized < 0, 0, z3.If(normalized > length, length, normalized))


def _model_len(args: Sequence[SymbolicTerm]) -> Optional[SymbolicTerm]:
    if len(args) != 1 or args[0].expression is None:
        return None
    arg = args[0]
    if isinstance(arg.expression, StructuredValue):
        return SymbolicTerm(z3.IntVal(len(arg.expression.children)), arg.guards)
    if z3.is_string(arg.expression):
        return SymbolicTerm(z3.Length(arg.expression), arg.guards)
    return None


def _model_abs(args: Sequence[SymbolicTerm]) -> Optional[SymbolicTerm]:
    if len(args) != 1 or args[0].expression is None:
        return None
    arg = args[0]
    if z3.is_int(arg.expression) or z3.is_real(arg.expression):
        return SymbolicTerm(
            z3.If(arg.expression >= 0, arg.expression, -arg.expression),
            arg.guards,
        )
    return None


def _model_bool(args: Sequence[SymbolicTerm]) -> Optional[SymbolicTerm]:
    return truthy(args[0]) if len(args) == 1 else None


def _model_string_method(
    owner: SymbolicTerm,
    args: Sequence[SymbolicTerm],
    operation: str,
) -> Optional[SymbolicTerm]:
    if owner.expression is None or not z3.is_string(owner.expression):
        return None
    if operation in {"startswith", "endswith"} and len(args) == 1:
        argument = args[0].expression
        if argument is None or not z3.is_string(argument):
            return None
        guards = merge_guards(owner, args[0])
        if operation == "startswith":
            return SymbolicTerm(z3.PrefixOf(argument, owner.expression), guards)
        return SymbolicTerm(z3.SuffixOf(argument, owner.expression), guards)
    if operation == "find" and 1 <= len(args) <= 3:
        argument = args[0].expression
        if argument is None or not z3.is_string(argument):
            return None
        start = args[1].expression if len(args) >= 2 else z3.IntVal(0)
        if start is None or not z3.is_int(start):
            return None
        guards = list(merge_guards(owner, *args))
        guards.append(start >= 0)
        if len(args) == 3:
            end = args[2].expression
            if end is None or not z3.is_int(end):
                return None
            guards.extend((end >= start, end <= z3.Length(owner.expression)))
            sliced = z3.SubString(owner.expression, start, end - start)
            relative = z3.IndexOf(sliced, argument, 0)
            return SymbolicTerm(
                z3.If(relative < 0, -1, relative + start), tuple(guards)
            )
        return SymbolicTerm(
            z3.IndexOf(owner.expression, argument, start), tuple(guards)
        )
    if operation in {"strip", "lstrip"} and len(args) == 1:
        characters = _string_literal(args[0])
        if characters is None:
            return None
        guards = list(merge_guards(owner, args[0]))
        guards.extend(
            z3.Not(z3.Contains(owner.expression, z3.StringVal(character)))
            for character in characters
        )
        return SymbolicTerm(owner.expression, tuple(guards))
    if operation == "replace" and len(args) == 2:
        old = _string_literal(args[0])
        new = _string_literal(args[1])
        if old is None or new is None or not old:
            return None
        first = z3.IndexOf(owner.expression, z3.StringVal(old), 0)
        second = z3.IndexOf(owner.expression, z3.StringVal(old), first + len(old))
        return SymbolicTerm(
            z3.Replace(owner.expression, z3.StringVal(old), z3.StringVal(new)),
            (*merge_guards(owner, *args), second == -1),
        )
    if operation == "isascii" and not args:
        ascii_re = z3.Star(z3.Range(chr(0), chr(127)))
        return SymbolicTerm(z3.InRe(owner.expression, ascii_re), owner.guards)
    if operation == "isalpha" and not args:
        code = z3.StrToCode(owner.expression)
        alpha = z3.Or(
            z3.And(code >= ord("A"), code <= ord("Z")),
            z3.And(code >= ord("a"), code <= ord("z")),
        )
        return SymbolicTerm(
            alpha,
            (*owner.guards, z3.Length(owner.expression) == 1, code >= 0, code < 128),
        )
    if operation == "lower" and not args:
        uppercase_guards = tuple(
            z3.Not(z3.Contains(owner.expression, z3.StringVal(character)))
            for character in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )
        return SymbolicTerm(owner.expression, (*owner.guards, *uppercase_guards))
    if operation == "split" and len(args) == 2:
        separator = _string_literal(args[0])
        maximum = args[1].expression
        if separator is None or not separator or not z3.is_int_value(maximum):
            return None
        if maximum.as_long() != 1:
            return None
        separator_expr = z3.StringVal(separator)
        index = z3.IndexOf(owner.expression, separator_expr, 0)
        before = z3.SubString(owner.expression, 0, index)
        after = z3.SubString(
            owner.expression,
            index + len(separator),
            z3.Length(owner.expression) - index - len(separator),
        )
        return SymbolicTerm(
            StructuredValue(
                "list",
                {0: SymbolicTerm(before), 1: SymbolicTerm(after)},
            ),
            (*merge_guards(owner, *args), index >= 0),
        )
    return None


def _string_literal(term: SymbolicTerm) -> Optional[str]:
    if term.expression is not None and z3.is_string_value(term.expression):
        rendered = term.expression.as_string()
        return decode_z3_string_escapes(rendered)
    return None


def _model_split_result(args: Sequence[SymbolicTerm]) -> Optional[SymbolicTerm]:
    if len(args) != 5:
        return None
    return SymbolicTerm(
        StructuredValue(
            "object",
            {
                name: term
                for name, term in zip(
                    ("scheme", "netloc", "path", "query", "fragment"), args
                )
            },
        ),
        merge_guards(*args),
    )


DEFAULT_MODEL_REGISTRY.register_function("len", _model_len, replace=True)
DEFAULT_MODEL_REGISTRY.register_function("abs", _model_abs, replace=True)
DEFAULT_MODEL_REGISTRY.register_function("bool", _model_bool, replace=True)
for _method_name in (
    "startswith",
    "endswith",
    "find",
    "strip",
    "lstrip",
    "replace",
    "isascii",
    "isalpha",
    "lower",
    "split",
):
    DEFAULT_MODEL_REGISTRY.register_method(
        _method_name,
        lambda owner, args, name=_method_name: _model_string_method(owner, args, name),
        replace=True,
    )
for _result_name in ("SplitResult", "urllib.parse.SplitResult"):
    DEFAULT_MODEL_REGISTRY.register_function(
        _result_name, _model_split_result, replace=True
    )
