"""Fixed-shape symbolic container inputs and model reconstruction."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import z3

from .models import DiagnosticCode, PythonConstant, StructuredValue, SymbolicTerm
from .symbolic import concrete_to_z3, symbolic_input

InputPath = Tuple[str, Tuple[Any, ...]]
DiagnosticSink = Callable[[DiagnosticCode, str], None]


def symbolize_input(
    root_name: str,
    value: Any,
    symbolic: bool,
    diagnose: DiagnosticSink,
) -> Tuple[SymbolicTerm, Dict[str, z3.ExprRef], Dict[str, InputPath]]:
    """Build a symbolic tree and flattened solver declarations for one input."""
    symbols: Dict[str, z3.ExprRef] = {}
    paths: Dict[str, InputPath] = {}
    counter = [0]

    def visit(current: Any, path: Tuple[Any, ...]) -> SymbolicTerm:
        concrete = concrete_to_z3(current)
        if concrete is not None:
            if not symbolic:
                return SymbolicTerm(concrete)
            symbol_name = (
                root_name
                if not path
                else f"__aria_input_{len(root_name)}_{root_name}_{counter[0]}"
            )
            counter[0] += 1
            expression = symbolic_input(symbol_name, current)
            assert expression is not None
            symbols[symbol_name] = expression
            paths[symbol_name] = (root_name, path)
            return SymbolicTerm(expression)

        if isinstance(current, list):
            children = {
                index: visit(item, (*path, index)) for index, item in enumerate(current)
            }
            return SymbolicTerm(StructuredValue("list", children))
        if isinstance(current, tuple):
            children = {
                index: visit(item, (*path, index)) for index, item in enumerate(current)
            }
            return SymbolicTerm(StructuredValue("tuple", children))
        if isinstance(current, dict):
            children = {
                key: visit(current[key], (*path, key))
                for key in sorted(current, key=lambda item: repr(item))
            }
            return SymbolicTerm(StructuredValue("dict", children))
        if isinstance(current, bytearray):
            children = {
                index: visit(item, (*path, index)) for index, item in enumerate(current)
            }
            return SymbolicTerm(StructuredValue("bytearray", children))
        if isinstance(current, set):
            ordered = sorted(current, key=repr)
            children = {
                index: visit(item, (*path, ("@set", index)))
                for index, item in enumerate(ordered)
            }
            return SymbolicTerm(StructuredValue("set", children))
        if dataclasses.is_dataclass(current) and not isinstance(current, type):
            children = {
                field.name: visit(
                    getattr(current, field.name),
                    (*path, ("@attr", field.name)),
                )
                for field in dataclasses.fields(current)
            }
            return SymbolicTerm(StructuredValue("dataclass", children))
        if hasattr(current, "__dict__"):
            public_fields = {
                name: item
                for name, item in vars(current).items()
                if not name.startswith("_")
            }
            if public_fields:
                children = {
                    name: visit(item, (*path, ("@attr", name)))
                    for name, item in sorted(public_fields.items())
                }
                return SymbolicTerm(StructuredValue("object", children))

        if not symbolic:
            return SymbolicTerm(PythonConstant(current))
        diagnose(
            DiagnosticCode.UNSUPPORTED_INPUT,
            (
                f"input {root_name!r} path {path!r} has unsupported type "
                f"{type(current).__name__}"
            ),
        )
        return SymbolicTerm(None, reason="unsupported structured input leaf")

    return visit(value, ()), symbols, paths


def concrete_at_path(inputs: Mapping[str, Any], input_path: InputPath) -> Any:
    root, path = input_path
    value = inputs[root]
    for component in path:
        if _is_attribute_component(component):
            value = getattr(value, component[1])
        elif _is_set_component(component):
            value = sorted(value, key=repr)[component[1]]
        else:
            value = value[component]
    return value


def materialize_candidate(
    base_inputs: Mapping[str, Any],
    flat_candidate: Mapping[str, Any],
    input_paths: Mapping[str, InputPath],
) -> Dict[str, Any]:
    """Rebuild nested Python inputs from a flattened solver model."""
    result = copy.deepcopy(dict(base_inputs))
    root_set_replacements: Dict[str, Dict[int, Any]] = {}
    for symbol_name, leaf_value in flat_candidate.items():
        root, path = input_paths.get(symbol_name, (symbol_name, ()))
        if len(path) == 1 and _is_set_component(path[0]):
            root_set_replacements.setdefault(root, {})[path[0][1]] = leaf_value

    for symbol_name, leaf_value in flat_candidate.items():
        root, path = input_paths.get(symbol_name, (symbol_name, ()))
        if len(path) == 1 and _is_set_component(path[0]):
            continue
        concrete_leaf = concrete_at_path(base_inputs, (root, path))
        leaf_value = _coerce_leaf(concrete_leaf, leaf_value)
        if not path:
            result[root] = leaf_value
        else:
            result[root] = _replace_path(result[root], path, leaf_value)
    for root, replacements in root_set_replacements.items():
        values = sorted(base_inputs[root], key=repr)
        for index, replacement in replacements.items():
            values[index] = replacement
        result[root] = set(values)
    return result


def _replace_path(container: Any, path: Tuple[Any, ...], value: Any) -> Any:
    component = path[0]
    if _is_attribute_component(component):
        copied = copy.deepcopy(container)
        attribute = component[1]
        if len(path) == 1:
            setattr(copied, attribute, value)
        else:
            setattr(
                copied,
                attribute,
                _replace_path(getattr(copied, attribute), path[1:], value),
            )
        return copied
    if _is_set_component(component):
        copied = set(container)
        original = sorted(copied, key=repr)[component[1]]
        if len(path) == 1:
            replacement = value
        else:
            replacement = _replace_path(original, path[1:], value)
        copied.remove(original)
        copied.add(replacement)
        return copied
    if len(path) == 1:
        if isinstance(container, tuple):
            mutable = list(container)
            mutable[component] = value
            return tuple(mutable)
        copied = copy.deepcopy(container)
        copied[component] = value
        return copied

    child = _replace_path(container[component], path[1:], value)
    if isinstance(container, tuple):
        mutable = list(container)
        mutable[component] = child
        return tuple(mutable)
    copied = copy.deepcopy(container)
    copied[component] = child
    return copied


def _is_attribute_component(component: Any) -> bool:
    return (
        isinstance(component, tuple)
        and len(component) == 2
        and component[0] == "@attr"
        and isinstance(component[1], str)
    )


def _is_set_component(component: Any) -> bool:
    return (
        isinstance(component, tuple)
        and len(component) == 2
        and component[0] == "@set"
        and isinstance(component[1], int)
    )


def _coerce_leaf(original: Any, value: Any) -> Any:
    if isinstance(original, bytes) and isinstance(value, str):
        return value.encode("latin-1")
    if isinstance(original, bytearray) and isinstance(value, (bytes, str)):
        return bytearray(value if isinstance(value, bytes) else value.encode("latin-1"))
    return value


def expand_input_shapes(
    seeds: Sequence[Mapping[str, Any]],
    max_variants: int = 32,
) -> list[Dict[str, Any]]:
    """Generate bounded concrete shape variants before symbolic exploration."""
    if max_variants <= 0:
        raise ValueError("max_variants must be positive")
    variants: list[Dict[str, Any]] = []
    seen = set()
    for seed in seeds:
        candidates = [copy.deepcopy(dict(seed))]
        for name, value in seed.items():
            expanded = _shape_variants(value)
            if len(expanded) <= 1:
                continue
            next_candidates = []
            for candidate in candidates:
                for replacement in expanded:
                    updated = copy.deepcopy(candidate)
                    updated[name] = replacement
                    next_candidates.append(updated)
                    if len(next_candidates) >= max_variants:
                        break
                if len(next_candidates) >= max_variants:
                    break
            candidates = next_candidates
        for candidate in candidates:
            key = repr(candidate)
            if key not in seen:
                seen.add(key)
                variants.append(candidate)
            if len(variants) >= max_variants:
                return variants
    return variants


def _shape_variants(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, bytearray, bytes)):
        items = list(value)
        filler = items[-1] if items else 0
        lengths = sorted({0, 1, len(items), len(items) + 1})
        variants = []
        for length in lengths:
            expanded = (items + [filler] * max(0, length - len(items)))[:length]
            if isinstance(value, tuple):
                variants.append(tuple(expanded))
            elif isinstance(value, bytes):
                variants.append(bytes(expanded))
            elif isinstance(value, bytearray):
                variants.append(bytearray(expanded))
            else:
                variants.append(expanded)
        return variants
    if isinstance(value, dict):
        keys = list(value)
        return [
            {},
            copy.deepcopy(value),
            {key: copy.deepcopy(value[key]) for key in keys[:1]},
        ]
    if isinstance(value, set):
        ordered = sorted(value, key=repr)
        return [set(), set(ordered[:1]), set(value)]
    return [copy.deepcopy(value)]
