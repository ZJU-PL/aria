"""Signature-aware invocation helpers shared by execution and artifacts."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Mapping, Tuple


def build_call_arguments(
    signature: inspect.Signature,
    inputs: Mapping[str, Any],
) -> Tuple[List[Any], Dict[str, Any]]:
    """Map named campaign inputs to Python's full parameter-kind semantics."""
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        and name not in inputs
    ]
    if missing:
        raise TypeError(f"missing concolic inputs: {', '.join(missing)}")
    unknown = set(inputs) - set(signature.parameters)
    if unknown:
        raise TypeError(f"unknown concolic inputs: {', '.join(sorted(unknown))}")

    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            if name in inputs:
                args.append(inputs[name])
            elif parameter.default is not inspect.Parameter.empty:
                args.append(parameter.default)
        elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            args.extend(inputs.get(name, ()))
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            kwargs.update(inputs.get(name, {}))
        elif name in inputs:
            kwargs[name] = inputs[name]
    return args, kwargs


async def invoke_target_async(
    target: Callable[..., Any],
    inputs: Mapping[str, Any],
    max_yields: int = 1_000,
) -> Any:
    """Invoke any supported target shape and drive lazy results to completion."""
    if max_yields <= 0:
        raise ValueError("max_yields must be positive")
    args, kwargs = build_call_arguments(inspect.signature(target), inputs)
    result = target(*args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    if inspect.isasyncgen(result):
        values = []
        for _ in range(max_yields):
            try:
                values.append(await result.__anext__())
            except StopAsyncIteration:
                return values
        await result.aclose()
        return values
    if inspect.isgenerator(result):
        values = []
        for _ in range(max_yields):
            try:
                values.append(next(result))
            except StopIteration:
                return values
        result.close()
        return values
    return result


def invoke_target_sync(
    target: Callable[..., Any],
    inputs: Mapping[str, Any],
    max_yields: int = 1_000,
) -> Any:
    """Synchronous entry point for regression tests and command-line replay."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(invoke_target_async(target, inputs, max_yields))
    raise RuntimeError(
        "invoke_target_sync() cannot run inside an event loop; await "
        "invoke_target_async() instead"
    )
