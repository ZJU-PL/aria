"""Thread-safe extension registry for symbolic function and method models."""

from __future__ import annotations

from threading import RLock
from typing import Callable, Dict, Optional, Sequence

from .models import SymbolicTerm

FunctionModel = Callable[[Sequence[SymbolicTerm]], Optional[SymbolicTerm]]
MethodModel = Callable[[SymbolicTerm, Sequence[SymbolicTerm]], Optional[SymbolicTerm]]


class SymbolicModelRegistry:
    """Resolve symbolic semantics without coupling them to instrumentation."""

    def __init__(self) -> None:
        self._functions: Dict[str, FunctionModel] = {}
        self._methods: Dict[str, MethodModel] = {}
        self._lock = RLock()

    def register_function(
        self,
        name: str,
        model: FunctionModel,
        *,
        replace: bool = False,
    ) -> None:
        if not name:
            raise ValueError("function model name cannot be empty")
        with self._lock:
            if name in self._functions and not replace:
                raise ValueError(
                    f"a symbolic function model is already registered for {name!r}"
                )
            self._functions[name] = model

    def register_method(
        self,
        name: str,
        model: MethodModel,
        *,
        replace: bool = False,
    ) -> None:
        if not name:
            raise ValueError("method model name cannot be empty")
        with self._lock:
            if name in self._methods and not replace:
                raise ValueError(
                    f"a symbolic method model is already registered for {name!r}"
                )
            self._methods[name] = model

    def function_model(self, name: str) -> Optional[FunctionModel]:
        with self._lock:
            return self._functions.get(name)

    def method_model(self, name: str) -> Optional[MethodModel]:
        with self._lock:
            return self._methods.get(name)

    def clone(self) -> "SymbolicModelRegistry":
        clone = SymbolicModelRegistry()
        with self._lock:
            clone._functions.update(self._functions)
            clone._methods.update(self._methods)
        return clone

    @property
    def function_names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._functions))

    @property
    def method_names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._methods))


DEFAULT_MODEL_REGISTRY = SymbolicModelRegistry()


def register_function_model(
    name: str,
    model: FunctionModel,
    *,
    replace: bool = False,
) -> None:
    DEFAULT_MODEL_REGISTRY.register_function(name, model, replace=replace)


def register_method_model(
    name: str,
    model: MethodModel,
    *,
    replace: bool = False,
) -> None:
    DEFAULT_MODEL_REGISTRY.register_method(name, model, replace=replace)
