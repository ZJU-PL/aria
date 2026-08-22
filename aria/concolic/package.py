"""Restorable package instrumentation for interprocedural campaigns."""

from __future__ import annotations

import builtins
import fnmatch
import importlib
import inspect
import pkgutil
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

from .instrumentation import (
    InstrumentationError,
    InstrumentedCallable,
    instrument_callable,
)

_PACKAGE_PATCH_LOCK = RLock()


@dataclass
class PackageInstrumentationReport:
    packages: Tuple[str, ...]
    modules: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packages": list(self.packages),
            "modules": list(self.modules),
            "functions": list(self.functions),
            "skipped": dict(self.skipped),
        }


@dataclass
class _Patch:
    owner: Any
    attribute: str
    replacement: Any


class PackageInstrumentor:
    """Compile package functions and patch them only during target execution."""

    def __init__(
        self,
        packages: Sequence[str],
        include: Sequence[str] = (),
        omit: Sequence[str] = (),
        import_submodules: bool = True,
    ) -> None:
        if not packages:
            raise ValueError("at least one instrumentation package is required")
        self.packages = tuple(dict.fromkeys(packages))
        self.include = tuple(include)
        self.omit = tuple(omit)
        self.import_submodules = import_submodules
        self.report = PackageInstrumentationReport(self.packages)
        self._compiled: Dict[Any, InstrumentedCallable] = {}
        self._patches: Dict[Tuple[int, str], _Patch] = {}
        self._instrumented_modules: set[str] = set()
        self._active_depth = 0
        self._active_originals: List[Tuple[Any, str, Any]] = []
        self._import_guard = False

    def instrument(self, target: Any) -> InstrumentedCallable:
        self._import_packages()
        self._instrument_modules(self._selected_modules())
        target_function = target.__func__ if inspect.ismethod(target) else target
        instrumented = self._compiled.get(target_function)
        if instrumented is None:
            instrumented = instrument_callable(target)
            self._rewire_private_namespace(instrumented)
        if inspect.ismethod(target) and target.__self__ is not None:
            instrumented = replace(
                instrumented,
                callable=types.MethodType(instrumented.callable, target.__self__),
            )
        return instrumented

    @contextmanager
    def activated(self) -> Iterator[None]:
        """Apply patches reentrantly and restore exact prior state afterward."""
        with _PACKAGE_PATCH_LOCK:
            if self._active_depth:
                self._active_depth += 1
                try:
                    yield
                finally:
                    self._active_depth -= 1
                return
            self._active_depth = 1
            self._active_originals = []
            original_import = builtins.__import__
            try:
                for patch in self._patches.values():
                    self._apply_patch(patch)
                builtins.__import__ = self._import_hook(original_import)
                yield
            finally:
                builtins.__import__ = original_import
                for owner, attribute, original in reversed(self._active_originals):
                    setattr(owner, attribute, original)
                self._active_originals = []
                self._active_depth = 0

    def _import_hook(self, original_import):
        def hooked(name, globals=None, locals=None, fromlist=(), level=0):
            result = original_import(name, globals, locals, fromlist, level)
            if self._import_guard:
                return result
            self._import_guard = True
            try:
                self._instrument_modules(self._selected_modules())
            finally:
                self._import_guard = False
            return result

        return hooked

    def _import_packages(self) -> None:
        for package_name in self.packages:
            try:
                package = importlib.import_module(package_name)
            except Exception as exc:
                self.report.skipped[package_name] = f"package import failed: {exc}"
                continue
            if not self.import_submodules or not hasattr(package, "__path__"):
                continue
            prefix = f"{package.__name__}."
            for module_info in pkgutil.walk_packages(package.__path__, prefix):
                if not self._allowed(module_info.name):
                    continue
                try:
                    importlib.import_module(module_info.name)
                except Exception as exc:
                    self.report.skipped[module_info.name] = (
                        f"submodule import failed: {exc}"
                    )

    def _selected_modules(self) -> List[types.ModuleType]:
        modules = []
        for name, module in tuple(sys.modules.items()):
            if module is None or not any(
                name == package or name.startswith(f"{package}.")
                for package in self.packages
            ):
                continue
            spec = getattr(module, "__spec__", None)
            if spec is not None and getattr(spec, "_initializing", False):
                continue
            if self._allowed(name):
                modules.append(module)
        return modules

    def _instrument_modules(self, modules: Sequence[types.ModuleType]) -> None:
        new_modules = [
            module
            for module in modules
            if module.__name__ not in self._instrumented_modules
        ]
        if not new_modules:
            return
        for module in new_modules:
            self._instrumented_modules.add(module.__name__)
            if module.__name__ not in self.report.modules:
                self.report.modules.append(module.__name__)
        candidates = list(self._candidates(new_modules))
        for _, _, _, function in candidates:
            identifier = f"{function.__module__}.{function.__qualname__}"
            if not self._allowed(identifier) or function in self._compiled:
                continue
            try:
                self._compiled[function] = instrument_callable(function)
                self.report.functions.append(identifier)
            except (InstrumentationError, SyntaxError, ValueError) as exc:
                self.report.skipped[identifier] = str(exc)
        self._record_candidate_patches(candidates)
        self._record_module_aliases(modules)
        for instrumented in self._compiled.values():
            self._rewire_private_namespace(instrumented)

    def _candidates(
        self, modules: Sequence[types.ModuleType]
    ) -> Iterable[Tuple[Any, str, str, Any]]:
        for module in modules:
            for name, value in tuple(vars(module).items()):
                if inspect.isfunction(value) and value.__module__ == module.__name__:
                    yield module, name, "function", value
                elif inspect.isclass(value) and value.__module__ == module.__name__:
                    yield from self._class_candidates(value)

    def _class_candidates(self, owner: type):
        for name, descriptor in tuple(vars(owner).items()):
            if inspect.isfunction(descriptor):
                yield owner, name, "function", descriptor
            elif isinstance(descriptor, staticmethod):
                yield owner, name, "staticmethod", descriptor.__func__
            elif isinstance(descriptor, classmethod):
                yield owner, name, "classmethod", descriptor.__func__
            elif isinstance(descriptor, property):
                for kind, function in (
                    ("property_get", descriptor.fget),
                    ("property_set", descriptor.fset),
                    ("property_del", descriptor.fdel),
                ):
                    if function is not None:
                        yield owner, name, kind, function

    def _record_candidate_patches(self, candidates) -> None:
        properties: Dict[Tuple[Any, str], Dict[str, Any]] = {}
        for owner, attribute, kind, original in candidates:
            instrumented = self._compiled.get(original)
            if instrumented is None:
                continue
            if kind.startswith("property_"):
                properties.setdefault((owner, attribute), {})[
                    kind
                ] = instrumented.callable
                continue
            replacement = instrumented.callable
            if kind == "staticmethod":
                replacement = staticmethod(replacement)
            elif kind == "classmethod":
                replacement = classmethod(replacement)
            self._record_patch(owner, attribute, replacement)
        for (owner, attribute), replacements in properties.items():
            original = inspect.getattr_static(owner, attribute)
            self._record_patch(
                owner,
                attribute,
                property(
                    replacements.get("property_get", original.fget),
                    replacements.get("property_set", original.fset),
                    replacements.get("property_del", original.fdel),
                    original.__doc__,
                ),
            )

    def _record_module_aliases(self, modules: Sequence[types.ModuleType]) -> None:
        replacements = {
            id(original): instrumented.callable
            for original, instrumented in self._compiled.items()
        }
        for module in modules:
            for name, value in tuple(vars(module).items()):
                replacement = replacements.get(id(value))
                if replacement is not None:
                    self._record_patch(module, name, replacement)

    def _rewire_private_namespace(self, instrumented: InstrumentedCallable) -> None:
        replacements = {
            id(original): compiled.callable
            for original, compiled in self._compiled.items()
        }
        for name, value in tuple(instrumented.namespace.items()):
            replacement = replacements.get(id(value))
            if replacement is not None:
                instrumented.namespace[name] = replacement

    def _record_patch(self, owner: Any, attribute: str, replacement: Any) -> None:
        patch = _Patch(owner, attribute, replacement)
        self._patches[(id(owner), attribute)] = patch
        if self._active_depth:
            self._apply_patch(patch)

    def _apply_patch(self, patch: _Patch) -> None:
        key = (id(patch.owner), patch.attribute)
        if not any(
            id(owner) == key[0] and attribute == key[1]
            for owner, attribute, _ in self._active_originals
        ):
            self._active_originals.append(
                (
                    patch.owner,
                    patch.attribute,
                    inspect.getattr_static(patch.owner, patch.attribute),
                )
            )
        setattr(patch.owner, patch.attribute, patch.replacement)

    def _allowed(self, identifier: str) -> bool:
        if self.include and not any(
            fnmatch.fnmatch(identifier, pattern) for pattern in self.include
        ):
            return False
        return not any(fnmatch.fnmatch(identifier, pattern) for pattern in self.omit)
