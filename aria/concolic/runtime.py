"""Execution-local runtime hooks used by instrumented Python code."""

from __future__ import annotations

import copy
import inspect
import operator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import z3

from .models import (
    BranchObservation,
    Diagnostic,
    DiagnosticCode,
    DiagnosticSeverity,
    SourceLocation,
    StructuredValue,
    SymbolicTerm,
)
from .locking import z3_global_lock
from .symbolic import (
    Descriptor,
    SymbolicEvaluator,
    concrete_to_z3,
    truthy,
)
from .structured import concrete_at_path, symbolize_input


@dataclass
class ShadowFrame:
    function_id: str
    values: Dict[str, SymbolicTerm]
    concrete_values: Dict[str, Any]
    return_term: Optional[SymbolicTerm] = None


@dataclass
class PendingCall:
    positional: Sequence[SymbolicTerm]
    keywords: Mapping[str, SymbolicTerm]
    result: Optional[SymbolicTerm] = None


@dataclass
class RuntimeSession:
    """Mutable state belonging to one logical concolic execution."""

    seed_inputs: Mapping[str, Any]
    frames: List[ShadowFrame] = field(default_factory=list)
    branches: List[BranchObservation] = field(default_factory=list)
    diagnostics: List[Diagnostic] = field(default_factory=list)
    path_constraints: List[z3.BoolRef] = field(default_factory=list)
    input_symbols: Dict[str, z3.ExprRef] = field(default_factory=dict)
    input_paths: Dict[str, tuple[str, tuple[Any, ...]]] = field(default_factory=dict)
    branch_occurrences: Dict[str, int] = field(default_factory=dict)
    current_location: Optional[SourceLocation] = None
    current_branch_id: Optional[str] = None
    pending_calls: List[PendingCall] = field(default_factory=list)
    call_results: Dict[str, SymbolicTerm] = field(default_factory=dict)

    def enter(
        self,
        function_id: str,
        arguments: Mapping[str, Any],
        input_specs: Sequence[Sequence[str]],
    ) -> int:
        parameter_kinds = {
            (item if isinstance(item, str) else item[0]): (
                "positional" if isinstance(item, str) else item[1]
            )
            for item in input_specs
        }
        values: Dict[str, SymbolicTerm] = {}
        if self.frames:
            if self.pending_calls:
                pending = self.pending_calls[-1]
                positional_index = 0
                for name, concrete_value in arguments.items():
                    kind = parameter_kinds.get(name)
                    if kind is None:
                        constant = concrete_to_z3(concrete_value)
                        values[name] = SymbolicTerm(
                            constant,
                            reason=(
                                None if constant is not None else "opaque environment"
                            ),
                        )
                    elif kind == "var_positional":
                        remaining = pending.positional[positional_index:]
                        values[name] = SymbolicTerm(
                            StructuredValue(
                                "tuple",
                                {index: term for index, term in enumerate(remaining)},
                            )
                        )
                        positional_index = len(pending.positional)
                    elif kind == "var_keyword":
                        values[name] = SymbolicTerm(
                            StructuredValue("dict", dict(pending.keywords))
                        )
                    elif name in pending.keywords:
                        values[name] = pending.keywords[name]
                    elif positional_index < len(pending.positional):
                        values[name] = pending.positional[positional_index]
                        positional_index += 1
                    else:
                        constant = concrete_to_z3(concrete_value)
                        values[name] = SymbolicTerm(
                            constant,
                            reason=None if constant is not None else "opaque default",
                        )
            else:
                self.diagnose(
                    DiagnosticCode.NESTED_SYMBOLIC_FRAME,
                    (
                        "a nested instrumented call was not routed through the "
                        "symbolic call bridge; its local values are opaque"
                    ),
                )
                values = {
                    name: SymbolicTerm(None, reason="nested symbolic frame")
                    for name in arguments
                }
        else:
            for name, concrete_value in arguments.items():
                term, symbols, paths = symbolize_input(
                    name,
                    concrete_value,
                    symbolic=name in self.seed_inputs,
                    diagnose=self.diagnose,
                )
                values[name] = term
                self.input_symbols.update(symbols)
                self.input_paths.update(paths)
        self.frames.append(ShadowFrame(function_id, values, dict(arguments)))
        return len(self.frames)

    def leave(self, frame_marker: int) -> None:
        if self.frames and frame_marker == len(self.frames):
            frame = self.frames.pop()
            if self.pending_calls:
                self.pending_calls[-1].result = frame.return_term

    @property
    def frame(self) -> Optional[ShadowFrame]:
        return self.frames[-1] if self.frames else None

    def evaluator(self) -> SymbolicEvaluator:
        shadow = self.frame.values if self.frame else {}
        return SymbolicEvaluator(
            shadow,
            self.diagnose,
            self.term_truth,
            self.call_results.get,
        )

    def term_truth(self, term: SymbolicTerm) -> Optional[bool]:
        """Evaluate a Boolean symbolic term under the current concrete seed."""
        condition = truthy(term)
        if condition.expression is None:
            return None
        solver = z3.Solver()
        for name, symbol in self.input_symbols.items():
            concrete_input = concrete_to_z3(
                concrete_at_path(self.seed_inputs, self.input_paths[name])
            )
            if concrete_input is not None:
                solver.add(symbol == concrete_input)
        solver.add(*condition.guards)
        solver.push()
        solver.add(condition.expression)
        can_be_true = solver.check() == z3.sat
        solver.pop()
        solver.add(z3.Not(condition.expression))
        can_be_false = solver.check() == z3.sat
        if can_be_true and not can_be_false:
            return True
        if can_be_false and not can_be_true:
            return False
        return None

    def diagnose(
        self,
        code: DiagnosticCode,
        message: str,
        severity: DiagnosticSeverity = DiagnosticSeverity.WARNING,
    ) -> None:
        diagnostic = Diagnostic(
            code=code,
            message=message,
            severity=severity,
            location=self.current_location,
            branch_id=self.current_branch_id,
        )
        if diagnostic not in self.diagnostics:
            self.diagnostics.append(diagnostic)

    def validate(self, term: SymbolicTerm, concrete: Any) -> bool:
        if term.expression is None:
            return False
        concrete_expr = concrete_to_z3(concrete)
        if concrete_expr is None or concrete_expr.sort() != term.expression.sort():
            return False
        solver = z3.Solver()
        for name, symbol in self.input_symbols.items():
            concrete_input = concrete_to_z3(
                concrete_at_path(self.seed_inputs, self.input_paths[name])
            )
            if concrete_input is not None:
                solver.add(symbol == concrete_input)
        for guard in term.guards:
            solver.push()
            solver.add(z3.Not(guard))
            guard_result = solver.check()
            solver.pop()
            if guard_result != z3.unsat:
                self.diagnose(
                    DiagnosticCode.DEFINEDNESS_MISMATCH,
                    (
                        f"definedness guard {guard} is not true for concrete "
                        f"value {concrete!r}"
                    ),
                    DiagnosticSeverity.ERROR,
                )
                return False
            solver.add(guard)
        solver.add(term.expression != concrete_expr)
        result = solver.check()
        if result == z3.unsat:
            return True
        if result == z3.sat:
            self.diagnose(
                DiagnosticCode.SYMBOLIC_CONCRETE_MISMATCH,
                (
                    f"symbolic value {term.expression} does not match concrete "
                    f"value {concrete!r}"
                ),
                DiagnosticSeverity.ERROR,
            )
        return False

    def record_branch(
        self,
        branch_id: str,
        outcome: bool,
        condition: SymbolicTerm,
        location: SourceLocation,
    ) -> None:
        self.current_location = location
        self.current_branch_id = branch_id
        try:
            predicate: Optional[z3.BoolRef] = None
            guards: Sequence[z3.BoolRef] = condition.guards
            if condition.expression is not None and self.validate(condition, outcome):
                predicate = condition.expression
            occurrence = self.branch_occurrences.get(branch_id, 0)
            self.branch_occurrences[branch_id] = occurrence + 1
            self.branches.append(
                BranchObservation(
                    branch_id=branch_id,
                    occurrence=occurrence,
                    taken=outcome,
                    location=location,
                    predicate=predicate,
                    guards=tuple(guards),
                    path_prefix=tuple(self.path_constraints),
                )
            )
            if predicate is not None:
                self.path_constraints.extend(guards)
                self.path_constraints.append(
                    predicate if outcome else z3.Not(predicate)
                )
        finally:
            self.current_location = None
            self.current_branch_id = None


_CURRENT_SESSION: ContextVar[Optional[RuntimeSession]] = ContextVar(
    "aria_concolic_runtime_session", default=None
)


@contextmanager
def session_scope(seed_inputs: Mapping[str, Any]) -> Iterator[RuntimeSession]:
    """Install a runtime session for the current thread/task context."""
    with z3_global_lock():
        session = RuntimeSession(copy.deepcopy(dict(seed_inputs)))
        token = _CURRENT_SESSION.set(session)
        try:
            yield session
        finally:
            _CURRENT_SESSION.reset(token)


class RuntimeHooks:
    """Stateless facade injected into transformed function globals."""

    def enter(
        self, function_id: str, arguments: Mapping[str, Any], input_names: Sequence[str]
    ) -> int:
        session = _CURRENT_SESSION.get()
        if session is None:
            return 0
        return session.enter(function_id, arguments, input_names)

    def leave(self, marker: int) -> None:
        session = _CURRENT_SESSION.get()
        if session is not None:
            session.leave(marker)

    def assign(self, name: str, concrete: Any, descriptor: Descriptor) -> Any:
        session = _CURRENT_SESSION.get()
        if session is None or session.frame is None:
            return concrete
        term = session.evaluator().evaluate(descriptor)
        if term.expression is not None and not session.validate(term, concrete):
            term = SymbolicTerm(None, term.guards, "symbolic/concrete mismatch")
        session.frame.values[name] = term
        session.frame.concrete_values[name] = concrete
        return concrete

    def assign_many(
        self, names: Sequence[str], concrete: Any, descriptor: Descriptor
    ) -> Any:
        session = _CURRENT_SESSION.get()
        if session is None or session.frame is None:
            return concrete
        term = session.evaluator().evaluate(descriptor)
        if term.expression is not None and not session.validate(term, concrete):
            term = SymbolicTerm(None, term.guards, "symbolic/concrete mismatch")
        for name in names:
            session.frame.values[name] = term
            session.frame.concrete_values[name] = concrete
        return concrete

    def unpack_assign(
        self,
        names: Sequence[str],
        concrete: Any,
        descriptor: Descriptor,
    ) -> Any:
        session = _CURRENT_SESSION.get()
        if session is None or session.frame is None:
            return concrete
        term = session.evaluator().evaluate(descriptor)
        structured = term.expression
        for index, name in enumerate(names):
            child = (
                structured.children.get(index)
                if isinstance(structured, StructuredValue)
                else None
            )
            session.frame.values[name] = child or SymbolicTerm(
                None, reason="unpack source is opaque"
            )
            session.frame.concrete_values[name] = concrete[index]
        return concrete

    def prepare(self, descriptor: Descriptor) -> SymbolicTerm:
        session = _CURRENT_SESSION.get()
        if session is None or session.frame is None:
            return SymbolicTerm(None, reason="no active concolic session")
        return session.evaluator().evaluate(descriptor)

    def commit(self, name: str, concrete: Any, term: SymbolicTerm) -> None:
        session = _CURRENT_SESSION.get()
        if session is None or session.frame is None:
            return
        if term.expression is not None and not session.validate(term, concrete):
            term = SymbolicTerm(None, term.guards, "symbolic/concrete mismatch")
        session.frame.values[name] = term
        session.frame.concrete_values[name] = concrete

    def branch(
        self,
        branch_id: str,
        concrete: Any,
        descriptor: Descriptor,
        location_data: Tuple[Any, ...],
    ) -> bool:
        outcome = bool(concrete)
        session = _CURRENT_SESSION.get()
        if session is None or session.frame is None:
            return outcome

        location = SourceLocation(*location_data)
        condition = truthy(session.evaluator().evaluate(descriptor))
        session.record_branch(branch_id, outcome, condition, location)
        return outcome

    def range_iter(
        self,
        branch_id: str,
        concrete_range: range,
        descriptors: Sequence[Descriptor],
        location_data: Tuple[Any, ...],
        target_name: str,
    ) -> Iterator[int]:
        """Iterate a concrete range while tracing its symbolic loop condition."""
        session = _CURRENT_SESSION.get()
        if session is None or session.frame is None:
            yield from concrete_range
            return
        terms = [session.evaluator().evaluate(descriptor) for descriptor in descriptors]
        if len(terms) == 1:
            start = SymbolicTerm(z3.IntVal(0))
            stop = terms[0]
            step = SymbolicTerm(z3.IntVal(1))
        elif len(terms) == 2:
            start, stop = terms
            step = SymbolicTerm(z3.IntVal(1))
        else:
            start, stop, step = terms
        symbolic_available = all(
            term.expression is not None and z3.is_int(term.expression)
            for term in (start, stop, step)
        )
        guards = tuple(guard for term in (start, stop, step) for guard in term.guards)
        location = SourceLocation(*location_data)
        for index, value in enumerate(concrete_range):
            if symbolic_available:
                symbolic_value = start.expression + index * step.expression
                condition_expr = z3.If(
                    step.expression > 0,
                    symbolic_value < stop.expression,
                    symbolic_value > stop.expression,
                )
                condition = SymbolicTerm(
                    condition_expr, (*guards, step.expression != 0)
                )
                session.frame.values[target_name] = SymbolicTerm(
                    symbolic_value, condition.guards
                )
            else:
                condition = SymbolicTerm(
                    None, guards, "range expression is not symbolic"
                )
                session.frame.values[target_name] = SymbolicTerm(
                    None, reason="opaque range target"
                )
            session.record_branch(branch_id, True, condition, location)
            yield value
        final_index = len(concrete_range)
        if symbolic_available:
            symbolic_value = start.expression + final_index * step.expression
            condition_expr = z3.If(
                step.expression > 0,
                symbolic_value < stop.expression,
                symbolic_value > stop.expression,
            )
            condition = SymbolicTerm(condition_expr, (*guards, step.expression != 0))
        else:
            condition = SymbolicTerm(None, guards, "range expression is not symbolic")
        session.record_branch(branch_id, False, condition, location)

    def iterate(
        self,
        branch_id: str,
        concrete_iterable: Any,
        descriptor: Descriptor,
        location_data: Tuple[Any, ...],
        target_name: str,
    ) -> Iterator[Any]:
        session = _CURRENT_SESSION.get()
        if session is None or session.frame is None:
            yield from concrete_iterable
            return
        iterable = session.evaluator().evaluate(descriptor)
        location = SourceLocation(*location_data)
        count = 0
        for index, value in enumerate(concrete_iterable):
            count = index + 1
            term = SymbolicTerm(None, reason="opaque iterable element")
            condition = SymbolicTerm(None, reason="opaque iterable length")
            if z3.is_string(iterable.expression):
                term = SymbolicTerm(z3.SubString(iterable.expression, index, 1))
                condition = SymbolicTerm(
                    z3.IntVal(index) < z3.Length(iterable.expression)
                )
            elif isinstance(iterable.expression, StructuredValue):
                child = iterable.expression.children.get(index)
                if child is not None:
                    term = child
                condition = SymbolicTerm(
                    z3.IntVal(index) < len(iterable.expression.children)
                )
            session.frame.values[target_name] = term
            session.frame.concrete_values[target_name] = value
            session.record_branch(branch_id, True, condition, location)
            yield value
        final_condition = SymbolicTerm(None, reason="opaque iterable length")
        if z3.is_string(iterable.expression):
            final_condition = SymbolicTerm(
                z3.IntVal(count) < z3.Length(iterable.expression)
            )
        elif isinstance(iterable.expression, StructuredValue):
            final_condition = SymbolicTerm(
                z3.IntVal(count) < len(iterable.expression.children)
            )
        session.record_branch(branch_id, False, final_condition, location)

    def returning(self, concrete: Any, descriptor: Descriptor) -> Any:
        session = _CURRENT_SESSION.get()
        if session is None or session.frame is None:
            return concrete
        term = session.evaluator().evaluate(descriptor)
        if (
            isinstance(term.expression, z3.ExprRef)
            and concrete_to_z3(concrete) is not None
            and not session.validate(term, concrete)
        ):
            term = SymbolicTerm(None, term.guards, "symbolic return mismatch")
        session.frame.return_term = term
        return concrete

    def interprocedural_call(
        self,
        call_id: str,
        function: Any,
        positional: Sequence[Any],
        keywords: Mapping[str, Any],
        positional_descriptors: Sequence[Descriptor],
        keyword_descriptors: Sequence[Sequence[Any]],
    ) -> Any:
        """Invoke a call while propagating shadow arguments and its return."""
        session = _CURRENT_SESSION.get()
        if session is None or session.frame is None:
            return function(*positional, **keywords)
        pending = self._expanded_pending_call(
            session,
            function,
            positional_descriptors,
            keyword_descriptors,
        )
        session.pending_calls.append(pending)
        try:
            concrete = function(*positional, **keywords)
            if pending.result is not None:
                session.call_results[call_id] = pending.result
            else:
                modeled_result = self._apply_registered_function(
                    session, function, pending, concrete
                )
                if modeled_result is not None:
                    session.call_results[call_id] = modeled_result
                else:
                    mutation_result = self._apply_mutating_method(
                        session,
                        function,
                        positional,
                        pending,
                    )
                    if mutation_result is not None:
                        session.call_results[call_id] = mutation_result
                    else:
                        session.call_results.pop(call_id, None)
            return concrete
        finally:
            session.pending_calls.pop()

    def getattr(
        self,
        attribute_id: str,
        owner: Any,
        attribute: str,
        owner_descriptor: Descriptor,
    ) -> Any:
        session = _CURRENT_SESSION.get()
        if session is None or session.frame is None:
            return getattr(owner, attribute)
        owner_term = session.evaluator().evaluate(owner_descriptor)
        pending = PendingCall((owner_term,), {})
        session.pending_calls.append(pending)
        try:
            concrete = getattr(owner, attribute)
            if pending.result is not None:
                session.call_results[attribute_id] = pending.result
            else:
                session.call_results.pop(attribute_id, None)
            return concrete
        finally:
            session.pending_calls.pop()

    @staticmethod
    def _expanded_pending_call(
        session: RuntimeSession,
        function: Any,
        positional_descriptors: Sequence[Descriptor],
        keyword_descriptors: Sequence[Sequence[Any]],
    ) -> PendingCall:
        evaluator = session.evaluator()
        positional = []
        for kind, descriptor in positional_descriptors:
            term = evaluator.evaluate(descriptor)
            if kind == "star" and isinstance(term.expression, StructuredValue):
                structured = term.expression
                if structured.kind in {"list", "tuple"}:
                    positional.extend(
                        structured.children[index]
                        for index in sorted(structured.children)
                    )
                    continue
            positional.append(term)

        keywords: Dict[str, SymbolicTerm] = {}
        for name, descriptor in keyword_descriptors:
            term = evaluator.evaluate(descriptor)
            if name is None and isinstance(term.expression, StructuredValue):
                structured = term.expression
                if structured.kind == "dict":
                    keywords.update(structured.children)
                    continue
            if name is not None:
                keywords[name] = term

        receiver = getattr(function, "__self__", None)
        should_prepend = inspect.ismethod(function)
        if (
            receiver is None
            and not inspect.isfunction(function)
            and not inspect.isbuiltin(function)
        ):
            if not inspect.isclass(function) and callable(function):
                receiver = function
                should_prepend = True
        if should_prepend:
            structured = _find_structured_value(session.frame, receiver)
            receiver_term = (
                SymbolicTerm(structured)
                if structured is not None
                else SymbolicTerm(None)
            )
            positional.insert(0, receiver_term)
        return PendingCall(tuple(positional), keywords)

    def setitem(
        self,
        owner: Any,
        key: Any,
        concrete_value: Any,
        value_descriptor: Descriptor,
    ) -> None:
        owner[key] = concrete_value
        session = _CURRENT_SESSION.get()
        if session is None or session.frame is None:
            return
        structured = _find_structured_value(session.frame, owner)
        if structured is None or not isinstance(structured.children, dict):
            session.diagnose(
                DiagnosticCode.UNSUPPORTED_OPERATION,
                "symbolic indexed assignment requires a tracked container",
            )
            return
        symbolic_value = session.evaluator().evaluate(value_descriptor)
        normalized = _normalized_container_key(structured, key)
        structured.children[normalized] = symbolic_value

    def delitem(self, owner: Any, key: Any) -> None:
        del owner[key]
        session = _CURRENT_SESSION.get()
        if session is None or session.frame is None:
            return
        structured = _find_structured_value(session.frame, owner)
        if structured is None or not isinstance(structured.children, dict):
            session.diagnose(
                DiagnosticCode.UNSUPPORTED_OPERATION,
                "symbolic indexed deletion requires a tracked container",
            )
            return
        normalized = _normalized_container_key(structured, key)
        if structured.kind in {"list", "bytearray"}:
            _sequence_delete(structured, normalized)
        else:
            structured.children.pop(normalized, None)

    def setattr(
        self,
        owner: Any,
        attribute: str,
        concrete_value: Any,
        value_descriptor: Descriptor,
    ) -> None:
        session = _CURRENT_SESSION.get()
        if session is None or session.frame is None:
            setattr(owner, attribute, concrete_value)
            return
        structured = _find_structured_value(session.frame, owner)
        owner_term = (
            SymbolicTerm(structured) if structured is not None else SymbolicTerm(None)
        )
        value_term = session.evaluator().evaluate(value_descriptor)
        pending = PendingCall((owner_term, value_term), {})
        session.pending_calls.append(pending)
        try:
            setattr(owner, attribute, concrete_value)
        finally:
            session.pending_calls.pop()
        if structured is None or structured.kind not in {"object", "dataclass"}:
            session.diagnose(
                DiagnosticCode.UNSUPPORTED_OPERATION,
                "symbolic attribute assignment requires a tracked object",
            )
            return
        assert isinstance(structured.children, dict)
        class_descriptor = inspect.getattr_static(type(owner), attribute, None)
        if not isinstance(class_descriptor, property):
            structured.children[attribute] = value_term

    def augattr(
        self,
        owner: Any,
        attribute: str,
        concrete_rhs: Any,
        operation: str,
        descriptor: Descriptor,
    ) -> None:
        operations = {
            "+": operator.iadd,
            "-": operator.isub,
            "*": operator.imul,
            "//": operator.ifloordiv,
            "%": operator.imod,
            "&": operator.iand,
            "|": operator.ior,
            "^": operator.ixor,
            "<<": operator.ilshift,
            ">>": operator.irshift,
        }
        symbolic = None
        session = _CURRENT_SESSION.get()
        if session is not None and session.frame is not None:
            symbolic = session.evaluator().evaluate(descriptor)
        concrete = operations[operation](getattr(owner, attribute), concrete_rhs)
        setattr(owner, attribute, concrete)
        if session is None or session.frame is None or symbolic is None:
            return
        structured = _find_structured_value(session.frame, owner)
        if structured is None or structured.kind not in {"object", "dataclass"}:
            session.diagnose(
                DiagnosticCode.UNSUPPORTED_OPERATION,
                "symbolic augmented attribute assignment requires a tracked object",
            )
            return
        assert isinstance(structured.children, dict)
        if (
            isinstance(symbolic.expression, z3.ExprRef)
            and concrete_to_z3(concrete) is not None
            and not session.validate(symbolic, concrete)
        ):
            symbolic = SymbolicTerm(None, symbolic.guards, "attribute mismatch")
        structured.children[attribute] = symbolic

    @staticmethod
    def _apply_registered_function(
        session: RuntimeSession,
        function: Any,
        pending: PendingCall,
        concrete_result: Any,
    ) -> Optional[SymbolicTerm]:
        if function is isinstance:
            return SymbolicTerm(z3.BoolVal(bool(concrete_result)))
        registry = session.evaluator().registry
        module = getattr(function, "__module__", "")
        qualified_name = getattr(function, "__qualname__", "")
        simple_name = getattr(function, "__name__", "")
        names = [
            name
            for name in (
                f"{module}.{qualified_name}" if module and qualified_name else "",
                f"{module}.{simple_name}" if module and simple_name else "",
                simple_name,
            )
            if name
        ]
        for name in names:
            model = registry.function_model(name)
            if model is None:
                continue
            result = model(pending.positional)
            if result is not None:
                return result
        return None

    @staticmethod
    def _apply_mutating_method(
        session: RuntimeSession,
        function: Any,
        positional: Sequence[Any],
        pending: PendingCall,
    ) -> Optional[SymbolicTerm]:
        receiver = getattr(function, "__self__", None)
        method = getattr(function, "__name__", "")
        if receiver is None:
            return None
        structured = _find_structured_value(session.frame, receiver)
        if structured is None or not isinstance(structured.children, dict):
            return None
        children = structured.children

        if structured.kind in {"list", "bytearray"}:
            if method == "append" and pending.positional:
                children[len(children)] = pending.positional[0]
            elif method == "extend" and pending.positional:
                extension = pending.positional[0].expression
                if isinstance(extension, StructuredValue):
                    for index in sorted(extension.children):
                        children[len(children)] = extension.children[index]
            elif method == "insert" and len(pending.positional) >= 2:
                index = _normalize_insert_index(int(positional[0]), len(children))
                _sequence_insert(structured, index, pending.positional[1])
            elif method == "pop":
                index = int(positional[0]) if positional else -1
                normalized = _normalized_container_key(structured, index)
                return _sequence_delete(structured, normalized)
            elif method == "clear":
                children.clear()
            else:
                return None
            return None

        if structured.kind == "dict":
            if method == "update":
                if pending.positional:
                    update = pending.positional[0].expression
                    if isinstance(update, StructuredValue) and update.kind == "dict":
                        children.update(update.children)
                children.update(pending.keywords)
                return None
            if method == "setdefault" and pending.positional:
                key = positional[0]
                if key in children:
                    return children[key]
                default = (
                    pending.positional[1]
                    if len(pending.positional) > 1
                    else SymbolicTerm(None, reason="None default is opaque")
                )
                children[key] = default
                return default
            if method == "pop" and positional:
                return children.pop(positional[0], None)
            if method == "clear":
                children.clear()
                return None
        return None


def _find_structured_value(
    frame: ShadowFrame, receiver: Any
) -> Optional[StructuredValue]:
    def walk(concrete: Any, symbolic: SymbolicTerm) -> Optional[StructuredValue]:
        if concrete is receiver and isinstance(symbolic.expression, StructuredValue):
            return symbolic.expression
        structured = symbolic.expression
        if not isinstance(structured, StructuredValue):
            return None
        for key, child in structured.children.items():
            try:
                if structured.kind in {"object", "dataclass"}:
                    concrete_child = getattr(concrete, key)
                else:
                    concrete_child = concrete[key]
            except (AttributeError, KeyError, IndexError, TypeError):
                continue
            found = walk(concrete_child, child)
            if found is not None:
                return found
        return None

    for name, concrete in frame.concrete_values.items():
        symbolic = frame.values.get(name)
        if symbolic is None:
            continue
        found = walk(concrete, symbolic)
        if found is not None:
            return found
    return None


def _normalized_container_key(structured: StructuredValue, key: Any) -> Any:
    if structured.kind in {"list", "bytearray"} and isinstance(key, int) and key < 0:
        return len(structured.children) + key
    return key


def _normalize_insert_index(index: int, length: int) -> int:
    if index < 0:
        return max(0, length + index)
    return min(index, length)


def _sequence_insert(
    structured: StructuredValue, index: int, value: SymbolicTerm
) -> None:
    children = structured.children
    assert isinstance(children, dict)
    for current in range(len(children), index, -1):
        children[current] = children[current - 1]
    children[index] = value


def _sequence_delete(structured: StructuredValue, index: int) -> Optional[SymbolicTerm]:
    children = structured.children
    assert isinstance(children, dict)
    removed = children.get(index)
    for current in range(index, len(children) - 1):
        children[current] = children[current + 1]
    children.pop(len(children) - 1, None)
    return removed


RUNTIME_HOOKS = RuntimeHooks()
