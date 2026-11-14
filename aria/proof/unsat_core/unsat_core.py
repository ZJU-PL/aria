"""Unsat core computation interface."""

from __future__ import annotations

from enum import Enum
import importlib.util
import os
from typing import List, Set, Dict, Any, Optional, Callable, Union

try:
    from z3 import Bool, Not, Or, Solver, unsat
except ImportError:
    pass


class Algorithm(Enum):
    """Enumeration of available unsat core algorithms."""

    MARCO = "marco"
    MUSX = "musx"
    OPTUX = "optux"

    @classmethod
    def from_string(cls, name: str) -> "Algorithm":
        """Convert string to Algorithm enum."""
        name = name.lower()
        for alg in cls:
            if alg.value == name:
                return alg
        raise ValueError(f"Unknown algorithm: {name}")


class UnsatCoreResult:
    """Result of unsat core computation."""

    def __init__(
        self,
        cores: List[Set[int]],
        is_minimal: bool = False,
        stats: Optional[Dict[str, Any]] = None,
    ):
        self.cores = cores
        self.is_minimal = is_minimal
        self.stats = stats or {}

    def __str__(self) -> str:
        cores_str = "\n".join(
            [f"Core {i + 1}: {sorted(core)}" for i, core in enumerate(self.cores)]
        )
        minimal_str = "minimal" if self.is_minimal else "not necessarily minimal"
        return f"Found {len(self.cores)} {minimal_str} unsat cores:\n{cores_str}"


class UnsatCoreComputer:
    """Computer for unsat cores using various algorithms."""

    def __init__(self, algorithm: Union[str, Algorithm] = Algorithm.MARCO):
        if isinstance(algorithm, str):
            self.algorithm = Algorithm.from_string(algorithm)
        else:
            self.algorithm = algorithm
        self._load_algorithm_module()

    def _load_algorithm_module(self):
        """Load the algorithm module dynamically."""
        module_name = self.algorithm.value
        try:
            module_path = os.path.join(os.path.dirname(__file__), f"{module_name}.py")
            if os.path.exists(module_path):
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec and spec.loader:
                    self.module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(self.module)
                else:
                    raise ImportError(f"Failed to load {module_name} module")
            else:
                self.module = importlib.import_module(f"aria.proof.unsat_core.{module_name}")
        except ImportError as import_err:
            raise ImportError(
                f"Failed to import algorithm module {module_name}: {import_err}"
            ) from import_err

    def compute_unsat_core(
        self,
        constraints: List[Any],
        solver_factory: Callable[[], Any],
        timeout: Optional[int] = None,
        **kwargs,
    ) -> UnsatCoreResult:
        """Compute unsat core using the configured algorithm."""
        if self.algorithm == Algorithm.MARCO:
            return self._run_marco(constraints, solver_factory, timeout, **kwargs)
        if self.algorithm == Algorithm.MUSX:
            return self._run_musx(constraints, solver_factory, timeout, **kwargs)
        if self.algorithm == Algorithm.OPTUX:
            return self._run_optux(constraints, solver_factory, timeout, **kwargs)
        raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _run_marco(
        self,
        constraints: List[Any],
        solver_factory: Callable[[], Any],  # noqa: ARG002
        timeout: Optional[int] = None,  # noqa: ARG002
        max_cores: int = 1,
        **kwargs,
    ) -> UnsatCoreResult:  # noqa: ARG002
        """Run MARCO algorithm to compute unsat core."""
        z3_constraints = []
        for constraint in constraints:
            if isinstance(constraint, str):
                if constraint.startswith("not "):
                    var_name = constraint[4:].split()[0]
                    z3_constraints.append(Not(Bool(var_name)))
                elif " or " in constraint:
                    parts = constraint.split(" or ")
                    if len(parts) == 2:
                        left, right = parts[0].strip(), parts[1].strip()
                        left_expr = (
                            Not(Bool(left[4:]))
                            if left.startswith("not ")
                            else Bool(left)
                        )
                        right_expr = (
                            Not(Bool(right[4:]))
                            if right.startswith("not ")
                            else Bool(right)
                        )
                        z3_constraints.append(Or(left_expr, right_expr))
                    else:
                        z3_constraints.append(Bool(constraint))
                else:
                    z3_constraints.append(Bool(constraint))
            else:
                z3_constraints.append(constraint)

        csolver = self.module.SubsetSolver(z3_constraints)
        msolver = self.module.MapSolver(n=csolver.n)

        cores = []
        count = 0
        for orig, lits in self.module.enumerate_sets(csolver, msolver):
            if orig == "MUS":
                core_indices = set()
                for lit in lits:
                    if hasattr(lit, "children") and len(lit.children()) == 1:
                        var_name = str(lit.children()[0])
                    else:
                        var_name = str(lit)
                    for i, constraint in enumerate(constraints):
                        if var_name in str(constraint):
                            core_indices.add(i)
                            break
                cores.append(core_indices)
                count += 1
                if count >= max_cores:
                    break

        return UnsatCoreResult(cores=cores, is_minimal=True)

    def _make_checker(
        self,
        constraints: List[Any],
        solver_factory: Callable[[], Any],
        timeout: Optional[int],
    ) -> Callable[[List[int]], Any]:
        """Build a helper that checks satisfiability of a subset of constraints."""

        def _check(indices: List[int]):
            solver = solver_factory()
            if timeout is not None:
                solver.set("timeout", int(timeout * 1000))
            for idx in indices:
                solver.add(constraints[idx])
            return solver.check()

        return _check

    def _run_musx(
        self,
        constraints: List[Any],
        solver_factory: Callable[[], Any],
        timeout: Optional[int] = None,
        **kwargs,  # noqa: ARG002
    ) -> UnsatCoreResult:
        """Run a deletion-based MUS algorithm to compute a minimal unsat core.

        This mirrors the algorithm in ``aria.proof.unsat_core.musx`` (deletion
        based MUS extraction) but works directly on the given solver
        constraints, so arbitrary Z3 formulas are supported without a CNF
        conversion.
        """
        if not constraints:
            return UnsatCoreResult(cores=[set()], is_minimal=True)

        _check = self._make_checker(constraints, solver_factory, timeout)

        # The full set must be unsatisfiable for a MUS to exist.
        if _check(list(range(len(constraints)))) != unsat:
            return UnsatCoreResult(cores=[], is_minimal=True)

        core = list(range(len(constraints)))
        i = 0
        while i < len(core):
            trial = core[:i] + core[i + 1 :]
            # Shrink only when the remaining constraints are still unsat.
            if _check(trial) == unsat:
                core = trial
            else:
                i += 1

        return UnsatCoreResult(cores=[set(core)], is_minimal=True)

    def _shrink_core(
        self,
        core: Set[int],
        constraints: List[Any],
        solver_factory: Callable[[], Any],
        timeout: Optional[int],
    ) -> Set[int]:
        """Drop constraints from ``core`` while the remainder stays unsatisfiable."""
        _check = self._make_checker(constraints, solver_factory, timeout)
        ordered = sorted(core)
        i = 0
        while i < len(ordered):
            trial = ordered[:i] + ordered[i + 1 :]
            if _check(trial) == unsat:
                ordered = trial
            else:
                i += 1
        return set(ordered)

    def _run_optux(
        self,
        constraints: List[Any],
        solver_factory: Callable[[], Any],
        timeout: Optional[int] = None,
        **kwargs,  # noqa: ARG002
    ) -> UnsatCoreResult:
        """Run OptUx algorithm to compute an optimal (smallest) unsat core.

        ``OptUx`` (a vendored PySAT example) expects a CNF/WCNF over integer
        literals, while callers pass Z3 expressions or plain strings. Each
        constraint is therefore Tseitin-encoded into clauses that are relaxed by
        a dedicated selector literal, and the selector is the (unit) soft clause
        handed to ``OptUx``. The MUS that ``OptUx`` returns is thus a set of
        selector literals, which map back to original constraint indices.
        """
        from pysat.formula import WCNF  # noqa: PLC0415

        # Numeric constraints (or numeric strings) can be used verbatim.
        if all(self._is_numeric_constraint(c) for c in constraints):
            wcnf = WCNF()
            for constraint in constraints:
                wcnf.append(self._numeric_clause(constraint))
            return self._optux_from_wcnf(wcnf, constraints, len(constraints))

        # Z3 / string constraints: encode propositionally with selectors.
        encoded = self._encode_constraints_as_wcnf(constraints)
        if encoded is None:
            # Not expressible as a propositional formula (e.g. theory atoms
            # whose propositional abstraction would be unsound). Fall back to
            # the sound deletion-based MUS algorithm.
            return self._run_musx(constraints, solver_factory, timeout)

        wcnf, selector_indices = encoded
        result = self._optux_from_wcnf(wcnf, constraints, len(constraints), selector_indices)

        # Guard against a propositional core that is not a genuine theory-level
        # core: verify with Z3, shrink to a minimal unsat subset, and fall back
        # to deletion-based extraction if the abstraction produced a spurious
        # core.
        core = result.cores[0] if result.cores else set()
        _check = self._make_checker(constraints, solver_factory, timeout)
        if not core or _check(sorted(core)) != unsat:
            return self._run_musx(constraints, solver_factory, timeout)

        minimized = self._shrink_core(core, constraints, solver_factory, timeout)
        result.cores = [minimized]
        return result

    @staticmethod
    def _is_numeric_constraint(constraint: Any) -> bool:
        """Whether ``constraint`` is already a numeric clause."""
        if isinstance(constraint, (list, tuple)):
            return all(isinstance(lit, int) for lit in constraint)
        if isinstance(constraint, str):
            return constraint.replace("-", "").isdigit()
        return False

    @staticmethod
    def _numeric_clause(constraint: Any) -> List[int]:
        """Normalize a numeric constraint to a list of integer literals."""
        if isinstance(constraint, str):
            return [int(constraint)]
        return list(constraint)

    @staticmethod
    def _encode_constraints_as_wcnf(constraints: List[Any]):
        """Tseitin-encode Z3 constraints into a selector-relaxed WCNF.

        Returns ``(wcnf, selector_index)`` or ``None`` when a constraint is not
        purely propositional (i.e. when treating theory atoms as Boolean
        variables would give an unsound answer).
        """
        try:
            from pysat.formula import WCNF  # noqa: PLC0415
            import z3  # noqa: PLC0415
        except ImportError:  # pragma: no cover - optional dependency
            return None

        var_ids: Dict[str, int] = {}

        def literal_id(literal) -> int:
            negated = z3.is_not(literal)
            atom = literal.arg(0) if negated else literal
            if not z3.is_const(atom) or atom.decl().kind() != z3.Z3_OP_UNINTERPRETED:
                raise ValueError("non-propositional literal")
            if atom.sort() != z3.BoolSort():
                raise ValueError("non-propositional literal")
            key = atom.sexpr()
            if key not in var_ids:
                var_ids[key] = len(var_ids) + 1
            vid = var_ids[key]
            return -vid if negated else vid

        encoder = z3.Tactic("tseitin-cnf")
        clauses_per_constraint: List[List[List[int]]] = []
        try:
            for constraint in constraints:
                if isinstance(constraint, str):
                    constraint = z3.Bool(constraint)
                goal = z3.Goal()
                goal.add(constraint)
                result = encoder(goal)
                clauses: List[List[int]] = []
                # ``result`` is an ``ApplyResult``; iterating it yields
                # sub-goals, each of which holds the derived clauses.
                for subgoal in result:
                    for clause in subgoal:
                        if z3.is_true(clause):
                            continue
                        if z3.is_false(clause):
                            raise ValueError("trivially false constraint")
                        parts = clause.children() if z3.is_or(clause) else [clause]
                        clauses.append([literal_id(part) for part in parts])
                clauses_per_constraint.append(clauses)
        except (ValueError, z3.Z3Exception):
            return None

        wcnf = WCNF()
        selector_index: Dict[int, int] = {}
        next_id = len(var_ids) + 1
        for idx, clauses in enumerate(clauses_per_constraint):
            selector = next_id
            next_id += 1
            selector_index[selector] = idx
            # Every clause of the constraint is relaxed by ``not selector``,
            # so the constraint is enforced exactly when the selector is true.
            for clause in clauses:
                wcnf.append(clause + [-selector], weight=None)
            wcnf.append([selector], weight=1)

        return wcnf, selector_index

    def _optux_from_wcnf(
        self,
        wcnf,
        constraints: List[Any],
        num_constraints: int,
        selector_index: Optional[Dict[int, int]] = None,
    ) -> UnsatCoreResult:
        """Run ``OptUx`` on ``wcnf`` and map its answer back to constraints."""
        optux = self.module.OptUx(wcnf, verbose=0)
        try:
            core = optux.compute()
            cost = getattr(optux, "cost", 0)
        finally:
            optux.delete()

        core_indices = set()
        if core:
            for clause_id in core:
                if selector_index is not None:
                    # ``clause_id`` is an index into the soft clauses, which are
                    # one selector per constraint, in constraint order.
                    idx = clause_id - 1
                    if 0 <= idx < num_constraints:
                        core_indices.add(idx)
                elif 0 <= clause_id < num_constraints:
                    core_indices.add(clause_id)

        stats = {"cost": cost}
        return UnsatCoreResult(cores=[core_indices], is_minimal=True, stats=stats)

    def enumerate_all_mus(
        self,
        constraints: List[Any],
        solver_factory: Callable[[], Any],
        timeout: Optional[int] = None,
        **kwargs,
    ) -> UnsatCoreResult:
        """Enumerate all MUSes using MARCO algorithm."""
        if self.algorithm != Algorithm.MARCO:
            self.algorithm = Algorithm.MARCO
            self._load_algorithm_module()

        return self._run_marco(
            constraints,
            solver_factory,
            timeout,
            max_cores=float("inf"),
            **kwargs,
        )


def get_unsat_core(
    constraints: List[Any],
    solver_factory: Callable[[], Any],
    algorithm: Union[str, Algorithm] = "marco",
    timeout: Optional[int] = None,
    **kwargs,
) -> UnsatCoreResult:
    """Get unsat core using specified algorithm."""
    computer = UnsatCoreComputer(algorithm)
    return computer.compute_unsat_core(constraints, solver_factory, timeout, **kwargs)


def enumerate_all_mus(
    constraints: List[Any],
    solver_factory: Callable[[], Any],
    timeout: Optional[int] = None,
    **kwargs,
) -> UnsatCoreResult:
    """Enumerate all MUSes using MARCO algorithm."""
    computer = UnsatCoreComputer(Algorithm.MARCO)
    return computer.enumerate_all_mus(constraints, solver_factory, timeout, **kwargs)


def enumerate_minimal_unsat_subsets(
    constraints: List[Any], max_cores: Optional[int] = None
) -> List[Set[int]]:
    """Enumerate minimal unsatisfiable subsets using the MARCO backend."""

    module_name = "marco"
    module_path = os.path.join(os.path.dirname(__file__), f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load marco module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    csolver = module.SubsetSolver(list(constraints))
    msolver = module.MapSolver(n=csolver.n)

    cores: List[Set[int]] = []
    for orig, lits in module.enumerate_sets(csolver, msolver):
        if orig != "MUS":
            continue
        core_indices = set()
        for lit in lits:
            if hasattr(lit, "children") and len(lit.children()) == 1:
                control_var = lit.children()[0]
            else:
                control_var = lit
            core_indices.add(csolver.idcache[module.get_id(control_var)])
        cores.append(core_indices)
        if max_cores is not None and len(cores) >= max_cores:
            break

    return cores


def main():
    """Main function for testing unsat core computation."""
    from z3 import Bools

    x, y, z = Bools("x y z")
    constraints = [
        x,  # x must be true
        y,  # y must be true
        z,  # z must be true
        Or(Not(x), Not(y)),  # x and y cannot both be true
        Or(Not(y), Not(z)),  # y and z cannot both be true
        Or(Not(x), Not(z)),  # x and z cannot both be true
    ]

    def solver_factory():
        return Solver()

    print("Example: Computing unsat core")
    print("=" * 40)
    for i, constraint in enumerate(constraints):
        print(f"  {i}: {constraint}")

    try:
        print("\nTrying MARCO algorithm...")
        computer = UnsatCoreComputer(Algorithm.MARCO)
        result = computer.compute_unsat_core(constraints, solver_factory, timeout=10)
        print(f"MARCO Result: {result}")
    except (ImportError, AttributeError, ValueError) as err:
        print(f"MARCO failed: {err}")

    print("\nTrying simple Z3 approach...")
    solver = Solver()
    for constraint in constraints:
        solver.add(constraint)

    if solver.check() == unsat:
        core = solver.unsat_core()
        print(f"Z3 unsat core: {core}")
    else:
        print("Formula is satisfiable")


if __name__ == "__main__":
    main()
