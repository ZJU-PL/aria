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
    def __init__(self, cores: List[Set[int]], is_minimal: bool = False,
                 stats: Optional[Dict[str, Any]] = None):
        self.cores = cores
        self.is_minimal = is_minimal
        self.stats = stats or {}

    def __str__(self) -> str:
        cores_str = "\n".join(
            [f"Core {i + 1}: {sorted(core)}" for i, core in enumerate(self.cores)])
        minimal_str = ("minimal" if self.is_minimal
                       else "not necessarily minimal")
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
            module_path = os.path.join(
                os.path.dirname(__file__), f"{module_name}.py")
            if os.path.exists(module_path):
                spec = importlib.util.spec_from_file_location(
                    module_name, module_path)
                if spec and spec.loader:
                    self.module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(self.module)
                else:
                    raise ImportError(f"Failed to load {module_name} module")
            else:
                self.module = importlib.import_module(
                    f"aria.unsat_core.{module_name}")
        except ImportError as import_err:
            raise ImportError(
                f"Failed to import algorithm module {module_name}: {import_err}"
            ) from import_err

    def compute_unsat_core(self, constraints: List[Any],
                           solver_factory: Callable[[], Any],
                           timeout: Optional[int] = None,
                           **kwargs) -> UnsatCoreResult:
        """Compute unsat core using the configured algorithm."""
        if self.algorithm == Algorithm.MARCO:
            return self._run_marco(constraints, solver_factory, timeout, **kwargs)
        if self.algorithm == Algorithm.MUSX:
            return self._run_musx(constraints, solver_factory, timeout, **kwargs)
        if self.algorithm == Algorithm.OPTUX:
            return self._run_optux(constraints, solver_factory, timeout, **kwargs)
        raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _run_marco(self, constraints: List[Any],
                   solver_factory: Callable[[], Any],  # noqa: ARG002
                   timeout: Optional[int] = None,  # noqa: ARG002
                   max_cores: int = 1, **kwargs) -> UnsatCoreResult:  # noqa: ARG002
        """Run MARCO algorithm to compute unsat core."""
        z3_constraints = []
        for constraint in constraints:
            if isinstance(constraint, str):
                if constraint.startswith('not '):
                    var_name = constraint[4:].split()[0]
                    z3_constraints.append(Not(Bool(var_name)))
                elif ' or ' in constraint:
                    parts = constraint.split(' or ')
                    if len(parts) == 2:
                        left, right = parts[0].strip(), parts[1].strip()
                        left_expr = (Not(Bool(left[4:]))
                                     if left.startswith('not ')
                                     else Bool(left))
                        right_expr = (Not(Bool(right[4:]))
                                      if right.startswith('not ')
                                      else Bool(right))
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
                    if (hasattr(lit, 'children') and
                            len(lit.children()) == 1):
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

    def _run_musx(self, constraints: List[Any],
                  solver_factory: Callable[[], Any],  # noqa: ARG002
                  timeout: Optional[int] = None, **kwargs) -> UnsatCoreResult:  # noqa: ARG002
        """Run MUSX algorithm to compute unsat core."""
        from pysat.formula import CNF  # noqa: PLC0415

        cnf = CNF()
        for constraint in constraints:
            if isinstance(constraint, str):
                if constraint.startswith('not '):
                    var_name = constraint[4:].split()[0]
                    lit = (-int(var_name) if var_name.isdigit()
                           else -hash(var_name) % 1000)
                    cnf.append([lit])
                elif ' or ' in constraint:
                    parts = constraint.split(' or ')
                    clause = []
                    for part in parts:
                        part = part.strip()
                        if part.startswith('not '):
                            var_name = part[4:].split()[0]
                            lit = (-int(var_name) if var_name.isdigit()
                                   else -hash(var_name) % 1000)
                        else:
                            var_name = part
                            lit = (int(var_name) if var_name.isdigit()
                                   else hash(var_name) % 1000)
                        clause.append(lit)
                    cnf.append(clause)
                else:
                    var_name = constraint
                    lit = (int(var_name) if var_name.isdigit()
                           else hash(var_name) % 1000)
                    cnf.append([lit])
            else:
                cnf.append(constraint)

        musx = self.module.MUSX(cnf, verbosity=0)
        core = musx.compute()

        core_indices = set()
        if core:
            for clause_id in core:
                if clause_id < len(constraints):
                    core_indices.add(clause_id)

        return UnsatCoreResult(cores=[core_indices], is_minimal=True)

    def _run_optux(self, constraints: List[Any],
                   solver_factory: Callable[[], Any],  # noqa: ARG002
                   timeout: Optional[int] = None, **kwargs) -> UnsatCoreResult:  # noqa: ARG002
        """Run OptUx algorithm to compute unsat core."""
        from pysat.formula import WCNF  # noqa: PLC0415

        wcnf = WCNF()
        for constraint in constraints:
            if isinstance(constraint, str):
                if constraint.startswith('not '):
                    var_name = constraint[4:].split()[0]
                    lit = (-int(var_name) if var_name.isdigit()
                           else -hash(var_name) % 1000)
                    wcnf.append([lit])
                elif ' or ' in constraint:
                    parts = constraint.split(' or ')
                    clause = []
                    for part in parts:
                        part = part.strip()
                        if part.startswith('not '):
                            var_name = part[4:].split()[0]
                            lit = (-int(var_name) if var_name.isdigit()
                                   else -hash(var_name) % 1000)
                        else:
                            var_name = part
                            lit = (int(var_name) if var_name.isdigit()
                                   else hash(var_name) % 1000)
                        clause.append(lit)
                    wcnf.append(clause)
                else:
                    var_name = constraint
                    lit = (int(var_name) if var_name.isdigit()
                           else hash(var_name) % 1000)
                    wcnf.append([lit])
            else:
                wcnf.append(constraint)

        optux = self.module.OptUx(wcnf, verbose=0)
        core = optux.compute()

        core_indices = set()
        if core:
            for clause_id in core:
                if clause_id < len(constraints):
                    core_indices.add(clause_id)

        stats = {"cost": getattr(optux, 'cost', 0)}
        return UnsatCoreResult(cores=[core_indices], is_minimal=True, stats=stats)

    def enumerate_all_mus(self, constraints: List[Any],
                          solver_factory: Callable[[], Any],
                          timeout: Optional[int] = None,
                          **kwargs) -> UnsatCoreResult:
        """Enumerate all MUSes using MARCO algorithm."""
        if self.algorithm != Algorithm.MARCO:
            self.algorithm = Algorithm.MARCO
            self._load_algorithm_module()

        cores = self.module.find_unsat_cores(
            constraints=constraints,
            solver_factory=solver_factory,
            timeout=timeout,
            enumerate_all=True,
            **kwargs
        )
        return UnsatCoreResult(cores=cores, is_minimal=True)


def get_unsat_core(constraints: List[Any], solver_factory: Callable[[], Any],
                   algorithm: Union[str, Algorithm] = "marco",
                   timeout: Optional[int] = None,
                   **kwargs) -> UnsatCoreResult:
    """Get unsat core using specified algorithm."""
    computer = UnsatCoreComputer(algorithm)
    return computer.compute_unsat_core(constraints, solver_factory, timeout, **kwargs)


def enumerate_all_mus(constraints: List[Any], solver_factory: Callable[[], Any],
                      timeout: Optional[int] = None,
                      **kwargs) -> UnsatCoreResult:
    """Enumerate all MUSes using MARCO algorithm."""
    computer = UnsatCoreComputer(Algorithm.MARCO)
    return computer.enumerate_all_mus(constraints, solver_factory, timeout, **kwargs)



def main():
    """Main function for testing unsat core computation."""
    from z3 import Bools
    x, y, z = Bools('x y z')
    constraints = [
        x,                    # x must be true
        y,                    # y must be true
        z,                    # z must be true
        Or(Not(x), Not(y)),   # x and y cannot both be true
        Or(Not(y), Not(z)),   # y and z cannot both be true
        Or(Not(x), Not(z)),   # x and z cannot both be true
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
        result = computer.compute_unsat_core(
            constraints, solver_factory, timeout=10)
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
