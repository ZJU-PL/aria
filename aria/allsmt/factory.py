"""
Factory for creating AllSMT solvers.

This module provides a factory for creating instances of different AllSMT solver implementations.
"""

from typing import Dict, Type, Optional, Any

from aria.allsmt.base import AllSMTSolver


class AllSMTSolverFactory:
    """
    Factory for creating AllSMT solvers.

    This class provides methods for registering and creating instances of different
    AllSMT solver implementations.
    """

    _solvers: Dict[str, Type[AllSMTSolver]] = {}

    @classmethod
    def register(cls, name: str, solver_class: Type[AllSMTSolver]) -> None:
        """
        Register a solver class with the factory.

        Args:
            name: The name to register the solver under
            solver_class: The solver class to register
        """
        cls._solvers[name.lower()] = solver_class

    @classmethod
    def create(cls, name: str, **kwargs) -> AllSMTSolver:
        """
        Create an instance of the specified solver.

        Args:
            name: The name of the solver to create
            **kwargs: Additional arguments to pass to the solver constructor

        Returns:
            An instance of the specified solver

        Raises:
            ValueError: If the specified solver is not registered
        """
        name = name.lower()
        if name not in cls._solvers:
            available = ", ".join(cls._solvers.keys())
            raise ValueError(f"Unknown solver: {name}. Available solvers: {available}")

        return cls._solvers[name](**kwargs)

    @classmethod
    def available_solvers(cls) -> list:
        """
        Get a list of available solvers.

        Returns:
            List of available solver names
        """
        return list(cls._solvers.keys())


# Register the available solvers
try:
    from .z3_solver import Z3AllSMTSolver

    AllSMTSolverFactory.register("z3", Z3AllSMTSolver)
except ImportError:
    pass

try:
    from .pysmt_solver import PySMTAllSMTSolver

    AllSMTSolverFactory.register("pysmt", PySMTAllSMTSolver)
except ImportError:
    pass

try:
    from .mathsat_solver import MathSATAllSMTSolver

    AllSMTSolverFactory.register("mathsat", MathSATAllSMTSolver)
except ImportError:
    pass


def create_allsmt_solver(name: str = "z3", **kwargs) -> AllSMTSolver:
    """
    Convenience function to create an AllSMT solver instance.

    Args:
        name: The name of the solver to create
        **kwargs: Additional arguments to pass to the solver constructor

    Returns:
        An instance of the specified AllSMT solver
    """
    return AllSMTSolverFactory.create(name, **kwargs)


# For backward compatibility, but will be deprecated in future versions
def create_solver(name: str = "z3", **kwargs) -> AllSMTSolver:
    """
    Convenience function to create an AllSMT solver instance.

    This function is deprecated and will be removed in a future version.
    Please use create_allsmt_solver() instead.

    Args:
        name: The name of the solver to create
        **kwargs: Additional arguments to pass to the solver constructor

    Returns:
        An instance of the specified AllSMT solver
    """
    import warnings
    warnings.warn(
        "create_solver() is deprecated and will be removed in a future version. "
        "Please use create_allsmt_solver() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_allsmt_solver(name, **kwargs)


def demo() -> None:
    """Demonstrate the usage of the AllSMT solver factory."""
    # Create a Z3 solver
    z3_solver = create_allsmt_solver("z3")
    print(f"Created Z3 solver: {z3_solver}")

    # List available solvers
    print(f"Available solvers: {AllSMTSolverFactory.available_solvers()}")

    # Try to create each available solver
    for solver_name in AllSMTSolverFactory.available_solvers():
        try:
            solver = create_allsmt_solver(solver_name)
            print(f"Successfully created {solver_name} solver: {solver}")
        except Exception as e:
            print(f"Failed to create {solver_name} solver: {e}")


if __name__ == "__main__":
    demo()
