"""Path and configuration management for SMT solvers and project directories.

This module provides utilities for locating solver executables and managing
project-wide paths and configurations.
"""
from pathlib import Path
import shutil
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SolverConfig:
    """Configuration container for an SMT solver.

    Attributes:
        name: The name of the solver.
        exec_name: The executable name of the solver.
        exec_path: The full path to the solver executable, if found.
        is_available: Whether the solver executable is available on the system.
    """
    def __init__(self, name: str, exec_name: str):
        self.name = name
        self.exec_name = exec_name
        self.exec_path: Optional[str] = None
        self.is_available: bool = False

    def __repr__(self) -> str:
        """Return a string representation of the solver configuration."""
        status = "available" if self.is_available else "unavailable"
        return f"SolverConfig(name={self.name}, exec_name={self.exec_name}, status={status})"


class SolverRegistry(type):
    """Metaclass implementing singleton pattern for GlobalConfig.

    Ensures only one instance of GlobalConfig exists throughout the application.
    """
    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class GlobalConfig(metaclass=SolverRegistry):
    """Global configuration manager for SMT solvers and project paths.

    This singleton class manages solver configurations and provides access
    to project-wide paths. It automatically locates solver executables in
    the bin_solvers directory or system PATH.

    Attributes:
        SOLVERS: Dictionary mapping solver names to their configurations.
    """
    SOLVERS = {
        "z3": SolverConfig("z3", "z3"),
        "cvc5": SolverConfig("cvc5", "cvc5"),
        "mathsat": SolverConfig("mathsat", "mathsat"),
        "yices2": SolverConfig("yices2", "yices-smt2"),
        "sharp_sat": SolverConfig("sharp_sat", "sharpSAT")
    }

    def __init__(self):
        """Initialize the global configuration and locate all solvers."""
        self._bin_solver_path = Path(__file__).parent.parent.parent / "bin_solvers"
        self._locate_all_solvers()

    def _locate_solver(self, solver_config: SolverConfig) -> None:
        """Locate a solver executable and update its configuration.

        Searches for the solver in the local bin_solvers directory first,
        then in the system PATH.

        Args:
            solver_config: The solver configuration to update.
        """
        local_path = self._bin_solver_path / solver_config.exec_name
        if shutil.which(str(local_path)):
            solver_config.exec_path = str(local_path)
            solver_config.is_available = True
            return

        system_path = shutil.which(solver_config.exec_name)
        if system_path:
            solver_config.exec_path = system_path
            solver_config.is_available = True
            return

        logger.warning("Could not locate %s solver executable", solver_config.name)

    def _locate_all_solvers(self) -> None:
        """Locate all configured solvers."""
        for solver_config in self.SOLVERS.values():
            self._locate_solver(solver_config)

    def set_solver_path(self, solver_name: str, path: str) -> None:
        """Set a custom path for a solver.

        Args:
            solver_name: Name of the solver to configure.
            path: Path to the solver executable.

        Raises:
            ValueError: If the solver name is unknown or the path doesn't exist.
        """
        if solver_name not in self.SOLVERS:
            raise ValueError(f"Unknown solver: {solver_name}")
        if not Path(path).exists():
            raise ValueError(f"Path does not exist: {path}")
        self._locate_solver(self.SOLVERS[solver_name])

    def get_solver_path(self, solver_name: str) -> Optional[str]:
        """Get the path to a solver executable.

        Args:
            solver_name: Name of the solver.

        Returns:
            Path to the solver executable, or None if not found.

        Raises:
            ValueError: If the solver name is unknown.
        """
        if solver_name not in self.SOLVERS:
            raise ValueError(f"Unknown solver: {solver_name}")
        return self.SOLVERS[solver_name].exec_path

    def is_solver_available(self, solver_name: str) -> bool:
        """Check if a solver is available on the system.

        Args:
            solver_name: Name of the solver to check.

        Returns:
            True if the solver is available, False otherwise.

        Raises:
            ValueError: If the solver name is unknown.
        """
        if solver_name not in self.SOLVERS:
            raise ValueError(f"Unknown solver: {solver_name}")
        return self.SOLVERS[solver_name].is_available

    def get_smt_solvers_config(self) -> Dict:
        """Get configuration dictionary for SMT solvers.

        Returns:
            Dictionary mapping solver names to their configuration,
            including availability, path, and command-line arguments.
        """
        return {
            'z3': {
                'available': self.is_solver_available("z3"),
                'path': self.get_solver_path("z3"),
                'args': "-in"
            },
            'cvc5': {
                'available': self.is_solver_available("cvc5"),
                'path': self.get_solver_path("cvc5"),
                'args': "-q -i"
            },
            'mathsat': {
                'available': self.is_solver_available("mathsat"),
                'path': self.get_solver_path("mathsat"),
                'args': ""
            }
        }

    @property
    def project_root(self) -> Path:
        """Get the root directory of the project.

        Returns:
            Path to the project root directory.
        """
        return Path(__file__).parent.parent.parent

    @property
    def bin_solvers_path(self) -> Path:
        """Get the path to the bin_solvers directory.

        Returns:
            Path to the bin_solvers directory.
        """
        return self.project_root / "bin_solvers"

    @property
    def benchmarks_path(self) -> Path:
        """Get the path to the benchmarks directory.

        Returns:
            Path to the benchmarks directory.
        """
        return self.project_root / "benchmarks"


global_config = GlobalConfig()

SMT_SOLVERS_PATH = global_config.get_smt_solvers_config()
PROJECT_ROOT = global_config.project_root
BIN_SOLVERS_PATH = global_config.bin_solvers_path
BENCHMARKS_PATH = global_config.benchmarks_path
