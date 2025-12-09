from pathlib import Path
import shutil
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SolverConfig:
    def __init__(self, name: str, exec_name: str):
        self.name = name
        self.exec_name = exec_name
        self.exec_path: Optional[str] = None
        self.is_available: bool = False


class SolverRegistry(type):
    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class GlobalConfig(metaclass=SolverRegistry):
    SOLVERS = {
        "z3": SolverConfig("z3", "z3"),
        "cvc5": SolverConfig("cvc5", "cvc5"),
        "mathsat": SolverConfig("mathsat", "mathsat"),
        "yices2": SolverConfig("yices2", "yices-smt2"),
        "sharp_sat": SolverConfig("sharp_sat", "sharpSAT")
    }

    def __init__(self):
        self._bin_solver_path = Path(__file__).parent.parent.parent / "bin_solvers"
        self._locate_all_solvers()

    def _locate_solver(self, solver_config: SolverConfig) -> None:
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

        logger.warning(f"Could not locate {solver_config.name} solver executable")

    def _locate_all_solvers(self) -> None:
        for solver_config in self.SOLVERS.values():
            self._locate_solver(solver_config)

    def set_solver_path(self, solver_name: str, path: str) -> None:
        if solver_name not in self.SOLVERS:
            raise ValueError(f"Unknown solver: {solver_name}")
        if not Path(path).exists():
            raise ValueError(f"Path does not exist: {path}")
        self._locate_solver(self.SOLVERS[solver_name])

    def get_solver_path(self, solver_name: str) -> Optional[str]:
        if solver_name not in self.SOLVERS:
            raise ValueError(f"Unknown solver: {solver_name}")
        return self.SOLVERS[solver_name].exec_path

    def is_solver_available(self, solver_name: str) -> bool:
        if solver_name not in self.SOLVERS:
            raise ValueError(f"Unknown solver: {solver_name}")
        return self.SOLVERS[solver_name].is_available

    def get_smt_solvers_config(self) -> Dict:
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
        return Path(__file__).parent.parent.parent

    @property
    def bin_solvers_path(self) -> Path:
        return self.project_root / "bin_solvers"

    @property
    def benchmarks_path(self) -> Path:
        return self.project_root / "benchmarks"


global_config = GlobalConfig()

SMT_SOLVERS_PATH = global_config.get_smt_solvers_config()
PROJECT_ROOT = global_config.project_root
BIN_SOLVERS_PATH = global_config.bin_solvers_path
BENCHMARKS_PATH = global_config.benchmarks_path
