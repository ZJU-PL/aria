from pathlib import Path
from typing import Dict, Optional


class EFMCConfig:
    _instance: Optional["EFMCConfig"] = None
    _solvers: Dict[str, str] = {
        "z3": "z3",
        "cvc5": "cvc5",
        "btor": "boolector",
        "bitwuzla": "bitwuzla",
        "yices": "yices-smt2",
        "mathsat": "mathsat",
        "caqe": "caqe",
        "q3b": "q3b",
    }

    def __new__(cls) -> "EFMCConfig":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self.project_root_dir: str = str(Path(__file__).parent.parent.parent)
        bin_dir = Path(self.project_root_dir) / "bin_solvers" / "bin"
        for name, fname in self._solvers.items():
            path = bin_dir / fname
            setattr(self, f"{name}_exec", str(path) if path.exists() else None)
        self.bin_solver_timeout: int = 30
        self.verifier_args: Optional[str] = None
        self.efsmt_args: Optional[str] = None

    def check_available(self, solver_name: str) -> bool:
        exec_path = getattr(self, f"{solver_name}_exec", None)
        return exec_path is not None and Path(exec_path).exists()


config = EFMCConfig()

# For backward compatibility (optional, can be removed if not needed)
project_root_dir = config.project_root_dir
z3_exec = config.z3_exec
cvc5_exec = config.cvc5_exec
btor_exec = config.btor_exec
bitwuzla_exec = config.bitwuzla_exec
yices_exec = config.yices_exec
math_exec = config.mathsat_exec
caqe_exec = config.caqe_exec
q3b_exec = config.q3b_exec

g_bin_solver_timeout = config.bin_solver_timeout
g_verifier_args = config.verifier_args
g_efsmt_args = config.efsmt_args

if __name__ == "__main__":
    print(config.project_root_dir)
    for name in EFMCConfig._solvers:
        print(f"{name}: {getattr(config, f'{name}_exec')}")
