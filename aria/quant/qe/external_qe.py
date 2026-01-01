"""
Unified interface for external quantifier elimination tools.
Supports QEPCAD, Mathematica, and Redlog backends.
"""

import subprocess
import tempfile
import logging
from typing import List, Optional, Tuple, Dict
from enum import Enum
from aria.utils.misc import run_external_tool

logger = logging.getLogger(__name__)


class QEBackend(Enum):
    """Available QE backends."""

    QEPCAD = "qepcad"
    MATHEMATICA = "mathematica"
    REDLOG = "redlog"


class QESolverConfig:
    """Configuration for QE solvers."""

    def __init__(self):
        # Default paths
        self.paths = {
            QEBackend.QEPCAD: "qepcad",
            QEBackend.MATHEMATICA: "math",
            QEBackend.REDLOG: "redlog",
        }

        # Default timeouts
        self.timeouts = {
            QEBackend.QEPCAD: 300,
            QEBackend.MATHEMATICA: 300,
            QEBackend.REDLOG: 300,
        }

        # Backend-specific options
        self.options = {
            QEBackend.QEPCAD: ["+N500000000", "+L100000", "+H1000000"],
            QEBackend.MATHEMATICA: ["-noprompt", "-run"],
            QEBackend.REDLOG: ["-w"],
        }

    def set_path(self, backend: QEBackend, path: str):
        """Set the path for a specific backend."""
        self.paths[backend] = path

    def set_timeout(self, backend: QEBackend, timeout: int):
        """Set the timeout for a specific backend."""
        self.timeouts[backend] = timeout

    def set_options(self, backend: QEBackend, options: List[str]):
        """Set the command line options for a specific backend."""
        self.options[backend] = options


class ExternalQESolver:
    """
    Unified interface for external quantifier elimination tools.
    Supports multiple backends with automatic fallback.
    """

    def __init__(self, config: Optional[QESolverConfig] = None):
        self.config = config or QESolverConfig()
        self._available_backends = self._detect_available_backends()

    def _detect_available_backends(self) -> List[QEBackend]:
        """Detect which backends are available on the system."""
        available = []

        for backend in QEBackend:
            try:
                cmd = (
                    [self.config.paths[backend], "--version"]
                    if backend != QEBackend.MATHEMATICA
                    else [self.config.paths[backend], "-version"]
                )
                result = subprocess.run(
                    cmd, capture_output=True, timeout=5, check=False
                )
                if (
                    result.returncode == 0
                    or "version" in result.stdout.decode().lower()
                ):
                    available.append(backend)
                    logger.info("Backend %s is available", backend.value)
            except (
                subprocess.TimeoutExpired,
                FileNotFoundError,
                subprocess.SubprocessError,
            ):
                logger.debug("Backend %s is not available", backend.value)

        return available

    def get_available_backends(self) -> List[QEBackend]:
        """Get list of available backends."""
        return self._available_backends.copy()

    def eliminate_quantifiers(
        self, formula: str, backend: Optional[QEBackend] = None, **kwargs
    ) -> Tuple[bool, str]:
        """
        Eliminate quantifiers using the specified or best available backend.

        Args:
            formula: The formula with quantifiers to eliminate
            backend: Specific backend to use (if None, auto-selects best available)
            **kwargs: Backend-specific parameters

        Returns:
            Tuple of (success, result_formula)
        """
        if not self._available_backends:
            return False, "No QE backends available"

        if backend is None:
            # Auto-select best available backend
            backend = self._select_best_backend(formula, **kwargs)
        if backend not in self._available_backends:
            return False, f"Backend {backend.value} is not available"

        try:
            if backend == QEBackend.QEPCAD:
                return self._call_qepcad(formula, **kwargs)
            if backend == QEBackend.MATHEMATICA:
                return self._call_mathematica(formula, **kwargs)
            if backend == QEBackend.REDLOG:
                return self._call_redlog(formula, **kwargs)
            return False, f"Unknown backend: {backend}"
        except (
            subprocess.TimeoutExpired,
            subprocess.SubprocessError,
            OSError,
            ValueError,
            KeyError,
        ) as e:
            logger.error("Error calling %s: %s", backend.value, e)
            return False, str(e)

    def _select_best_backend(self, _formula: str, **kwargs) -> QEBackend:
        """Select the best available backend based on formula characteristics."""
        # Simple heuristic: prefer QEPCAD for real arithmetic, Redlog for general
        domain = kwargs.get("domain", "").lower()
        if "real" in domain or "reals" in domain:
            if QEBackend.QEPCAD in self._available_backends:
                return QEBackend.QEPCAD

        # Default priority order
        priority_order = [QEBackend.QEPCAD, QEBackend.REDLOG, QEBackend.MATHEMATICA]

        for backend in priority_order:
            if backend in self._available_backends:
                return backend

        return self._available_backends[0]  # Fallback to first available

    def _call_qepcad(self, formula: str, **kwargs) -> Tuple[bool, str]:
        """Call QEPCAD backend."""
        qepcad_formula = self._convert_to_qepcad_syntax(formula)
        input_content = f"[]\n{qepcad_formula}.\nfinish\n"

        cmd = [self.config.paths[QEBackend.QEPCAD]] + self.config.options[
            QEBackend.QEPCAD
        ]
        timeout = kwargs.get("timeout", self.config.timeouts[QEBackend.QEPCAD])

        success, stdout, stderr = run_external_tool(cmd, input_content, timeout)
        if not success:
            return False, stderr or stdout

        return True, self._parse_qepcad_output(stdout)

    def _call_mathematica(self, formula: str, **kwargs) -> Tuple[bool, str]:
        """Call Mathematica backend."""
        domain = kwargs.get("domain", "Reals")
        timeout = kwargs.get("timeout", self.config.timeouts[QEBackend.MATHEMATICA])

        input_content = self._make_mathematica_input(formula, domain)
        cmd = [self.config.paths[QEBackend.MATHEMATICA]] + self.config.options[
            QEBackend.MATHEMATICA
        ]
        cmd.append(f"<<{input_content}")

        success, stdout, stderr = run_external_tool(cmd, None, timeout)
        if not success:
            return False, stderr or stdout

        return True, self._parse_mathematica_output(stdout)

    def _call_redlog(self, formula: str, **kwargs) -> Tuple[bool, str]:
        """Call Redlog backend."""
        logic = kwargs.get("logic", "real")
        timeout = kwargs.get("timeout", self.config.timeouts[QEBackend.REDLOG])

        input_content = self._make_redlog_input(formula, logic)
        cmd = [self.config.paths[QEBackend.REDLOG]] + self.config.options[
            QEBackend.REDLOG
        ]

        success, stdout, stderr = run_external_tool(cmd, input_content, timeout)
        if not success:
            return False, stderr or stdout

        return True, self._parse_redlog_output(stdout)

    def _convert_to_qepcad_syntax(self, formula: str) -> str:
        """Convert formula to QEPCAD syntax."""
        replacements = {
            "∧": "/\\",
            "∨": "\\/",
            "¬": "~",
            "→": "==>",
            "∀": "A",
            "∃": "E",
        }
        for old, new in replacements.items():
            formula = formula.replace(old, new)
        return formula

    def _make_mathematica_input(self, formula: str, domain: str) -> str:
        """Create Mathematica input script."""
        lines = [
            f"formula = {formula};",
            f"result = Quiet[Resolve[formula, {domain}]];",
            'Print["RESULT_START"];',
            "Print[result];",
            'Print["RESULT_END"];',
            "Exit[];",
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
            f.write("\n".join(lines))
            return f.name

    def _make_redlog_input(self, formula: str, logic: str) -> str:
        """Create Redlog input script."""
        lines = ["load_package redlog;"]

        if logic.lower() == "real":
            lines.append("rlset ofsf;")
        elif logic.lower() == "integer":
            lines.append("rlset pasf;")
        else:
            lines.append(f"rlset {logic};")

        lines.extend(
            [
                f"formula := {formula};",
                "result := rlqe formula;",
                "print_with result;",
                "quit;",
            ]
        )

        return "\n".join(lines)

    def _parse_qepcad_output(self, output: str) -> str:
        """Parse QEPCAD output to extract result."""
        lines = output.strip().split("\n")
        result_lines = []
        capture = False

        for line in lines:
            if "An equivalent quantifier-free formula:" in line:
                capture = True
                continue
            if capture and "=====================" in line:
                break
            if capture and line.strip() and not line.startswith("="):
                result_lines.append(line.strip())

        return " ".join(result_lines)

    def _parse_mathematica_output(self, output: str) -> str:
        """Parse Mathematica output to extract result."""
        lines = output.strip().split("\n")
        result_lines = []
        capture = False

        for line in lines:
            if "RESULT_START" in line:
                capture = True
                continue
            if "RESULT_END" in line:
                break
            if capture:
                result_lines.append(line.strip())

        return " ".join(result_lines)

    def _parse_redlog_output(self, output: str) -> str:
        """Parse Redlog output to extract result."""
        return output.strip()

    def try_all_backends(
        self, formula: str, **kwargs
    ) -> Dict[QEBackend, Tuple[bool, str]]:
        """
        Try all available backends and return results from each.

        Args:
            formula: The formula to process
            **kwargs: Backend-specific parameters

        Returns:
            Dictionary mapping backends to their results
        """
        results = {}

        for backend in self._available_backends:
            try:
                result = self.eliminate_quantifiers(formula, backend, **kwargs)
                results[backend] = result
            except (
                subprocess.TimeoutExpired,
                subprocess.SubprocessError,
                OSError,
                ValueError,
                KeyError,
            ) as e:
                results[backend] = (False, str(e))

        return results


# Convenience functions for backward compatibility
def eliminate_quantifiers_qepcad(
    formula: str, qepcad_path: str = None, timeout: int = 300
):
    """QEPCAD-specific QE function for backward compatibility."""
    config = QESolverConfig()
    if qepcad_path:
        config.set_path(QEBackend.QEPCAD, qepcad_path)
    if timeout != 300:
        config.set_timeout(QEBackend.QEPCAD, timeout)

    solver = ExternalQESolver(config)
    return solver.eliminate_quantifiers(formula, QEBackend.QEPCAD, timeout=timeout)


def eliminate_quantifiers_mathematica(
    formula: str,
    domain: str = "Reals",
    math_path: Optional[str] = None,
    timeout: int = 300,
):
    """Mathematica-specific QE function for backward compatibility."""
    config = QESolverConfig()
    if math_path:
        config.set_path(QEBackend.MATHEMATICA, math_path)
    if timeout != 300:
        config.set_timeout(QEBackend.MATHEMATICA, timeout)

    solver = ExternalQESolver(config)
    return solver.eliminate_quantifiers(
        formula, QEBackend.MATHEMATICA, domain=domain, timeout=timeout
    )


def eliminate_quantifiers_redlog(
    formula: str, logic: str = "real", redlog_path: str = None, timeout: int = 300
):
    """Redlog-specific QE function for backward compatibility."""
    config = QESolverConfig()
    if redlog_path:
        config.set_path(QEBackend.REDLOG, redlog_path)
    if timeout != 300:
        config.set_timeout(QEBackend.REDLOG, timeout)

    solver = ExternalQESolver(config)
    return solver.eliminate_quantifiers(
        formula, QEBackend.REDLOG, logic=logic, timeout=timeout
    )


def demo():
    """Demonstration of the unified external QE functionality."""
    # Example formula: ∃x. (x^2 - 3*x + 2 = 0)
    formula = "(E x)[x^2 - 3*x + 2 = 0]"

    # Create solver with default configuration
    solver = ExternalQESolver()

    print(f"Available backends: {[b.value for b in solver.get_available_backends()]}")
    print(f"Original formula: {formula}")

    # Try auto-selection
    success, result = solver.eliminate_quantifiers(formula)
    if success:
        print(f"Auto-selected backend result: {result}")
    else:
        print(f"Auto-selection failed: {result}")

    # Try all backends
    print("\nTrying all backends:")
    all_results = solver.try_all_backends(formula)
    for backend, (success, result) in all_results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {backend.value}: {status} - {result}")


if __name__ == "__main__":
    demo()
