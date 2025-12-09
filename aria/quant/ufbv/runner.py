"""
Execution helpers to call Z3 externally on SMT2 content.

Workers interact with these helpers to evaluate approximated formulas and the
original formula. Using the external solver process avoids interference with
Z3's internal global state across multiprocessing.
"""

from __future__ import annotations

import os
import random
import subprocess
import tempfile
from typing import Optional

import z3

from aria.global_params import global_config

# Resolve the Z3 binary path from global configuration
Z3_PATH = global_config.get_solver_path("z3")


def run_z3_on_smt2_text(smt2_text: str, timeout_sec: int = 60, randomize: bool = False) -> str:
    """Run Z3 on an SMT2 string and return one of {"sat","unsat","unknown"}.

    Writes the text into a temporary file to avoid CLI quoting pitfalls.
    """
    temp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".smt2", mode="w+", delete=False) as temp_file:
            temp_path = temp_file.name
            # Quiet options and no extra proof/trace output
            temp_file.write("(set-option :print-success false)\n")
            temp_file.write("(set-option :produce-unsat-cores false)\n")
            temp_file.write("(set-option :produce-proofs false)\n")
            temp_file.write("(set-option :trace false)\n")
            temp_file.write("(set-option :global-decls false)\n")
            temp_file.write("(set-option :verbose 0)\n")
            temp_file.write(smt2_text)
            if "(check-sat)" not in smt2_text:
                temp_file.write("\n(check-sat)")
            if randomize:
                seed = random.randint(0, 10)
                temp_file.write(f"\n(set-option :smt.random_seed {seed})")

        proc = subprocess.run(
            [Z3_PATH, "-smt2", "-q", temp_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        # Parse the last occurrence of a sat-status token
        result = "unknown"
        for line in proc.stdout.strip().split("\n"):
            t = line.strip()
            if t in {"sat", "unsat", "unknown"}:
                result = t
        return result
    except subprocess.TimeoutExpired:
        return "unknown"
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


def run_direct(formula_str: str, timeout_sec: int = 60, randomize: bool = False) -> str:
    """Shortcut for running Z3 directly on the given SMT2 text string."""
    return run_z3_on_smt2_text(formula_str, timeout_sec=timeout_sec, randomize=randomize)


def to_checksat_result(status: str) -> z3.CheckSatResult:
    """Convert a sat-status string to Z3's CheckSatResult."""
    if status == "sat":
        return z3.CheckSatResult(z3.Z3_L_TRUE)
    if status == "unsat":
        return z3.CheckSatResult(z3.Z3_L_FALSE)
    return z3.CheckSatResult(z3.Z3_L_UNDEF)


__all__ = [
    "run_z3_on_smt2_text",
    "run_direct",
    "to_checksat_result",
]
