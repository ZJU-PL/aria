"""Regression tests for explicit failure reporting in optimization wrappers."""

import pytest

from aria.optimization.bin_solver import (
    SolverOutputError,
    get_solver_command,
    run_solver,
)
from aria.optimization.omtbv.bit_blast_omt_solver import BitBlastOMTBVSolver


def test_get_solver_command_rejects_unknown_solver_name():
    with pytest.raises(ValueError, match="Unsupported smt solver 'bogus'"):
        get_solver_command("smt", "bogus", "/tmp/example.smt2")


def test_get_solver_command_rejects_unknown_solver_type():
    with pytest.raises(ValueError, match="Unsupported solver type: bogus"):
        get_solver_command("bogus", "z3", "/tmp/example.smt2")


def test_run_solver_raises_on_unrecognized_output():
    with pytest.raises(SolverOutputError, match="unrecognized output"):
        run_solver(["/bin/sh", "-c", "printf 'maybe\\n'"])


def test_bit_blast_solver_rejects_unknown_engine():
    solver = BitBlastOMTBVSolver()
    solver.set_engine("UNKNOWN")

    with pytest.raises(ValueError, match="Unknown MaxSAT engine 'UNKNOWN'"):
        solver._solve_with_engine(  # pylint: disable=protected-access
            maxsat_sol=None,  # type: ignore[arg-type]
            obj_str="x",
            total_score=0,
            bool_vars=[],
            is_signed=False,
        )
