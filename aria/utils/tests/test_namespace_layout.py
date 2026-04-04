"""Regression tests for organized utility namespaces."""

from aria.utils import SolverResult
from aria.utils.sexpr2 import SList, symbols_used
from aria.utils.solver import SolverResult as NamespacedSolverResult
from aria.utils.solver import solve_with_sat_solver
from aria.utils.z3.bv import Signedness
from aria.utils.z3.expr import get_variables


def test_solver_result_is_reexported_from_solver_namespace():
    assert NamespacedSolverResult is SolverResult


def test_z3_namespace_reexports_existing_helpers():
    assert Signedness.UNKNOWN.name == "UNKNOWN"
    assert callable(get_variables)


def test_top_level_specialized_modules_remain_importable():
    assert symbols_used(SList(["x"])) == {"x"}
    assert callable(solve_with_sat_solver)
