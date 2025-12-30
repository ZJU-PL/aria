#!/usr/bin/env python3
"""
ff_regress.py  –  Regression test driver for finite-field solvers

Usage:
    python ff_regress.py bv <benchmark_dir>    # Test BV encoding solver (wide-BV)
    python ff_regress.py bv2 <benchmark_dir>   # Test BV bridge solver (Int/BV bridge)
    python ff_regress.py int <benchmark_dir>   # Test integer encoding solver
    python ff_regress.py both <benchmark_dir>  # Test both solvers
"""
from __future__ import annotations
import sys
import pathlib
import subprocess
import z3

# Handle both direct execution and module import
if __name__ == "__main__":
    # Add parent directories to path for direct execution
    import os as _os
    _script_dir = _os.path.dirname(_os.path.abspath(__file__))
    _aria_dir = _os.path.dirname(_os.path.dirname(_os.path.dirname(_script_dir)))
    if _aria_dir not in sys.path:
        sys.path.insert(0, _aria_dir)

try:
    from aria.smt.ff.ff_parser import parse_ff_file
    from aria.smt.ff.ff_bv_solver import FFBVSolver
    from aria.smt.ff.ff_bv_solver2 import FFBVBridgeSolver
    from aria.smt.ff.ff_int_solver import FFIntSolver
    from aria.smt.ff.ff_ast import (
        FieldAdd, FieldMul, FieldEq, FieldVar, FieldConst, ParsedFormula
    )
except ImportError:
    # Fallback to relative imports when used as a module
    from .ff_parser import parse_ff_file
    from .ff_bv_solver import FFBVSolver
    from .ff_bv_solver2 import FFBVBridgeSolver
    from .ff_int_solver import FFIntSolver
    from .ff_ast import (
        FieldAdd, FieldMul, FieldEq, FieldVar, FieldConst, ParsedFormula
    )


def tiny_demo() -> ParsedFormula:
    """Create a simple demo formula for testing."""
    # m * x + 16 == is_zero     ∧     is_zero * x == 0
    p = 17
    x, m, z = "x", "m", "is_zero"
    variables = {x: "ff", m: "ff", z: "ff"}
    f1 = FieldEq(
        FieldAdd(FieldMul(FieldVar(m), FieldVar(x)),
                 FieldConst(16),
                 FieldVar(z)),
        FieldConst(0))
    f2 = FieldEq(FieldMul(FieldVar(z), FieldVar(x)), FieldConst(0))
    return ParsedFormula(p, variables, [f1, f2])


def run_demo(solver_type: str = "bv"):
    """Run a simple demo with the specified solver."""
    formula = tiny_demo()
    if solver_type == "bv":
        solver = FFBVSolver()
    elif solver_type == "bv2":
        solver = FFBVBridgeSolver()
    elif solver_type == "int":
        solver = FFIntSolver()
    else:
        print(f"Unknown solver type: {solver_type}")
        return

    res = solver.check(formula)
    print(f"Result ({solver_type}):", res)
    if res == z3.sat:
        print("Model:", solver.model())


def solve_in_subprocess(
    file_path: str, solver_type: str, timeout: float = 5.0
) -> tuple[str, str]:
    """Solve a formula in a subprocess with timeout.

    Returns:
        (result, error_message) where result is "sat", "unsat", "unknown",
        "timeout", or "error"
    """
    import os as _os_module  # pylint: disable=import-outside-toplevel
    script_path = pathlib.Path(__file__).resolve()
    # Ensure subprocess can find aria by setting PYTHONPATH
    # (same calculation as __main__)
    env = _os_module.environ.copy()
    script_dir = script_path.parent
    # Go up 3 levels: smt/ff -> smt -> aria -> root
    root_dir = script_dir.parent.parent.parent
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{root_dir}{_os_module.pathsep}{pythonpath}"
    else:
        env["PYTHONPATH"] = str(root_dir)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "solve", solver_type, file_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False
        )
        if result.returncode == 0:
            verdict = result.stdout.strip()
            return (verdict, "")
        return ("error", result.stderr.strip() or "Unknown error")
    except subprocess.TimeoutExpired:
        return ("timeout", "")
    except Exception as ex:  # pylint: disable=broad-exception-caught
        # Catch all exceptions for subprocess errors
        return ("error", str(ex))


def solve_single(file_path: str, solver_type: str) -> str:
    """Solve a single formula (called from subprocess)."""
    try:
        formula = parse_ff_file(file_path)
        if solver_type == "bv":
            solver = FFBVSolver()
        elif solver_type == "bv2":
            solver = FFBVBridgeSolver()
        elif solver_type == "int":
            solver = FFIntSolver()
        else:
            return "error"
        result = solver.check(formula)
        return str(result)
    except Exception:  # pylint: disable=broad-exception-caught
        # Catch all exceptions to return "error" for any solver failure
        return "error"


def regress(dir_path: str, solver_type: str = "bv", timeout: float = 5.0) -> None:  # pylint: disable=too-many-locals
    """Walk a directory containing .smt2 finite-field benchmarks.

    Args:
        dir_path: Directory containing .smt2 files
        solver_type: "bv", "bv2", "int", or "both"
        timeout: Timeout per test in seconds (default 5.0)
    """
    if solver_type not in ("bv", "bv2", "int", "both"):
        print(f"Unknown solver type: {solver_type}. Use 'bv', 'bv2', 'int', or 'both'")
        return

    stats = {"total": 0, "passed": 0, "failed": 0, "parse_errors": 0, "timeouts": 0}

    for fn in sorted(pathlib.Path(dir_path).rglob("*.smt2")):
        txt = fn.read_text()
        expect = "unknown"
        # First try to get expected status from parsed formula (set-info :status)
        try:
            formula = parse_ff_file(str(fn))
            if formula.expected_status:
                expect = formula.expected_status
        except Exception as ex:  # pylint: disable=broad-exception-caught
            print(f"{fn.name:<50}  parse error: {ex}")
            stats["parse_errors"] += 1
            continue

        # Fallback to comment-based status if not found in set-info
        if expect == "unknown":
            if "; EXPECT: sat" in txt:
                expect = "sat"
            elif "; EXPECT: unsat" in txt:
                expect = "unsat"

        stats["total"] += 1
        file_path = str(fn.resolve())

        if solver_type == "both":
            verdict_bv, _ = solve_in_subprocess(file_path, "bv", timeout)
            verdict_int, _ = solve_in_subprocess(file_path, "int", timeout)

            has_timeout = verdict_bv == "timeout" or verdict_int == "timeout"
            if has_timeout:
                stats["timeouts"] += 1

            ok_bv = (verdict_bv != "timeout" and
                     (verdict_bv == expect or expect in ("unknown",)))
            ok_int = (verdict_int != "timeout" and
                      (verdict_int == expect or expect in ("unknown",)))
            passed = ok_bv and ok_int

            stats["passed" if passed else "failed"] += 1
            agree = verdict_bv == verdict_int
            print(f"{fn.name:<50}  expect={expect:7}  "
                  f"bv={verdict_bv:7}  int={verdict_int:7}  "
                  f"{'✓' if passed else '✗'}  agree={'✓' if agree else '✗'}")
        else:
            verdict, _ = solve_in_subprocess(file_path, solver_type, timeout)
            if verdict == "timeout":
                stats["timeouts"] += 1
            passed = (verdict != "timeout" and
                      (verdict == expect or expect in ("unknown",)))
            stats["passed" if passed else "failed"] += 1
            print(f"{fn.name:<50}  expect={expect:7}  got={verdict:7}  "
                  f"{'✓' if passed else '✗'}")

    # Summary
    print("\n" + "=" * 70)
    print(f"Summary: {stats['total']} benchmarks, {stats['passed']} passed, "
          f"{stats['failed']} failed, {stats['parse_errors']} parse errors, "
          f"{stats['timeouts']} timeouts")


# -----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "solve":
        # Called from subprocess to solve a single file
        print(solve_single(sys.argv[3], sys.argv[2]))
    elif len(sys.argv) == 2 and sys.argv[1] == "demo":
        run_demo("bv")
    elif len(sys.argv) == 3 and sys.argv[1] == "demo":
        run_demo(sys.argv[2])
    elif len(sys.argv) == 3:
        # python ff_regress.py <solver_type> <benchmark_dir>
        regress(sys.argv[2], sys.argv[1])
    elif len(sys.argv) == 4:
        # python ff_regress.py <solver_type> <benchmark_dir> <timeout>
        regress(sys.argv[2], sys.argv[1], float(sys.argv[3]))
    else:
        print("Usage:")
        print("  python ff_regress.py demo [bv|bv2|int]")
        print("  python ff_regress.py <bv|bv2|int|both> <benchmark_dir> [timeout]")
        print("\nExamples:")
        print("  python ff_regress.py demo bv")
        print("  python ff_regress.py bv2 benchmarks/smtlib2/ff/")
        print("  python ff_regress.py both benchmarks/smtlib2/ff/ 10")
