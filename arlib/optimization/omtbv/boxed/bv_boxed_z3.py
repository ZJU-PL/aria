"""
Z3-based solver for boxed optimization problems.
"""

from typing import List, Optional

import subprocess

from arlib.optimization.omtbv.bv_opt_utils import res_z3_trans


def _ensure_get_objectives(script: str) -> str:
    """Ensure the SMT2 script requests objectives."""
    if "(get-objectives)" in script:
        return script
    # Try to insert before (get-model) if present, otherwise append.
    if "(get-model)" in script:
        return script.replace("(get-model)", "(get-objectives)\n(get-model)", 1)
    return script + "\n(get-objectives)\n"


def solve_boxed_z3(file_path: str, objective_order: Optional[List[str]] = None) -> List[int]:
    """Solve boxed optimization problems using the Z3 CLI and parse objectives.

    Args:
        file_path: Path to an SMT-LIB2 file containing asserts and maximize/minimize commands.
        objective_order: Optional explicit ordering of objective variable names.

    Returns:
        List of optimized objective values (as integers) in the requested order.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        script = f.read()

    script = _ensure_get_objectives(script)

    result = subprocess.run(
        ["z3", "opt.priority=box", "-in"],
        input=script,
        text=True,
        capture_output=True,
        check=True,
    )

    return res_z3_trans(result.stdout, objective_order=objective_order)
