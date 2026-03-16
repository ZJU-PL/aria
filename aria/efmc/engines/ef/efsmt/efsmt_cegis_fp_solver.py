"""
Solving Exists-Forall problems over floating-point variables using CEGIS.

As pysmt does not support QF_FP, we only use z3py here.
"""

from typing import List, Optional, Tuple
import logging
import os
import re
import z3

logger = logging.getLogger(__name__)


def _dump_challenging_query(  # pylint: disable=too-many-arguments,too-many-locals
    synthesis_solver: z3.Solver,
    verification_solver: z3.Solver,
    phi: z3.ExprRef,
    x: List[z3.ExprRef],
    y: List[z3.ExprRef],
    dump_dir: Optional[str],
    iteration: int,
    reason: str,
):
    """Dump a challenging query to SMT2 file for later analysis."""
    if not dump_dir:
        return

    os.makedirs(dump_dir, exist_ok=True)

    # Create filename with iteration and reason
    filename = f"challenging_query_iter{iteration}_{reason}.smt2"
    filepath = os.path.join(dump_dir, filename)

    # Create a solver with all assertions to get faithful SMT2 output
    dump_solver = z3.Solver()

    # Add all synthesis solver constraints
    for assertion in synthesis_solver.assertions():
        dump_solver.add(assertion)

    # Add verification solver constraints if available
    if verification_solver:
        for assertion in verification_solver.assertions():
            dump_solver.add(assertion)

    # Get SMT2 content - this includes all necessary variable declarations
    smt2_content = dump_solver.to_smt2()
    has_bitvec = "BitVec" in smt2_content

    # Post-process to fix Z3-specific syntax that cvc5 doesn't support
    # Z3 uses '/' for real division and '(- x)' for unary negation in
    # to_fp conversions but cvc5 may not accept these formats
    def fix_z3_specific_syntax(content: str) -> str:
        """Fix Z3-specific syntax for cvc5 compatibility."""

        # First, handle unary negation in to_fp:
        # ((_ to_fp ...) ... (- number)) -> (fp.neg ((_ to_fp ...) ... number))
        # cvc5 doesn't accept (- number) or -number in to_fp, so we need fp.neg
        def fix_negation_in_to_fp(match):
            try:
                fp_params = match.group(1)  # e.g., "8 24"
                number = match.group(2)  # The number
                # Remove any leading negative sign from number if present
                # (double negation case)
                if number.startswith("-"):
                    number = number[1:]
                # Wrap the to_fp expression with fp.neg
                return (
                    f"(fp.neg ((_ to_fp {fp_params}) roundNearestTiesToEven {number}))"
                )
            except Exception:  # pylint: disable=broad-exception-caught
                # Catch any regex processing errors and return original match
                return match.group(0)

        # Pattern 1: ((_ to_fp X Y) roundNearestTiesToEven (- number))
        # This matches the entire to_fp expression that contains a negated
        # number in parentheses. Make pattern more flexible with whitespace
        to_fp_negation_pattern = (
            r"\(\(_\s+to_fp\s+(\d+\s+\d+)\)\s+roundNearestTiesToEven\s+"
            r"\(-\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*\)\)"
        )
        content = re.sub(to_fp_negation_pattern, fix_negation_in_to_fp, content)

        # Pattern 2: ((_ to_fp X Y) roundNearestTiesToEven -number)
        # Handle case where negative number is already written as -number
        # (from previous fix attempt)
        def fix_negation_in_to_fp_direct(match):
            try:
                fp_params = match.group(1)
                # This will be the number (may or may not have negative sign)
                number = match.group(2)
                # Remove the leading negative sign if present
                if number.startswith("-"):
                    number = number[1:]
                return (
                    f"(fp.neg ((_ to_fp {fp_params}) roundNearestTiesToEven {number}))"
                )
            except Exception:  # pylint: disable=broad-exception-caught
                # Catch any regex processing errors and return original match
                return match.group(0)

        to_fp_negation_direct_pattern = (
            r"\(\(_\s+to_fp\s+(\d+\s+\d+)\)\s+roundNearestTiesToEven\s+"
            r"-([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\)"
        )
        content = re.sub(
            to_fp_negation_direct_pattern, fix_negation_in_to_fp_direct, content
        )

        # Pattern 3: Also handle cases where the expression might be nested
        # differently. Look for any occurrence of (- number) that appears after
        # roundNearestTiesToEven in a to_fp context. This is a more general
        # pattern that catches edge cases
        # Note: This pattern is currently commented out as it's not used

        # Second, handle real division: (/ x y) -> computed_value
        def compute_division(match):
            try:
                numerator = float(match.group(1))
                denominator = float(match.group(2))
                if denominator != 0:
                    result = numerator / denominator
                    # Format as decimal, avoiding scientific notation for
                    # readability. Use enough precision but avoid unnecessary
                    # trailing zeros
                    formatted = f"{result:.15g}".rstrip("0").rstrip(".")
                    if not formatted.replace(".", "").replace("-", ""):
                        formatted = "0.0"
                    return formatted
                # Keep original if division by zero
                return match.group(0)
            except Exception:  # pylint: disable=broad-exception-caught
                # If we can't compute, keep original (catch conversion/computation errors)
                return match.group(0)

        # Replace simple divisions like (/ 3.0 10.0) with computed values
        # This regex matches (/ followed by optional whitespace, a number,
        # optional whitespace, another number, and closing paren)
        # Handles integers, decimals, and scientific notation
        division_pattern = (
            r"\(/\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+"
            r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*\)"
        )
        content = re.sub(division_pattern, compute_division, content)
        return content

    smt2_content = fix_z3_specific_syntax(smt2_content)

    # Write to file, using to_smt2() output but with our logic
    with open(filepath, "w", encoding="utf-8") as f:
        # Use QF_FP if no BitVec, otherwise QF_BVFP (supports both BitVector and FloatingPoint)
        logic = "QF_BVFP" if has_bitvec else "QF_FP"
        f.write(f"(set-logic {logic})\n\n")

        # Get lines from to_smt2() output
        lines = smt2_content.split("\n")

        # Write all lines except set-logic (we set it manually)
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("(set-logic"):
                f.write(line + "\n")

    logger.info("Dumped challenging query to %s", filepath)


def simple_cegis_efsmt_fp(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    logic: str,
    x: List[z3.ExprRef],
    y: List[z3.ExprRef],
    phi: z3.ExprRef,
    maxloops=None,
    timeout=None,
    solver_timeout: int = 10,  # pylint: disable=unused-argument
    dump_dir: Optional[str] = None,
    dump_threshold: int = 5,
) -> Tuple[str, Optional[z3.ModelRef]]:
    """
    Solve EFSMT over floating-point using CEGIS.

    Args:
        logic: Logic string (e.g., "FP", "QF_FP")
        x: List of existential variables
        y: List of universal variables
        phi: Formula to solve
        maxloops: Maximum number of CEGIS iterations
        timeout: Overall timeout in seconds (unused, kept for compatibility)
        solver_timeout: Timeout per solver call in seconds (default: 10)
        dump_dir: Directory to dump challenging queries (None = no dumping)
        dump_threshold: Dump queries that timeout or exceed this many iterations

    Returns:
        Tuple of (result_string, model) where result is "sat", "unsat", or "unknown"
    """
    maxloops = maxloops or 100

    logger.info(
        "Solving EFSMT over floating-point with logic: %s, maxloops: %s, "
        "solver_timeout: %ss",
        logic,
        maxloops,
        solver_timeout,
    )

    # Create synthesis solver for existential variables
    synthesis_solver = z3.Solver()
    synthesis_solver.set("timeout", solver_timeout * 1000)  # Convert to milliseconds

    # Start with no constraints (just True)
    synthesis_solver.add(z3.BoolVal(True))

    timeout_count = 0

    for loop in range(maxloops):
        logger.debug("CEGIS iteration %s", loop + 1)

        # Synthesize: Find a candidate solution for existential variables
        result = synthesis_solver.check()
        if result == z3.unsat:
            logger.info("Synthesis solver returned UNSAT - no solution exists")
            return "unsat", None
        if result == z3.unknown:
            # Check if it's a timeout or keyboard interrupt
            try:
                reason = synthesis_solver.reason_unknown()
                reason_str = str(reason).lower() if reason else ""
            except Exception:  # pylint: disable=broad-exception-caught
                # Z3 may raise various exceptions, catch all for robustness
                reason_str = ""

            # Check for keyboard interrupt
            if "interrupted" in reason_str or "keyboard" in reason_str:
                logger.warning("Synthesis solver interrupted by user (Ctrl+C)")
                raise KeyboardInterrupt("CEGIS interrupted by user")

            if "timeout" in reason_str or "time" in reason_str:
                timeout_count += 1
                logger.warning(
                    "Synthesis solver timed out at iteration %s (timeout count: %s)",
                    loop + 1,
                    timeout_count,
                )

                # Dump challenging query if enabled
                if dump_dir and (timeout_count == 1 or loop + 1 >= dump_threshold):
                    _dump_challenging_query(
                        synthesis_solver,
                        None,
                        phi,
                        x,
                        y,
                        dump_dir,
                        loop + 1,
                        "synthesis_timeout",
                    )

                # Continue to next iteration (counts towards maxloops)
                continue
            logger.warning(
                "Synthesis solver returned UNKNOWN: %s",
                reason_str if reason_str else "unknown reason",
            )
            # For non-timeout unknowns, we might want to continue or return
            # Let's continue for now to see if we can make progress
            if dump_dir:
                _dump_challenging_query(
                    synthesis_solver,
                    None,
                    phi,
                    x,
                    y,
                    dump_dir,
                    loop + 1,
                    "synthesis_unknown",
                )
            continue

        # Extract candidate solution
        model = synthesis_solver.model()
        candidate = {}
        for var in x:
            try:
                val = model.eval(var, True)
                if val is not None:
                    candidate[var] = val
            except Exception:  # pylint: disable=broad-exception-caught
                # Z3 model evaluation may raise various exceptions, use default
                if var.sort().name() == "FPSort":
                    candidate[var] = z3.FPVal(0.0, var.sort())
                else:
                    # For non-FP variables, use a reasonable default
                    sort_name = var.sort().name()
                    candidate[var] = (
                        z3.IntVal(0) if sort_name == "Int" else z3.RealVal(0)
                    )

        # Verify: Check if candidate works for all universal variables
        # Substitute candidate values into phi
        phi_substituted = z3.substitute(phi, list(candidate.items()))

        # Check if Not(phi_substituted) is satisfiable (i.e., if there's a counterexample)
        verification_solver = z3.Solver()
        verification_solver.set(
            "timeout", solver_timeout * 1000
        )  # Convert to milliseconds

        verification_solver.add(z3.Not(phi_substituted))

        result = verification_solver.check()
        if result == z3.unsat:
            # No counterexample found - candidate is valid
            logger.info("Found valid solution after %s iterations", loop + 1)
            return "sat", model
        if result == z3.unknown:
            # Check if it's a timeout or keyboard interrupt
            try:
                reason = verification_solver.reason_unknown()
                reason_str = str(reason).lower() if reason else ""
            except Exception:  # pylint: disable=broad-exception-caught
                # Z3 may raise various exceptions, catch all for robustness
                reason_str = ""

            # Check for keyboard interrupt
            if "interrupted" in reason_str or "keyboard" in reason_str:
                logger.warning("Verification solver interrupted by user (Ctrl+C)")
                raise KeyboardInterrupt("CEGIS interrupted by user")

            if "timeout" in reason_str or "time" in reason_str:
                timeout_count += 1
                logger.warning(
                    "Verification solver timed out at iteration %s (timeout count: %s)",
                    loop + 1,
                    timeout_count,
                )

                # Dump challenging query if enabled
                if dump_dir and (timeout_count == 1 or loop + 1 >= dump_threshold):
                    _dump_challenging_query(
                        synthesis_solver,
                        verification_solver,
                        phi,
                        x,
                        y,
                        dump_dir,
                        loop + 1,
                        "verification_timeout",
                    )

                # Continue to next iteration (counts towards maxloops)
                continue
            logger.warning(
                "Verification solver returned UNKNOWN: %s",
                reason_str if reason_str else "unknown reason",
            )
            if dump_dir:
                _dump_challenging_query(
                    synthesis_solver,
                    verification_solver,
                    phi,
                    x,
                    y,
                    dump_dir,
                    loop + 1,
                    "verification_unknown",
                )
            continue

        # Extract counterexample and add as constraint
        counterexample_model = verification_solver.model()
        counterexample = {}
        for var in y:
            try:
                val = counterexample_model.eval(var, True)
                if val is not None:
                    counterexample[var] = val
            except Exception:  # pylint: disable=broad-exception-caught
                # Z3 model evaluation may raise various exceptions, skip this var
                pass

        # Add counterexample as constraint to synthesis solver
        phi_with_counterexample = z3.substitute(phi, list(counterexample.items()))
        synthesis_solver.add(phi_with_counterexample)

        logger.debug(
            "Added counterexample constraint, synthesis solver now has %s "
            "constraints",
            len(synthesis_solver.assertions()),
        )

        # Dump if we've reached many iterations
        if dump_dir and loop + 1 >= dump_threshold:
            _dump_challenging_query(
                synthesis_solver,
                verification_solver,
                phi,
                x,
                y,
                dump_dir,
                loop + 1,
                "many_iterations",
            )

    logger.warning(
        "CEGIS reached maximum iterations (%s), total timeouts: %s",
        maxloops,
        timeout_count,
    )
    if dump_dir:
        _dump_challenging_query(
            synthesis_solver,
            None,
            phi,
            x,
            y,
            dump_dir,
            maxloops,
            "max_iterations_reached",
        )
    return "unknown", None


# Keep the old function name for backward compatibility
def cegis_efsmt_fp(  # pylint: disable=too-many-arguments
    x: List[z3.ExprRef],
    y: List[z3.ExprRef],
    phi: z3.ExprRef,
    max_loops: Optional[int] = None,
    timeout: Optional[int] = None,
    solver_timeout: int = 10,
    dump_dir: Optional[str] = None,
    dump_threshold: int = 5,
) -> Tuple[str, Optional[z3.ModelRef]]:
    """
    Legacy function - now just calls simple_cegis_efsmt_fp for backward
    compatibility.
    """
    return simple_cegis_efsmt_fp(
        "QF_FP",
        x,
        y,
        phi,
        maxloops=max_loops,
        timeout=timeout,
        solver_timeout=solver_timeout,
        dump_dir=dump_dir,
        dump_threshold=dump_threshold,
    )


def test_simple_fp_problem():
    """Test a simple floating-point EFSMT problem."""
    fp_sort = z3.FPSort(8, 24)
    x, y = z3.FP("x", fp_sort), z3.FP("y", fp_sort)
    phi = z3.fpGEQ(z3.fpAdd(z3.RNE(), x, y), z3.FPVal(0.0, fp_sort))

    # CEGIS approach
    cegis_result, cegis_model = simple_cegis_efsmt_fp(
        "QF_FP", [x], [y], phi, maxloops=10
    )

    # Direct Z3 approach
    solver = z3.Solver()
    solver.add(z3.ForAll([y], phi))
    direct_result = solver.check()
    direct_result_str = (
        "sat"
        if direct_result == z3.sat
        else ("unsat" if direct_result == z3.unsat else "unknown")
    )

    print(
        f"CEGIS: {cegis_result}, Direct Z3: {direct_result_str}, "
        f"Match: {cegis_result == direct_result_str}"
    )
    if cegis_result == "sat" and cegis_model is not None:
        print(f"CEGIS model: {cegis_model}")
    return cegis_result, direct_result_str


if __name__ == "__main__":
    test_simple_fp_problem()
