"""
Sequential (solve one-by-one) optimization strategy for bit-vector objectives

FIXME: by LLM, to check if this is correct
"""
import logging
import time
from typing import List, Tuple

import z3

from arlib.optimization.omtbv.bv_opt_qsmt import bv_opt_with_qsmt
from arlib.optimization.omtbv.bv_opt_maxsat import bv_opt_with_maxsat
from arlib.optimization.omtbv.bv_opt_iterative_search import bv_opt_with_linear_search, bv_opt_with_binary_search

logger = logging.getLogger(__name__)


def _parse_smtlib_value(result: str) -> int:
    """Parse SMT-LIB value from solver output.

    Args:
        result: Raw solver output containing SMT-LIB model

    Returns:
        Integer value of the objective

    Examples:
        "sat\n((x #x7f))" -> 127
        "sat\n((y #b01111111))" -> 127
    """
    import re

    # Find SMT-LIB value patterns: #x[0-9a-f]+ or #b[0-1]+
    hex_match = re.search(r'#x([0-9a-fA-F]+)', result)
    if hex_match:
        return int(hex_match.group(1), 16)

    bin_match = re.search(r'#b([0-1]+)', result)
    if bin_match:
        return int(bin_match.group(1), 2)

    # Fallback: try to parse as decimal
    dec_match = re.search(r'\b(\d+)\b', result)
    if dec_match:
        return int(dec_match.group(1))

    raise ValueError(f"No valid SMT-LIB value found in: {result}")


def solve_boxed_sequential(formula: z3.BoolRef,
                         objectives: List[z3.ExprRef],
                         minimize: bool = False,
                         engine: str = "qsmt",
                         solver_name: str = "z3") -> List[int]:
    """
    Solve multiple objectives sequentially using boxed optimization strategy.

    Args:
        formula: The base formula (constraints)
        objectives: List of objectives to optimize
        minimize: Whether to minimize (True) or maximize (False) objectives
        engine: Optimization engine to use ("qsmt", "maxsat", or "iter")
        solver_name: Specific solver to use within the chosen engine

    Returns:
        List of optimal values for each objective
    """
    results = []
    current_formula = formula

    for i, obj in enumerate(objectives):
        logger.info(f"Optimizing objective {i+1}/{len(objectives)}: {obj}")
        start_time = time.time()

        # Select optimization method based on engine
        if engine == "qsmt":
            result = bv_opt_with_qsmt(current_formula, obj, minimize, solver_name)
        elif engine == "maxsat":
            result = bv_opt_with_maxsat(current_formula, obj, minimize, solver_name)
        elif engine == "iter":
            if solver_name.endswith("-ls"):
                solver_type = solver_name.split('-')[0]
                result = bv_opt_with_linear_search(current_formula, obj, minimize, solver_type)
            elif solver_name.endswith("-bs"):
                solver_type = solver_name.split('-')[0]
                result = bv_opt_with_binary_search(current_formula, obj, minimize, solver_type)
            else:
                raise ValueError(f"Invalid solver name for iterative engine: {solver_name}")
        else:
            raise ValueError(f"Unsupported engine: {engine}")

        logger.info(f"Objective {i+1} optimization completed in {time.time() - start_time:.2f}s")

        if result is None or result == "unknown":
            logger.warning(f"Could not optimize objective {i+1}")
            results.append(None)
            continue

        # Parse result and add constraint for the current objective
        if isinstance(result, str):
            if "sat" in result.lower():
                # Extract value from SMT-LIB model output
                try:
                    value = _parse_smtlib_value(result)
                    results.append(value)
                    # Add constraint to maintain this objective's value
                    current_formula = z3.And(current_formula, obj == value)
                except (ValueError, IndexError) as e:
                    logger.error(f"Could not parse result: {result} ({e})")
                    results.append(None)
            else:
                logger.warning(f"Unexpected result format: {result}")
                results.append(None)
        else:
            # Numeric result
            results.append(result)
            # Add constraint to maintain this objective's value
            current_formula = z3.And(current_formula, obj == result)

    return results

def demo():
    """Demo usage of sequential boxed optimization"""
    x = z3.BitVec('x', 8)
    y = z3.BitVec('y', 8)

    # Create a simple formula with two objectives
    formula = z3.And(x >= 0, y >= 0, x + y <= 10)
    objectives = [x, y]

    # Try different engines
    engines = [
        ("qsmt", "z3"),
        ("maxsat", "FM"),
        ("iter", "z3-ls")
    ]

    for engine, solver in engines:
        print(f"\nTrying {engine} engine with {solver} solver:")
        try:
            results = solve_boxed_sequential(formula, objectives, False, engine, solver)
            print(f"Results: {results}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
