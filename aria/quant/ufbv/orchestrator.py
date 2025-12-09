"""
Parallel orchestration of under-/over-approximation workers for UFBV.

This module exposes user-facing entry points that accept Z3 formulas or SMT2
strings/paths and coordinate worker processes.
"""

from __future__ import annotations

import multiprocessing
import os
from typing import List

import z3

from .enums import Quantification, Polarity, ReductionType
from .approximation import (
    rec_go,
    extract_max_bits_for_formula,
    next_approx,
)
from .runner import run_z3_on_smt2_text, to_checksat_result


# Track spawned processes to ensure cleanup on exit
process_queue: List[multiprocessing.Process] = []


def split_list(lst, n):
    """Split a list into n nearly equal slices."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def solve_with_approx_partitioned(formula_str: str,
                                  reduction_type: ReductionType,
                                  q_type: Quantification,
                                  bit_places: int,
                                  polarity: Polarity,
                                  result_queue: multiprocessing.Queue,
                                  local_max_bit_width: int) -> None:
    """Worker loop: iterate approximations and report the first conclusive result."""
    try:
        formula = z3.And(z3.parse_smt2_string(formula_str))
    except Exception:
        result_queue.put("unknown")
        return

    while (bit_places < (local_max_bit_width - 2)):
        try:
            approximated_formula = rec_go(formula, [], reduction_type, q_type, bit_places, polarity)
            formula_text = approximated_formula.sexpr()
            result = run_z3_on_smt2_text(formula_text, timeout_sec=60)

            if q_type == Quantification.UNIVERSAL:
                if result in {"sat", "unknown"}:
                    reduction_type, bit_places = next_approx(reduction_type, bit_places)
                else:  # unsat
                    result_queue.put("unsat")
                    return
            else:
                if result == "sat":
                    result_queue.put("sat")
                    return
                else:  # unsat or unknown
                    reduction_type, bit_places = next_approx(reduction_type, bit_places)
        except Exception:
            reduction_type, bit_places = next_approx(reduction_type, bit_places)

    # Fallback after exhausting approximations: direct solve on original
    result = run_z3_on_smt2_text(formula_str, timeout_sec=60, randomize=True)
    result_queue.put(result)


def solve_qbv_parallel(formula: z3.AstRef) -> z3.CheckSatResult:
    """Solve a quantified BV formula using parallel approximation strategies."""
    formula_str = formula.sexpr()

    reduction_type = ReductionType.ZERO_EXTENSION
    timeout = 60
    workers = 4

    if workers == 1:
        result = run_z3_on_smt2_text(formula_str, timeout_sec=120)
        return to_checksat_result(result)

    m_max_bit_width = extract_max_bits_for_formula(formula)
    partitioned_bits = list(range(1, m_max_bit_width + 1))
    over_parts = split_list(partitioned_bits, int(workers / 2))
    under_parts = split_list(partitioned_bits, int(workers / 2))

    with multiprocessing.Manager():
        result_queue: multiprocessing.Queue = multiprocessing.Queue()

        for nth in range(int(workers)):
            bits_id = int(nth / 2)
            if nth % 2 == 0:
                start_width = over_parts[bits_id][0] if len(over_parts[bits_id]) > 0 else 1
                end_width = over_parts[bits_id][-1] if len(over_parts[bits_id]) > 0 else m_max_bit_width
                process_queue.append(
                    multiprocessing.Process(
                        target=solve_with_approx_partitioned,
                        args=(formula_str, reduction_type, Quantification.UNIVERSAL,
                              start_width, Polarity.POSITIVE, result_queue, end_width),
                    )
                )
            else:
                start_width = under_parts[bits_id][0] if len(under_parts[bits_id]) > 0 else 1
                end_width = under_parts[bits_id][-1] if len(under_parts[bits_id]) > 0 else m_max_bit_width
                process_queue.append(
                    multiprocessing.Process(
                        target=solve_with_approx_partitioned,
                        args=(formula_str, reduction_type, Quantification.EXISTENTIAL,
                              start_width, Polarity.POSITIVE, result_queue, end_width),
                    )
                )

        for p in process_queue:
            p.start()

        try:
            result_str = result_queue.get(timeout=timeout)
            result = to_checksat_result(result_str)
        except Exception:
            result = to_checksat_result("unknown")
        finally:
            for p in process_queue:
                try:
                    p.terminate()
                except Exception:
                    pass

    return result


def solve_qbv_file_parallel(formula_file: str) -> z3.CheckSatResult:
    """Parse and solve a formula from an SMT2 file path."""
    try:
        with open(formula_file, "r") as f:
            formula_str = f.read()
        if not formula_str or "(assert" not in formula_str:
            return to_checksat_result("unknown")
        formula = z3.And(z3.parse_smt2_file(formula_file))
        return solve_qbv_parallel(formula)
    except Exception:
        return to_checksat_result("unknown")


def solve_qbv_str_parallel(fml_str: str) -> z3.CheckSatResult:
    """Parse and solve a formula from an SMT2 string."""
    try:
        if not fml_str or "(assert" not in fml_str:
            fml_str = f"(assert {fml_str})"
        formula = z3.And(z3.parse_smt2_string(fml_str))
        return solve_qbv_parallel(formula)
    except Exception:
        return to_checksat_result("unknown")


__all__ = [
    "solve_qbv_parallel",
    "solve_qbv_file_parallel",
    "solve_qbv_str_parallel",
]
