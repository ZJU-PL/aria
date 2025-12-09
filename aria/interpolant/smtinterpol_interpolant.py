"""SMTInterpol-based interpolant synthesis module.

Supports both binary and sequence interpolants.
"""
import logging
import os
from typing import Optional, List
import z3
from aria.utils.smtlib_solver import SmtlibProc
from aria.global_params import global_config

logger = logging.getLogger(__name__)


class SMTInterpolInterpolantSynthesizer:
    """SMTInterpol-based interpolant synthesizer."""
    SOLVER_NAME = "smtinterpol"

    def __init__(self, timeout: int = 300) -> None:
        smtinterpol_path = global_config.get_solver_path("smtinterpol")
        if not smtinterpol_path or not os.path.exists(smtinterpol_path):
            logger.warning("SMTInterpol not found in global config. Using 'smtinterpol' from PATH.")
            smtinterpol_path = "smtinterpol"
        self.smtinterpol_path = smtinterpol_path
        self.timeout = timeout

    def interpolate(self, A: z3.BoolRef, B: z3.BoolRef, logic=None) -> Optional[z3.ExprRef]:
        """Generate a binary interpolant for formulas A and B."""
        if not A or not B:
            raise ValueError("Both formulas A and B must be provided")

        # Extract signature from combined formula
        unify_solver = z3.Solver()
        unify_solver.add(z3.And(A, B))
        signature = ""
        for line in unify_solver.to_smt2().split("\n"):
            if line.startswith("(as"):
                break
            signature += f"{line}\n"

        itp_cmd = f"(get-interpol A {B.sexpr()})"
        smtlib = SmtlibProc(self.smtinterpol_path, debug=False)
        smtlib.start()
        try:
            if logic:
                smtlib.send(f"(set-logic {logic})")
            smtlib.send(signature)
            smtlib.send(f"(assert {A.sexpr()})\n")
            smtlib.send("(check-sat)")
            status = smtlib.recv()
            if status != "unsat":
                logger.warning(f"Expected unsat for interpolation, got {status}")
                return None
            smtlib.send(itp_cmd)
            itp = smtlib.recv()
            if "error" in itp or "none" in itp:
                return None
            return z3.And(z3.parse_smt2_string(signature + itp + "\n(assert A)"))
        except Exception as e:
            logger.error(f"Interpolant generation failed: {e}")
            return None
        finally:
            smtlib.stop()

    def sequence_interpolate(self, formulas: List[z3.ExprRef], logic=None) -> Optional[List[z3.ExprRef]]:
        """Generate a sequence interpolant for a list of formulas."""
        if not formulas:
            raise ValueError("At least one formula must be provided")

        # Extract signature from all formulas
        combined = z3.And(*formulas)
        unify_solver = z3.Solver()
        unify_solver.add(combined)
        signature = ""
        for line in unify_solver.to_smt2().split("\n"):
            if line.startswith("(as"):
                break
            signature += f"{line}\n"

        # Build sequence interpolant command using named assertions
        # Format: (get-interpolants IP_0 IP_1 ... IP_n)
        partition_names = [f"IP_{i}" for i in range(len(formulas))]
        itp_cmd = f"(get-interpolants {' '.join(partition_names)})"

        smtlib = SmtlibProc(self.smtinterpol_path, debug=False)
        smtlib.start()
        try:
            if logic:
                smtlib.send(f"(set-logic {logic})")
            smtlib.send(signature)
            for i, fml in enumerate(formulas):
                smtlib.send(f"(assert (! {fml.sexpr()} :named {partition_names[i]}))\n")
            smtlib.send("(check-sat)")
            status = smtlib.recv()
            if status != "unsat":
                logger.warning(f"Expected unsat for interpolation, got {status}")
                return None
            smtlib.send(itp_cmd)
            itp_response = smtlib.recv()
            if "error" in itp_response or "none" in itp_response:
                return None
            # Parse the sequence of interpolants
            parsed = z3.parse_smt2_string(signature + itp_response)
            return list(parsed) if isinstance(parsed, list) else [parsed]
        except Exception as e:
            logger.error(f"Sequence interpolant generation failed: {e}")
            return None
        finally:
            smtlib.stop()


# Backward compatibility alias
InterpolantSynthesizer = SMTInterpolInterpolantSynthesizer
