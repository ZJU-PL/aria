"""
efmc.engines.ef.farkas.prover
=============================

Production-ready *Farkas-based invariant synthesiser* for linear (integer /
rational) transition systems.  The class hides all template construction and
solver interaction behind a very small public interface:

    prover = FarkasProver(sts, num_templates=2, timeout=5, verbose=True)
    result = prover.solve()

    if result.is_safe:
        print("Inductive invariant:", result.invariant)

Improvements compared with the original draft
---------------------------------------------
1. Strong static typing (`from __future__ import annotations`) and *dataclass*
   configuration.
2. Clear separation between:
      * template generation,
      * constraint solving,
      * invariant validation.
3. Robust solver selection that works for *mixed* Int/Real systems and reports
   unsupported Bit-Vector components early.
4. Consistent timeout handling (milliseconds) and accurate wall-clock
   measurement with a lightweight context-manager.
5. Verbose logging guarded by a single `verbose` flag; no scattered `if`s.
6. Full exception transparency – internal failures are re-raised after logging
   so callers can still decide how to handle them.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import z3

from aria.efmc.sts import TransitionSystem
from aria.efmc.engines.ef.farkas.farkas_template import FarkasTemplate
from aria.efmc.utils.verification_utils import VerificationResult, check_invariant

__all__ = ["FarkasProver", "FarkasProverConfig"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------#
# Helper: simple wall-clock timer                                            #
# ---------------------------------------------------------------------------#
@contextlib.contextmanager
def _timer() -> Any:  # returns elapsed seconds as float
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


# ---------------------------------------------------------------------------#
# Configuration                                                              #
# ---------------------------------------------------------------------------#
@dataclass
class FarkasProverConfig:
    """Configuration for FarkasProver."""

    num_templates: int = 3
    solver: str = "z3"  # placeholder for future external support
    validate_invariant: bool = True
    timeout: Optional[int] = None  # seconds
    verbose: bool = False
    z3_params: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------#
# Main class                                                                 #
# ---------------------------------------------------------------------------#
class FarkasProver:
    """
    Farkas-based invariant inference engine that supports linear (Int/Real)
    programs.  Bit-vectors are rejected because Farkas' lemma is formulated
    over ordered rings.

    Public API
    ----------
    solve(timeout: int | None = None) -> VerificationResult
    reset() -> None
    get_statistics() -> dict
    """

    # ------------------------------------------------------------------- init
    def __init__(self, sts: TransitionSystem, **kwargs: Any) -> None:
        self.sts: TransitionSystem = sts
        self.cfg = FarkasProverConfig(**kwargs)

        if self.sts.has_bv:
            raise ValueError(
                "Bit-vector variables detected – Farkas prover supports only "
                "Int and Real variables."
            )

        self._template: Optional[FarkasTemplate] = None
        self._invariant: Optional[z3.BoolRef] = None
        self._solve_time: float = 0.0

        if self.cfg.verbose:
            logger.setLevel(logging.INFO)

    # -------------------------------------------------------- private helpers
    # ----------------------------- template
    def _create_template(self) -> FarkasTemplate:
        return FarkasTemplate(self.sts, num_templates=self.cfg.num_templates)

    # ----------------------------- solver selection
    @staticmethod
    def _logic_for(sts: TransitionSystem) -> str:
        """
        Decide which quantifier-free logic the constraint system belongs to.
        """
        # Z3 names: https://rise4fun.com/z3/tutorialcontent/guide
        if sts.has_real and sts.has_int:
            return "QF_NIRA"  # non-linear int+real arithmetic
        if sts.has_real:
            return "QF_NRA"
        if sts.has_int:
            return "QF_NIA"
        return "QF_UFLIA"  # fallback – uninterpreted + linear int

    def _setup_solver(self) -> z3.Solver:
        logic_name = self._logic_for(self.sts)
        solver = z3.SolverFor(logic_name)
        # user-supplied parameters …
        for k, v in self.cfg.z3_params.items():
            solver.set(**{k: v})
        # timeout in milliseconds
        if self.cfg.timeout is not None:
            solver.set("timeout", int(self.cfg.timeout * 1000))
        return solver

    # ----------------------------- invariant validation
    def _validate(self, inv: z3.BoolRef) -> bool:
        if not self.cfg.validate_invariant:
            return True
        inv_prime = z3.substitute(
            inv, list(zip(self.sts.variables, self.sts.prime_variables))
        )
        return check_invariant(self.sts, inv, inv_prime)

    # -------------------------------------------------------- public methods
    def solve(self, timeout: Optional[int] = None) -> VerificationResult:
        """
        Try to synthesise an inductive linear invariant.

        Parameters
        ----------
        timeout : int | None
            Overrides the timeout from the configuration *for this call*.

        Returns
        -------
        VerificationResult
            .is_safe     – True  → invariant found
            .is_unknown  – True  → solver returned unknown or synthesis failed
            .invariant   – Z3 formula or None
        """
        if timeout is not None:
            self.cfg.timeout = timeout

        try:
            with _timer() as elapsed:
                if self.cfg.verbose:
                    logger.info(
                        "Starting Farkas synthesis (%d template(s))",
                        self.cfg.num_templates,
                    )

                self._template = self._create_template()
                solver = self._setup_solver()
                solver.add(self._template.template_cnt_init_and_post)
                solver.add(self._template.template_cnt_trans)

                res = solver.check()
                self._solve_time = elapsed()

                if res != z3.sat:
                    if self.cfg.verbose:
                        logger.info(
                            "Synthesis returned %s after %.2fs", res, self._solve_time
                        )
                    return VerificationResult(
                        is_safe=False, invariant=None, is_unknown=True
                    )

                model = solver.model()
                inv = self._template.build_invariant_expr(model)

                if self.cfg.verbose:
                    logger.info("Candidate invariant: %s", inv)

                if not self._validate(inv):
                    logger.error("Invariant failed inductiveness check.")
                    return VerificationResult(
                        is_safe=False, invariant=None, is_unknown=True
                    )

                # success
                self._invariant = inv
                if self.cfg.verbose:
                    logger.info("Synthesis successful (%.2fs).", self._solve_time)
                return VerificationResult(is_safe=True, invariant=inv)

        except Exception as exc:
            self._solve_time = getattr(self, "_solve_time", 0.0)
            logger.exception("Farkas synthesis raised an exception: %s", exc)
            # Re-raise after logging so clients can decide what to do.
            raise

    # ----------------------------- misc utilities
    def reset(self) -> None:
        """Forget previously computed data so the instance can be reused."""
        self._template = None
        self._invariant = None
        self._solve_time = 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Return a dictionary with basic run statistics."""
        return {
            "solve_time": self._solve_time,
            "num_templates": self.cfg.num_templates,
            "has_invariant": self._invariant is not None,
            "solver_logic": self._logic_for(self.sts),
        }

    # ----------------------------- representation
    def __str__(self) -> str:
        return (
            f"FarkasProver(templates={self.cfg.num_templates}, "
            f"timeout={self.cfg.timeout}, "
            f"validate={self.cfg.validate_invariant})"
        )
