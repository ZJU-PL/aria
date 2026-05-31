"""
NeuroSym standalone solver — no Z3, no Bitwuzla.

Pipeline:
  1. Parse with ns_parser (own SMT-LIB2 parser)
  2. GAN fast path (if PyTorch available):
       QF_BV/QF_ABV → bv_encode → BVIterativeGenerator → bv_decode → ns_evaluator verify
       QF_LIA       → encode    → IterativeGenerator   → decode    → ns_evaluator verify
  3. Symbolic fallback (own solvers):
       QF_BV/QF_ABV → ns_bitblaster → ns_dpll
       QF_LIA       → ns_lia
"""

import time
from typing import Optional, Tuple

from .ns_parser    import parse_file, parse_string
from .ns_ast       import NsFormula, BVSort, IntSort
from .ns_evaluator import evaluate
from .ns_encoder   import encode, decode_assignment
from .ns_bv_encoder import bv_encode, bv_decode_assignment

# PyTorch and trained GAN models are optional
try:
    import torch
    from .gan    import IterativeGenerator,   MAX_VARS, NOISE_DIM
    from .bv_gan import BVIterativeGenerator, BV_NOISE_DIM
    _TORCH = True
except ImportError:
    _TORCH = False

from .ns_bitblaster import blast, reconstruct
from .ns_dpll       import solve_cnf
from .ns_lia        import solve_lia

RESULT_SAT     = "sat"
RESULT_UNSAT   = "unsat"
RESULT_UNKNOWN = "unknown"

_BV_LOGICS  = {"QF_BV", "QF_ABV", "QF_AUFBV", "BV"}
_LIA_LOGICS = {"QF_LIA", "QF_NIA", "QF_LRA", "LIA"}


class NeuroSymSolver:
    def __init__(
        self,
        model_path:     Optional[str] = None,
        bv_model_path:  Optional[str] = None,
        lia_model_path: Optional[str] = None,
        n_candidates:   int  = 8,
        timeout_ms:     int  = 20_000,
        device:         str  = "cpu",
    ):
        self.n_candidates = n_candidates
        self.timeout_ms   = timeout_ms

        if _TORCH:
            self.device  = torch.device(device)
            self.lia_gen = IterativeGenerator().to(self.device)
            self.lia_gen.eval()
            lia_path = lia_model_path or model_path
            if lia_path:
                state = torch.load(lia_path, map_location=self.device,
                                   weights_only=True)
                self.lia_gen.load_state_dict(state)

            self.bv_gen = BVIterativeGenerator().to(self.device)
            self.bv_gen.eval()
            if bv_model_path:
                state = torch.load(bv_model_path, map_location=self.device,
                                   weights_only=True)
                self.bv_gen.load_state_dict(state)
        else:
            self.device  = None
            self.lia_gen = None
            self.bv_gen  = None

    # ── Public API ────────────────────────────────────────────────────────────

    def solve_file(self, path: str) -> Tuple[str, Optional[dict], float]:
        formula = parse_file(path)
        return self._solve(formula)

    def solve_string(self, smtlib_str: str) -> Tuple[str, Optional[dict], float]:
        formula = parse_string(smtlib_str)
        return self._solve(formula)

    # ── Internal dispatch ─────────────────────────────────────────────────────

    def _solve(self, formula: NsFormula) -> Tuple[str, Optional[dict], float]:
        t0    = time.time()
        logic = formula.logic.upper()
        is_bv = logic in _BV_LOGICS

        deadline = t0 + self.timeout_ms / 1000.0

        # ── GAN fast path ─────────────────────────────────────────────────────
        if _TORCH:
            try:
                if is_bv:
                    r, m = self._bv_gan_path(formula)
                elif logic in _LIA_LOGICS or formula.variables:
                    r, m = self._lia_gan_path(formula)
                else:
                    r, m = RESULT_UNKNOWN, None

                if r == RESULT_SAT:
                    return r, m, (time.time() - t0) * 1000
            except Exception:
                pass

        # ── Symbolic fallback — own solvers only ──────────────────────────────
        remaining = deadline - time.time()
        if remaining <= 0:
            return RESULT_UNKNOWN, None, (time.time() - t0) * 1000

        if is_bv:
            result, model = self._bv_solve(formula, deadline)
        else:
            result, model = self._lia_solve(formula, deadline)

        return result, model, (time.time() - t0) * 1000

    # ── GAN paths ─────────────────────────────────────────────────────────────

    def _lia_gan_path(self, formula: NsFormula) -> Tuple[str, Optional[dict]]:
        enc   = encode(formula)
        enc_t = torch.tensor(enc, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        with torch.no_grad():
            candidates = self.lia_gen.sample(enc_t, n_samples=self.n_candidates)

        for i in range(self.n_candidates):
            vec        = candidates[0, i].cpu().numpy()
            assignment = decode_assignment(vec, formula)
            if evaluate(formula, assignment):
                return RESULT_SAT, assignment
        return RESULT_UNKNOWN, None

    def _bv_gan_path(self, formula: NsFormula) -> Tuple[str, Optional[dict]]:
        enc   = bv_encode(formula)
        enc_t = torch.tensor(enc, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        with torch.no_grad():
            candidates = self.bv_gen.sample(enc_t, n_samples=self.n_candidates)

        for i in range(self.n_candidates):
            vec        = candidates[0, i].cpu().numpy()
            assignment = bv_decode_assignment(vec, formula)
            if evaluate(formula, assignment):
                return RESULT_SAT, assignment
        return RESULT_UNKNOWN, None

    # ── Symbolic solvers ──────────────────────────────────────────────────────

    def _bv_solve(self, formula: NsFormula,
                  deadline: float) -> Tuple[str, Optional[dict]]:
        try:
            clauses, n_vars, var_map = blast(formula)
        except Exception:
            return RESULT_UNKNOWN, None

        if not clauses and not var_map:
            return RESULT_SAT, {}

        sat_assign = solve_cnf(clauses, n_vars, deadline=deadline)
        if sat_assign is None:
            # DPLL returned None — could be UNSAT or timeout
            remaining = deadline - time.time()
            if remaining <= 0:
                return RESULT_UNKNOWN, None
            return RESULT_UNSAT, None

        bv_assign = reconstruct(sat_assign, var_map)
        # Verify with our own evaluator
        if evaluate(formula, bv_assign):
            return RESULT_SAT, bv_assign
        # Assignment found by DPLL but evaluator disagrees — encoding mismatch
        return RESULT_UNKNOWN, None

    def _lia_solve(self, formula: NsFormula,
                   deadline: float) -> Tuple[str, Optional[dict]]:
        try:
            result, assignment = solve_lia(formula, deadline)
        except Exception:
            return RESULT_UNKNOWN, None

        if result == RESULT_SAT and assignment is not None:
            if evaluate(formula, assignment):
                return RESULT_SAT, assignment
            # LIA solver returned SAT but evaluator disagrees (e.g. non-linear
            # constraints skipped) — fall back to unknown
            return RESULT_UNKNOWN, None

        return result, assignment


# ── Output formatting ─────────────────────────────────────────────────────────

def format_output(result: str, model: Optional[dict] = None) -> str:
    lines = [result]
    if result == RESULT_SAT and model:
        lines.append("(model")
        for name, val in sorted(model.items()):
            lines.append(f"  (define-fun {name} () Int {val})")
        lines.append(")")
    return "\n".join(lines)
