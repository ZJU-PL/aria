"""Smoke tests — verify the full pipeline end-to-end without a trained model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import z3

from gansat.parser import parse_string
from gansat.encoder import encode, decode_assignment, feature_dim
from gansat.solver import GANSATSolver, RESULT_SAT, RESULT_UNSAT
from gansat.gan import (
    IterativeGenerator, Discriminator, ViolationComputer,
    RefinementStep, InitialGuesser, assignment_to_tensor,
    FORMULA_DIM, ASSIGN_DIM, CONSTRAINT_DIM,
)

SIMPLE_SAT = """
(set-logic QF_LIA)
(declare-fun x () Int)
(declare-fun y () Int)
(assert (>= x 0))
(assert (<= x 10))
(assert (>= y 0))
(assert (<= y 10))
(assert (= (+ x y) 7))
(check-sat)
"""

SIMPLE_UNSAT = """
(set-logic QF_LIA)
(declare-fun x () Int)
(assert (>= x 5))
(assert (<= x 3))
(check-sat)
"""


def test_parser():
    formula = parse_string(SIMPLE_SAT)
    assert formula.logic == "QF_LIA"
    assert "x" in formula.variables and "y" in formula.variables
    print("[PASS] parser")


def test_encoder():
    formula = parse_string(SIMPLE_SAT)
    enc = encode(formula)
    assert enc.shape == (feature_dim(),)
    assert enc.dtype == np.float32
    print(f"[PASS] encoder — dim={feature_dim()}")


def test_violation_computer():
    formula = parse_string(SIMPLE_SAT)
    enc = torch.tensor(encode(formula), dtype=torch.float32).unsqueeze(0)

    vc = ViolationComputer()

    # All-zero assignment should violate x+y=7
    zero_assign = torch.zeros(1, ASSIGN_DIM)
    c_viol, v_viol = vc(enc, zero_assign)
    assert c_viol.shape == (1, 128)
    assert v_viol.shape == (1, 64)
    assert c_viol.sum() > 0, "Zero assignment should have violations"

    print(f"[PASS] ViolationComputer — violations sum={c_viol.sum().item():.4f}")


def test_initial_guesser():
    formula = parse_string(SIMPLE_SAT)
    enc   = torch.tensor(encode(formula), dtype=torch.float32).unsqueeze(0)
    noise = torch.randn(1, 128)

    ig = InitialGuesser()
    out = ig(enc, noise)
    assert out.shape == (1, ASSIGN_DIM)
    assert out.abs().max() <= 1.0 + 1e-5, "Output must be in [-1,1] (Tanh)"
    print("[PASS] InitialGuesser")


def test_refinement_step():
    formula = parse_string(SIMPLE_SAT)
    enc    = torch.tensor(encode(formula), dtype=torch.float32).unsqueeze(0)
    assign = torch.zeros(1, ASSIGN_DIM)
    c_viol = torch.rand(1, CONSTRAINT_DIM)
    v_viol = torch.rand(1, ASSIGN_DIM)

    rs = RefinementStep()
    refined = rs(enc, assign, c_viol, v_viol)
    assert refined.shape == (1, ASSIGN_DIM)
    assert refined.abs().max() <= 1.0 + 1e-5
    print("[PASS] RefinementStep")


def test_iterative_generator_trajectory():
    formula = parse_string(SIMPLE_SAT)
    enc = torch.tensor(encode(formula), dtype=torch.float32).unsqueeze(0)

    G = IterativeGenerator(n_rounds=3)
    final, traj = G(enc, return_trajectory=True)

    assert final.shape == (1, ASSIGN_DIM)
    assert len(traj) == 4, f"Expected 4 steps (round 0 + 3 refinements), got {len(traj)}"

    # Violation should decrease (or at least not explode) across refinement rounds
    vc = ViolationComputer()
    viol_scores = [vc(enc, t)[0].sum().item() for t in traj]
    print(f"[PASS] IterativeGenerator trajectory — violations: {[f'{v:.4f}' for v in viol_scores]}")


def test_generator_sample():
    formula = parse_string(SIMPLE_SAT)
    enc = torch.tensor(encode(formula), dtype=torch.float32).unsqueeze(0)

    G = IterativeGenerator()
    samples = G.sample(enc, n_samples=8)
    assert samples.shape == (1, 8, ASSIGN_DIM)
    print("[PASS] Generator.sample — shape correct")


def test_discriminator():
    formula = parse_string(SIMPLE_SAT)
    enc    = torch.tensor(encode(formula), dtype=torch.float32).unsqueeze(0)
    assign = torch.zeros(1, ASSIGN_DIM)

    D = Discriminator()
    logit = D(enc, assign)
    assert logit.shape == (1,)
    print(f"[PASS] Discriminator — logit={logit.item():.4f}")


def test_violation_score():
    formula = parse_string(SIMPLE_SAT)
    enc = torch.tensor(encode(formula), dtype=torch.float32).unsqueeze(0)

    G = IterativeGenerator()
    assign = G(enc)
    score  = G.violation_score(enc, assign)
    assert score.shape == (1,)
    print(f"[PASS] violation_score — score={score.item():.4f}")


def test_solver_sat():
    solver = GANSATSolver(model_path=None, n_candidates=8, timeout_ms=5000)
    result, model, elapsed = solver.solve_string(SIMPLE_SAT)
    assert result == RESULT_SAT, f"Expected sat, got {result}"
    assert "x" in model or "y" in model
    print(f"[PASS] solver SAT — {elapsed:.1f}ms  model={model}")


def test_solver_unsat():
    solver = GANSATSolver(model_path=None, n_candidates=8, timeout_ms=5000)
    result, model, elapsed = solver.solve_string(SIMPLE_UNSAT)
    assert result == RESULT_UNSAT, f"Expected unsat, got {result}"
    print(f"[PASS] solver UNSAT — {elapsed:.1f}ms")


if __name__ == "__main__":
    test_parser()
    test_encoder()
    test_violation_computer()
    test_initial_guesser()
    test_refinement_step()
    test_iterative_generator_trajectory()
    test_generator_sample()
    test_discriminator()
    test_violation_score()
    test_solver_sat()
    test_solver_unsat()
    print("\n[ALL TESTS PASSED]")
