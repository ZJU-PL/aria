"""
QF_BV pipeline smoke tests.
Verifies the full bit-vector path: parse → encode → GAN → solver → verify.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from gansat.parser      import parse_string
from gansat.bv_encoder  import bv_encode, bv_decode_assignment, bv_feature_dim, MAX_VARS
from gansat.bv_gan      import (
    BVIterativeGenerator, BVDiscriminator, BVViolationComputer,
    BVInitialGuesser, BVRefinementStep,
    BV_FORMULA_DIM, BV_ASSIGN_DIM, BV_NOISE_DIM,
)
from gansat.solver      import GANSATSolver, RESULT_SAT, RESULT_UNSAT

# ── Test formulas ─────────────────────────────────────────────────────────────

BV32_SAT = """
(set-logic QF_BV)
(declare-fun x () (_ BitVec 32))
(declare-fun y () (_ BitVec 32))
(assert (bvult x (_ bv100 32)))
(assert (bvugt x (_ bv10 32)))
(assert (bvult y (_ bv100 32)))
(assert (bvugt y (_ bv10 32)))
(assert (= (bvadd x y) (_ bv50 32)))
(check-sat)
"""

BV8_SAT = """
(set-logic QF_BV)
(declare-fun a () (_ BitVec 8))
(declare-fun b () (_ BitVec 8))
(assert (bvult a (_ bv200 8)))
(assert (bvugt b (_ bv5 8)))
(assert (= (bvand a b) (_ bv0 8)))
(check-sat)
"""

BV_UNSAT = """
(set-logic QF_BV)
(declare-fun x () (_ BitVec 4))
(assert (bvugt x (_ bv15 4)))
(check-sat)
"""

BV_KLEE_STYLE = """
(set-logic QF_ABV)
(declare-fun symb () (_ BitVec 32))
(assert (bvult symb (_ bv7 32)))
(assert (bvugt symb (_ bv0 32)))
(assert (not (= symb (_ bv3 32))))
(check-sat)
"""


def test_bv_parser():
    f = parse_string(BV32_SAT)
    assert f.logic == "QF_BV"
    assert "x" in f.variables and "y" in f.variables
    import z3
    assert z3.is_bv_sort(f.variables["x"].sort())
    print(f"[PASS] BV parser — logic={f.logic}, vars={list(f.variables)}")


def test_bv_encoder():
    f   = parse_string(BV32_SAT)
    enc = bv_encode(f)
    assert enc.shape == (bv_feature_dim(),), f"expected {bv_feature_dim()}, got {enc.shape}"
    assert enc.dtype == np.float32
    print(f"[PASS] BV encoder — dim={bv_feature_dim()}")


def test_bv_violation_computer():
    f   = parse_string(BV32_SAT)
    enc = torch.tensor(bv_encode(f), dtype=torch.float32).unsqueeze(0)
    vc  = BVViolationComputer()

    zero_assign = torch.zeros(1, BV_ASSIGN_DIM)
    cv, vv = vc(enc, zero_assign)
    assert cv.shape == (1, 128)
    assert vv.shape == (1, 64)
    print(f"[PASS] BV ViolationComputer — viol_sum={cv.sum().item():.4f}")


def test_bv_initial_guesser():
    f   = parse_string(BV32_SAT)
    enc = torch.tensor(bv_encode(f), dtype=torch.float32).unsqueeze(0)
    z   = torch.randn(1, BV_NOISE_DIM)

    ig  = BVInitialGuesser()
    out = ig(enc, z)
    assert out.shape == (1, BV_ASSIGN_DIM)
    assert out.abs().max().item() <= 1.0 + 1e-5
    print("[PASS] BV InitialGuesser")


def test_bv_refinement_step():
    f   = parse_string(BV32_SAT)
    enc = torch.tensor(bv_encode(f), dtype=torch.float32).unsqueeze(0)
    x   = torch.zeros(1, BV_ASSIGN_DIM)
    cv  = torch.rand(1, 128)
    vv  = torch.rand(1, 64)

    rs      = BVRefinementStep()
    refined = rs(enc, x, cv, vv)
    assert refined.shape == (1, BV_ASSIGN_DIM)
    assert refined.abs().max().item() <= 1.0 + 1e-5
    print("[PASS] BV RefinementStep")


def test_bv_generator_trajectory():
    f   = parse_string(BV32_SAT)
    enc = torch.tensor(bv_encode(f), dtype=torch.float32).unsqueeze(0)

    G = BVIterativeGenerator(n_rounds=3)
    final, traj = G(enc, return_trajectory=True)

    assert final.shape == (1, BV_ASSIGN_DIM)
    assert len(traj) == 4
    vc     = BVViolationComputer()
    viols  = [vc(enc, t)[0].sum().item() for t in traj]
    print(f"[PASS] BV Generator trajectory — violations: {[f'{v:.4f}' for v in viols]}")


def test_bv_generator_sample():
    f   = parse_string(BV32_SAT)
    enc = torch.tensor(bv_encode(f), dtype=torch.float32).unsqueeze(0)

    G       = BVIterativeGenerator()
    samples = G.sample(enc, n_samples=8)
    assert samples.shape == (1, 8, BV_ASSIGN_DIM)
    print("[PASS] BV Generator.sample")


def test_bv_discriminator():
    f   = parse_string(BV32_SAT)
    enc = torch.tensor(bv_encode(f), dtype=torch.float32).unsqueeze(0)
    x   = torch.zeros(1, BV_ASSIGN_DIM)

    D     = BVDiscriminator()
    logit = D(enc, x)
    assert logit.shape == (1,)
    print(f"[PASS] BV Discriminator — logit={logit.item():.4f}")


def test_bv_decode():
    f   = parse_string(BV32_SAT)
    vec = np.zeros(64, dtype=np.float32)
    assignment = bv_decode_assignment(vec, f)
    assert isinstance(assignment, dict)
    for val in assignment.values():
        assert val >= 0
    print(f"[PASS] BV decode — {assignment}")


def test_solver_bv_sat():
    solver = GANSATSolver(n_candidates=16, timeout_ms=5000)
    result, model, elapsed = solver.solve_string(BV32_SAT)
    assert result == RESULT_SAT, f"Expected sat, got {result}"
    assert "x" in model and "y" in model
    x, y = model["x"], model["y"]
    assert 10 < x < 100, f"x={x} out of range"
    assert 10 < y < 100, f"y={y} out of range"
    assert x + y == 50,  f"x+y={x+y} != 50"
    print(f"[PASS] BV Solver SAT — {elapsed:.1f}ms  x={x} y={y}  x+y={x+y}")


def test_solver_bv_unsat():
    solver = GANSATSolver(n_candidates=16, timeout_ms=5000)
    result, model, elapsed = solver.solve_string(BV_UNSAT)
    assert result == RESULT_UNSAT, f"Expected unsat, got {result}"
    print(f"[PASS] BV Solver UNSAT — {elapsed:.1f}ms")


def test_solver_klee_style():
    solver = GANSATSolver(n_candidates=16, timeout_ms=5000)
    result, model, elapsed = solver.solve_string(BV_KLEE_STYLE)
    assert result == RESULT_SAT, f"Expected sat, got {result}"
    val = model.get("symb", -1)
    assert 0 < val < 7 and val != 3, f"symb={val} violates constraints"
    print(f"[PASS] BV Solver KLEE-style — {elapsed:.1f}ms  symb={val}")


def test_solver_bv8():
    solver = GANSATSolver(n_candidates=16, timeout_ms=5000)
    result, model, elapsed = solver.solve_string(BV8_SAT)
    assert result == RESULT_SAT, f"Expected sat, got {result}"
    print(f"[PASS] BV8 Solver — {elapsed:.1f}ms  model={model}")


def test_bridge_script():
    """Verify the KLEE bridge script works as a subprocess."""
    import subprocess, sys
    bridge = str(Path(__file__).resolve().parent.parent / "klee_plugin" / "gansat_bridge.py")
    proc   = subprocess.run(
        [sys.executable, bridge],
        input=BV32_SAT, capture_output=True, text=True, timeout=30,
    )
    assert "sat" in proc.stdout.lower(), f"Bridge output: {proc.stdout}"
    print(f"[PASS] gansat_bridge.py subprocess — output: {proc.stdout[:40].strip()}")


if __name__ == "__main__":
    test_bv_parser()
    test_bv_encoder()
    test_bv_violation_computer()
    test_bv_initial_guesser()
    test_bv_refinement_step()
    test_bv_generator_trajectory()
    test_bv_generator_sample()
    test_bv_discriminator()
    test_bv_decode()
    test_solver_bv_sat()
    test_solver_bv_unsat()
    test_solver_klee_style()
    test_solver_bv8()
    test_bridge_script()
    print("\n[ALL BV TESTS PASSED]")
