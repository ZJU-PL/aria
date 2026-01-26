"""Integration tests for PS_SMTO with SMT-LIB file loading."""

import tempfile
import os

import z3

from aria.ml.llm.smto import PS_SMTOConfig, PS_SMTOSolver, SolvingMode


def test_load_smtlib_string():
    """Test loading SMT-LIB content from string."""
    content = """
(declare-const x Int)
(declare-const y Int)

(declare-nl abs ((Int)) Int
  (nldesc "returns absolute value of input")
  (examples [(10), (-5)])
  (library libfun.so))

(assert (> (abs x) 5))
(assert (< (abs y) 3))
"""
    
    config = PS_SMTOConfig(
        model="gpt-4",
        enable_spec_synthesis=False,  # Disable for faster tests
        mode=SolvingMode.BIDIRECTIONAL,
    )
    
    solver = PS_SMTOSolver(config)
    solver.load_smtlib_string(content)
    
    # Check that oracle was registered
    assert "abs" in solver.oracles
    assert len(solver.oracles) == 1
    
    # Check that constraints were added (solver should have some assertions)
    # Note: We can't easily check the exact assertions without accessing internals
    # but we can verify the solver was set up


def test_load_smtlib_file():
    """Test loading SMT-LIB content from file."""
    content = """
(declare-const x Int)

(declare-nl max ((Int) (Int)) Int
  (nldesc "returns maximum of two inputs")
  (examples [(5 3), (2 7)]))

(assert (> (max x 10) 5))
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
        f.write(content)
        temp_file = f.name
    
    try:
        config = PS_SMTOConfig(
            model="gpt-4",
            enable_spec_synthesis=False,
            mode=SolvingMode.BIDIRECTIONAL,
        )
        
        solver = PS_SMTOSolver(config)
        solver.load_smtlib_file(temp_file)
        
        # Check that oracle was registered
        assert "max" in solver.oracles
        assert len(solver.oracles) == 1
        
    finally:
        os.unlink(temp_file)


def test_multiple_oracles():
    """Test loading multiple oracles from SMT-LIB."""
    content = """
(declare-const t1 Int)
(declare-const t2 Int)
(declare-const t3 Int)

(declare-nl abs ((Int)) Int
  (nldesc "returns absolute value")
  (examples [(10), (-5)]))

(declare-nl max ((Int) (Int) (Int)) Int
  (nldesc "returns maximum")
  (examples [(10 11 12)]))

(declare-nl median ((Int) (Int) (Int)) Int
  (nldesc "returns median")
  (examples [(10 11 12)]))
"""
    
    config = PS_SMTOConfig(
        model="gpt-4",
        enable_spec_synthesis=False,
        mode=SolvingMode.BIDIRECTIONAL,
    )
    
    solver = PS_SMTOSolver(config)
    solver.load_smtlib_string(content)
    
    # Check all oracles were registered
    assert len(solver.oracles) == 3
    assert "abs" in solver.oracles
    assert "max" in solver.oracles
    assert "median" in solver.oracles


def test_constraints_with_oracles():
    """Test that constraints are properly added alongside oracles."""
    content = """
(declare-const x Int)
(declare-const y Int)

(declare-nl abs ((Int)) Int
  (nldesc "returns absolute value")
  (examples [(10), (-5)]))

(assert (> x 0))
(assert (< y 10))
(assert (= (abs x) x))
"""
    
    config = PS_SMTOConfig(
        model="gpt-4",
        enable_spec_synthesis=False,
        mode=SolvingMode.BIDIRECTIONAL,
    )
    
    solver = PS_SMTOSolver(config)
    solver.load_smtlib_string(content)
    
    # Check oracle was registered
    assert "abs" in solver.oracles
    
    # The solver should have assertions (though we can't easily verify exact content)
    # We can at least verify the solver is in a valid state
    assert solver.solver is not None


if __name__ == "__main__":
    test_load_smtlib_string()
    print("✓ test_load_smtlib_string passed")
    
    test_load_smtlib_file()
    print("✓ test_load_smtlib_file passed")
    
    test_multiple_oracles()
    print("✓ test_multiple_oracles passed")
    
    test_constraints_with_oracles()
    print("✓ test_constraints_with_oracles passed")
    
    print("\nAll integration tests passed!")
