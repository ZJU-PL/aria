"""Tests for SMT-LIB parser with oracle declarations."""

import tempfile
import os

import z3

from aria.ml.llm.smto.smtlib_parser import parse_smtlib_file, parse_smtlib_string


def test_parse_simple_oracle():
    """Test parsing a simple oracle declaration."""
    content = """
(declare-nl abs ((Int)) Int
  (nldesc "returns absolute value of input")
  (examples [(10), (-5)])
  (library libfun.so))
"""
    
    oracles, remaining = parse_smtlib_string(content)
    
    assert len(oracles) == 1
    oracle = oracles[0]
    assert oracle.name == "abs"
    assert len(oracle.input_types) == 1
    assert oracle.input_types[0] == z3.IntSort()
    assert oracle.output_type == z3.IntSort()
    assert oracle.description == "returns absolute value of input"
    assert len(oracle.examples) == 2


def test_parse_multiple_oracles():
    """Test parsing multiple oracle declarations."""
    content = """
(declare-nl abs ((Int)) Int
  (nldesc "returns absolute value")
  (examples [(10), (-5)]))

(declare-nl max ((Int) (Int) (Int)) Int
  (nldesc "returns maximum of three inputs")
  (examples [(10 11 12), (5 4 6)]))
"""
    
    oracles, remaining = parse_smtlib_string(content)
    
    assert len(oracles) == 2
    assert oracles[0].name == "abs"
    assert oracles[1].name == "max"
    assert len(oracles[1].input_types) == 3


def test_parse_with_constraints():
    """Test parsing oracles with SMT-LIB constraints."""
    content = """
(declare-const x Int)
(declare-const y Int)

(declare-nl abs ((Int)) Int
  (nldesc "returns absolute value")
  (examples [(10), (-5)]))

(assert (> x 0))
(assert (< y 10))
"""
    
    oracles, remaining = parse_smtlib_string(content)
    
    assert len(oracles) == 1
    assert "declare-const x" in remaining
    assert "declare-const y" in remaining
    assert "assert (> x 0)" in remaining or "(assert (> x 0))" in remaining


def test_parse_file():
    """Test parsing from a file."""
    content = """
(declare-nl abs ((Int)) Int
  (nldesc "returns absolute value")
  (examples [(10), (-5)]))
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
        f.write(content)
        temp_file = f.name
    
    try:
        oracles, remaining = parse_smtlib_file(temp_file)
        assert len(oracles) == 1
        assert oracles[0].name == "abs"
    finally:
        os.unlink(temp_file)


def test_parse_bitvector():
    """Test parsing bit-vector types."""
    content = """
(declare-nl ctz ((_ BitVec 16)) (_ BitVec 16)
  (nldesc "returns number of trailing zeros")
  (examples [(8), (3)]))
"""
    
    oracles, remaining = parse_smtlib_string(content)
    
    assert len(oracles) == 1
    oracle = oracles[0]
    assert oracle.name == "ctz"
    assert z3.is_bv_sort(oracle.input_types[0])
    assert oracle.input_types[0].size() == 16
    assert z3.is_bv_sort(oracle.output_type)
    assert oracle.output_type.size() == 16


def test_parse_examples_with_commas():
    """Test parsing examples with comma-separated values."""
    content = """
(declare-nl max ((Int) (Int) (Int)) Int
  (nldesc "returns maximum")
  (examples [[10, 11, 12], [5, 4, 6]]))
"""
    
    oracles, remaining = parse_smtlib_string(content)
    
    assert len(oracles) == 1
    oracle = oracles[0]
    assert len(oracle.examples) == 2
    # Check first example
    ex1 = oracle.examples[0]
    assert ex1["input"]["arg0"] == 10
    assert ex1["input"]["arg1"] == 11
    assert ex1["input"]["arg2"] == 12


def test_parse_no_examples():
    """Test parsing oracle without examples."""
    content = """
(declare-nl func ((Int)) Int
  (nldesc "some function")
  (library lib.so))
"""
    
    oracles, remaining = parse_smtlib_string(content)
    
    assert len(oracles) == 1
    oracle = oracles[0]
    assert oracle.name == "func"
    assert len(oracle.examples) == 0


if __name__ == "__main__":
    test_parse_simple_oracle()
    print("✓ test_parse_simple_oracle passed")
    
    test_parse_multiple_oracles()
    print("✓ test_parse_multiple_oracles passed")
    
    test_parse_with_constraints()
    print("✓ test_parse_with_constraints passed")
    
    test_parse_file()
    print("✓ test_parse_file passed")
    
    test_parse_bitvector()
    print("✓ test_parse_bitvector passed")
    
    test_parse_examples_with_commas()
    print("✓ test_parse_examples_with_commas passed")
    
    test_parse_no_examples()
    print("✓ test_parse_no_examples passed")
    
    print("\nAll tests passed!")
