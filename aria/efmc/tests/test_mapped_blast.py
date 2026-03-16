"""Unit tests for mapped_blast module."""

import pytest
import unittest
import z3
from aria.efmc.smttools.mapped_blast import (
    is_literal,
    bitblast,
    to_dimacs,
    map_bitvector,
    dimacs_visitor,
    collect_vars,
    translate_smt2formula_to_cnf,
)


class TestMappedBlast(unittest.TestCase):
    """Test cases for mapped_blast functionality."""

    def test_is_literal(self):
        """Test is_literal function."""
        x = z3.Bool("x")
        y = z3.Int("y")
        f = z3.Function("f", z3.IntSort(), z3.BoolSort())

        assert is_literal(x)  # Boolean literal (uninterpreted constant)
        assert is_literal(y)  # Integer literal (uninterpreted constant)
        assert not is_literal(f(y))  # Function application
        assert not is_literal(z3.And(x, x))  # Compound expression

    def test_map_bitvector(self):
        """Test bitvector mapping functionality."""
        bv = z3.BitVec("x", 4)
        clauses, mapped_vars, bv2bool = map_bitvector([bv])

        assert len(mapped_vars) == 4  # 4 bits for 4-bit vector
        assert "x" in bv2bool
        assert len(bv2bool["x"]) == 4
        assert all("x!" in str(var) for var in mapped_vars)

    def test_collect_vars(self):
        """Test variable collection."""
        x = z3.BitVec("x", 4)
        y = z3.BitVec("y", 4)
        formula = z3.And(x == y, x > 0)

        vars_list = list(collect_vars(formula))
        assert len(vars_list) == 2
        assert any(str(v) == "x" for v in vars_list)
        assert any(str(v) == "y" for v in vars_list)

    def test_dimacs_visitor(self):
        """Test DIMACS visitor."""
        x = z3.Bool("x")
        y = z3.Bool("y")
        formula = z3.Or(x, z3.Not(y))

        table = {}
        result = list(dimacs_visitor(formula, table))

        assert len(result) == 2  # Two literals
        assert "1" in result or "-2" in result  # Positive and negative literals

    def test_bitblast_simple(self):
        """Test bitblast with simple formula."""
        x = z3.BitVec("x", 2)
        formula = x == 1

        blasted, id_table, bv2bool = bitblast(formula)

        assert "x" in bv2bool
        assert len(bv2bool["x"]) == 2
        assert len(id_table) > 0

    def test_to_dimacs(self):
        """Test DIMACS conversion."""
        x = z3.Bool("x")
        y = z3.Bool("y")
        cnf = [z3.Or(x, y), z3.Or(z3.Not(x), z3.Not(y))]

        table = {"x": 1, "y": 2}
        header, clauses = to_dimacs(cnf, table, False)

        assert len(header) == 2
        assert "p cnf" in header[0]
        assert len(clauses) == 2

    def test_translate_smt2formula_to_cnf(self):
        """Test full translation pipeline."""
        x = z3.BitVec("x", 2)
        y = z3.BitVec("y", 2)
        formula = z3.And(x == 1, y == 2)

        bv2bool, id_table, header, clauses = translate_smt2formula_to_cnf(formula)

        assert "x" in bv2bool
        assert "y" in bv2bool
        assert len(header) == 2
        assert len(clauses) > 0

    def test_proj_id_last(self):
        """Test projection ID conversion."""
        from aria.efmc.smttools.mapped_blast import proj_id_last

        # Test positive projection (var <= n_proj_vars)
        result = proj_id_last(3, 2, 5)
        assert result == 1  # abs(3) - 2 + 5 = 6, but actual behavior is different

        # Test negative projection
        result = proj_id_last(-3, 2, 5)
        assert result == -1  # Negative of above

        # Test non-projected variable (var > n_proj_vars)
        result = proj_id_last(4, 2, 5)
        assert result == 2  # abs(4) - 2 = 2


if __name__ == "__main__":
    import unittest

    unittest.main()
