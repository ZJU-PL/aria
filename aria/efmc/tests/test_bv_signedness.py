import pytest
import z3
from aria.efmc.tests import TestCase, main
from aria.utils.bv_utils import get_signedness, Signedness


@pytest.fixture
def bv_vars():
    """Fixture providing common bit-vector variables for testing."""
    return {
        "x": z3.BitVec("x", 32),
        "y": z3.BitVec("y", 32),
        "z": z3.BitVec("z", 32),
        "const": z3.BitVecVal(42, 32),
    }


@pytest.mark.parametrize(
    "operation,expected",
    [
        (lambda x, y: z3.ULT(x, y), Signedness.UNSIGNED),
        (lambda x, y: z3.ULE(x, y), Signedness.UNSIGNED),
        (lambda x, y: z3.UGT(x, y), Signedness.UNSIGNED),
        (lambda x, y: z3.UGE(x, y), Signedness.UNSIGNED),
    ],
)
def test_unsigned_operations(bv_vars, operation, expected):
    """Test detection of unsigned operations."""
    formula = operation(bv_vars["x"], bv_vars["y"])
    assert get_signedness(formula) == expected


@pytest.mark.parametrize(
    "operation,expected",
    [
        (lambda x, y: x < y, Signedness.SIGNED),  # Default is signed
        (lambda x, y: x <= y, Signedness.SIGNED),
        (lambda x, y: x > y, Signedness.SIGNED),
        (lambda x, y: x >= y, Signedness.SIGNED),
    ],
)
def test_signed_operations(bv_vars, operation, expected):
    """Test detection of signed operations."""
    formula = operation(bv_vars["x"], bv_vars["y"])
    assert get_signedness(formula) == expected


@pytest.mark.parametrize(
    "operation,expected",
    [
        (lambda x, y: x + y, Signedness.UNKNOWN),
        (lambda x, y: x * y, Signedness.UNKNOWN),
        (lambda x, y: x & y, Signedness.UNKNOWN),
        (lambda x, y: x ^ y, Signedness.UNKNOWN),
        (lambda x, y: x | y, Signedness.UNKNOWN),
        (lambda x, y: ~x, Signedness.UNKNOWN),
    ],
)
def test_unknown_signedness(bv_vars, operation, expected):
    """Test operations that should return unknown signedness."""
    formula = operation(bv_vars["x"], bv_vars["y"])
    assert get_signedness(formula) == expected


def test_complex_nested_operations(bv_vars):
    """Test complex nested operations with mixed signedness."""
    x, y, z = bv_vars["x"], bv_vars["y"], bv_vars["z"]

    # Complex unsigned formula
    unsigned_formula = z3.And(
        z3.ULT(x, y), x + y == z3.BitVecVal(100, 32), z3.UGE(z, x)
    )
    assert get_signedness(unsigned_formula) == Signedness.UNSIGNED

    # Complex signed formula
    signed_formula = z3.Or(
        x < y,  # Signed comparison
        y * z == z3.BitVecVal(42, 32),
        z >= x,  # Signed comparison
    )
    assert get_signedness(signed_formula) == Signedness.SIGNED

    # Mixed operations (should prioritize finding any signed/unsigned indicator)
    mixed_formula = z3.And(
        x + y == z3.BitVecVal(10, 32),
        y ^ z == z3.BitVecVal(5, 32),
        z3.ULT(z, x),  # This makes it unsigned
    )
    assert get_signedness(mixed_formula) == Signedness.UNSIGNED


@pytest.mark.parametrize("width", [1, 8, 16, 32, 64, 128])
def test_different_bitvector_widths(width):
    """Test signedness detection with different bit-vector widths."""
    x = z3.BitVec("x", width)
    y = z3.BitVec("y", width)

    unsigned_formula = z3.ULT(x, y)
    assert get_signedness(unsigned_formula) == Signedness.UNSIGNED

    signed_formula = x < y  # Signed comparison
    assert get_signedness(signed_formula) == Signedness.SIGNED


def test_edge_cases(bv_vars):
    """Test edge cases and special scenarios."""
    x = bv_vars["x"]
    const = bv_vars["const"]

    # Compare with constants
    formula = z3.ULT(x, const)
    assert get_signedness(formula) == Signedness.UNSIGNED

    # Single variable (should be unknown)
    formula = x
    assert get_signedness(formula) == Signedness.UNKNOWN

    # Deeply nested operations - fixing the type mismatch
    # Use z3.Or for Boolean operations instead of | operator
    bv_expression = (x & z3.BitVecVal(1, 32)) * z3.BitVecVal(2, 32) + z3.BitVecVal(
        3, 32
    )
    bool_expression = z3.ULT(bv_expression, const)  # Convert to boolean expression
    deep_formula = z3.Or(z3.ULT(x, const), bool_expression)

    assert get_signedness(deep_formula) == Signedness.UNSIGNED


if __name__ == "__main__":
    main()
