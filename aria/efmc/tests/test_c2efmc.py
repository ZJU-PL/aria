import os
import textwrap
import tempfile

import z3

from aria.efmc.frontends.c2efmc import c_to_efmc


def _write_tmp_c(code: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".c")
    with os.fdopen(fd, "w") as f:
        f.write(textwrap.dedent(code))
    return path


def test_simple_while_translation():
    path = _write_tmp_c(
        """
        int main() {
            int x;
            x = 0;
            while (x < 3) { x = x + 1; }
            assert(x == 3);
        }
        """
    )
    try:
        ts = c_to_efmc(path)
        x = ts.variables[0]
        xp = ts.prime_variables[0]

        assert len(ts.variables) == 1
        assert str(x) == "x"
        assert str(xp) == "x!"

        _assert_equiv(ts.init, x == 0)
        _assert_equiv(ts.trans, z3.And(x < 3, xp == x + 1))
        _assert_equiv(ts.post, z3.Implies(z3.Not(x < 3), x == 3))
    finally:
        os.unlink(path)


def test_for_loop_normalization():
    path = _write_tmp_c(
        """
        int main() {
            int i;
            for (i = 0; i < 2; i = i + 1) { }
            assert(i == 2);
        }
        """
    )
    try:
        ts = c_to_efmc(path)
        i = ts.variables[0]
        ip = ts.prime_variables[0]

        _assert_equiv(ts.init, i == 0)
        _assert_equiv(ts.trans, z3.And(i < 2, ip == i + 1))
        _assert_equiv(ts.post, z3.Implies(z3.Not(i < 2), i == 2))
    finally:
        os.unlink(path)


def test_assume_added_to_guard():
    path = _write_tmp_c(
        """
        int main() {
            int x;
            x = 1;
            while (x < 5) { __VERIFIER_assume(x >= 0); x = x + 2; }
        }
        """
    )
    try:
        ts = c_to_efmc(path)
        x = ts.variables[0]
        xp = ts.prime_variables[0]

        # Transition must imply the assume guard
        solver = z3.Solver()
        solver.add(ts.trans, z3.Not(x >= 0))
        assert solver.check() == z3.unsat

        # And still carry the expected update
        _assert_equiv(ts.trans, z3.And(x < 5, x >= 0, xp == x + 2))
    finally:
        os.unlink(path)


def test_if_else_paths_in_transition():
    path = _write_tmp_c(
        """
        int main() {
            int x = 0;
            while (x < 5) {
                if (x % 2 == 0) {
                    x = x + 2;
                } else {
                    x = x + 3;
                }
            }
        }
        """
    )
    try:
        ts = c_to_efmc(path)
        x, xp = ts.variables[0], ts.prime_variables[0]

        even_path = z3.And(x % 2 == 0, xp == x + 2)
        odd_path = z3.And(z3.Not(x % 2 == 0), xp == x + 3)
        _assert_equiv(ts.init, x == 0)
        _assert_equiv(ts.trans, z3.And(x < 5, z3.Or(even_path, odd_path)))
    finally:
        os.unlink(path)


def test_post_assertions_are_guarded():
    path = _write_tmp_c(
        """
        int main() {
            int x = 2;
            while (x < 3) { x = x + 1; }
            if (x > 1) {
                __VERIFIER_assert(x == 3);
            } else {
                __VERIFIER_assert(x == 1);
            }
        }
        """
    )
    try:
        ts = c_to_efmc(path)
        x = ts.variables[0]
        post = ts.post

        exit_guard = z3.Not(x < 3)
        guarded = z3.And(
            z3.Implies(z3.And(exit_guard, x > 1), x == 3),
            z3.Implies(z3.And(exit_guard, z3.Not(x > 1)), x == 1),
        )
        _assert_equiv(post, guarded)
    finally:
        os.unlink(path)


def test_unmodified_variables_are_preserved():
    path = _write_tmp_c(
        """
        int main() {
            int x = 1;
            int y = 2;
            while (x < 4) { x += 1; }
        }
        """
    )
    try:
        ts = c_to_efmc(path)
        x, y = ts.variables
        xp, yp = ts.prime_variables

        expected_trans = z3.And(x < 4, z3.And(xp == x + 1, yp == y))
        _assert_equiv(ts.trans, expected_trans)
    finally:
        os.unlink(path)


def test_no_loop_raises_error():
    path = _write_tmp_c(
        """
        int main() {
            int x = 0;
            x = x + 1;
            assert(x == 1);
        }
        """
    )
    try:
        try:
            c_to_efmc(path)
        except ValueError as exc:
            assert "No loop found" in str(exc)
        else:
            raise AssertionError("Expected ValueError for missing loop")
    finally:
        os.unlink(path)


def test_unsupported_assignment_raises():
    path = _write_tmp_c(
        """
        int main() {
            int x = 1;
            while (x < 3) { x *= 2; }
        }
        """
    )
    try:
        try:
            c_to_efmc(path)
        except NotImplementedError as exc:
            assert "*=" in str(exc)
        else:
            raise AssertionError("Expected NotImplementedError for '*='")
    finally:
        os.unlink(path)


def _assert_equiv(e1: z3.ExprRef, e2: z3.ExprRef):
    """Assert two boolean expressions are equivalent via a solver check."""
    s = z3.Solver()
    s.add(z3.Xor(e1, e2))
    assert s.check() == z3.unsat


if __name__ == "__main__":
    test_simple_while_translation()
    test_for_loop_normalization()
    test_assume_added_to_guard()
    test_if_else_paths_in_transition()
    test_post_assertions_are_guarded()
    test_unmodified_variables_are_preserved()
    test_no_loop_raises_error()
    test_unsupported_assignment_raises()
