#!/usr/bin/env python3
"""
Test script for Boogie while loop support in the EFMC converter.
"""

import tempfile
import os
from aria.efmc.frontends.boogie2efmc import BoogieToEFMCConverter


def get_while_loop_programs():
    """Return test programs with while loops."""
    return [
        (
            "Simple While",
            """
implementation SimpleWhile() {
    var x: int;
    x := 10;
    while (x > 0) { x := x - 1; }
    assert x == 0;
}""",
        ),
        (
            "Multiple Variables",
            """
implementation MultipleVars() {
    var x, y: int;
    x := 5; y := 0;
    while (x > 0) { x := x - 1; y := y + 1; }
    assert x + y == 5;
}""",
        ),
        (
            "With Invariant",
            """
implementation WithInvariant() {
    var n, sum, i: int;
    assume n >= 0; sum := 0; i := 0;
    while (i < n) invariant sum >= 0; invariant i >= 0; {
        sum := sum + i; i := i + 1;
    }
    assert sum >= 0;
}""",
        ),
        (
            "Complex Condition",
            """
implementation ComplexWhile() {
    var a, b, counter: int;
    a := 10; b := 5; counter := 0;
    while (a > 0 && b > 0) {
        a := a - 1; b := b - 1; counter := counter + 1;
    }
    assert counter <= 10;
}""",
        ),
        (
            "With Havoc",
            """
implementation WithHavoc() {
    var x, random: int;
    x := 0;
    while (x < 10) {
        havoc random; assume random >= 0 && random <= 1;
        x := x + random;
    }
    assert x >= 10;
}""",
        ),
    ]


def run_while_loop_tests():
    """Run all while loop tests."""
    print("Testing Boogie While Loop Support")
    print("-" * 40)

    programs = get_while_loop_programs()
    passed = 0

    for name, code in programs:
        print(f"Testing: {name}")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bpl", delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            converter = BoogieToEFMCConverter()
            ts = converter.convert_file_to_transition_system(temp_file)
            print(f"✅ {name} - Variables: {len(ts.variables)}")
            passed += 1
        except Exception as e:
            print(f"❌ {name} - {e}")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    print(f"\nResults: {passed} passed, {len(programs) - passed} failed")
    if passed == len(programs):
        print("🎉 All tests passed!")


if __name__ == "__main__":
    run_while_loop_tests()
