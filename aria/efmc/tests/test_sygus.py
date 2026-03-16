import z3
from aria.efmc.smttools.sygus_solver import synthesize_function, replace_func_with_template


def test_chc_template():
    print("Testing CHC solving with template")
    print("================================")

    # Define integer variables
    x, y = z3.Ints("x y")

    # Define a formula
    fml = z3.And(x > 0, y > 0, x + y < 10)

    # Define a template for the invariant
    template = z3.And(x >= 0, y >= 0, x + y <= z3.Int("c"))

    # Print the formula and template
    print(f"Formula: {fml}")
    print(f"Template: {template}")

    # Create a simple CHC problem
    inv = z3.Function("inv", z3.IntSort(), z3.IntSort(), z3.BoolSort())

    # Replace the uninterpreted function with the template
    chc = z3.And(
        z3.ForAll([x, y], z3.Implies(z3.And(x == 1, y == 1), inv(x, y))),
        z3.ForAll([x, y], z3.Implies(z3.And(inv(x, y), x < 5), inv(x + 1, y))),
        z3.ForAll([x, y], z3.Implies(inv(x, y), x + y < 10)),
    )

    # Replace the inv function with the template
    result = replace_func_with_template(chc, inv, template)

    print("\nResult of replacing inv with template:")
    print(result)
    print()


def test_max_function():
    print("Testing max function synthesis with CVC5")
    print("=======================================")

    # Define a max function
    max_func = z3.Function("max", z3.IntSort(), z3.IntSort(), z3.IntSort())
    a, b = z3.Ints("a b")

    # Define constraints for the max function
    constraints = [
        max_func(a, a) == a,  # Identity when inputs are equal
        max_func(a, b) == max_func(b, a),  # Symmetry
        z3.Implies(a >= b, max_func(a, b) == a),  # Definition of max
    ]

    print("Constraints:")
    for c in constraints:
        print(f"  {c}")

    # Synthesize the max function using CVC5
    print("\nSynthesizing max function using CVC5...")
    max_expr = synthesize_function(
        max_func, constraints, [a, b], logic="LIA", timeout=30, force_cvc5=True
    )

    # Check if max_expr is not None
    if max_expr is not None:
        print(f"\nSynthesized max function: {max_expr}")

        # Test the function with concrete values
        print("\nTesting with concrete values:")
        test_cases = [(1, 2), (3, 2), (5, 5), (-1, -2)]
        for x_val, y_val in test_cases:
            # Create a solver to evaluate the expression with concrete values
            s = z3.Solver()
            x_val_expr = z3.IntVal(x_val)
            y_val_expr = z3.IntVal(y_val)
            result_expr = z3.substitute(max_expr, (a, x_val_expr), (b, y_val_expr))
            s.add(result_expr == z3.IntVal(max(x_val, y_val)))
            print(
                f"  max({x_val}, {y_val}) = {max(x_val, y_val)}, Correct: {s.check() == z3.sat}"
            )
    else:
        print("Failed to synthesize max function")


def test_bv_xor_function():
    print("\nTesting bit-vector XOR function synthesis with CVC5")
    print("==================================================")

    # Define a bit-vector XOR function
    bv8_sort = z3.BitVecSort(8)
    bvxor_func = z3.Function("bvxor_func", bv8_sort, bv8_sort, bv8_sort)
    x_bv, y_bv = z3.BitVecs("x_bv y_bv", 8)

    # Define constraints for the XOR function
    bv_constraints = [
        bvxor_func(z3.BitVecVal(0, 8), z3.BitVecVal(0, 8)) == z3.BitVecVal(0, 8),
        bvxor_func(z3.BitVecVal(0, 8), z3.BitVecVal(1, 8)) == z3.BitVecVal(1, 8),
        bvxor_func(z3.BitVecVal(1, 8), z3.BitVecVal(0, 8)) == z3.BitVecVal(1, 8),
        bvxor_func(z3.BitVecVal(1, 8), z3.BitVecVal(1, 8)) == z3.BitVecVal(0, 8),
    ]

    print("Constraints:")
    for c in bv_constraints:
        print(f"  {c}")

    # Synthesize the XOR function using CVC5
    print("\nSynthesizing bit-vector XOR function using CVC5...")
    bvxor_expr = synthesize_function(
        bvxor_func,
        bv_constraints,
        [x_bv, y_bv],
        logic="BV",
        timeout=30,
        force_cvc5=True,
    )

    # Check if bvxor_expr is not None
    if bvxor_expr is not None:
        print(f"\nSynthesized bit-vector XOR function: {bvxor_expr}")

        # Test the function with concrete values
        print("\nTesting with concrete values:")
        test_cases = [(0, 0), (0, 1), (1, 0), (1, 1), (3, 5), (255, 0)]
        for x_val, y_val in test_cases:
            # Create a solver to evaluate the expression with concrete values
            s = z3.Solver()
            x_val_expr = z3.BitVecVal(x_val, 8)
            y_val_expr = z3.BitVecVal(y_val, 8)
            result_expr = z3.substitute(
                bvxor_expr, (x_bv, x_val_expr), (y_bv, y_val_expr)
            )
            s.add(result_expr == z3.BitVecVal(x_val ^ y_val, 8))
            print(
                f"  {x_val} XOR {y_val} = {x_val ^ y_val}, Correct: {s.check() == z3.sat}"
            )
    else:
        print("Failed to synthesize bit-vector XOR function")


def test_string_concat_function():
    print("\nTesting string concatenation function synthesis with CVC5")
    print("=======================================================")

    # Define a string concatenation function
    str_sort = z3.StringSort()
    concat_func = z3.Function("concat_func", str_sort, str_sort, str_sort)
    s1, s2 = z3.Strings("s1 s2")

    # Define constraints for the concatenation function
    str_constraints = [
        concat_func(z3.StringVal(""), z3.StringVal("")) == z3.StringVal(""),
        concat_func(z3.StringVal("a"), z3.StringVal("")) == z3.StringVal("a"),
        concat_func(z3.StringVal(""), z3.StringVal("b")) == z3.StringVal("b"),
        concat_func(z3.StringVal("a"), z3.StringVal("b")) == z3.StringVal("ab"),
    ]

    print("Constraints:")
    for c in str_constraints:
        print(f"  {c}")

    # Synthesize the concatenation function using CVC5
    print("\nSynthesizing string concatenation function using CVC5...")
    concat_expr = synthesize_function(
        concat_func, str_constraints, [s1, s2], logic="S", timeout=30, force_cvc5=True
    )

    # Check if concat_expr is not None
    if concat_expr is not None:
        print(f"\nSynthesized string concatenation function: {concat_expr}")

        # Test the function with concrete values
        print("\nTesting with concrete values:")
        test_cases = [("", ""), ("a", ""), ("", "b"), ("a", "b"), ("hello", "world")]
        for x_val, y_val in test_cases:
            # Create a solver to evaluate the expression with concrete values
            s = z3.Solver()
            x_val_expr = z3.StringVal(x_val)
            y_val_expr = z3.StringVal(y_val)
            result_expr = z3.substitute(concat_expr, (s1, x_val_expr), (s2, y_val_expr))
            s.add(result_expr == z3.StringVal(x_val + y_val))
            print(
                f'  "{x_val}" + "{y_val}" = "{x_val + y_val}", Correct: {s.check() == z3.sat}'
            )
    else:
        print("Failed to synthesize string concatenation function")


if __name__ == "__main__":
    test_chc_template()
    test_max_function()
    test_bv_xor_function()
    test_string_concat_function()
