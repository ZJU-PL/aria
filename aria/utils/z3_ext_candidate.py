"""
Some APIs/functions for playing with Z3 expr (cont.)

- absolute_value_bv: Compute absolute value for bitvectors
- absolute_value_int: Compute absolute value for integers
- ground_quantifier: Extract body and variables from a single quantifier
- ground_quantifier_all: Extract body and variables from nested quantifiers (preserves quantifier structure)
- reconstruct_quantified_formula: Reconstruct a quantified formula from its body and quantifier info
- native_to_dnf: Convert a Z3 expression to DNF (may have limitations for complex expressions)
- to_dnf_boolean: Convert a boolean Z3 expression to DNF (more reliable for boolean expressions)
"""
import z3



def ground_quantifier(qexpr):
    """
    It seems this can only handle exists x . fml, or forall x.fml, that is
    the quantifier is at the outermost level of the formula.
    """
    body = qexpr.body()
    var_list = list()
    for i in range(qexpr.num_vars()):
        vi_name = qexpr.var_name(i)
        vi_sort = qexpr.var_sort(i)
        vi = z3.Const(vi_name, vi_sort)
        var_list.append(vi)

    body = z3.substitute_vars(body, *var_list)
    return body, var_list


def ground_quantifier_all(qexpr):
    """
    Handle nested quantifiers like Exists x . Forall y . Exists z . fml
    This version preserves:
    - order of the quantifiers
    - which variables are associated with which quantifiers

    Args:
        qexpr: A Z3 expression with quantifiers (ForAll or Exists)

    Returns:
        A tuple (body, quantifier_info) where:
        - body: The innermost expression with all quantifiers removed
        - quantifier_info: A list of tuples (quantifier_type, variables) where:
          * quantifier_type is a boolean (True for ForAll, False for Exists)
          * variables is a list of Z3 constants for that quantifier
          * The list is ordered from outermost to innermost quantifier

    Example:
        For the formula ForAll x, Exists y, ForAll z, body
        This returns (body, [(True, [x]), (False, [y]), (True, [z])])

    Note: This implementation does not preserve patterns in quantifiers.
    Patterns are used for controlling the SMT solver's instantiation
    heuristics and are not part of the logical meaning of the formula.
    """
    quantifier_info = []
    exp = qexpr

    while z3.is_quantifier(exp):
        is_forall = exp.is_forall()
        var_list = []

        # Extract variables from this quantifier level
        for i in range(exp.num_vars()):
            vi_name = exp.var_name(i)
            vi_sort = exp.var_sort(i)
            vi = z3.Const(vi_name, vi_sort)
            var_list.append(vi)

        # Store quantifier type and its variables
        quantifier_info.append((is_forall, var_list))

        # Get the body and continue with the next level
        body = exp.body()
        body = z3.substitute_vars(body, *var_list)
        exp = body

    # Return the final body and the quantifier information
    return exp, quantifier_info


def reconstruct_quantified_formula(body, quantifier_info):
    """
    Reconstruct a quantified formula from its body and quantifier information.

    Args:
        body: The innermost expression (without quantifiers)
        quantifier_info: List of tuples (is_forall, var_list) as returned by
                        ground_quantifier_all, ordered from outermost
                        to innermost quantifier

    Returns:
        A Z3 expression with the original quantifier structure restored

    Example:
        Given body and [(True, [x]), (False, [y]), (True, [z])]
        This returns ForAll x, Exists y, ForAll z, body

    Note:
        The reconstructed formula will be semantically equivalent to the original,
        but may have syntactic differences due to how Z3 handles term reordering.
    """
    result = body

    # Process quantifiers from innermost to outermost (reverse order)
    for is_forall, var_list in reversed(quantifier_info):
        if is_forall:
            result = z3.ForAll(var_list, result)
        else:
            result = z3.Exists(var_list, result)

    return result


def to_dnf_boolean(expr):
    """
    Convert a boolean Z3 expression to DNF (Disjunctive Normal Form).

    This function implements a more reliable approach to DNF conversion
    for boolean expressions. It does not use the Tseitin transformation
    and avoids introducing auxiliary variables.

    Args:
        expr: A Z3 boolean expression

    Returns:
        A Z3 expression in DNF form

    Note:
        This function only works for boolean expressions, not for
        expressions involving integers, reals, or other types.
    """
    # Base cases
    if z3.is_const(expr) or (z3.is_not(expr) and z3.is_const(expr.arg(0))):
        return expr

    # Handle OR: already in DNF if all arguments are literals or AND of literals
    if z3.is_or(expr):
        return z3.Or([to_dnf_boolean(arg) for arg in expr.children()])

    # Handle AND: distribute AND over OR
    if z3.is_and(expr):
        args = [to_dnf_boolean(arg) for arg in expr.children()]

        # Find any OR arguments
        or_args = []
        for arg in args:
            if z3.is_or(arg):
                or_args.append(arg)

        if not or_args:
            # No OR arguments, already in DNF
            return z3.And(args)

        # Take the first OR argument and distribute
        or_arg = or_args[0]
        other_args = []
        for arg in args:
            if arg.eq(or_arg) is not True:  # Use eq() instead of !=
                other_args.append(arg)

        # Distribute AND over OR
        distributed_terms = []
        for or_child in or_arg.children():
            distributed_terms.append(z3.And([or_child] + other_args))

        distributed = z3.Or(distributed_terms)

        # Recursively convert the distributed expression to DNF
        return to_dnf_boolean(distributed)

    # Handle NOT: apply De Morgan's laws
    if z3.is_not(expr):
        arg = expr.arg(0)

        # NOT(NOT(p)) = p
        if z3.is_not(arg):
            return to_dnf_boolean(arg.arg(0))

        # NOT(AND(p, q, ...)) = OR(NOT(p), NOT(q), ...)
        if z3.is_and(arg):
            return to_dnf_boolean(z3.Or([z3.Not(child) for child in arg.children()]))

        # NOT(OR(p, q, ...)) = AND(NOT(p), NOT(q), ...)
        if z3.is_or(arg):
            return to_dnf_boolean(z3.And([z3.Not(child) for child in arg.children()]))

    # Handle other operations by converting to basic operations
    if z3.is_implies(expr):
        # p => q is equivalent to NOT(p) OR q
        p, q = expr.children()
        return to_dnf_boolean(z3.Or(z3.Not(p), q))

    if expr.decl().kind() == z3.Z3_OP_XOR:
        # p XOR q is equivalent to (p AND NOT(q)) OR (NOT(p) AND q)
        p, q = expr.children()
        return to_dnf_boolean(z3.Or(z3.And(p, z3.Not(q)), z3.And(z3.Not(p), q)))

    if expr.decl().kind() == z3.Z3_OP_IFF:
        # p <=> q is equivalent to (p AND q) OR (NOT(p) AND NOT(q))
        p, q = expr.children()
        return to_dnf_boolean(z3.Or(z3.And(p, q), z3.And(z3.Not(p), z3.Not(q))))

    # For other expressions, return as is
    return expr


def test_quant():
    """
    Demonstrate the usage of ground_quantifier_all
    with a few essential test cases.
    """
    print("\n=== Basic API Usage ===\n")

    # Test case 1: Simple nested quantifiers
    print("Test Case 1: Simple nested quantifiers")
    x, y, z = z3.Ints("x y z")
    fml = z3.And(x < y, y < z)
    qfml = z3.ForAll(x, z3.Exists(y, z3.ForAll(z, fml)))

    print("Original formula:")
    print(qfml)

    # Use ground_quantifier_all
    print("\nUsing ground_quantifier_all:")
    body, quant_info = ground_quantifier_all(qfml)
    print("Body:", body)
    print("Quantifier info:")
    for i, (is_forall, vars_list) in enumerate(quant_info):
        quant_type = "ForAll" if is_forall else "Exists"
        print(f"  Level {i + 1}: {quant_type} {vars_list}")

    # Demonstrate reconstruction
    print("\nReconstructed formula:")
    reconstructed = reconstruct_quantified_formula(body, quant_info)
    print(reconstructed)

    # Verify semantic equivalence
    s = z3.Solver()
    s.add(qfml != reconstructed)
    print("Semantically equivalent:", s.check() == z3.unsat)

    # Test case 2: Multiple variables per quantifier
    print("\n\nTest Case 2: Multiple variables per quantifier")
    a, b, c, d = z3.Ints("a b c d")

    complex_fml = z3.And(x > a, y < b, z == c, d >= 0)
    complex_qfml = z3.ForAll([x, a], z3.Exists([y, b], z3.ForAll([z, c, d], complex_fml)))

    print("Original formula:")
    print(complex_qfml)

    body2, quant_info2 = ground_quantifier_all(complex_qfml)
    print("\nQuantifier info:")
    for i, (is_forall, vars_list) in enumerate(quant_info2):
        quant_type = "ForAll" if is_forall else "Exists"
        print(f"  Level {i + 1}: {quant_type} {vars_list}")

    reconstructed2 = reconstruct_quantified_formula(body2, quant_info2)
    print("\nReconstructed formula:")
    print(reconstructed2)

    # Verify semantic equivalence
    s2 = z3.Solver()
    s2.add(complex_qfml != reconstructed2)
    print("Semantically equivalent:", s2.check() == z3.unsat)

    # Test case 3: Different variable types
    print("\n\nTest Case 3: Different variable types (Int, Real, Bool)")
    x = z3.Int("x")
    y = z3.Real("y")
    p = z3.Bool("p")

    fml3 = z3.ForAll([x, p], z3.Exists(y, z3.And(y > x, z3.Implies(p, y < 10))))

    print("Original formula:")
    print(fml3)

    body3, quant_info3 = ground_quantifier_all(fml3)
    print("\nQuantifier info:")
    for i, (is_forall, vars_list) in enumerate(quant_info3):
        quant_type = "ForAll" if is_forall else "Exists"
        print(f"  Level {i + 1}: {quant_type} {vars_list}")

    reconstructed3 = reconstruct_quantified_formula(body3, quant_info3)
    print("\nReconstructed formula:")
    print(reconstructed3)

    # Verify semantic equivalence
    s3 = z3.Solver()
    s3.add(fml3 != reconstructed3)
    print("Semantically equivalent:", s3.check() == z3.unsat)

    # Practical example: Modifying a quantified formula
    print("\n\n=== Practical Example: Modifying a Quantified Formula ===\n")

    # Create a formula: ForAll x, Exists y, x < y
    x, y = z3.Ints("x y")
    original_formula = z3.ForAll(x, z3.Exists(y, x < y))
    print("Original formula:", original_formula)

    # Extract the body and quantifier information
    body, quant_info = ground_quantifier_all(original_formula)
    print("Body:", body)
    print("Quantifier info:", quant_info)

    # Modify the body: change x < y to x <= y
    modified_body = x <= y
    print("Modified body:", modified_body)

    # Reconstruct the formula with the modified body
    modified_formula = reconstruct_quantified_formula(modified_body, quant_info)
    print("Modified formula:", modified_formula)

    # Another example: Swapping quantifiers
    print("\nSwapping quantifiers:")
    # Swap ForAll and Exists
    swapped_quant_info = [(not is_forall, vars_list) for is_forall, vars_list in quant_info]
    print("Swapped quantifier info:", swapped_quant_info)

    # Reconstruct with swapped quantifiers
    swapped_formula = reconstruct_quantified_formula(body, swapped_quant_info)
    print("Formula with swapped quantifiers:", swapped_formula)

    print("\nNote: The ground_quantifier_all function preserves the quantifier structure")
    print("and logical meaning of formulas, even if the syntactic representation might differ.")


def test_dnf():
    x, y, z = z3.Ints("x y z")
    fml = z3.Xor(z3.Or(x > 3, y + z < 100), z3.And(x < 100, y == 3), z3.Or(x + y < 100, y - z > 3))
    cnf_fml = z3.Then("simplify", "tseitin-cnf")(fml).as_expr()
    print(cnf_fml)
    # FIXME: this triggers an assertion error in prime_implicant
    # maybe caused by skelom constant(but the algo should be independent of the form)
    # print(exclusive_to_dnf(cnf_fml))


def test_to_dnf_boolean():
    """
    Test the to_dnf_boolean function with various boolean expressions.
    """
    print("\n=== Testing to_dnf_boolean ===\n")

    x, y, z = z3.Bools("x y z")

    # Test case 1: Already in DNF
    fml1 = z3.Or(z3.And(x, y), z)
    print("Original formula 1:", fml1)
    dnf1 = to_dnf_boolean(fml1)
    print("DNF result 1:", dnf1)

    # Check equivalence
    s1 = z3.Solver()
    s1.add(fml1 != dnf1)
    print("Is equivalent:", s1.check() == z3.unsat)

    # Test case 2: AND over OR
    fml2 = z3.And(x, z3.Or(y, z))
    print("\nOriginal formula 2:", fml2)
    dnf2 = to_dnf_boolean(fml2)
    print("DNF result 2:", dnf2)

    # Check equivalence
    s2 = z3.Solver()
    s2.add(fml2 != dnf2)
    print("Is equivalent:", s2.check() == z3.unsat)

    # Check if in expected form
    expected2 = z3.Or(z3.And(x, y), z3.And(x, z))
    s2e = z3.Solver()
    s2e.add(dnf2 != expected2)
    print("Is in expected form:", s2e.check() == z3.unsat)

    # Test case 3: NOT with AND
    fml3 = z3.Not(z3.And(x, y))
    print("\nOriginal formula 3:", fml3)
    dnf3 = to_dnf_boolean(fml3)
    print("DNF result 3:", dnf3)

    # Check equivalence
    s3 = z3.Solver()
    s3.add(fml3 != dnf3)
    print("Is equivalent:", s3.check() == z3.unsat)

    # Test case 4: XOR
    fml4 = z3.Xor(x, y)
    print("\nOriginal formula 4:", fml4)
    dnf4 = to_dnf_boolean(fml4)
    print("DNF result 4:", dnf4)

    # Check equivalence
    s4 = z3.Solver()
    s4.add(fml4 != dnf4)
    print("Is equivalent:", s4.check() == z3.unsat)

    # Test case 5: Complex nested expression
    fml5 = z3.And(z3.Or(x, y), z3.Or(z3.Not(x), z))
    print("\nOriginal formula 5:", fml5)
    dnf5 = to_dnf_boolean(fml5)
    print("DNF result 5:", dnf5)

    # Check equivalence
    s5 = z3.Solver()
    s5.add(fml5 != dnf5)
    print("Is equivalent:", s5.check() == z3.unsat)


if __name__ == "__main__":
    # test_quant()
    # test_dnf()
    # test_native_to_dnf()
    test_to_dnf_boolean()
