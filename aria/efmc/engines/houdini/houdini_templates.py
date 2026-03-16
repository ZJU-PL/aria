"""
Houdini templates for generating candidate invariants

This module provides template-based generation of candidate lemmas
for the Houdini invariant inference algorithm.
"""

from itertools import combinations
from typing import List, Set

import z3

from aria.efmc.utils.z3_expr_utils import get_variables


def get_selector_var(idx: int) -> z3.ExprRef:
    """Create a temp symbol using the given index"""
    return z3.Bool(f"SEL_{idx}")


def prime(exp: z3.ExprRef) -> z3.ExprRef:
    """Replace all symbols in the formula with their primed version"""
    exp_vars = get_variables(exp)
    substitutions = []
    for v in exp_vars:
        sort = v.sort()
        if z3.is_bv_sort(sort):
            substitutions.append((v, z3.BitVec(f"{v}_p", sort.size())))
        elif sort == z3.RealSort():
            substitutions.append((v, z3.Real(f"{v}_p")))
        elif z3.is_fp_sort(sort):
            substitutions.append((v, z3.FP(f"{v}_p", sort)))
        else:  # default to Int
            substitutions.append((v, z3.Int(f"{v}_p")))
    return z3.substitute(exp, substitutions)


def generate_fp_lemmas(var: z3.ExprRef) -> List[z3.ExprRef]:
    """Generate floating point specific lemmas"""
    if not z3.is_fp_sort(var.sort()):
        return []

    sort = var.sort()
    zero = z3.FPVal(0.0, sort)
    large_val = z3.FPVal(1e38, sort)
    small_val = z3.FPVal(-1e38, sort)

    return [
        z3.fpIsNormal(var),
        z3.fpIsSubnormal(var),
        z3.fpIsZero(var),
        z3.fpIsPositive(var),
        z3.fpIsNegative(var),
        var <= large_val,
        var >= small_val,
        z3.fpAbs(var) <= large_val,
        z3.Or(var >= zero, var <= zero),
    ]


def generate_basic_lemmas(
    vars_list: List[z3.ExprRef],
    post_condition: z3.ExprRef = None,
    existing_invariants: List[z3.ExprRef] = None,
) -> List[z3.ExprRef]:
    """Generate basic candidate lemmas for invariant inference"""
    lemmas = []

    if post_condition is not None:
        lemmas.append(post_condition)
    if existing_invariants:
        lemmas.extend(existing_invariants)

    for var in vars_list:
        if var.sort() == z3.IntSort():
            lemmas.append(var >= 0)
        elif z3.is_fp_sort(var.sort()):
            lemmas.extend(generate_fp_lemmas(var))

    # Add relationships between variables
    for idx, var1 in enumerate(vars_list):
        for var2 in vars_list[idx + 1 :]:
            if var1.sort() == var2.sort():
                lemmas.extend([var1 <= var2, var1 >= var2])

    return lemmas


class HoudiniTemplates:
    """Template generator for Houdini candidate lemmas"""

    def __init__(self, variables: List[z3.ExprRef]):
        self.variables = variables
        self.int_vars = [v for v in variables if v.sort() == z3.IntSort()]
        self.real_vars = [v for v in variables if v.sort() == z3.RealSort()]
        self.bv_vars = [v for v in variables if z3.is_bv_sort(v.sort())]
        self.bool_vars = [v for v in variables if v.sort() == z3.BoolSort()]
        self.fp_vars = [v for v in variables if z3.is_fp_sort(v.sort())]

    def generate_all(self) -> List[z3.ExprRef]:
        """Generate all template-based candidate lemmas"""
        return (
            self.bounds_templates()
            + self.ordering_templates()
            + self.arithmetic_templates()
            + self.equality_templates()
            + self.sign_templates()
            + self.floating_point_templates()
        )

    def bounds_templates(self) -> List[z3.ExprRef]:
        """Generate bound templates: x >= c, x <= c"""
        lemmas = []
        constants = [0, 1, -1, 2, -2, 10, -10]

        for var in self.int_vars + self.real_vars:
            for c in constants:
                lemmas.extend([var >= c, var <= c])

        # Bit-vector bounds
        for var in self.bv_vars:
            size = var.sort().size()
            max_val = (1 << (size - 1)) - 1
            min_val = -(1 << (size - 1))
            lemmas.extend([var >= 0, var <= max_val, var >= min_val])

        return lemmas

    def ordering_templates(self) -> List[z3.ExprRef]:
        """Generate ordering templates: x <= y, x >= y"""
        lemmas = []

        for v1, v2 in combinations(self.int_vars + self.real_vars, 2):
            lemmas.extend([v1 <= v2, v1 >= v2])

        for v1, v2 in combinations(self.bv_vars, 2):
            if v1.sort() == v2.sort():
                lemmas.extend([v1 <= v2, v1 >= v2])

        return lemmas

    def arithmetic_templates(self) -> List[z3.ExprRef]:
        """Generate arithmetic relationship templates"""
        lemmas = []

        # Linear relationships
        for v1, v2 in combinations(self.int_vars + self.real_vars, 2):
            lemmas.extend([v1 + v2 >= 0, v1 - v2 >= 0, v1 + v2 <= 0, v1 - v2 <= 0])

        # Multiplication by constants
        for var in self.int_vars + self.real_vars:
            for c in [2, -1]:
                lemmas.extend([c * var >= 0, c * var <= 0])

        return lemmas

    def equality_templates(self) -> List[z3.ExprRef]:
        """Generate equality templates"""
        lemmas = []

        # Variable equals constant
        for var in self.int_vars + self.real_vars:
            for c in [0, 1, -1]:
                lemmas.append(var == c)

        # Variable equality
        for v1, v2 in combinations(self.variables, 2):
            if v1.sort() == v2.sort():
                lemmas.append(v1 == v2)

        return lemmas

    def sign_templates(self) -> List[z3.ExprRef]:
        """Generate sign-based templates"""
        lemmas = []
        for var in self.int_vars + self.real_vars:
            lemmas.extend([var > 0, var < 0, var >= 0, var <= 0])
        return lemmas

    def floating_point_templates(self) -> List[z3.ExprRef]:
        """Generate floating point specific templates"""
        lemmas = []

        for var in self.fp_vars:
            lemmas.extend(generate_fp_lemmas(var))

        # Floating point relationships
        for v1, v2 in combinations(self.fp_vars, 2):
            if v1.sort() == v2.sort():
                lemmas.extend([v1 <= v2, v1 >= v2])

        return lemmas

    def modular_templates(self) -> List[z3.ExprRef]:
        """Generate modular arithmetic templates"""
        lemmas = []
        for var in self.int_vars:
            for m in [2, 3, 4]:
                for r in range(m):
                    lemmas.append(var % m == r)
        return lemmas

    def divisibility_templates(self) -> List[z3.ExprRef]:
        """Generate divisibility templates"""
        lemmas = []
        for var in self.int_vars:
            for d in [2, 3, 4, 5]:
                lemmas.append(var % d == 0)
        return lemmas

    def boolean_templates(self) -> List[z3.ExprRef]:
        """Generate boolean templates"""
        lemmas = []

        # Simple boolean values
        for var in self.bool_vars:
            lemmas.extend([var, z3.Not(var)])

        # Boolean combinations
        for v1, v2 in combinations(self.bool_vars, 2):
            lemmas.extend([z3.And(v1, v2), z3.Or(v1, v2), z3.Implies(v1, v2)])

        return lemmas


def generate_templates(
    vars_list: List[z3.ExprRef], template_types: Set[str] = None
) -> List[z3.ExprRef]:
    """Generate templates for given variables and template types"""
    if template_types is None:
        template_types = {
            "bounds",
            "ordering",
            "arithmetic",
            "equality",
            "sign",
            "floating_point",
        }

    generator = HoudiniTemplates(vars_list)
    method_map = {
        "bounds": generator.bounds_templates,
        "ordering": generator.ordering_templates,
        "arithmetic": generator.arithmetic_templates,
        "equality": generator.equality_templates,
        "sign": generator.sign_templates,
        "floating_point": generator.floating_point_templates,
        "modular": generator.modular_templates,
        "divisibility": generator.divisibility_templates,
        "boolean": generator.boolean_templates,
    }

    lemmas = []
    for template_type in template_types:
        if template_type in method_map:
            lemmas.extend(method_map[template_type]())

    return lemmas


if __name__ == "__main__":
    # Example with integer variables
    x, y, z = z3.Ints("x y z")
    vars_list = [x, y, z]

    templates = generate_templates(vars_list, {"bounds", "ordering"})
    print(f"Generated {len(templates)} templates for variables {vars_list}")

    # Show first few templates
    for idx, template in enumerate(templates[:10]):
        print(f"  {idx+1}: {template}")

    # Example with floating point variables
    a = z3.FP("a", z3.FPSort(8, 24))
    b = z3.FP("b", z3.FPSort(8, 24))
    fp_vars_list = [a, b]

    fp_templates = generate_templates(fp_vars_list, {"floating_point"})
    print(f"\nGenerated {len(fp_templates)} floating point templates")

    # Show first few floating point templates
    for idx, template in enumerate(fp_templates[:5]):
        print(f"  {idx+1}: {template}")
