"""
Randomly generating a transition system
FIXME: to check (gnerated by LLM)
"""

import random
import z3

from aria.efmc.tests.formula_generator import FormulaGenerator
from aria.efmc.sts import TransitionSystem


class TransitionSystemGenerator:
    """Generator for random transition systems of different types"""

    def __init__(self):
        self.var_count = random.randint(2, 5)  # Number of variables to generate
        self.fml_gene = FormulaGenerator(
            init_vars=[]
        )  # Initialize with empty vars list

    def _create_vars(self, prefix: str, sort_fn) -> tuple:
        """Helper to create variables with given prefix and sort"""
        vars = [sort_fn(f"{prefix}{i}") for i in range(self.var_count)]
        prime_vars = [sort_fn(f"{prefix}{i}!") for i in range(self.var_count)]
        return vars, prime_vars

    def _create_transition_system(
        self, vars, prime_vars, init, trans, post
    ) -> TransitionSystem:
        """Helper to create and initialize a transition system"""
        sts = TransitionSystem()
        sts.from_z3_cnts([vars + prime_vars, init, trans, post])
        return sts

    def gen_bool_program(self) -> TransitionSystem:
        """Generate boolean program transition system"""
        vars, prime_vars = self._create_vars("b", z3.Bool)

        # Generate random boolean combinations
        init = z3.And(*[v == z3.BoolVal(random.choice([True, False])) for v in vars])

        # Random transitions
        trans_conditions = []
        for v, pv in zip(vars, prime_vars):
            cond = random.choice(
                [
                    pv == z3.Not(v),
                    pv == v,
                    pv == z3.Or(v, vars[random.randint(0, len(vars) - 1)]),
                ]
            )
            trans_conditions.append(cond)
        trans = z3.And(*trans_conditions)

        # Safety property
        post = z3.Or(*[v for v in vars])  # At least one variable is true

        return self._create_transition_system(vars, prime_vars, init, trans, post)

    def gen_lra_program(self) -> TransitionSystem:
        """Generate linear real arithmetic transition system"""
        vars, prime_vars = self._create_vars("x", z3.Real)

        # Initial conditions
        init = z3.And(*[v == random.uniform(-10, 10) for v in vars])

        # Linear transitions
        trans_conditions = []
        for v, pv in zip(vars, prime_vars):
            coeff = random.uniform(-2, 2)
            const = random.uniform(-5, 5)
            trans_conditions.append(pv == coeff * v + const)
        trans = z3.And(*trans_conditions)

        # Linear inequality property
        post = vars[0] <= vars[1] + random.uniform(-10, 10)

        return self._create_transition_system(vars, prime_vars, init, trans, post)

    def gen_lia_program(self) -> TransitionSystem:
        """Generate linear integer arithmetic transition system"""
        vars, prime_vars = self._create_vars("i", z3.Int)

        # Initial conditions
        init = z3.And(*[v == random.randint(-10, 10) for v in vars])

        # Integer transitions
        trans_conditions = []
        for v, pv in zip(vars, prime_vars):
            coeff = random.randint(-2, 2)
            const = random.randint(-5, 5)
            trans_conditions.append(pv == coeff * v + const)
        trans = z3.And(*trans_conditions)

        # Integer inequality property
        post = vars[0] <= 2 * vars[1] + random.randint(-10, 10)

        return self._create_transition_system(vars, prime_vars, init, trans, post)

    def gen_auflia_program(self) -> TransitionSystem:
        """Generate array and linear integer arithmetic transition system"""
        # Base integer variables
        vars, prime_vars = self._create_vars("i", z3.Int)

        # Array variables
        array = z3.Array("arr", z3.IntSort(), z3.IntSort())
        array_p = z3.Array("arr!", z3.IntSort(), z3.IntSort())
        vars.append(array)
        prime_vars.append(array_p)

        # Initial conditions for integers and array
        init_int = z3.And(*[v == random.randint(0, 5) for v in vars[:-1]])
        init_array = array == z3.Store(z3.K(z3.IntSort(), 0), 0, random.randint(1, 10))
        init = z3.And(init_int, init_array)

        # Transitions with array updates
        trans_int = [pv == v + 1 for v, pv in zip(vars[:-1], prime_vars[:-1])]
        trans_array = array_p == z3.Store(array, vars[0], vars[1])
        trans = z3.And(z3.And(*trans_int), trans_array)

        # Property involving array elements
        post = z3.Select(array, vars[0]) >= 0

        return self._create_transition_system(vars, prime_vars, init, trans, post)

    def gen_bv_program(self, width: int = 8) -> TransitionSystem:
        """Generate bitvector transition system"""
        vars, prime_vars = self._create_vars("bv", lambda x: z3.BitVec(x, width))

        # Initial conditions
        init = z3.And(*[v == random.randint(0, 2**width - 1) for v in vars])

        # Bitvector operations
        trans_conditions = []
        for v, pv in zip(vars, prime_vars):
            op = random.choice(
                [
                    lambda x: x + 1,  # Increment
                    lambda x: x - 1,  # Decrement
                    lambda x: x << 1,  # Left shift
                    lambda x: x >> 1,  # Right shift
                ]
            )
            trans_conditions.append(pv == op(v))
        trans = z3.And(*trans_conditions)

        # Bitvector property
        post = z3.ULT(vars[0], 2 ** (width - 1))  # Value stays below half of max

        return self._create_transition_system(vars, prime_vars, init, trans, post)

    def gen_nra_program(self) -> TransitionSystem:
        """Generate non-linear real arithmetic transition system"""
        vars, prime_vars = self._create_vars("x", z3.Real)

        # Initial conditions
        init = z3.And(*[v == random.uniform(0, 5) for v in vars])

        # Non-linear transitions
        trans_conditions = []
        for v, pv in zip(vars, prime_vars):
            other_var = vars[random.randint(0, len(vars) - 1)]
            trans_conditions.append(pv == v * other_var + random.uniform(-1, 1))
        trans = z3.And(*trans_conditions)

        # Non-linear property
        post = vars[0] * vars[0] <= 100

        return self._create_transition_system(vars, prime_vars, init, trans, post)

    def gen_nia_program(self) -> TransitionSystem:
        """Generate non-linear integer arithmetic transition system"""
        vars, prime_vars = self._create_vars("i", z3.Int)

        # Initial conditions
        init = z3.And(*[v == random.randint(0, 5) for v in vars])

        # Non-linear transitions
        trans_conditions = []
        for v, pv in zip(vars, prime_vars):
            other_var = vars[random.randint(0, len(vars) - 1)]
            trans_conditions.append(pv == v * other_var + random.randint(-2, 2))
        trans = z3.And(*trans_conditions)

        # Non-linear property
        post = vars[0] * vars[0] <= 100

        return self._create_transition_system(vars, prime_vars, init, trans, post)


def demo():
    gen = TransitionSystemGenerator()
    # Generate and print a boolean program
    sts = gen.gen_bv_program()
    print("Generated Program:")
    print(f"Init: {sts.init}")
    print(f"Trans: {sts.trans}")
    print(f"Post: {sts.post}")


if __name__ == "__main__":  # Fixed typo in __name__
    demo()
