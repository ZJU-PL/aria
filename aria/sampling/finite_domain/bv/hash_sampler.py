# Enumerate models with some random parity constraints as suggested for approximate model counting.
# Taken from https://github.com/Z3Prover/z3/issues/4675#issuecomment-686880139
"""
XOR-based (hash-based) sampling for bit-vector formulas.

This implements approximate uniform sampling using random parity (XOR) constraints.
"""

from typing import List, Dict, Any, Set
import z3
from random import randrange, seed as set_seed

from aria.sampling.base import Sampler, Logic, SamplingMethod, SamplingOptions, SamplingResult
from aria.utils.z3_expr_utils import get_variables, is_bv_sort


def _get_uniform_samples_with_xor(vars: List[z3.ExprRef], cnt: z3.ExprRef, num_samples: int):
    """
    Internal implementation: Get num_samples models (projected to vars) using XOR constraints.

    Args:
        vars: List of bit-vector variables to project on
        cnt: The constraints/formula
        num_samples: Number of samples to generate

    Returns:
        List of models (each model is a list of values for vars)
    """
    res = []
    s = z3.Solver()
    s.add(cnt)
    bits = []
    for var in vars:
        bits = bits + [z3.Extract(i, i, var) == 1 for i in range(var.size())]
    num_success = 0
    while True:
        s.push()
        rounds = 3  # why 3?
        for x in range(rounds):
            trials = 10
            fml = z3.BoolVal(randrange(0, 2))
            for i in range(trials):
                fml = z3.Xor(fml, bits[randrange(0, len(bits))])
            s.add(fml)
        if s.check() == z3.sat:
            res.append([s.model().eval(var, True) for var in vars])
            num_success += 1
            if num_success == num_samples:
                break
        s.pop()
    return res


class HashBasedBVSampler(Sampler):
    """
    XOR-based (hash-based) sampler for bit-vector formulas.

    This sampler uses random parity (XOR) constraints to achieve approximate
    uniform sampling over the solution space.

    Reference:
        https://github.com/Z3Prover/z3/issues/4675#issuecomment-686880139
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the hash-based bit-vector sampler."""
        self.formula: z3.ExprRef = None
        self.variables: List[z3.ExprRef] = []

    def supports_logic(self, logic: Logic) -> bool:
        """
        Check if this sampler supports the given logic.

        Args:
            logic: The logic to check

        Returns:
            True if the sampler supports the logic, False otherwise
        """
        return logic == Logic.QF_BV

    def init_from_formula(self, formula: z3.ExprRef) -> None:
        """
        Initialize the sampler with a formula.

        Args:
            formula: The Z3 formula to sample from
        """
        self.formula = formula

        # Extract bit-vector variables from the formula
        self.variables = []
        for var in get_variables(formula):
            if is_bv_sort(var):
                self.variables.append(var)

    def sample(self, options: SamplingOptions) -> SamplingResult:
        """
        Generate samples using XOR-based hashing.

        Args:
            options: The sampling options

        Returns:
            A SamplingResult containing the generated samples
        """
        if self.formula is None:
            raise ValueError("Sampler not initialized with a formula")

        if not self.variables:
            raise ValueError("No bit-vector variables found in formula")

        # Set random seed if provided
        if options.random_seed is not None:
            set_seed(options.random_seed)

        # Generate samples using XOR constraints
        raw_samples = _get_uniform_samples_with_xor(
            self.variables,
            self.formula,
            options.num_samples
        )

        # Convert to standard format
        samples: List[Dict[str, Any]] = []
        for raw_sample in raw_samples:
            sample: Dict[str, Any] = {}
            for var, value in zip(self.variables, raw_sample):
                sample[str(var)] = value.as_long() if hasattr(value, 'as_long') else int(value)
            samples.append(sample)

        stats = {
            "time_ms": 0,
            "iterations": options.num_samples,
            "method": "hash_based_xor"
        }

        return SamplingResult(samples, stats)

    def get_supported_methods(self) -> Set[SamplingMethod]:
        """
        Get the sampling methods supported by this sampler.

        Returns:
            A set of supported sampling methods
        """
        return {SamplingMethod.HASH_BASED}

    def get_supported_logics(self) -> Set[Logic]:
        """
        Get the logics supported by this sampler.

        Returns:
            A set of supported logics
        """
        return {Logic.QF_BV}


def test_api():
    """Test the hash-based sampler."""
    x, y, z = z3.BitVecs('x y z', 32)
    fml = z3.And(z3.ULT(x, 13), z3.ULT(y, x), z3.ULE(y, z))

    sampler = HashBasedBVSampler()
    sampler.init_from_formula(fml)
    result = sampler.sample(SamplingOptions(num_samples=8))

    print(f"Generated {len(result)} samples:")
    for i, sample in enumerate(result):
        print(f"  Sample {i+1}: {sample}")


if __name__ == "__main__":
    test_api()
