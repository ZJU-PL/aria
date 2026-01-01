"""
For formulas with different types of variables
"""

from aria.sampling.base import (
    Sampler,
    SamplingOptions,
    SamplingResult,
    Logic,
    SamplingMethod,
)
import z3


class MixedSampler(Sampler):
    """
    The formula can have different types of variables, e.g.,
    bool, bit-vec, real, int (and even string?)
    """

    def __init__(self, **options):
        super().__init__()
        self.conjuntion_sampler = None
        self.number_samples = 0

    def supports_logic(self, logic: Logic) -> bool:
        """Check if this sampler supports the given logic."""
        return logic in [
            Logic.QF_LRA,
            Logic.QF_LIA,
            Logic.QF_LIRA,
            Logic.QF_BV,
            Logic.QF_BOOL,
        ]

    def init_from_formula(self, formula: z3.ExprRef) -> None:
        """Initialize the sampler with a formula."""
        raise NotImplementedError

    def sample(self, options: SamplingOptions) -> SamplingResult:
        """
        External interface - generate samples.

        Args:
            options: Sampling options (num_samples will be used)
        """
        self.number_samples = options.num_samples
        return self.sample_via_enumeration()

    def sample_via_enumeration(self):
        """
        Call an SMT solver iteratively (block sampled models).
        """
        raise NotImplementedError

    def sample_via_smt_enumeration(self):
        """
        Call an SMT solver iteratively (block sampled models)
        """
        raise NotImplementedError

    def sample_via_smt_random_seed(self):
        """
        Call an SMT solver iteratively (no blocking, but give the solver different
        random seeds)
        """
        raise NotImplementedError
