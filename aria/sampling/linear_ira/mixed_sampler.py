"""
For formulas with different types of variables
"""

from aria.sampling.base import Sampler


# from exceptions import *


class MixedSampler(Sampler):
    """
    The formula can have different types of variables, e.g.,
    bool, bit-vec, real, int (and even string?)
    """

    def __init__(self, **options):
        Sampler.__init__(self, **options)

        self.conjuntion_sampler = None
        self.number_samples = 0

    def sample(self, number=1):
        """
        External interface
        """
        self.number_samples = number
        return self.sample_via_enumeration()

    def sample_via_smt_enumeration(self):
        """
        Call an SMT solver iteratively (block sampled models)
        """
        raise NotImplementedError

    def sample_via_smt_random_seed(self):
        """
        Call an SMT solver iteratively (no blocking, but give the solver diferent
        random seeds)
        """
        raise NotImplementedError
