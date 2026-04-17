"""
Factory for creating samplers.
"""

from typing import Dict, Set, Type, Optional, List

import z3

from .base import Sampler, Logic, SamplingMethod, SamplingOptions, SamplingResult
from aria.sampling.finite_domain.bool.base import BooleanSampler
from aria.sampling.finite_domain.bv.base import BitVectorSampler
from aria.sampling.finite_domain.dt.base import DatatypeSampler
from aria.sampling.finite_domain.fp.base import FloatingPointSampler
from aria.sampling.finite_domain.fp.hash_sampler import HashBasedFPSampler
from aria.sampling.finite_domain.fp.total_order_sampler import TotalOrderFPSampler
from aria.sampling.finite_domain.uf.base import UninterpretedFunctionSampler
from aria.sampling.finite_domain.ufdt.base import MixedUFDatatypeSampler
from aria.sampling.dtlia import ADTLIASampler
from aria.sampling.linear_ira.lira_sampler import LIRASampler
from aria.sampling.general_sampler.mcmc_sampler import MCMCSampler
from aria.sampling.nonlinear_ira import NASampler


class SamplerFactory:
    """Factory for creating samplers."""

    _samplers: Dict[Logic, List[Type[Sampler]]] = {}

    @classmethod
    def register(cls, logic: Logic, sampler_class: Type[Sampler]) -> None:
        cls._samplers.setdefault(logic, []).append(sampler_class)

    @classmethod
    def create(
        cls, logic: Logic, method: Optional[SamplingMethod] = None, **kwargs
    ) -> Sampler:
        if logic not in cls._samplers or not cls._samplers[logic]:
            available = ", ".join(str(l) for l in cls._samplers.keys())
            raise ValueError(
                f"No sampler available for logic {logic}. Available logics: {available}"
            )

        if method:
            for sampler_class in cls._samplers[logic]:
                sampler = sampler_class(**kwargs)
                if method in sampler.get_supported_methods():
                    return sampler
            raise ValueError(
                f"No sampler available for logic {logic} and method {method}"
            )

        return cls._samplers[logic][0](**kwargs)

    @classmethod
    def available_logics(cls) -> Set[Logic]:
        return set(cls._samplers.keys())

    @classmethod
    def available_methods(cls, logic: Logic) -> Set[SamplingMethod]:
        if logic not in cls._samplers:
            return set()
        methods: Set[SamplingMethod] = set()
        for sampler_class in cls._samplers[logic]:
            methods.update(sampler_class().get_supported_methods())
        return methods


SamplerFactory.register(Logic.QF_BOOL, BooleanSampler)
SamplerFactory.register(Logic.QF_BV, BitVectorSampler)
SamplerFactory.register(Logic.QF_FP, FloatingPointSampler)
SamplerFactory.register(Logic.QF_FP, HashBasedFPSampler)
SamplerFactory.register(Logic.QF_FP, TotalOrderFPSampler)
SamplerFactory.register(Logic.QF_UF, UninterpretedFunctionSampler)
SamplerFactory.register(Logic.QF_UFLIA, UninterpretedFunctionSampler)
SamplerFactory.register(Logic.QF_DT, DatatypeSampler)
SamplerFactory.register(Logic.QF_UFDT, MixedUFDatatypeSampler)
SamplerFactory.register(Logic.QF_DTLIA, ADTLIASampler)

for _logic in (Logic.QF_LRA, Logic.QF_LIA, Logic.QF_LIRA):
    SamplerFactory.register(_logic, LIRASampler)

for _logic in (Logic.QF_NRA, Logic.QF_NIA):
    SamplerFactory.register(_logic, NASampler)

for _logic in (
    Logic.QF_LRA,
    Logic.QF_LIA,
    Logic.QF_NRA,
    Logic.QF_NIA,
    Logic.QF_LIRA,
    Logic.QF_BOOL,
    Logic.QF_ALL,
):
    SamplerFactory.register(_logic, MCMCSampler)


def create_sampler(
    logic: Logic, method: Optional[SamplingMethod] = None, **kwargs
) -> Sampler:
    """Create a sampler instance."""
    return SamplerFactory.create(logic, method, **kwargs)


def sample_models_from_formula(
    formula: z3.ExprRef, logic: Logic, options: Optional[SamplingOptions] = None
) -> SamplingResult:
    """Sample models from a formula."""
    sampler = create_sampler(logic, options.method if options else None)
    sampler.init_from_formula(formula)
    return sampler.sample(options or SamplingOptions())


def sample_formula(
    formula: z3.ExprRef, logic: Logic, options: Optional[SamplingOptions] = None
) -> SamplingResult:
    """Deprecated: Use sample_models_from_formula() instead."""
    import warnings

    warnings.warn(
        "sample_formula() is deprecated. Please use sample_models_from_formula() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return sample_models_from_formula(formula, logic, options)


def demo() -> None:
    """Demonstrate the usage of the sampler factory."""
    x, y = z3.Reals("x y")
    formula = z3.And(x + y > 0, x - y < 1)
    assert isinstance(formula, z3.ExprRef)
    result = sample_models_from_formula(
        formula, Logic.QF_LRA, SamplingOptions(num_samples=5)
    )
    print(f"Generated {len(result)} models:")
    for i, sample in enumerate(result, 1):
        print(f"Model {i}: {sample}")


if __name__ == "__main__":
    demo()
