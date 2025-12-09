#!/usr/bin/env python3
import z3
from aria.sampling import sample_models_from_formula, Logic, SamplingOptions, SamplingMethod, create_sampler

def bool_sampling():
    a, b, c, d = z3.Bools('a b c d')
    f = z3.And(z3.Or(a, b, c), z3.Or(z3.Not(a), d), z3.Or(z3.Not(b), z3.Not(c)), z3.Or(a, z3.Not(d)))
    for m in [SamplingMethod.ENUMERATION, SamplingMethod.HASH_BASED]:
        try:
            opts = SamplingOptions(method=m, num_samples=3, random_seed=42)
            r = sample_models_from_formula(f, Logic.QF_BOOL, opts)
            print(f"{m.value}: {len(r)} samples")
        except: print(f"{m.value}: not available")

def lra_sampling():
    x, y, z = z3.Reals('x y z')
    f = z3.And(x + y + z <= 10, x >= 0, y >= 0, z >= 0, 2*x + y <= 8, x + 3*y <= 12)
    for m in [SamplingMethod.ENUMERATION, SamplingMethod.REGION, SamplingMethod.DIKIN_WALK]:
        try:
            opts = SamplingOptions(method=m, num_samples=2, random_seed=42)
            r = sample_models_from_formula(f, Logic.QF_LRA, opts)
            print(f"{m.value}: {len(r)} samples")
        except: print(f"{m.value}: not available")

def lia_sampling():
    x, y = z3.Ints('x y')
    f = z3.And(x + 2*y <= 10, x >= 0, y >= 0, x <= 5, y <= 4, x + y >= 2)
    opts = SamplingOptions(method=SamplingMethod.ENUMERATION, num_samples=5, random_seed=42)
    r = sample_models_from_formula(f, Logic.QF_LIA, opts)
    print(f"LIA: {len(r)} samples")

def mcmc_sampling():
    x, y, z = z3.Reals('x y z')
    f = z3.And(x*x + y*y <= 4, z >= x + y, z <= 3, x >= -2, y >= -2, z >= -2)
    try:
        opts = SamplingOptions(method=SamplingMethod.MCMC, num_samples=3, random_seed=42, burn_in=100, thin=10)
        r = sample_models_from_formula(f, Logic.QF_NRA, opts)
        print(f"MCMC: {len(r)} samples")
    except: print("MCMC not available")

def custom_sampler():
    p, q, r = z3.Bools('p q r')
    f = z3.And(z3.Or(p, q), z3.Or(z3.Not(p), r), z3.Or(q, z3.Not(r)))
    try:
        s = create_sampler(Logic.QF_BOOL, SamplingMethod.ENUMERATION)
        s.init_from_formula(f)
        opts = SamplingOptions(method=SamplingMethod.ENUMERATION, num_samples=5, random_seed=123)
        r = s.sample(opts)
        print(f"Custom: {len(r)} samples")
    except: print("Custom sampler failed")

def main():
    print("Sampling Examples\n" + "="*20)
    bool_sampling(); lra_sampling(); lia_sampling(); mcmc_sampling(); custom_sampler()
    print("Done!")

if __name__ == "__main__":
    main() 