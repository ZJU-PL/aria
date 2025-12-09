"""
Sampling solutions of nonliner arithmetic  (QF_NRA and QF_NIA) formulas
"""
import random
from z3 import *
from aria.sampling.general_sampler.searchtree_sampler import search_tree_sample


class NASampler:
    def __init__(self, formula, num_samples=10):
        self.formula = formula
        self.num_samples = num_samples
        self.solver = Solver()
        self.solver.add(formula)

    def sample(self):
        samples = []
        for _ in range(self.num_samples):
            if self.solver.check() == sat:
                model = self.solver.model()
                sample = {d: model[d] for d in model.decls()}
                samples.append(sample)
                # Add a constraint to avoid getting the same model again
                self.solver.add(Or([d() != model[d] for d in model.decls()]))
            else:
                break
        return samples

    def searchtree_sample(self):
        variables = list(self.formula.decls())
        samples = search_tree_sample(variables, self.formula, self.num_samples)
        return samples


if __name__ == "__main__":
    # Example usage
    x, y = Reals('x y')
    formula = And(x ** 2 + y ** 2 < 1, x > 0, y > 0)
    sampler = NASampler(formula, num_samples=5)
    samples = sampler.sample()
    for sample in samples:
        print(sample)
