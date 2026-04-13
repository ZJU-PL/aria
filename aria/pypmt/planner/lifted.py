import time

import z3

from .base import Search
from .plan.smt_sequential_plan import SMTSequentialPlan
from .utilities import dumpProblem
from ..utilities import log

class LiftedSearch(Search):

    def search(self):

        log(f'Starting to solve', 1)
        total_time = 0
        start_time = time.time()
        formula = self.encoder.encode()

        if not self.solver:
            self.solver = z3.Solver(ctx=self.encoder.ctx)

        horizon = z3.Int("horizon", ctx=self.encoder.ctx)
        body = z3.And([self.encoder.z3_timestep_var > z3.IntVal(0, ctx=self.encoder.ctx),
        horizon == self.encoder.z3_timestep_var,
        formula['initial'],
        formula['goal'],
        formula['actions'],
        formula['frame'],
        formula['typing']
        ])
        full_formula = z3.Exists([self.encoder.z3_timestep_var], body)
        self.solver.add(full_formula)

        end_time = time.time()
        encoding_time = end_time - start_time
        start_time = time.time()
        res = self.solver.check()
        end_time = time.time()
        solving_time = end_time - start_time  # Calculate time elapsed for this iteration
        total_time = total_time + solving_time + encoding_time
        log(f'Encoding full lifted formula: {encoding_time:.2f}s, solving: {solving_time:.2f}s', 2)

        if res == z3.sat:
            log(f'Satisfiable model found. Took:{total_time:.2f}s', 3)
            log(f'Z3 statistics:\n{self.solver.statistics()}', 4)
            model = self.solver.model()
            self.horizon = model.evaluate(horizon).as_long()
            self.solution = self.encoder.extract_plan(model, self.horizon)
        return self.solution

    def dump_smtlib_to_file(self, t, path):
        log(f'Encoding problem into a SMTLIB file', 1)
        start_time = time.time()
        formula = self.encoder.encode()

        if not self.solver:
            self.solver = z3.Solver(ctx=self.encoder.ctx)

        horizon = z3.Int("horizon", ctx=self.encoder.ctx)
        body = z3.And([
            self.encoder.z3_timestep_var > z3.IntVal(t, ctx=self.encoder.ctx),
            horizon == self.encoder.z3_timestep_var,
            formula['initial'],
            formula['goal'],
            formula['actions'],
            formula['frame'],
            formula['typing']
        ])
        full_formula = z3.Exists([self.encoder.z3_timestep_var], body)
        self.solver.add(full_formula)

        end_time = time.time()
        encoding_time = end_time - start_time
        dumpProblem(self.solver, path, add_check_sat=True)
        log(f'Encoding full lifted formula took: {encoding_time:.2f}s', 2)
