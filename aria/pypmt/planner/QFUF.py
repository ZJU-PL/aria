
import time

import z3

from .base import Search
from .plan.smt_sequential_plan import SMTSequentialPlan
from .utilities import dumpProblem
from ..utilities import log

class QFUFSearch(Search):
    """
    Base class defining search schemes.
    """
    
    def search(self):
        self.horizon = 0

        log(f'Starting to solve', 1)
        total_time = 0
        for horizon in self.scheduler:
            self.horizon  = horizon
            start_time = time.time()
            formula = self.encoder.encode(self.horizon)

            if not self.solver:
                self.solver =  z3.Solver(ctx=self.encoder.ctx) if 'objective' not in formula else z3.Optimize(ctx=self.encoder.ctx)
            
            # deal with the initial state
            if self.horizon == 0:
                self.solver.add(formula['initial'])
                self.solver.add(formula['typing'])
            
            # deal with the goal state
            g = z3.Bool(f"g{self.horizon}", self.encoder.ctx) # Now create a Boolean variable for assuming the goal
            reified_goal = z3.Implies(g, z3.And(formula['goal']))
            self.solver.add(reified_goal) # Add the goal

            # deal with the objective
            if 'objective' in formula:
                self.solver.minimize(formula['objective'])

            # We assert the rest of formulas to the solver
            self.solver.add(formula['actions'])
            self.solver.add(formula['frame'])

            # Check for satisfiability assuming the goal
            end_time = time.time()
            encoding_time = end_time - start_time
            start_time = time.time()
            res = self.solver.check(g)
            end_time = time.time()
            solving_time = end_time - start_time
            total_time = total_time + solving_time + encoding_time
            log(f'Step {horizon+1}/{(self.scheduler[-1]+1)} encoding: {encoding_time:.2f}s, solving: {solving_time:.2f}s', 2)
            
            if res == z3.sat:
                log(f'Satisfiable model found. Took:{total_time:.2f}s', 3)
                log(f'Z3 statistics:\n{self.solver.statistics()}', 4)
                self.solution = self.encoder.extract_plan(self.solver.model(), self.horizon)
                break
        return self.solution

    def dump_smtlib_to_file(self, t, path):
        self.horizon = 0
        start_time = time.time()
        log(f'Encoding problem into a SMTLIB file', 1)
        for horizon in range(0, t):
            self.horizon  = horizon
            formula = self.encoder.encode(self.horizon)

            if not self.solver:
                self.solver =  z3.Solver(ctx=self.encoder.ctx) if 'objective' not in formula else z3.Optimize(ctx=self.encoder.ctx)
            
            # deal with the initial state
            if self.horizon == 0:
                self.solver.add(formula['initial'])
                self.solver.add(formula['typing'])
            
            # deal with the goal state
            g = z3.Bool(f"g{self.horizon}", self.encoder.ctx) # Now create a Boolean variable for assuming the goal
            reified_goal = z3.Implies(g, z3.And(formula['goal']))
            self.solver.add(reified_goal) # Add the goal

            # deal with the objective
            if 'objective' in formula:
                self.solver.minimize(formula['objective'])

            # We assert the rest of formulas to the solver
            self.solver.add(formula['actions'])
            self.solver.add(formula['frame'])

        end_time = time.time()
        encoding_time = end_time - start_time
        self.solver.add(g) # we assert the goal happens in the last step (which would normally be an assumption)
        dumpProblem(self.solver, path, add_check_sat=True)
        log(f'Encoding the formula took: {encoding_time:.2f}s', 2)
