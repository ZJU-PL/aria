import time
import z3

from .utilities import dumpProblem
from ..utilities import log
from .base import Search

class SMTSearchActionPropagator(Search):
    """
    Basic grounded incremental search
    """
    def search(self):
        self.horizon = 0

        log(f'Starting to solve', 1)
        total_time = 0
        for horizon in self.scheduler:
            self.horizon  = horizon

            start_time = time.time()
            formula    = self.encoder.encode(self.horizon)
            context    = self.encoder.ctx

            if not self.solver:
                self.solver = z3.Solver(ctx=context) if 'objective' not in formula else z3.Optimize(ctx=context)
                self.solver.set("smt.up.persist_clauses", True)
            if not self.propagator:
                # If we did not instantiate the propagator, we do it here and 
                from ..config import global_config
                action_propagator = global_config.get('propagator')
                self.propagator = action_propagator(e=self.encoder, s=self.solver)
                # then we add the action variables to the propagator
                for a in self.encoder.task.actions:
                    action = self.encoder.get_action_var(a.name, 0)
                    self.propagator.add(action)
            else:
                for a in self.encoder.task.actions:
                    action = self.encoder.get_action_var(a.name, horizon)
                    self.propagator.add(action)
            
            # deal with the initial state
            if self.horizon == 0:
                self.solver.add(formula['initial'])
            del formula['initial']
            
            # deal with the goal state
            g = z3.Bool(f"g{self.horizon}", ctx=context) # Now create a Boolean variable for assuming the goal
            reified_goal = z3.Implies(g, z3.And(formula['goal']))
            self.solver.add(reified_goal) # Add the goal
            del formula['goal']

            # deal with the objective
            if 'objective' in formula:
                self.solver.minimize(formula['objective']) # type: ignore
                del formula['objective']

            # We assert the rest of formulas to the solver
            for _, v in formula.items():
                if v is not None:
                    self.solver.add(v)

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
            formula    = self.encoder.encode(self.horizon)
            context    = self.encoder.ctx

            if not self.solver:
                self.solver = z3.Solver(ctx=context) if 'objective' not in formula else z3.Optimize(ctx=context)
            
            # deal with the initial state
            if self.horizon == 0:
                self.solver.add(formula['initial'])
            del formula['initial']
            
            # deal with the goal state
            g = z3.Bool(f"g{self.horizon}", ctx=context) # Now create a Boolean variable for assuming the goal
            reified_goal = z3.Implies(g, z3.And(formula['goal']))
            self.solver.add(reified_goal) # Add the goal
            del formula['goal']

            # deal with the objective
            if 'objective' in formula:
                self.solver.minimize(formula['objective']) # type: ignore
                del formula['objective']

            # We assert the rest of formulas to the solver
            for _, v in formula.items():
                if v is not None:
                    self.solver.add(v)

        end_time = time.time()
        encoding_time = end_time - start_time
        self.solver.add(g) # type: ignore # we assert the goal happens in the last step (which would normally be an assumption)
        dumpProblem(self.solver, path, add_check_sat=True)
        log(f'Encoding the formula took: {encoding_time:.2f}s', 2)
