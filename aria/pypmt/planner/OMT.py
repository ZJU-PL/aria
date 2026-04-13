import time

import z3

from .utilities import dumpProblem
from ..utilities import log
from .base import Search
from .plan.smt_sequential_plan import SMTSequentialPlan

class OMTSearch(Search):
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
                self.solver = z3.Optimize(ctx=context)
            
            # deal with the initial state
            if self.horizon == 0:
                self.solver.add(formula['initial'])
            del formula['initial']
            
            # deal with the goal state
            g = z3.Bool(f"g{self.horizon}", ctx=context) # Now create a Boolean variable for assuming the goal
            reified_goal = z3.Implies(g, z3.And(formula['abstract_goal']))
            self.solver.add(reified_goal) # Add the goal
            del formula['abstract_goal']

            # deal with abstract formulas 
            formula['loop_formulas']    = z3.BoolVal(True, ctx=context) if len(formula['loop_formulas'].children()) == 0    else formula['loop_formulas']
            formula['abstract_actions'] = z3.BoolVal(True, ctx=context) if len(formula['abstract_actions'].children()) == 0 else formula['abstract_actions']
            reified_abstract_formulas = z3.Implies(g, z3.And(formula['abstract_actions'],formula['loop_formulas'] ))
            self.solver.add(reified_abstract_formulas) # Add the abstract transition formulas
            del formula['abstract_actions']
            del formula['loop_formulas']

            
            # deal with the objective
            if 'objective' in formula:
                self.solver.minimize(formula['objective'])
                del formula['objective']

            # We assert the rest of formulas to the solver
            for k, v in formula.items():
                if k == 'goal':
                    continue
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
                model = self.solver.model()
                if model.eval(formula['goal']):
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
                self.solver = z3.Optimize(ctx=context)
            
            # deal with the initial state
            if self.horizon == 0:
                self.solver.add(formula['initial'])
            del formula['initial']
            
            # deal with the goal state
            g = z3.Bool(f"g{self.horizon}", ctx=context) # Now create a Boolean variable for assuming the goal
            reified_goal = z3.Implies(g, z3.And(formula['abstract_goal']))
            self.solver.add(reified_goal) # Add the goal
            del formula['abstract_goal']

            # deal with abstract formulas 
            reified_abstract_formulas = z3.Implies(g, z3.And(formula['abstract_actions'],formula['loop_formulas'] ))
            self.solver.add(reified_abstract_formulas) # Add the abstract transition formulas
            del formula['abstract_actions']
            del formula['loop_formulas']

            # deal with the objective
            if 'objective' in formula:
                self.solver.minimize(formula['objective'])
                del formula['objective']

            # We assert the rest of formulas to the solver
            for k, v in formula.items():
                if k == 'goal':
                    continue
                if v is not None:
                    self.solver.add(v)


        end_time = time.time()
        encoding_time = end_time - start_time
        self.solver.add(g) # we assert the goal happens in the last step (which would normally be an assumption)
        dumpProblem(self.solver, path, add_check_sat=False)
        log(f'Encoding the formula took: {encoding_time:.2f}s', 2)
