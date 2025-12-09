"""
Implicit Hitting Set algorithm for MaxSMT solving.

This module implements the Implicit Hitting Set (IHS) approach for MaxSMT solving, 
based on the IJCAR'18 paper.
"""

from typing import Tuple, Optional, List

import z3

from .base import MaxSMTSolverBase, logger


class ImplicitHittingSetSolver(MaxSMTSolverBase):
    """
    Implicit Hitting Set algorithm for MaxSMT
    
    Iteratively finds optimal hitting sets for the collection of cores.
    """
    
    def solve(self) -> Tuple[bool, Optional[z3.ModelRef], float]:
        """Implicit Hitting Set algorithm for MaxSMT
        
        Returns:
            Tuple of (sat, model, optimal_cost)
        """
        # Check if hard constraints are satisfiable
        solver = z3.Solver()
        for hc in self.hard_constraints:
            solver.add(hc)
        
        if solver.check() != z3.sat:
            logger.warning("Hard constraints are unsatisfiable")
            return False, None, float('inf')
        
        # Create relaxation variables for soft constraints
        relax_vars = [z3.Bool(f"_relax_{i}") for i in range(len(self.soft_constraints))]
        
        # Solver for finding cores
        core_solver = z3.Solver()
        
        # Add hard constraints
        for hc in self.hard_constraints:
            core_solver.add(hc)
        
        # Add soft constraints with relaxation variables
        for i, sc in enumerate(self.soft_constraints):
            core_solver.add(z3.Or(sc, relax_vars[i]))
        
        # Set to keep track of all discovered cores
        all_cores = []  # type: List[List[int]]
        
        # Current best model and cost
        best_model = None
        best_cost = float('inf')
        
        # Solver for the hitting set problem
        hs_solver = z3.Optimize()
        
        # Add variables for the hitting set problem
        hs_vars = [z3.Bool(f"_hs_{i}") for i in range(len(self.soft_constraints))]
        
        # Add objective: minimize sum of weights of selected soft constraints
        obj_terms = [(hs_vars[i], self.weights[i]) for i in range(len(self.soft_constraints))]
        hs_solver.minimize(z3.Sum([z3.If(var, weight, 0) for var, weight in obj_terms]))
        
        while True:
            # Solve the hitting set problem
            if hs_solver.check() == z3.sat:
                hs_model = hs_solver.model()
                hitting_set = [i for i in range(len(self.soft_constraints)) 
                              if hs_model.evaluate(hs_vars[i])]
                
                current_cost = sum(self.weights[i] for i in hitting_set)
                
                # If the current cost equals the best cost, we're done
                if best_model is not None and abs(current_cost - best_cost) < 1e-6:
                    return True, best_model, best_cost
                
                # Check if the current hitting set gives a satisfiable formula
                assumptions = [z3.Not(relax_vars[i]) for i in range(len(self.soft_constraints)) 
                              if i not in hitting_set]
                
                # Check satisfiability with assumptions
                result = core_solver.check(assumptions)
                
                if result == z3.sat:
                    # Found a better solution
                    best_model = core_solver.model()
                    best_cost = current_cost
                    
                    # Add constraint to find a better solution
                    if hitting_set:  # If there are soft constraints violated
                        # We want to find a solution where fewer than current_violated soft constraints are violated
                        # Instead of using PbLe with float, use a logical constraint
                        # Exclude at least one of the current violated constraints
                        hs_solver.add(z3.Not(z3.And([hs_vars[i] for i in hitting_set])))
                    else:
                        # If no soft constraints are violated, we've found the optimal solution
                        return True, best_model, best_cost
                else:
                    # Extract the unsatisfiable core
                    core = core_solver.unsat_core()
                    core_indices = [i for i in range(len(self.soft_constraints)) 
                                   if z3.Not(relax_vars[i]) in core]
                    
                    if not core_indices:
                        # No soft constraints in the core
                        if best_model is not None:
                            return True, best_model, best_cost
                        else:
                            return False, None, float('inf')
                    
                    # Add the core to the set of all cores
                    all_cores.append(core_indices)
                    
                    # Add constraint: at least one variable from the core must be hit
                    hs_solver.add(z3.Or([hs_vars[i] for i in core_indices]))
            else:
                # Hitting set problem is unsatisfiable
                if best_model is not None:
                    return True, best_model, best_cost
                else:
                    return False, None, float('inf') 