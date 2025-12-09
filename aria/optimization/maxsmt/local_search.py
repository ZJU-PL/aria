"""
Local search algorithm for MaxSMT solving.

This module implements a local search approach for MaxSMT, 
particularly suitable for SMT formulas over linear integer arithmetic.
"""

from typing import Tuple, Optional, List

import z3

from .base import MaxSMTSolverBase, logger


class LocalSearchSolver(MaxSMTSolverBase):
    """
    Local search algorithm for MaxSMT(LIA)
    
    Uses local search techniques to find a solution.
    Currently only implemented for SMT formulas over linear integer arithmetic.
    """
    
    def solve(self, max_iterations: int = 1000) -> Tuple[bool, Optional[z3.ModelRef], float]:
        """Local search algorithm for MaxSMT
        
        Args:
            max_iterations: Maximum number of iterations for the local search
        
        Returns:
            Tuple of (sat, model, optimal_cost)
        """
        # First check if hard constraints are satisfiable
        solver = z3.Solver()
        for hc in self.hard_constraints:
            solver.add(hc)
        
        if solver.check() != z3.sat:
            logger.warning("Hard constraints are unsatisfiable")
            return False, None, float('inf')
        
        # Start with a model that satisfies hard constraints
        current_model = solver.model()
        
        # Get all variables in the formula
        all_vars = set()
        for hc in self.hard_constraints:
            all_vars.update(self._get_variables(hc))
        for sc in self.soft_constraints:
            all_vars.update(self._get_variables(sc))
        
        # Convert the set to a list for iteration
        all_vars = list(all_vars)
        
        # Initialize best model and cost
        best_model = current_model
        best_cost = self._calculate_cost(current_model)
        
        # Main local search loop
        for _ in range(max_iterations):
            # Try to improve the current model
            improved = False
            
            # Try to modify each variable
            for var in all_vars:
                if not z3.is_int(var):
                    continue  # Currently only supports integer variables
                
                try:
                    # Get current value
                    current_value = current_model.eval(var, model_completion=True).as_long()
                    
                    # Try neighboring values
                    for offset in [-2, -1, 1, 2]:
                        new_value = current_value + offset
                        
                        # Create a temporary solver and assign the new value
                        temp_solver = z3.Solver()
                        
                        # Add all hard constraints
                        for hc in self.hard_constraints:
                            temp_solver.add(hc)
                        
                        # Add the fixed value for this variable
                        temp_solver.add(var == new_value)
                        
                        # Add the values for all other variables from the current model
                        # (except the one we're modifying)
                        for other_var in all_vars:
                            if other_var != var and other_var in current_model.decls():
                                val = current_model.eval(other_var, model_completion=True)
                                temp_solver.add(other_var == val)
                        
                        # Check if satisfiable
                        if temp_solver.check() == z3.sat:
                            # Get the new model
                            new_model = temp_solver.model()
                            
                            # Calculate the new cost
                            new_cost = self._calculate_cost(new_model)
                            
                            # If better than current best, update
                            if new_cost < best_cost:
                                best_model = new_model
                                best_cost = new_cost
                                current_model = new_model
                                improved = True
                                break
                except:
                    # Skip if there are any errors (e.g., variable not in model)
                    continue
                
                if improved:
                    break
            
            # If no improvement was found, stop
            if not improved:
                break
        
        return True, best_model, best_cost 