"""
Statistics tracking module for LIA* (Linear Integer Arithmetic Star) solver.

This module contains global variables used to track various performance metrics
and statistics during the execution of the LIA* solving process. These statistics
help in analyzing solver performance and debugging.
"""

# Problem characteristics
problem_size = 0  # Number of variables/constraints in the input problem

# Solver operation counters
z3_calls = 0  # Total number of calls made to the Z3 SMT solver
interpolants_generated = 0  # Number of interpolants generated during solving
merges = 0  # Number of merge operations performed on solution sets
shiftdowns = 0  # Number of shift-down operations applied
offsets = 0  # Number of offset adjustments made

# Timing measurements (in seconds)
reduction_time = 0  # Time spent on problem reduction/preprocessing
augment_time = 0  # Time spent on augmenting solution sets
interpolation_time = 0  # Time spent generating interpolants
solution_time = 0  # Total time spent finding solutions