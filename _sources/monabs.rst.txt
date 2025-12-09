Monadic Predicate Abstraction
=================================

===============
Introduction to Monadic Predicate Abstraction
===============

Given a formula F and a set of predicates {P1, ..., Pn}, monadic predicate abstraction decides for
each Pi whether F and Pi is satisfiable or not. This is a fundamental operation in many program analysis
and verification tasks.

Applications:

- In symbolic execution, F is a path condition, and Pi is a predicate related to certain properties.
- In program verification, F is an invariant, and Pi represents properties to be verified.
- In static analysis, monadic predicate abstraction can help determine which branches are feasible.

==========
Monadic Predicate Abstraction in Aria
==========

Aria provides several implementations of monadic predicate abstraction with different performance characteristics:

1. **Unary Check** (`unary_check.py`): Basic implementation that checks each predicate individually.
   - `unary_check`: Standard implementation
   - `unary_check_cached`: Optimized version with caching

2. **Disjunctive Check** (`dis_check.py`): Checks predicates using a disjunctive approach.
   - `disjunctive_check`: Standard implementation
   - `disjunctive_check_incremental`: Incremental version for better performance

3. **UNSAT Check** (`unsat_check.py`): Leverages UNSAT core computation for efficient checking.

4. **Parallel Check** (`parallel_check.py`): Parallelized implementation for improved performance on multi-core systems.

The module also includes PySMT-specific implementations (`unary_check_pysmt.py`, `dis_check_pysmt.py`) for compatibility with the PySMT framework.

==========
Usage Example
==========

```python
import z3
from aria.monabs.unary_check import unary_check
from aria.monabs.dis_check import disjunctive_check_incremental

# Define variables
x, y, z = z3.Ints('x y z')

# Define formula and predicates
formula = z3.And(x > 0, y > x, z < y)
predicates = [x < 10, y < 20, z > 5, x + y > z]

# Run monadic predicate abstraction
result_unary = unary_check(formula, predicates)
result_incremental = disjunctive_check_incremental(formula, predicates)

# Results are boolean lists indicating satisfiability of each predicate
print("Unary check results:", result_unary)
print("Incremental check results:", result_incremental)
```

For more advanced usage and benchmarking, refer to the `run_monabs.py` script in the repository root.

==========
Performance Comparison
==========

The different implementations offer various performance trade-offs:

- `unary_check`: Simple but potentially slower for many predicates
- `unary_check_cached`: Improved performance through caching
- `disjunctive_check`: Better for related predicates
- `disjunctive_check_incremental`: Best for incremental solving
- `unsat_check`: Efficient for formulas with many unsatisfiable predicates
- `parallel_check`: Best for multi-core systems with many predicates

==========
References
==========

- T. Ball, R. Majumdar, T. Millstein, and S. K. Rajamani. "Automatic predicate abstraction of C programs." In PLDI, 2001.
- E. M. Clarke, O. Grumberg, S. Jha, Y. Lu, and H. Veith. "Counterexample-guided abstraction refinement." In CAV, 2000.
