UNSAT Core Extraction
=================

===============
Introduction to UNSAT Core Extraction
===============

An unsatisfiable core (UNSAT core) is a subset of constraints in an unsatisfiable formula that is still unsatisfiable.
A minimal unsatisfiable subset (MUS) is an UNSAT core that becomes satisfiable if any constraint is removed.

UNSAT core extraction is crucial in many applications:

- **Debugging**: Identifying the source of unsatisfiability in complex constraint systems
- **Verification**: Pinpointing the exact cause of property violations
- **Optimization**: Guiding search algorithms by identifying conflicting constraints
- **Explanation**: Providing human-readable explanations for why a system has no solution

===============
UNSAT Core Extraction in Aria
===============

Aria provides a comprehensive suite of algorithms for UNSAT core extraction:

1. **MARCO** (`marco.py`): A highly efficient algorithm for MUS enumeration that can find multiple MUSes quickly.

2. **MUSX** (`musx.py`): An algorithm focused on finding a single MUS with good performance characteristics.

3. **OPTUX** (`optux.py`): An optimized algorithm for finding minimal UNSAT cores.

4. **MSS** (`mss.py`): Algorithms for finding maximal satisfiable subsets, which are complementary to MUSes.

The module provides a unified interface through the `UnsatCoreComputer` class, which allows users to select the appropriate algorithm for their needs.

===============
Usage Example
===============

```python
import z3
from aria.unsat_core.unsat_core import get_unsat_core, Algorithm

# Define an unsatisfiable formula
x, y = z3.Ints('x y')
constraints = [
    x > 0,
    x < 0,
    y > 10,
    y < 5
]

# Create a solver factory function
def solver_factory():
    return z3.Solver()

# Extract an UNSAT core using the MARCO algorithm
result = get_unsat_core(
    constraints=constraints,
    solver_factory=solver_factory,
    algorithm=Algorithm.MARCO,
    timeout=10000  # 10 seconds
)

# Print the UNSAT core
print("UNSAT core:")
for core in result.cores:
    for idx in core:
        print(f"  {constraints[idx]}")

# Enumerate all MUSes
from aria.unsat_core.unsat_core import enumerate_all_mus

all_muses = enumerate_all_mus(
    constraints=constraints,
    solver_factory=solver_factory,
    timeout=10000
)

print(f"Found {len(all_muses.cores)} MUSes")
```

===============
Algorithm Selection
===============

The choice of algorithm depends on your specific needs:

- **MARCO**: Best for enumerating multiple MUSes, especially when you need to find as many as possible
- **MUSX**: Efficient for finding a single MUS quickly
- **OPTUX**: Optimized for finding minimal cores with specific characteristics

For most applications, the default MARCO algorithm provides a good balance of performance and quality.

===============
Advanced Features
===============

The UNSAT core extraction module provides several advanced features:

- **Timeout control**: Set maximum execution time for algorithms
- **Statistics collection**: Gather performance metrics during execution
- **Custom solvers**: Use any SMT solver that provides a compatible interface
- **Incremental solving**: Reuse solver state for improved performance

===============
References
===============

- M. H. Liffiton and K. A. Sakallah. "Algorithms for computing minimal unsatisfiable subsets of constraints." Journal of Automated Reasoning, 2008.
- J. Marques-Silva, F. Heras, M. Janota, A. Previti, and A. Belov. "On computing minimal correction subsets." IJCAI, 2013.
- A. Belov and J. Marques-Silva. "MUSer2: An efficient MUS extractor." Journal on Satisfiability, Boolean Modeling and Computation, 2012.
