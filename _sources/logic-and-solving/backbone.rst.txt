Backbone Computation
==================

===============
Introduction to Backbone Computation
===============

The backbone of a Boolean formula is the set of literals that take the same value (True or False) in all satisfying assignments.
These literals represent the "fixed" or "necessary" parts of any solution to the formula.

Backbone computation has important applications in:

- **Constraint solving**: Identifying variables that must take specific values
- **SAT solving**: Simplifying formulas by fixing backbone literals
- **Verification**: Finding invariants in system models
- **Optimization**: Reducing search space by fixing necessary assignments

===============
Backbone Computation in Aria
===============

Aria provides two main approaches to backbone computation:

1. **Literal-based Backbone** (`backbone_literals.py`): Computes the backbone by analyzing individual literals.

2. **Clause-based Backbone** (`backbone_clauses.py`): Computes the backbone by analyzing clauses in the formula.

Both implementations offer efficient algorithms for finding backbones in Boolean formulas.

===============
Usage Example
===============

```python
import z3
from aria.backbone import get_backbone

# Define a formula
x, y, z = z3.Bools('x y z')
formula = z3.And(z3.Or(x, y), z3.Or(z, ~y), z3.Or(~x, ~z))

# Compute the backbone
backbone = get_backbone(formula)

# Print the backbone literals
print("Backbone literals:")
for lit in backbone:
    print(f"  {lit}")

# Using the clause-based approach
from aria.backbone import get_backbone_clauses

backbone_clauses = get_backbone_clauses(formula)
print("Backbone literals (clause-based):")
for lit in backbone_clauses:
    print(f"  {lit}")
```

===============
Algorithm Details
===============

The backbone computation algorithms in Aria use several optimizations:

- **Incremental SAT solving**: Reusing solver state between iterations
- **Assumption-based solving**: Efficiently testing candidate backbone literals
- **Clause learning**: Leveraging learned clauses to speed up subsequent iterations
- **Early termination**: Detecting when no more backbone literals can be found

===============
Performance Considerations
===============

For most formulas, the literal-based approach (`get_backbone`) provides good performance. However, for formulas with specific structures, the clause-based approach (`get_backbone_clauses`) may be more efficient.

Factors affecting performance include:
- Formula size and structure
- Number of variables
- Density of the backbone (percentage of variables in the backbone)
- SAT solver implementation used

===============
References
===============

- I. Lynce and J. Marques-Silva. "Efficient haplotype inference with Boolean satisfiability." National Conference on Artificial Intelligence, 2006.
- J. Marques-Silva, M. Janota, and I. Lynce. "On computing backbones of propositional theories." ECAI, 2010.
- N. Narodytska and T. Walsh. "The importance of being structured: On the computation of backdoors for MaxSAT." AAAI, 2014.
