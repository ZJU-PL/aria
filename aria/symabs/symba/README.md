# SYMBA: Symbolic Optimization with SMT Solvers

This directory contains an implementation of the SYMBA algorithm for symbolic optimization using SMT solvers, based on the paper "Symbolic Optimization with SMT Solvers" by Li, Albarghouthi, Kincaid, Gurinkel, and Chechik.

## Overview

SYMBA is an SMT-based optimization algorithm for objective functions in the theory of linear real arithmetic (LRA). Given a formula φ and an objective function t, SYMBA finds a satisfying assignment of φ that exhibits the least upper bound κ (maximum value) of t such that φ ∧ t ≤ κ is satisfiable and φ ∧ t ≥ κ is unsatisfiable.

## Key Features

- **SMT-based optimization**: Uses SMT solvers as black boxes for satisfiability checking
- **Multi-objective support**: Can optimize multiple objective functions simultaneously
- **Pareto optimization**: Supports finding Pareto-optimal solutions
- **Incremental algorithm**: Uses inference rules to iteratively improve approximations
- **Integration with existing OMT infrastructure**: Compatible with aria's OMT engines

## Algorithm Overview

SYMBA maintains two key data structures:
- **U**: Under-approximation of points that are NOT optimal
- **O**: Over-approximation of points that ARE optimal

The algorithm applies inference rules until convergence:
1. **INIT**: Initialize U and O
2. **GLOBALPUSH**: Find models outside the current under-approximation
3. **UNBOUNDED**: Detect unbounded objectives
4. **UNBOUNDED-FAIL**: Handle bounded objectives
5. **BOUNDED**: Strengthen the over-approximation

## Usage

### Basic Usage

```python
import z3
from aria.symabs.symba import SYMBA

# Define variables and constraints
x = z3.Int('x')
y = z3.Int('y')
constraints = z3.And(x + y <= 10, x >= 0, y >= 0)

# Define objective function
objective = x + y

# Create and run SYMBA
symba = SYMBA(constraints, [objective])
state = symba.optimize()

# Get results
optimal_values = symba.get_optimal_values()
print(f"Optimal value: {optimal_values[objective]}")
```

### Multi-Objective Optimization

```python
from aria.symabs.symba import MultiSYMBA

# Multiple objectives
objectives = [x, y]  # maximize x and maximize y

# Create and run MultiSYMBA
multi_symba = MultiSYMBA(constraints, objectives)
state = multi_symba.optimize()

# Get Pareto front
pareto_front = multi_symba.get_pareto_front()
pareto_values = multi_symba.get_pareto_values()
```

## Implementation Details

### Core Classes

- **`SYMBA`**: Main optimization algorithm implementation
- **`MultiSYMBA`**: Specialized class for multi-objective optimization
- **`SYMBAState`**: Represents the state of the optimization process
- **`InferenceRule`**: Enumeration of SYMBA inference rules

### Key Methods

- **`optimize()`**: Run the SYMBA optimization algorithm
- **`get_optimal_values()`**: Get optimal values for each objective
- **`get_pareto_front()`**: Get Pareto-optimal solutions (MultiSYMBA only)
- **`is_optimal()`**: Check if a model represents an optimal solution

### Inference Rules Implementation

The implementation faithfully follows the paper's inference rules:

1. **INIT Rule**: Initialize under-approximation U and over-approximation O
2. **GLOBALPUSH Rule**: Find models of φ that are not captured by form_T(U)
3. **UNBOUNDED Rule**: Detect when an objective function is unbounded
4. **UNBOUNDED-FAIL Rule**: Handle cases where objectives are bounded
5. **BOUNDED Rule**: Strengthen the over-approximation O


## Examples

See `example.py` for comprehensive usage examples:

```bash
python example.py
```

This includes:
- Resource allocation problems
- Production optimization with multiple objectives
- Bounded optimization problems

## Testing

Run the test suite:

```bash
python test_symba.py
```

Tests cover:
- Simple maximization and minimization
- Multi-objective optimization
- Linear programming examples
- Edge cases and error conditions

## Performance Considerations

- SYMBA uses incremental SMT solving for efficiency
- The number of SMT queries depends on problem complexity
- Performance scales with the number of objectives and constraint complexity
- Early termination when optimal solutions are found

## Limitations

- Currently optimized for linear real arithmetic (LRA)
- Integer constraints may require special handling
- Performance may degrade with very large search spaces
