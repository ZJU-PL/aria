# Backbone Computation

Algorithms for computing backbones (also called "implied literals") of Boolean and SMT formulas.

## What is a Backbone?

The backbone of a formula consists of literals that are true in ALL satisfying assignments.
These literals are necessarily true and cannot be negated without making the formula unsatisfiable.

## Components

- `sat_backbone.py`: Backbone computation for SAT formulas
  - Multiple algorithms: iterative, chunking, refinement, approximation
  - `compute_backbone()`, `compute_backbone_iterative()`, `compute_backbone_chunking()`, etc.
- `smt_backbone_literals.py`: Backbone computation for SMT formulas (literal-level)
- `smt_backbone_clauses.py`: Backbone computation for SMT formulas (clause-level)

## Algorithms

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| Iterative | Repeated SAT calls with assumptions | Balanced precision/speed |
| Chunking | Process variables in chunks | Large formulas |
| Refinement | Start with approximation, refine | When near-complete solution needed |
| Approximation | Quick over-approximation | Fast initial bounds |

## Usage

```python
from aria.backbone import compute_backbone, BackboneAlgorithm

# Compute backbone of a CNF formula
backbone = compute_backbone(cnf, algorithm=BackboneAlgorithm.ITERATIVE)

# Check if a literal is in the backbone
is_backbone = is_backbone_literal(cnf, literal)
```
