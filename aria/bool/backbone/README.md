# Boolean Backbone

This package collects SAT-level backbone algorithms for Boolean formulas.

A backbone literal is a literal that is true in every satisfying assignment of
the formula. For CNF formulas, this is the Boolean analogue of an implied
literal.

## Included algorithms

- `compute_backbone_iterative`: checks each variable by flipping it.
- `compute_backbone_chunking`: flips variables in batches to reduce calls.
- `compute_backbone_refinement`: starts from a model and refines candidate
  literals.
- `compute_backbone_with_approximation`: returns definite and potential
  backbone literals from sampled models.
- `is_backbone_literal`: tests one literal directly.

## Usage

```python
from pysat.formula import CNF

from aria.bool.backbone import BackboneAlgorithm, compute_backbone

formula = CNF(from_clauses=[[1, 2], [-1, 3], [-2, 3]])
backbone, solver_calls = compute_backbone(
    formula, algorithm=BackboneAlgorithm.BACKBONE_REFINEMENT
)

print(backbone)      # [3]
print(solver_calls)  # solver-call count
```

## Notes

- These APIs operate on PySAT `CNF` formulas.
- For SMT-level backbone reasoning, the legacy `aria.backbone` package still
  contains the SMT-specific helpers.
