# Modal Reasoning

`aria.bool.modal` provides a lightweight finite-model modal logic toolkit.

## Current Scope

- modal AST nodes: `Atom`, `Not`, `And`, `Or`, `Implies`, `Iff`, `Box`, `Diamond`
- finite `KripkeModel` semantics with world-indexed evaluation
- rooted model utilities: predecessors, reachability, induced submodels, and
  generated submodels
- frame validation for `K`, `D`, `T`, `B`, `K4`, `S4`, and `S5`
- bounded finite witness search for satisfiability, non-validity, and
  entailment failure, with optional Z3-backed bounded search
- parsing, implication elimination, modal NNF conversion, and canonical
  pretty-printing, including biconditionals
- structural helpers such as formula size, modal depth, and subformula
  enumeration
- lightweight formula simplification

## Example

```python
from aria.bool.modal import (
    Box,
    Diamond,
    FrameLogic,
    Iff,
    Atom,
    find_countermodel,
    find_model,
    format_formula,
    parse_formula,
    simplify,
)

formula = simplify(Iff(parse_formula("[](p -> <>q)"), parse_formula("[]<>q")))
print(format_formula(formula, unicode=True))

witness = find_model(
    Diamond(Atom("p")), logic=FrameLogic.D, max_worlds=2, backend="z3"
)
countermodel = find_countermodel(
    formula, logic=FrameLogic.K, max_worlds=2, backend="z3"
)
```

## Notes

- witness search is exhaustive only up to the supplied world bound
- `backend="auto"` uses the Z3 encoding when available and otherwise falls
  back to exhaustive finite enumeration
- this module is designed for small examples, regression tests, and
  prototyping rather than large-scale modal solving
