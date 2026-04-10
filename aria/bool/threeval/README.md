# Three-Valued Propositional Logic

`aria.bool.threeval` provides a lightweight AST and reasoning toolkit for
three-valued propositional logics.

## Included Semantics

- `STRONG_KLEENE`: strong Kleene K3, with designated set `{TRUE}`
- `WEAK_KLEENE`: weak Kleene logic with infectious `UNKNOWN`
- `LUKASIEWICZ_K3`: Lukasiewicz three-valued implication with min/max lattice
- `GODEL_G3`: Godel/G3 implication with min/max lattice

All semantics are instances of `Semantics`, so designated values can be
customized when you want tolerant consequence relations such as
`{TRUE, UNKNOWN}`.

## Consequence Convention

By default, `entails` and `is_valid` use designated-value preservation:

- a premise counts as satisfied when it evaluates to a designated value
- a conclusion must also evaluate to a designated value
- under the default semantics, only `TRUE` is designated

This means `p | !p` is not valid in strong Kleene logic, because it evaluates
to `UNKNOWN` when `p` is `UNKNOWN`.

If you want tolerant entailment, either construct a new semantics object with
different designated values or pass `designated_values` explicitly.

## Formula Support

The AST includes:

- constants: `TRUE`, `FALSE`, `UNKNOWN`
- atoms: `Variable("p")`
- unary: `Not`
- binary: `And`, `Or`, `Implies`, `Iff`, `Xor`, `Nand`, `Nor`
- n-ary helpers: `conjoin`, `disjoin`, `NaryAnd`, `NaryOr`

Formulas also expose structural helpers such as `variables()`, `atoms()`,
`subformulas()`, `size()`, `depth()`, and `substitute()`.

## Reasoning Utilities

- `evaluate(formula, valuation, semantics=...)`
- `truth_table(formula, semantics=...)`
- `entails(premises, conclusion, semantics=...)`
- `is_valid(formula, semantics=...)`
- `is_satisfiable`, `is_unsatisfiable`, `is_contingent`
- `find_model`, `find_counterexample`
- `satisfying_valuations`, `counterexample_valuations`
- `is_equivalent`, `is_consistent`
- `is_classically_valid`
- `is_classically_valid_under_completions`
- `simplify(formula, valuation=...)`

## Semantic Minimization

The module also implements the semantic-minimization algorithms from Reps,
Loginov, and Sagiv (LICS 2002):

- `supervaluation(formula, valuation)`
- `constructive_semantic_minimize(formula)`
- `prime_implicant_semantic_minimize(formula)`
- `semantically_minimize(formula, method="primes")`

The constructive algorithm follows the improved Blamey realization from the
paper's Section 4, while the prime-implicant variant follows Figure 2.

For strong Kleene semantics, the Section 5 path now builds the Boolean pair
representation directly from formula structure with OBDDs, following the
paper's bottom-up translation rather than enumerating all Boolean assignments.

## Parsing and Pretty Printing

`parse_formula` accepts a small infix syntax:

```text
!(p & q) -> (r xor unknown)
p <-> q
p nand q
p nor q
```

Recognized constants:

- `true`
- `false`
- `unknown`
- `⊤`
- `⊥`
- `?`

Pretty printing is available via `format_formula(formula)` and
`format_formula(formula, unicode=True)`.

## Interoperability with `aria.bool.nnf`

`aria.bool.threeval.adapters` provides two helpers:

- `from_nnf(sentence)`: convert a Boolean NNF sentence into a three-valued AST
- `to_nnf(formula, unknown_policy=...)`: convert a three-valued formula into
  Boolean NNF

Important: `to_nnf` is a classical embedding. The produced NNF sentence agrees
with the original formula on classical valuations. It does not encode the full
three-valued semantics of `UNKNOWN` inside the NNF object model.

The supported `unknown_policy` modes are:

- `"error"`: reject formulas containing `UNKNOWN`
- `"false"`: map `UNKNOWN` to Boolean false
- `"true"`: map `UNKNOWN` to Boolean true

## Example

```python
from aria.bool.threeval import (
    Not,
    Or,
    STRONG_KLEENE,
    TruthValue,
    Variable,
    evaluate,
    is_classically_valid,
)

p = Variable("p")
formula = Or(p, Not(p))

assert evaluate(formula, {"p": TruthValue.UNKNOWN}, STRONG_KLEENE) == TruthValue.UNKNOWN
assert is_classically_valid(formula)
```
