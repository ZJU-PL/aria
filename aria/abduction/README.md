# Abduction

Abductive reasoning engine for computing explanations (abductive hypotheses) for observations.

## Components

- `abductor.py`: Main abductor implementation
- `abductor_parser.py`: SMT-LIB2 to Z3 expression parser
- `dillig_abduct.py`: Implementation based on Dillig et al.'s algorithm
- `qe_abduct.py`: Quantifier elimination based abduction
- `cvc5_sygus_abduct.py`: CVC5 SyGuS-based abduction interface
- `utils.py`: Utility functions
- `belief_revision.py`: Belief-base expansion, contraction, and revision

## Usage

```python
from aria.abduction import Abductor

# Create abductor for a theory
abductor = Abductor(theory='QF_LIA')

# Compute explanations for an observation
explanations = abductor.explain(observation)
```

## Belief Revision

The package also includes consistency-based belief change for finite belief
bases represented as lists of Z3 formulas.

```python
from z3 import Int
from aria.abduction import revise_belief_base

x = Int("x")
base = [x >= 0, x <= 5, x >= 10]
result = revise_belief_base(base, x <= 3)

print(result.kept_indices)
print(result.result_base)
print(result.conflict_sets)
print(result.incision_indices)
print(result.incision_cost)
print(result.rank_strata)
```

Revision treats the incoming belief as hard and keeps a maximal consistent
subset of the prior base. Weighted retention and optimal-outcome enumeration
are also supported.

Available revision operators:

- `max_retention`: keep a maximum-weight consistent subset of the old base
- `lexicographic`: prefer keeping earlier beliefs in the given base order
- `kernel`: compute minimal conflict sets and remove a minimum-weight incision
- `partial_order`: prefer higher-priority belief strata given by `ranks=[...]`
- `epistemic`: use epistemic rank strata, where smaller ranks are retained first

Conflict sets are extracted via `aria.unsat_core` and returned in
`BeliefRevisionResult.conflict_sets` and `BeliefRevisionResult.conflict_belief_sets`.
Chosen incisions are returned explicitly in `BeliefRevisionResult.incision_indices`
and `BeliefRevisionResult.incision_beliefs`. Their aggregate weight is exposed
as `BeliefRevisionResult.incision_cost`.

Rank-aware operators also populate `BeliefRevisionResult.rank_strata`, which
summarizes each stratum's belief indices, kept/removed indices, and retained vs.
removed weight.

Contraction supports the same operators:

- `contract_belief_base(..., operator="max_retention")`
- `contract_belief_base(..., operator="lexicographic")`
- `contract_belief_base(..., operator="kernel")`
- `contract_belief_base(..., operator="partial_order", ranks=[...])`
- `contract_belief_base(..., operator="epistemic", ranks=[...])`

Kernel outcomes can also be fully enumerated:

```python
from aria.abduction import enumerate_optimal_revisions, enumerate_optimal_contractions

revision_results = enumerate_optimal_revisions(base, phi, operator="kernel")
contraction_results = enumerate_optimal_contractions(base, phi, operator="kernel")
```

These enumeration APIs return every minimum-cost kernel incision up to the
optional `limit`.

For rank-based operators, pass one rank per belief. Smaller ranks mean stronger
epistemic priority, and beliefs in the same rank form one stratum.

## References

- Dillig et al., "Abduction with Fast Subsumption" - Algorithm used in `dillig_abduct.py`
