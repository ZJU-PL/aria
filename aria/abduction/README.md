# Abduction

Abductive reasoning engine for computing explanations (abductive hypotheses) for observations.

## Components

- `abductor.py`: Main abductor implementation
- `abductor_parser.py`: SMT-LIB2 to Z3 expression parser
- `dillig_abduct.py`: Implementation based on Dillig et al.'s algorithm
- `qe_abduct.py`: Quantifier elimination based abduction
- `cvc5_sygus_abduct.py`: CVC5 SyGuS-based abduction interface
- `utils.py`: Utility functions

## Usage

```python
from aria.abduction import Abductor

# Create abductor for a theory
abductor = Abductor(theory='QF_LIA')

# Compute explanations for an observation
explanations = abductor.explain(observation)
```

## References

- Dillig et al., "Abduction with Fast Subsumption" - Algorithm used in `dillig_abduct.py`
