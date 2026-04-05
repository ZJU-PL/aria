# Format Translators

Converters between various constraint/solver formats.

## CNF/Propositional

| File | Description |
|------|-------------|
| `dimacs2smt.py` | DIMACS CNF → SMT2 |
| `cnf2smt.py` | CNF → SMT2 encoding |
| `cnf2lp.py` | CNF → Linear Programming format |
| `smt2dimacs.py` | SMT-LIB2 propositional fragment → DIMACS |
| `opb2smt.py` | OPB / WBO-style pseudo-Boolean constraints, soft clauses, and products → SMT-LIB2 |
| `wcnf2z3.py` | Weighted CNF → Z3 optimization |
| `wcnf2smt.py` | Weighted CNF / MaxSAT text → SMT-LIB2 with soft constraints |

## QBF (Quantified Boolean Formulas)

| File | Description |
|------|-------------|
| `qbf2smt.py` | QBF → SMT2 encoding |
| `qcir2smt.py` | QCIR → SMT-LIB2 |

## SMT-LIB

| File | Description |
|------|-------------|
| `smt2c.py` | SMT-LIB → C code generation |
| `smt2sympy.py` | SMT-LIB → SymPy expressions |

## SyGuS

| File | Description |
|------|-------------|
| `sygus2smt.py` | SyGuS syntax → SMT2 |

## FlatZinc (from `fzn2omt/`)

| File | Description |
|------|-------------|
| `fzn2z3.py` | FlatZinc → Z3 |
| `fzn2cvc4.py` | FlatZinc → CVC4 |
| `fzn2optimathsat.py` | FlatZinc → Optimathsat |
| `smt2model2fzn.py` | SMT model → FlatZinc solution |

## Shared Infrastructure

| File | Description |
|------|-------------|
| `registry.py` | Translator capability registry used by the CLI |
| `parsing.py` | Shared parsing adapters reused by translator modules |

## Usage

```python
from aria.translator import dimacs2smt

# Convert DIMACS CNF to SMT2
dimacs2smt.convert_file('input.cnf', 'output.smt2')
```

The registry-backed CLI entrypoint is exposed through `aria.cli.fmldoc_cli` and
currently covers DIMACS, QDIMACS, QCIR, OPB, WCNF, SyGuS, and selected
SMT-LIB2 conversions.

The OPB translator now handles weighted constraints (`[w] ... ;`), `soft:`
headers, min/max objectives, richer comparators, and product terms by emitting
either `QF_LIA` or `QF_NIA` SMT-LIB2 as needed.
