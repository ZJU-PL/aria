# Format Translators

Converters between various constraint/solver formats.

## CNF/Propositional

| File | Description |
|------|-------------|
| `dimacs2smt.py` | DIMACS CNF → SMT2 |
| `cnf2smt.py` | CNF → SMT2 encoding |
| `cnf2lp.py` | CNF → Linear Programming format |
| `wcnf2z3.py` | Weighted CNF → Z3 optimization |

## QBF (Quantified Boolean Formulas)

| File | Description |
|------|-------------|
| `qbf2smt.py` | QBF → SMT2 encoding |

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

## Usage

```python
from aria.translator import dimacs2smt

# Convert DIMACS CNF to SMT2
dimacs2smt.convert_file('input.cnf', 'output.smt2')
```
