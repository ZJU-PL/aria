# Interactive Theorem Proving

An interactive proof assistant built on top of the Z3 theorem prover.

**Note:** "ITP" here means "Interactive Theorem Proving", not "Interpolation".

## Features

- Proof construction with tactics
- Axiom and definition management
- Theory support for multiple logics
- Integration with external solvers

## Components

### Core
- `kernel.py`: Core proof kernel
- `smt.py`: Z3 integration
- `tactics.py`: Proof tactics
- `notation.py`: Syntax and notation

### Theories
- `theories/bool.py`: Boolean logic
- `theories/int.py`: Integer arithmetic
- `theories/real.py`: Real arithmetic
- `theories/algebra/`: Algebraic structures
- `theories/logic/`: Various logics (Peano, ZF, etc.)
- `theories/set.py`: Set theory
- `theories/seq.py`: Sequence theory
- `theories/regex.py`: Regular expressions

### Parsers
- `parsers/smtlib.py`: SMT-LIB parser
- `parsers/tptp.py`: TPTP parser
- `parsers/sexp.py`: S-expression parser

### Solvers
- `solvers/egraph.py`: E-graph based solving
- `solvers/gappa.py`: Gappa integration
- `solvers/datalog.py`: Datalog solver

### Utilities
- `rewrite.py`: Term rewriting
- `datatype.py`: Datatype definitions
- `property.py`: Property checking

## Usage

```python
from aria.itp import prove, axiom, define, FreshVar

# Define an axiom
axiom("my_axiom", "forall x. P(x) -> Q(x)")

# Prove a theorem
result = prove("forall x. Q(x)")
```
