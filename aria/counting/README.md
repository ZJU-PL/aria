# Model Counting

Counting the number of satisfying assignments for logical formulas.

## Components

### Boolean Model Counting
- `bool/dimacs_counting.py`: Count models from DIMACS CNF files
- `bool/pysmt_expr_counting.py`: PySMT-based counting
- `bool/z3py_expr_counting.py`: Z3 Python API counting

### Bit-Vector Counting
- `qfbv_counting.py`: Model counting for QF_BV formulas

### Arithmetic Counting
- `arith/arith_counting_latte.py`: LIA model counting via LattE integration

### String Counting
- `string/string_counting.py`: Model counting for string constraints (placeholder)

### Main Interface
- `mc.py`: Main model counting interface

## Usage

```python
from aria.counting.bool import dimacs_counting

# Count models from DIMACS file
count = dimacs_counting.count_models('formula.cnf')

# Using PySMT
from aria.counting.bool.pysmt_expr_counting import PySMTModelCounter
counter = PySMTModelCounter()
count = counter.count(formula)
```
