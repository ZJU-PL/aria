# Model Counting

Counting the number of satisfying assignments for logical formulas.

## Components

### Main Interface
- `api.py`: Dispatches counting based on formula or file type
- `mc.py`: CLI model counting entry point

### Boolean Model Counting
- `bool/dimacs_counting.py`: Count models from DIMACS CNF files via sharpSAT or ApproxMC
- `bool/pysmt_expr_counting.py`: PySMT-based Boolean counting
- `bool/z3py_expr_counting.py`: Z3 Python API counting

### Bit-Vector Counting
- `bv/counter.py`: Model counting for QF_BV formulas

### Arithmetic Counting
- `arith/arith_counting_latte.py`: LIA model counting via LattE integration

### String Counting
- `string/string_counting.py`: Model counting for string constraints (placeholder)

## Usage

High-level interface:

```python
import z3
from aria.counting import count, count_from_file

# Count from formula
x, y = z3.Bools("x y")
print(count(z3.Or(x, y)))

# Count models from DIMACS or SMT file
count_val = count_from_file("formula.cnf")
```

Lower-level theory interfaces:

```python
# Direct DIMACS counting with sharpSAT
from aria.counting.bool.dimacs_counting import call_sharp_sat
count_val = call_sharp_sat("formula.cnf")

# Using PySMT
from aria.counting.bool.pysmt_expr_counting import count_pysmt_solutions
# count_val = count_pysmt_solutions(pysmt_formula)
```
