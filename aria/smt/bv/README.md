# Bit-Vector SMT

This package contains ARIA's bit-vector solving and bit-blasting utilities for
quantifier-free bit-vector fragments, plus a bit-vector interpolant prototype.

## Main Modules

- `qfbv_solver.py`: `QFBVSolver` for `QF_BV` formulas. It combines Z3
  preprocessing and bit-blasting with PySAT solving, with a Z3 fallback mode.
- `qfaufbv_solver.py`: `QFAUFBVSolver` for `QF_AUFBV` formulas. It reduces
  array/UF/bit-vector formulas using Z3 tactics and solves propositional cases
  with PySAT.
- `qfufbv_solver.py`: `QFUFBVSolver` for `QF_UFBV` formulas with the same
  general flattening-plus-SAT structure.
- `qfbv_itp.py`: bit-vector Craig interpolant prototype built on top of mapped
  bit-blasting.
- `mapped_blast.py`: bit-blast formulas while preserving a mapping from
  bit-vector variables to generated Boolean variables.
- `unmapped_blast.py`: simpler bit-blasting path for solving `QF_BV` formulas
  without tracking the SAT/BV correspondence.

## Main Ideas

- Use Z3 tactics such as `simplify`, `reduce-bv-size`, `max-bv-sharing`,
  `ackermannize_bv`, and `bit-blast` to reduce SMT formulas to propositional
  structure.
- Export CNF through Z3 or custom DIMACS conversion.
- Solve the resulting SAT problem with PySAT when possible.

## Typical Usage

```python
from aria.smt.bv.qfbv_solver import QFBVSolver

solver = QFBVSolver()
result = solver.solve_smt_file("example.smt2")
print(result)
```

For direct bit-blasting utilities:

```python
from aria.smt.bv.unmapped_blast import qfbv_to_sat

result = qfbv_to_sat(formula)
print(result)
```

## Notes

- The default PySAT engine in these solvers is `mgh`.
- `mapped_blast.py` is the right entry point when later phases need to recover
  which Boolean variables correspond to each bit-vector bit.
- The interpolation code in `qfbv_itp.py` is experimental and oriented toward
  research workflows rather than polished production APIs.
