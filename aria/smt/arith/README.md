# Arithmetic


### Incremental Lineralization

Basic idea: Abstraction/refinement to SMT(`QF_UFLLA`/`QF_UFLRA`)

- Non-linear multiplication, sin() and exp() modeled by uninterpreted functions
- Incrementally axiomatization on demand by linear constraints

1. Abstract a non-linear (e.g., `QF_NRA`) formula as a linear formula (`QF_UFLRA`)
2. If the abstracted formula is UNSAT, then the original formula is also UNSAT
3. Else, validate the model of the `QF_UFLRA` formula.
    + If the model is infeasible in the non-linear world, then
      we can refine the `QF_UFLRA` formula. Go to Step 2.
    + If the model is feasible in the non-linear world, then
      the original formula is also satisfiable.

Publications

- Icremental Linearization for Satisfiability and Verification Modulo Nonlinear Arithmetic and Transcendental Functions. Ahmed Irfan. PhD thesis


## Maathmatica
