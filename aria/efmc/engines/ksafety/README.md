# k-safety Verification


## Properties Implemented 

### Non-Interference 

- Purpose: Ensures that low-security outputs do not depend on high-security inputs
- ∀t1, t2. (high1 = high2) → (low1 = low2)
- Implementation: Compares two execution traces where high-security variables are kept equal and checks if low-security variables remain equal



### Determinism

- Purpose: Ensures that for the same inputs, the system always produces the same outputs
- Formula: ∀t1, t2. (inputs1 = inputs2) → (outputs1 = outputs2)
- Implementation: Compares two execution traces with identical inputs and verifies outputs are identical

Additionally, the engine now supports:

### Symmetry (Input Permutation)

- Purpose: Verify that under a permutation of inputs, outputs are permuted accordingly
- Formula: ∀t1, t2. perm_in(inputs1) = inputs2 → perm_out(outputs1) = outputs2
- API: `SymmetryProver.verify_input_permutation_symmetry(input_perm, output_perm)`

### Epsilon-Sensitivity (Deterministic DP analogue)

- Purpose: Ensure bounded change in outputs under bounded change in inputs
- Formula: ∀t1, t2. |inputs1 - inputs2| ≤ δ → |outputs1 - outputs2| ≤ ε
- API: `DifferentialPrivacyProver.verify_epsilon_sensitivity(input_vars, output_vars, adjacency_bounds, epsilon)`


## Properties Not Implemented 


### Program Equivalence/Refinement

- Purpose: Verify that two program versions are equivalent or that one refines the other
- Functional Equivalence (implemented via two-system BMC): for same inputs, outputs match at bound B
- API: `EquivalenceProver.verify_functional_equivalence(input_vars, output_vars, inputs_equal_all_steps=True)`
- Refinement (current equals equivalence for deterministic outputs): `RefinementProver.verify_refinement(...)`

### Symmetry Properties

Note: Basic input-permutation symmetry is implemented. Other symmetry classes (parameter, geometric) are TBD.


## Engine Enhancements 

- Time-unrolled BMC: BMC now unrolls transitions up to bound B and checks property at step B
- k-induction: Added a proper inductive step (assume property holds for steps < k, prove at k)

### HyperLTL-like formulas (bounded)

- Minimal DSL and prover to express relational temporal properties across k traces
- Operators: G, F, X, U with bounded semantics
- API: `HyperLTLProver.set_formula(phi)` then `solve()` with `method="bounded_model_checking"`
- Example usage:

```python
from efmc.engines.ksafety import HyperLTLProver
from efmc.engines.ksafety.hyperltl import Var, Atom, G, Implies

phi = Implies(
    G(Atom('==', Var('high',0), Var('high',1))),
    G(Atom('==', Var('low',0), Var('low',1)))
)
prover = HyperLTLProver(sts, k=2, method='bounded_model_checking', max_bound=10)
prover.set_formula(phi)
res = prover.solve()
```

## More 

- Differential Privacy: ∀t1, t2. |output1 - output2| ≤ ε
- Statistical Independence: Output distributions are independent of certain inputs
- ...?


## Related Work 

[1] Clarkson, M.R. and Schneider, F.B. "Hyperproperties." Journal of Computer Security, 2010.

[2] Clarkson, M.R., Finkbeiner, B., Koleini, M., et al. "Temporal logics for hyperproperties." POST 2014.

[5] Terauchi, T. and Aiken, A. "Secure information flow as a safety problem." SAS 2005.

[6] Clarkson, M.R., Finkbeiner, B., Koleini, M., et al. "Temporal logics for hyperproperties." CAV 2014.

[7] Coenen, N., Finkbeiner, B., Hahn, C., and Hofmann, J. "The hierarchy of hyperlogics." LICS 2019.

[9] Barthe, G., D'Argenio, P.R., and Rezk, T. "Secure information flow by self-composition." Mathematical Structures in Computer Science, 2011.

[10] Unno, H., Terauchi, T., and Koskinen, E. "Constraint-based relational verification." CAV 2021.

[12] Hsu, T.-C., Sánchez, C., and Bonakdarpour, B. "Bounded model checking for hyperproperties." TACAS 2021.

[13] Coenen, N., Finkbeiner, B., Hofmann, J., and Passing, Y. "Safe approximation of hyperproperties using fixpoint iteration." APLAS 2019.

[31] Finkbeiner, B., Hahn, C., Lukert, P., Stenger, M., and Tentrup, L. "Synthesis from hyperproperties." Acta Informatica, 2020.

[33] Coenen, N., Finkbeiner, B., Hahn, C., and Hofmann, J. "The hierarchy of hyperlogics." LICS 2019.