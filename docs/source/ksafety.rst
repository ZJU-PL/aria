K-Safety Verification
======================

Overview
--------

K-safety verification is a technique for verifying relational properties that relate multiple execution traces. Unlike traditional safety properties that hold for a single execution, k-safety properties express relationships between k different executions of a program.

The k-safety engine in EFMC supports verification of various hyperproperties including:

- **Non-Interference**: Ensures that low-security outputs do not depend on high-security inputs
- **Determinism**: Ensures that identical inputs always produce identical outputs
- **Symmetry**: Verifies that outputs are permuted according to input permutations
- **Differential Privacy**: Ensures bounded change in outputs under bounded change in inputs
- **Program Equivalence**: Verifies that two program versions are functionally equivalent
- **Refinement**: Verifies that one program refines another
- **HyperLTL Properties**: Supports temporal hyperproperties using a bounded HyperLTL logic

Architecture
------------

The k-safety engine is built on a base prover that:

1. Creates k copies of the transition system (one for each trace)
2. Encodes the relational property as a hyperproperty
3. Uses bounded model checking or k-induction to verify the property

Available Provers
-----------------

- **BaseKSafetyProver**: Base class for k-safety verification
- **NonInterferenceProver**: Verifies non-interference properties
- **DeterminismProver**: Verifies determinism properties
- **SymmetryProver**: Verifies symmetry properties
- **DifferentialPrivacyProver**: Verifies differential privacy properties
- **EquivalenceProver**: Verifies program equivalence
- **RefinementProver**: Verifies program refinement
- **HyperLTLProver**: Verifies HyperLTL temporal properties

Usage
-----

The k-safety engine is primarily used programmatically through the Python API:

.. code-block:: python

   from efmc.engines.ksafety import NonInterferenceProver
   from efmc.sts import TransitionSystem
   
   # Create transition system
   sts = TransitionSystem()
   # ... initialize sts ...
   
   # Create prover
   prover = NonInterferenceProver(sts, k=2, method='bounded_model_checking', max_bound=10)
   
   # Set high and low security variables
   prover.set_high_variables([high_var])
   prover.set_low_variables([low_var])
   
   # Verify
   result = prover.solve()

HyperLTL Example
----------------

.. code-block:: python

   from efmc.engines.ksafety import HyperLTLProver
   from efmc.engines.ksafety.hyperltl import Var, Atom, G, Implies
   
   # Define HyperLTL formula: G(high[0] == high[1] → G(low[0] == low[1]))
   phi = Implies(
       G(Atom('==', Var('high', 0), Var('high', 1))),
       G(Atom('==', Var('low', 0), Var('low', 1)))
   )
   
   prover = HyperLTLProver(sts, k=2, method='bounded_model_checking', max_bound=10)
   prover.set_formula(phi)
   result = prover.solve()

Verification Methods
--------------------

**Bounded Model Checking (BMC)**
- Unrolls transitions up to a bound B
- Checks property at step B
- Can find violations but cannot prove properties in general

**K-Induction**
- Base case: Property holds for initial states
- Inductive step: If property holds for steps < k, it holds at step k
- Can prove properties for all executions

Properties
----------

Non-Interference
~~~~~~~~~~~~~~~~

Ensures that low-security outputs do not depend on high-security inputs:

∀t₁, t₂. (high₁ = high₂) → (low₁ = low₂)

Determinism
~~~~~~~~~~~

Ensures that identical inputs always produce identical outputs:

∀t₁, t₂. (inputs₁ = inputs₂) → (outputs₁ = outputs₂)

Symmetry
~~~~~~~~

Verifies that outputs are permuted according to input permutations:

∀t₁, t₂. perm_in(inputs₁) = inputs₂ → perm_out(outputs₁) = outputs₂

Differential Privacy
~~~~~~~~~~~~~~~~~~~~

Ensures bounded change in outputs under bounded change in inputs:

∀t₁, t₂. |inputs₁ - inputs₂| ≤ δ → |outputs₁ - outputs₂| ≤ ε

Program Equivalence
~~~~~~~~~~~~~~~~~~~

Verifies functional equivalence between two program versions:

∀t₁, t₂. (inputs₁ = inputs₂) → (outputs₁ = outputs₂)

Related Work
------------

- Clarkson, M.R. and Schneider, F.B. "Hyperproperties." Journal of Computer Security, 2010.
- Clarkson, M.R., Finkbeiner, B., Koleini, M., et al. "Temporal logics for hyperproperties." POST 2014.
- Terauchi, T. and Aiken, A. "Secure information flow as a safety problem." SAS 2005.
- Hsu, T.-C., Sánchez, C., and Bonakdarpour, B. "Bounded model checking for hyperproperties." TACAS 2021.

