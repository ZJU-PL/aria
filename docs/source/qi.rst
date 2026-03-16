Quantifier Instantiation (QI)
============================

The Quantifier Instantiation (QI) module provides verification capabilities using SMT solvers with quantifier support. It supports multiple strategies including Model-Based Quantifier Instantiation (MBQI) and E-matching, and can work with various SMT solvers.

======================
Overview
======================

Quantifier Instantiation is a technique for handling quantified formulas in SMT solving. The QI prover in EFMC formulates verification problems as quantified formulas and uses advanced SMT solving techniques to find inductive invariants.

======================
Algorithm
======================

The QI prover works by:

1. **Problem Formulation**: Convert the verification problem into quantified formulas:
   - Initiation: ∀x. init(x) → inv(x)
   - Consecution: ∀x,x'. inv(x) ∧ trans(x,x') → inv(x')
   - Safety: ∀x. inv(x) → post(x)

2. **Solver Configuration**: Configure the SMT solver with appropriate quantifier instantiation strategies:
   - **MBQI (Model-Based Quantifier Instantiation)**: Uses concrete models to guide instantiation
   - **E-matching**: Pattern-based instantiation using triggers
   - **Combined**: Uses both MBQI and E-matching together

3. **Solving**: Use the configured solver to find a satisfying assignment for the invariant function

======================
Supported Strategies
======================

**MBQI (Model-Based Quantifier Instantiation):**
- Uses concrete models from the solver to guide quantifier instantiation


**E-matching:**
- Pattern-based instantiation using triggers


**Combined:**

======================
Supported Solvers
======================

- **Z3 API**: Direct integration with Z3's Python API
- **SMT-LIB2**: Dumps problems to SMT-LIB2 format for external solvers

======================
Usage
======================

The QI prover can be used as follows:

.. code-block:: python

    from efmc.engines.qi import QuantifierInstantiationProver
    from efmc.sts import TransitionSystem
    
    # Create transition system
    sts = TransitionSystem(...)
    
    # Create QI prover with MBQI strategy
    prover = QuantifierInstantiationProver(
        sts,
        qi_strategy='mbqi',
        solver='z3api',
        timeout=60
    )
    
    # Solve the verification problem
    result = prover.solve()
    
    if result.is_safe:
        print("System is safe")
        print(f"Invariant: {result.invariant}")
    else:
        print("System is unsafe or unknown")

======================
Configuration Options
======================

- **qi_strategy**: Strategy for quantifier instantiation ('mbqi', 'ematching', 'combined', 'auto')
- **solver**: SMT solver to use ('z3api', 'smtlib2', 'cvc5')
- **timeout**: Timeout in seconds for the verification process
- **verbose**: Enable verbose output for debugging
- **dump_file**: Optional file to dump SMT-LIB2 representation


======================
Future Improvements
======================

- Integration with more SMT solvers
- Advanced quantifier instantiation strategies
- Integration with counterexample-guided approaches

======================
References
======================

.. [1] L. de Moura and N. Bjørner, "Z3: An efficient SMT solver," in TACAS, 2008.

.. [2] A. Reynolds and C. Tinelli, "Model-based quantifier instantiation for induction," in CADE, 2015.

.. [3] L. de Moura and N. Bjørner, "Efficient E-matching for SMT solvers," in CADE, 2007. 