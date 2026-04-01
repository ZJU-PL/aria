Model Counting
=================================


==========
Introduction to Model Counting
==========

Model counting is the problem of determining the number of possible solutions
(models) to a given formula. It is a fundamental problem in computer
science and has applications in various fields such as artificial intelligence, cryptography, and verification.
The complexity of model counting varies depending on the logic and constraints involved.
For Boolean satisfiability (SAT), model counting is #P-complete.


==========
Model Counting in Aria
==========

Model Counting for SAT Formulas
-----


To count models for SAT formulas, please use ``sharpSAT`` or other third-party tools.
SharpSAT is an efficient model counter for Boolean formulas in CNF format.

Example usage: (to implement)

.. code-block:: python

    # FIXME: to make it more conveinent for the users
    # we should allow for counting the models of a z3 exper or pysmt expr
    from aria.bool.counting import count_models  # TBD
    from aria.logic import # TBD
    from pysat import ...
    import z3
    import pysmt

    # Create a CNF formula
    cnf = CNF()
    cnf.add_clause([1, 2])  # x1 OR x2
    cnf.add_clause([-1, 3]) # NOT x1 OR x3

    # Count models
    count = count_models(cnf)
    print(f"Number of solutions: {count}")

Model Counting for QF_BV Formulas
----------------------------------

QF_BV stands for the quantifier-free bit-vector logic. It is a subset of the SMT-LIB standard and is commonly used in the analysis and verification
of computer hardware and software systems.

To count the models of a QF_BV formula, refer to
- ``aria\bv\qfbv_counting.py``.
- ``aria\tests\test_bv_counting.py``


Example usage: (to implement)

.. code-block:: python

    from aria.bv import
    from aria.bv.qfbv_counting import count_bv_models
    # Create bit-vector variables

    # Create a formula: x + y < 10

    # Count models
    count = count_bv_models(formula)
    print(f"Number of solutions: {count}")

Note that we rely on sharpSAT for the implementation. Currently, you need to either copy a
binary version of sharpSAT to ``bin_solvers`` or install a sharpSAT globally.


==========
Advanced Features
==========

Projected Model Counting
------

Projected model counting involves counting models while considering only a subset of variables.
This is useful when you're only interested in specific variables' solutions.

Approximate Model Counting
-------

For large formulas where exact counting is impractical, approximate model counting can be used.


============
References
============
