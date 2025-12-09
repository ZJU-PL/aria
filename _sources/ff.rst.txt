SMT Solving for Finite Field
============================

.. _ff-solving:

================
Overview
================

Finite field arithmetic is fundamental to many cryptographic protocols.


Theory Reference: Finite Fields
===============================

.. note::
  Currently, only finite fields of prime order p are supported.
   Such a field is isomorphic to the integers modulo p.

Semantics
^^^^^^^^^

We interpret field sorts as prime fields and field terms as integers modulo :math:`p`. In the following, let:

* ``N`` be an integer numeral and :math:`N` be its integer
* ``p`` be a prime numeral and :math:`p` be its prime
* ``F`` be an SMT field sort (of order :math:`p`)
* ``x`` and ``y`` be SMT field terms (of the same sort ``F``) with interpretations :math:`x` and :math:`y`

+-----------------------+--------------------------------------------+----------------------------------------------+
| SMT construct         | Semantics                                  | Notes                                        |
+=======================+============================================+==============================================+
| ``(_ FiniteField p)`` | the field of order :math:`p`               | represented as the integers modulo :math:`p` |
+-----------------------+--------------------------------------------+----------------------------------------------+
| ``(as ffN F)``        | the integer :math:`(N \bmod p)`            |                                              |
+-----------------------+--------------------------------------------+----------------------------------------------+
| ``(ff.add x y)``      | the integer :math:`((x + y) \bmod p)`      | NB: ``ff.add`` is an n-ary operator          |
+-----------------------+--------------------------------------------+----------------------------------------------+
| ``(ff.mul x y)``      | the integer :math:`((x \times y) \bmod p)` | NB: ``ff.mul`` is an n-ary operator          |
+-----------------------+--------------------------------------------+----------------------------------------------+
| ``(= x y)``           | the Boolean :math:`x = y`                  |                                              |
+-----------------------+--------------------------------------------+----------------------------------------------+


Syntax
^^^^^^

+----------------------+----------------------------------------------+
| SMT construct        | SMT-LIB syntax                                |
+======================+==============================================+
| Logic String         | use `FF` for finite fields                    |
|                      |                                              |
|                      | ``(set-logic QF_FF)``                        |
+----------------------+----------------------------------------------+
| Sort                 | ``(_ FiniteField <Prime Order>)``            |
+----------------------+----------------------------------------------+
| Constants            | ``(declare-const X (_ FiniteField 7))``      |
+----------------------+----------------------------------------------+
| Finite Field Value   | ``(as ff3 (_ FiniteField 7))``               |
+----------------------+----------------------------------------------+
| Addition             | ``(ff.add x y)``                             |
+----------------------+----------------------------------------------+
| Multiplication       | ``(ff.mul x y)``                             |
+----------------------+----------------------------------------------+
| Equality             | ``(= x y)``                                  |
+----------------------+----------------------------------------------+


Examples
^^^^^^^^

.. code:: smtlib

  (set-logic QF_FF)
  (set-info :status unsat)
  (define-sort F () (_ FiniteField 3))
  (declare-const x F)
  (assert (= (ff.mul x x) (as ff-1 F)))
  (check-sat)
  ; unsat

.. code:: smtlib

  (set-logic QF_FF)
  (set-info :status sat)
  (define-sort F () (_ FiniteField 3))
  (declare-const x F)
  (assert (= (ff.mul x x) (as ff0 F)))
  (check-sat)
  ; sat: (= x (as ff0 F)) is the only model


Experimental Extensions
^^^^^^^^^^^^^^^^^^^^^^^

Experimental features (may be removed in the future):

* ``ff.bitsum``: n-ary operator for bitsums: ``(ff.bitsum x0 x1 x2)`` = :math:`x_0 + 2x_1 + 4x_2`
* ``ff.neg``: unary negation

================
Implementation
================

Aria provides SMT solving over finite fields in the ``aria/smt/ff`` module with two encoding strategies:

* **Bit-vector encoding** (``ff_bv_solver.py``): Encodes field elements as bit-vectors (width log₂(field_size)) with modular constraints
* **Integer encoding** (``ff_int_solver.py``): Encodes field elements as integers in [0, field_size-1] with explicit modulo operations

Core components include formula parsing (``ff_parser.py``) and translation to bit-vector or integer arithmetic.

================
Usage Example
================

.. code-block:: python

    from aria.smt.ff.ff_bv_solver import solve_qfff

    smt_input = """
    (set-logic QF_FF)
    (declare-fun x () (_ FiniteField 17))
    (declare-fun y () (_ FiniteField 17))
    (assert (= (ff.add x y) #f3m17))
    (assert (= (ff.mul x y) #f5m17))
    (check-sat)
    """
    solve_qfff(smt_input)  # Solves x + y = 3 and x * y = 5 in GF(17)

References
==========

* Hader, T., Kaufmann, D., Irfan, A., Graham-Lengrand, S., & Kovács, L. (2024).
    `MCSat-based Finite Field Reasoning in the Yices2 SMT Solver`_.
    ArXiv:2402.17927.

* Ozdemir, A., Pailoor, S., Bassa, A., Ferles, K., Barrett, C., & Dillig, I. (2024).
    Split Gröbner Bases for Satisfiability Modulo Finite Fields.
    In *Computer Aided Verification (CAV)*.

* Ozdemir, A., Kremer, G., Tinelli, C., & Barrett, C. (2023).
    Satisfiability modulo finite fields.
    In *Computer Aided Verification (CAV)*.

* Hader, T., Kaufmann, D., & Kovács, L. (2023). SMT solving over finite field arithmetic. In *LPAR*.

* Hader, T. (2022). Non-linear SMT-reasoning over finite fields. Master's thesis, TU Wien.

* Hader, T., & Kovács, L. (2022). Non-linear SMT-reasoning over finite fields. In *SMT Workshop*.

.. _MCSat-based Finite Field Reasoning in the Yices2 SMT Solver: https://arxiv.org/pdf/2402.17927
