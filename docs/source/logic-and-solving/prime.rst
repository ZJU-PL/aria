Prime Implicant / Implicate Enumeration
========================================

``aria.bool.prime`` provides algorithms for enumerating prime implicants and
prime implicants of Boolean formulas.

Directory structure
-------------------

::

   aria/bool/prime/
   └── enumeration.py      # Prime implicant / implicate enumeration algorithms

Overview
--------

A **prime implicant** of a Boolean function ``f`` is a minimal partial
assignment that implies ``f``; a **prime implicate** is a minimal clause
implied by ``f``. Enumerating these structures is fundamental to Boolean
analysis, circuit minimisation, and diagnostic reasoning.

The ``enumeration`` module implements enumeration algorithms that, given a
Boolean formula or CNF, produce the complete (or bounded) set of prime
implicants or implicates.

Programmatic usage
------------------

.. code-block:: python

   from aria.bool.prime.enumeration import enumerate_prime_implicants

   primes = enumerate_prime_implicants(cnf_clauses, variables)
