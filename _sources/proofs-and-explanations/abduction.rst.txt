Abductive Reasoning
===================

Abduction asks for hypotheses that, together with background assumptions, explain
an observation. In automated reasoning and verification, abductive workflows are
used for explanation generation, invariant discovery, diagnosis, and related
belief-change tasks.

Abduction in ARIA
-----------------

``aria.abduction`` contains abductive reasoning helpers together with
belief-revision operations over finite bases of Z3 formulas.

Current public entrypoints include:

* ``abduce``
* ``check_abduct``
* ``revise_belief_base``
* ``contract_belief_base``
* ``expand_belief_base``

Example imports
---------------

.. code-block:: python

   from aria.abduction import abduce, revise_belief_base

For concrete examples of belief revision and optimal revision/contraction
enumeration, see ``aria/abduction/README.md``.

References
----------

* Kakas, A. C., Kowalski, R. A., and Toni, F. (1992). Abductive logic programming.
* Eiter, T. and Gottlob, G. (1995). The complexity of logic-based abduction.
* Poole, D. (1993). Probabilistic Horn abduction and Bayesian networks.
