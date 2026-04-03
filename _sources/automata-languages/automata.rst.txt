Automata
===========================

Introduction
=====================

The automata module (``aria/automata``) provides implementations of finite automata and related algorithms for formal language processing and constraint solving. It includes both classical finite automata (DFA/NFA) and symbolic automata for string constraint solving.

Key Features
-------------

* **Finite Automata**: DFA and NFA manipulation and conversion
* **Symbolic Automata**: Symbolic finite automata with theory-based transitions
* **Automata Learning**: Active learning algorithms for automata inference
* **Vendored AALpy**: Upstream automata-learning toolkit under ARIA namespace
* **String Constraint Solving**: Integration with SMT solvers for string theories

Components
=====================

Finite Automata (``aria/automata/fa.py``)
--------------------------------------------

Basic finite automata operations:

.. code-block:: python

   from aria.automata.fa import DFA, NFA

   # Create a DFA
   dfa = DFA(
       states={'q0', 'q1', 'q2'},
       alphabet={'a', 'b'},
       transitions={('q0', 'a'): 'q1', ('q1', 'b'): 'q2'},
       initial_state='q0',
       accepting_states={'q2'}
   )

   # Accept strings
   result = dfa.accepts("ab")  # True

Symbolic Automata (``aria/automata/symautomata``)
---------------------------------------------------

Symbolic finite automata framework supporting predicate-guarded transitions. This module is adapted from the `symautomata <https://github.com/spencerwuwu/symautomata>`_ project.

The current symbolic API is centered on ``aria.automata.symautomata.SFA`` and the
guard types ``Predicate``, ``SetPredicate``, and ``Z3Predicate``.

``SetPredicate`` is the finite-alphabet implementation. ``Z3Predicate`` is the
solver-backed implementation for symbolic domains.

``SFA`` also bridges to the older concrete DFA workflows:

.. code-block:: python

   from aria.automata.symautomata import SFA
   from aria.automata.symautomata.dfa import DFA

   dfa = DFA(["a", "b"])
   dfa.add_arc(0, 1, "a")
   dfa[1].final = True

   sfa = SFA.from_acceptor(dfa)
   regex = sfa.to_regex()

Important semantic limits:

* ``SFA.concretize()``, ``SFA.save()``, and ``SFA.load()`` require a finite alphabet.
* ``SFA.complete()`` and ``SFA.complement()`` require either a finite alphabet or a
  ``predicate_factory`` that can produce a predicate over the intended symbolic universe.
* Finite save/load workflows serialize concrete symbols, not symbolic formulas.

Automata Learning (``aria/automata/fa_learning.py``)
------------------------------------------------------

Active learning algorithms for inferring automata from examples:

.. code-block:: python

   from aria.automata.fa_learning import learn_automaton

   # Learn DFA from membership queries
   automaton = learn_automaton(examples, membership_oracle)

Vendored AALpy (``aria/automata/aalpy``)
-----------------------------------------

ARIA also carries a vendored copy of the upstream `AALpy
<https://github.com/DES-Lab/AALpy>`_ package as
``aria.automata.aalpy``.

Use this namespace when you want the broader active/passive learning toolkit,
including reusable oracles, SUL wrappers, stochastic learners, and visibly
pushdown automata support.

.. code-block:: python

   from aria.automata.aalpy.learning_algs import run_Lstar
   from aria.automata.aalpy.oracles import RandomWalkEqOracle
   from aria.automata.aalpy.SULs import AutomatonSUL

See :doc:`aalpy` for the vendored package overview and API reference.

Applications
=====================

* String constraint solving in SMT
* Regular expression analysis
* Program verification with string operations
* Vulnerability detection in web applications

References
=====================

- Hopcroft, J. E., & Ullman, J. D. (1979). *Introduction to Automata Theory, Languages, and Computation*
- D'Antoni, L., & Veanes, M. (2014). *Minimization of Symbolic Automata*. POPL 2014
- Angluin, D. (1987). *Learning Regular Sets from Queries and Counterexamples*
