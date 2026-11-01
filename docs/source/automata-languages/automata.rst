Automata
========

Introduction
============

The automata module (``aria/automata``) provides implementations of finite automata
and related algorithms for formal language processing and constraint solving.
It includes concrete finite automata (``PythonDFA``), symbolic finite automata (``SFA``),
and vendored automata-learning toolkits.

Key Features
------------

* **Finite Automata**: DFA manipulation, minimisation, and conversion
* **Symbolic Automata**: Symbolic finite automata with predicate-guarded transitions
* **Automata Learning**: Active learning algorithms via vendored AALpy
* **CFL Reachability**: Context-free language reachability under ``aria.automata.cfl``
* **CFL-OBDD**: Bit-vector decision diagrams under ``aria.automata.cflobdd``

Components
==========

Finite Automata (``aria.automata.symautomata.pythondfa``)
---------------------------------------------------------

Basic concrete finite automata operations:

.. code-block:: python

   from aria.automata.symautomata.pythondfa import PythonDFA

   dfa = PythonDFA(["a", "b"])
   dfa.add_arc(0, 1, "a")
   dfa[1].final = True
   dfa[0].initial = True

Symbolic Automata (``aria.automata.symautomata``)
-------------------------------------------------

Symbolic finite automata framework supporting predicate-guarded transitions:

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

Vendored AALpy (``aria.automata.aalpy``)
----------------------------------------

ARIA carries a vendored copy of the upstream `AALpy
<https://github.com/DES-Lab/AALpy>`_ package as
``aria.automata.aalpy``.

Use this namespace when you want active/passive automata learning:

.. code-block:: python

   from aria.automata.aalpy.learning_algs import run_Lstar
   from aria.automata.aalpy.oracles import RandomWalkEqOracle
   from aria.automata.aalpy.SULs import AutomatonSUL

See :doc:`aalpy` for the vendored package overview and API reference.

Context-Free Language Reachability (``aria.automata.cfl``)
----------------------------------------------------------

See :doc:`cfl` for context-free language reachability algorithms and graph analyses.

CFL-OBDD (``aria.automata.cflobdd``)
------------------------------------

See :doc:`cflobdd` for ordered bitvector decision diagrams.

Applications
============

* String constraint solving in SMT
* Regular expression analysis
* Program verification with string operations
* Vulnerability detection in web applications

References
==========

- Hopcroft, J. E., & Ullman, J. D. (1979). *Introduction to Automata Theory, Languages, and Computation*
- D'Antoni, L., & Veanes, M. (2014). *Minimization of Symbolic Automata*. POPL 2014
- Angluin, D. (1987). *Learning Regular Sets from Queries and Counterexamples*
