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

Symbolic finite automata framework supporting predicates over infinite alphabets. This module is adapted from the `symautomata <https://github.com/spencerwuwu/symautomata>`_ project.

Automata Learning (``aria/automata/fa_learning.py``)
------------------------------------------------------

Active learning algorithms for inferring automata from examples:

.. code-block:: python

   from aria.automata.fa_learning import learn_automaton

   # Learn DFA from membership queries
   automaton = learn_automaton(examples, membership_oracle)

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
