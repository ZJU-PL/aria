AALpy
=====

The ``aria.automata.aalpy`` package vendors the upstream `AALpy
<https://github.com/DES-Lab/AALpy>`_ automata-learning toolkit under the ARIA
namespace.

Overview
--------

``aria.automata.aalpy`` provides active and passive automata-learning
algorithms, reusable equivalence oracles, systems-under-learning wrappers, and
automaton data structures for deterministic, nondeterministic, stochastic, and
visibly pushdown settings.

The vendored package keeps the upstream module layout while rewriting imports to
the ARIA namespace:

* ``aria.automata.aalpy.automata`` for automaton definitions
* ``aria.automata.aalpy.learning_algs`` for learning algorithms
* ``aria.automata.aalpy.oracles`` for equivalence oracles
* ``aria.automata.aalpy.SULs`` for SUL adapters
* ``aria.automata.aalpy.utils`` for file IO, generators, and helpers

Typical Usage
-------------

.. code-block:: python

   from aria.automata.aalpy.SULs import AutomatonSUL
   from aria.automata.aalpy.learning_algs import run_Lstar
   from aria.automata.aalpy.oracles import RandomWalkEqOracle
   from aria.automata.aalpy.utils import generate_random_deterministic_automata

   model = generate_random_deterministic_automata(
       automaton_type='dfa',
       num_states=4,
       input_alphabet_size=2,
       output_alphabet_size=2,
   )
   alphabet = model.get_input_alphabet()
   sul = AutomatonSUL(model)
   eq_oracle = RandomWalkEqOracle(alphabet, sul, num_steps=500)
   learned = run_Lstar(alphabet, sul, eq_oracle, automaton_type='dfa')

Notes
-----

* Visualization helpers rely on ``pydot`` and Graphviz.
* Model-checking helpers in ``aria.automata.aalpy.utils.ModelChecking`` use the
  configurable paths in ``aria.automata.aalpy.paths``.
* Upstream reference material is copied into
  ``aria/automata/aalpy/README.upstream.md`` and
  ``aria/automata/aalpy/LICENSE.upstream.txt``.

API Reference
-------------

.. automodule:: aria.automata.aalpy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aria.automata.aalpy.automata
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aria.automata.aalpy.learning_algs
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aria.automata.aalpy.oracles
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aria.automata.aalpy.SULs
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aria.automata.aalpy.utils
   :members:
   :undoc-members:
   :show-inheritance:
