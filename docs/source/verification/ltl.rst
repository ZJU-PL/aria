Linear Temporal Logic
=====================

Linear Temporal Logic (LTL) is a formalism for expressing properties of
reactive systems. It is a modal logic: it uses temporal operators to express
relationships between propositions across a single execution trace. LTL is
used in formal verification of software and hardware systems and is supported
by model checkers such as SPIN and NuSMV.

LTL is a linear-time temporal logic, which means that it describes the
behavior of a system as a sequence of states, and the temporal operators in
LTL operate on this sequence. The basic temporal operators in LTL are:

- **X** (Next): ``X p`` holds in a state ``s`` if ``p`` holds in the next state
  ``s'``.
- **G** (Globally): ``G p`` holds in a state ``s`` if ``p`` holds in ``s`` and
  in all subsequent states.
- **F** (Finally): ``F p`` holds in a state ``s`` if ``p`` holds in ``s`` or in
  some later state.
- **U** (Until): ``p U q`` holds in a state ``s`` if there is a state ``s'``
  at or after ``s`` where ``q`` holds, and ``p`` holds in every state from
  ``s`` up to (but not including) ``s'``.

LTL formulas are built using propositional connectives (and, or, not,
implies) and the temporal operators. For example, ``G (p -> F q)`` expresses
that whenever ``p`` becomes true, ``q`` becomes true eventually.

Status in aria
--------------

aria does **not** ship a general LTL model checker. There is no standalone
``aria.logic.ltl`` module and no LTL-to-automata translation.

The only temporal reasoning code in the repository is the *bounded*
HyperLTL-like encoding in ``aria.efmc.engines.ksafety.hyperltl``, which
supports ``X``, ``G``, ``F``, and ``U`` over a finite unrolling and is used by
``HyperLTLProver`` for k-safety/hyperproperty checks:

.. code-block:: python

   from aria.efmc.engines.ksafety import HyperLTLProver
   from aria.efmc.engines.ksafety.hyperltl import Var, Atom, G, Implies
   from aria.efmc.sts import TransitionSystem

   sts = TransitionSystem()
   # ... initialize sts ...

   phi = Implies(
       G(Atom('==', Var('high', 0), Var('high', 1))),
       G(Atom('==', Var('low', 0), Var('low', 1))),
   )

   prover = HyperLTLProver(sts, k=2, method='bounded_model_checking', max_bound=10)
   prover.set_formula(phi)
   result = prover.solve()

Note the limitations of that encoding:

- It is bounded: ``X`` at the last step is clamped to the final step, and
  ``G``/``F``/``U`` range only over steps up to the unrolling bound.
- It only supports ``bounded_model_checking`` (see :doc:`ksafety`).
- It is a hyperproperty encoding over ``k`` traces, not a single-trace LTL
  model checker.
