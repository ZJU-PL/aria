First-Order Logic (FOL)
=======================

The ``aria.smt.fol`` module provides **Miniprover**, an automated theorem prover for first-order logic. This is a pedagogical implementation that is guaranteed to find proofs for all provable formulae (though it may loop forever for some unprovable ones, per Gödel's Entscheidungsproblem).

.. contents:: Table of Contents
   :local:
   :depth: 2

Directory Structure
-------------------

.. code-block:: text

   smt/fol/
   └── miniprover/
       ├── __init__.py          (1 line - docstring)
       ├── language.py          (517 lines)
       ├── prover.py            (524 lines)
       ├── main.py              (525 lines)
       └── README.md            (94 lines)
       └── __pycache__/         (compiled Python files)

Module Purpose
--------------

Miniprover is an educational first-order logic theorem prover using sequent calculus. It serves as:

1. **Pedagogical tool** for teaching proof search strategies
2. **Reference implementation** for building efficient FOL provers
3. **Research foundation** for experimenting with inference rules

**Note**: This is intentionally slow for practical use - it's designed for correctness and educational value, not performance.

Key Components
--------------

Language Definitions (``language.py`` - 517 lines)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines the complete FOL language with support for unification and substitution.

**Terms:**

* **Variable**: FOL variables with name and instantiation time tracking
  - Tracks when variables were created for temporal reasoning
* **Function**: Represents function applications with recursive term structure
  - Example: ``f(g(x), h(f(a), b))``
* **UnificationTerm**: Metavariables used during unification
  - Supports occurs-check to prevent infinite unification

**Formulae:**

* **Predicate**: Atomic formulas with name and term arguments
  - Example: ``P(x), R(f(x), y)``
* **Not**: Logical negation (``¬φ``)
* **And**/``Or``: Logical conjunction and disjunction
* **Implies**: Logical implication (``φ → ψ``)
* **ForAll**/``ThereExists``: Universal and existential quantification

**Common Methods (across all classes):**

* ``freeVariables()``: Extract all free variables
* ``freeUnificationTerms()``: Extract all unification metavariables
* ``replace(old, new)``: Substitute terms/variables
* ``occurs(term)``: Occurs-check for unification safety
* ``setInstantiationTime(time)``: Track when terms were created

Theorem Prover (``prover.py`` - 524 lines)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implements complete sequent calculus with unification-based proof search.

**Unification Engine:**

* ``unify(term_a, term_b)``: Solves single unification equations
  - Uses occurs-check to prevent infinite loops
  - Returns unification substitution or failure
* ``unify_list(pairs)``: Solves multiple unification equations simultaneously
  - More efficient than pairwise solving

**Sequent Calculus:**

* **Sequent class**: Represents sequents (Γ ⊢ Δ)
  - **Left side (Γ)**: Set of formulae assumed true
  - **Right side (Δ)**: Set of formulae to be proven
  - **Sibling tracking**: Manages parallel proof branches
  - **Depth information**: Tracks proof depth for ordering

**Proof Search:**

* ``proveSequent(sequent)``: Main proof search algorithm. It expands sequents
  breadth-first, applying one rule per sequent before re-queueing it; the
  earliest-created formula (smallest instantiation counter) is selected first;
  and it manages sibling sequents for unification across branches.

**Key Inference Rules Implemented:**

* **Left rules** (for Γ):
  - Not-left: If ¬φ ∈ Γ, remove ¬φ from Γ and add φ to Δ
  - And-left: If φ∧ψ ∈ Γ, replace it by φ and ψ in Γ (single sequent)
  - Or-left: If φ∨ψ ∈ Γ, split into two branches (Γ,φ ⊢ Δ) and (Γ,ψ ⊢ Δ)
  - Implies-left: If φ→ψ ∈ Γ, split into (Γ ⊢ Δ,φ) and (Γ,ψ ⊢ Δ)
  - ForAll-left: If ∀x.φ ∈ Γ, instantiate with a fresh unification term
  - ThereExists-left: If ∃x.φ ∈ Γ, instantiate with a fresh variable

* **Right rules** (for Δ):
  - Not-right: If ¬φ ∈ Δ, remove ¬φ from Δ and add φ to Γ
  - And-right: If φ∧ψ ∈ Δ, split into two branches (Γ ⊢ Δ,φ) and (Γ ⊢ Δ,ψ)
  - Or-right: If φ∨ψ ∈ Δ, add both φ and ψ to Δ (single sequent)
  - Implies-right: If φ→ψ ∈ Δ, move φ to Γ and add ψ to Δ
  - ThereExists-right: If ∃x.φ ∈ Δ, instantiate with a fresh unification term
  - ForAll-right: If ∀x.φ ∈ Δ, instantiate with a fresh variable

**Proof Strategy:**

* **Formula selection**: Expands formulas in order of creation time; the
  earliest-created formula (smallest instantiation counter) is expanded first.
* **Universal quantification**: Skolem-style unification terms
  - Introduces unification metavariables for instantiations
* **Existential quantification**: Fresh variable instantiation
  - Ensures no variable capture during substitution

Command-Line Interface (``main.py`` - 525 lines)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interactive theorem prover REPL with formula management.

**Lexer/Parser:**

* ``lex(inp)``: Tokenizes input strings into tokens
  - Handles identifiers, operators, quantifiers, parentheses
  - Supports whitespace and comments
* ``parse(tokens)``: Recursive descent parser
  - **Operator precedence**: quantifiers > implication > disjunction > conjunction > not > atoms
  - **Type checking**: Validates well-formedness of formulae
  - **Error messages**: Clear feedback on parse errors

**Syntax Support:**

* **Variables**: ``x``, ``y``, ``z`` (lowercase identifiers)
* **Functions**: ``f(term, ...)``
* **Predicates**: ``P(term)`` (uppercase identifiers)
* **Logical operators**: ``not``, ``and``, ``or``, ``implies``
* **Quantifiers**: ``forall x. P``, ``exists x. P``

**Interactive Session:**

* Formula input and proof verification
* **Axiom management**: add, list, remove axioms
* **Lemma management**: prove lemmas, add them, list, remove
  - Tracks dependencies between lemmas
* **Reset functionality**: Clear session state

**Session Commands:**

.. list-table::
   :header-rows: 1
   :widths: 25 60

   * - Command
     - Description
   * - ``[formula]``
     - Input formula and check provability
   * - ``axiom <formula>``
     - Add an axiom to the current set
   * - ``axioms``
     - List all axioms
   * - ``lemma <formula>``
     - Prove a lemma and add it to the knowledge base
   * - ``lemmas``
     - List all lemmas
   * - ``remove <formula>``
     - Remove a matching axiom or lemma (dependent lemmas are removed too)
   * - ``reset``
     - Clear all axioms and lemmas
   * - ``exit``
     - Exit the prover

The command set is also printed at startup. Bare ``help`` is not a command.

Documentation (``README.md`` - 94 lines)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive guide including:

* Purpose and limitations
  - **Pedagogical tool**: Intentionally slow
  - **Completeness**: Guaranteed to find proofs for provable formulae
  - **Gödel limitation**: May loop forever on unprovable formulae
  - **Logic**: Classical sequent calculus with multi-conclusion right rules
* **Syntax reference**: Complete grammar with examples
* **Interactive session example**: Demonstrates proof steps with sequent numbering
* **Axiom and lemma usage**: Examples of building knowledge bases

Algorithms Implemented
----------------------

1. **Unification Algorithm**: First-order unification with occurs-check
   - Time-based instantiation constraints
   - Handles substitution composition

2. **Sequent Calculus**: Complete set of inference rules
   - Left rules (for assumptions in Γ)
   - Right rules (for goals in Δ)
   - Proper handling of quantifiers

3. **Proof Search**: Breadth-first exploration
   - Parallel branch management
   - Unification for closing branches
   - Skolemization for universal quantification
   - Fresh variable generation for existential quantification

4. **Formula Ordering**: Earliest-created formula first within a sequent; sequents are processed breadth-first

Usage Examples
--------------

Python API
~~~~~~~~~~

.. code-block:: python

   from aria.smt.fol.miniprover.language import *
   from aria.smt.fol.miniprover.prover import proveFormula

   # Define axioms and prove formulae
   axioms = {
       ForAll(Variable('x'), Predicate('P', [Variable('x')])),
       ForAll(Variable('y'), Predicate('Q', [Variable('y')]))
   }

   formula = Implies(
       And(Predicate('P', [Variable('a')]),
           Predicate('Q', [Variable('a')])),
       Predicate('P', [Variable('a')]),
   )

   result = proveFormula(axioms, formula)
   # Returns True when the formula is provable; False (or loops) otherwise

Interactive CLI
~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m aria.smt.fol.miniprover.main

   # Interactive session (lines after the prompt are the prover's output):
   > axiom P(a)
   Axiom added: P(a).

   > axiom P(a) implies Q(a)
   Axiom added: (P(a) → Q(a)).

   > Q(a)
   0. (P(a) → Q(a)), P(a) ⊢ Q(a)
   1. P(a) ⊢ Q(a), P(a)
   1. P(a), Q(a) ⊢ Q(a)
   Formula proven: Q(a).

   > axioms
   P(a)
   (P(a) → Q(a))

   > exit
   Goodbye!

Key Features
------------

1. **Complete**: Guaranteed to find proofs for all provable formulae
2. **Educational**: Shows detailed proof steps with sequent numbering
3. **Unification-based**: Handles equality and substitution automatically
4. **Axiom System**: Supports custom axioms and lemma management
   - Dependency tracking for lemmas
5. **Interactive REPL**: Rich command-line interface for experimentation

**Strengths:**

* Clear separation between language definitions, prover logic, and CLI
* Well-documented with comprehensive README
* Proper error handling and user feedback
* Comprehensive proof search with parallel branch handling

**Limitations (as stated in documentation):**

* **Performance**: Too slow for practical use (pedagogical tool only)
* **Termination**: May loop forever on unprovable formulae (Entscheidungsproblem)
* **Logic**: Implements a classical sequent calculus (multi-conclusion right rules)
* **No practical applications**: Intended for teaching and research, not production use

Size Statistics
---------------

* Total Python lines: ~1,567
* language.py: 517 lines
* prover.py: 524 lines
* main.py: 525 lines
* README.md: 94 lines

Testing
-------

While no explicit test files were found in the analysis, the module includes:

* Manual testing through interactive sessions
* Formula validation via type checking
* Example proofs demonstrating various inference patterns

Use Cases
---------

1. **Teaching first-order logic proof strategies**
2. **Understanding sequent calculus mechanics**
3. **Experimenting with axiom systems and lemma development**
4. **Reference implementation for building efficient provers**
5. **Research in automated deduction systems**

Notable Design Decisions
------------------------

* **Classical sequent calculus**: The multi-conclusion right side proves classical tautologies such as ``P or not P``
* **Breadth-first search**: Ensures completeness (won't miss shorter proofs)
* **Creation time ordering**: Reproduces typical textbook proofs
* **Unification terms**: Implements Skolemization naturally
