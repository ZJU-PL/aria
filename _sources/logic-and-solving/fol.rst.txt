First-Order Logic (FOL)
=========================

The ``aria.fol`` module provides **Miniprover**, an automated theorem prover for first-order logic. This is a pedagogical implementation that is guaranteed to find proofs for all provable formulae (though it may loop forever for some unprovable ones, per Gödel's Entscheidungsproblem).

.. contents:: Table of Contents
   :local:
   :depth: 2

Directory Structure
----------------

```
fol/
└── miniprover/
    ├── __init__.py          (0 lines - empty)
    ├── language.py          (517 lines)
    ├── prover.py            (525 lines)
    ├── main.py              (526 lines)
    └── README.md            (95 lines)
    └── __pycache__/         (compiled Python files)
```

Module Purpose
--------------

Miniprover is an educational first-order logic theorem prover using sequent calculus. It serves as:

1. **Pedagogical tool** for teaching proof search strategies
2. **Reference implementation** for building efficient FOL provers
3. **Research foundation** for experimenting with inference rules

**Note**: This is intentionally slow for practical use - it's designed for correctness and educational value, not performance.

Key Components
----------------

### 1. Language Definitions (``language.py`` - 517 lines)

Defines the complete FOL language with support for unification and substitution.

**Terms:**

* **Variable**: FOL variables with name and instantiation time tracking
  - Tracks when variables were created for temporal reasoning
* **Function**: Represents function applications with recursive term structure
  - Example: ``f(g(x), h(f(a), b)``
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

### 2. Theorem Prover (``prover.py`` - 525 lines)

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

* ``proveSequent(sequent)``: Main proof search algorithm
  - **Breadth-first expansion** of sequents
  - **Formula ordering**: Expands by creation time (depth-first within each sequent)
  - **Inference rules**: Complete set of left and right rules for all connectives
  - **Parallel branch handling**: Manages sibling sequents for unification across branches

**Key Inference Rules Implemented:**

* **Left rules** (for Γ):
  - Not-left: If ¬φ ∈ Γ, add φ to Γ and move ¬φ to Δ
  - And-left: If φ∧ψ ∈ Γ, create two branches (φ) and (ψ)
  - Or-left: If φ∨ψ ∈ Γ, add both to Γ
  - Implies-left: If φ→ψ ∈ Γ, add ψ to Γ, move φ to Δ
  - ForAll-left: If ∀x.φ ∈ Γ, instantiate with fresh term

* **Right rules** (for Δ):
  - Not-right: Move ¬φ to Γ, add φ to Δ
  - And-right: If φ∧ψ ∈ Δ, add both to Δ
  - Or-right: If φ∨ψ ∈ Δ, create two branches
  - Implies-right: If φ→ψ ∈ Δ, move φ to Γ, add ψ to Δ
  - ThereExists-right: If ∃x.φ ∈ Δ, instantiate with fresh term

**Proof Strategy:**

* **Formula selection**: Expands formulas in order of creation time
  - Within sequent: depth-first (most recent first)
* **Universal quantification**: Skolem-style unification terms
  - Introduces unification metavariables for instantiations
* **Existential quantification**: Fresh variable instantiation
  - Ensures no variable capture during substitution

### 3. Command-Line Interface (``main.py`` - 526 lines)

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
   :header: "Command, Description"
   :widths: 15, 60
   
   ``[formula]``, Input formula and check provability
   ``axiom add [formula]``, Add axiom to current set
   ``axiom list``, List all axioms
   ``axiom remove [n]``, Remove nth axiom
   ``lemma prove [formula]``, Prove lemma and add to knowledge base
   ``lemma list``, List all lemmas with dependencies
   ``lemma remove [n]``, Remove nth lemma
   ``reset``, Clear all axioms and lemmas
   ``help``, Show available commands
   ``exit``, Exit the prover

### 4. Documentation (``README.md`` - 95 lines)

Comprehensive guide including:

* Purpose and limitations
  - **Pedagogical tool**: Intentionally slow
  - **Completeness**: Guaranteed to find proofs for provable formulae
  - **Gödel limitation**: May loop forever on unprovable formulae
  - **Logic**: Implements intuitionistic sequent calculus (not classical)
* **Syntax reference**: Complete grammar with examples
* **Interactive session example**: Demonstrates proof steps with sequent numbering
* **Axiom and lemma usage**: Examples of building knowledge bases

Algorithms Implemented
---------------------

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

4. **Formula Ordering**: Depth-first within sequent, breadth-first globally

Usage Examples
--------------

### Python API
~~~~~~~~~~~~~~~

.. code-block:: python

   from aria.fol.miniprover.language import *
   from aria.fol.miniprover.prover import proveFormula

   # Define axioms and prove formulae
   axioms = {
       ForAll(Variable('x'), Predicate('P', [Variable('x')])),
       ForAll(Variable('y'), Predicate('Q', [Variable('y')]))
   }
   
   formula = Implies(
       And(Predicate('P', [Variable('a')]),
       Predicate('Q', [Variable('a')]))
   )
   
   result = proveFormula(axioms, formula)
   # Returns sequent or None if no proof found

### Interactive CLI
~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m aria.fol.miniprover.main
   
   # Interactive session:
   # Input: forall x. P(x)
   # Prover: Added to axioms: [1]
   # Input: P(a)
   # Prover: ✓ Provable
   # 
   # Input: implies (P(a)) (Q(a))
   # Prover: ✓ Provable (1 steps)

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
* **Logic**: Implements intuitionistic sequent calculus (not classical logic)
* **No practical applications**: Intended for teaching and research, not production use

Size Statistics
---------------

* Total Python lines: ~1,568
* language.py: 517 lines
* prover.py: 525 lines
* main.py: 526 lines
* README.md: 95 lines

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
----------------------

* **Intuitionistic over classical**: Matches pedagogical use case
* **Breadth-first search**: Ensures completeness (won't miss shorter proofs)
* **Creation time ordering**: Reproduces typical textbook proofs
* **Unification terms**: Implements Skolemization naturally
