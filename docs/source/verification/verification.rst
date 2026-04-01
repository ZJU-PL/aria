Program Verification
=====================

Program verification is the process of checking whether a program satisfies a given specification. The specification can be a set of properties that the program must satisfy, such as correctness, safety, or liveness. Verification can be done using various techniques, such as model checking, static analysis, and theorem proving.

Available Verification Engines
-------------------------------

EFMC provides multiple verification engines, each suited for different types of programs and properties:

**Main Engines:**
- **Template-based (EF)**: Constraint-based invariant generation using predefined templates
- **PDR**: Property-Directed Reachability using Z3's Spacer engine
- **K-Induction**: K-induction based verification with optional auxiliary invariants
- **LLM4Inv**: LLM-guided invariant synthesis using a CEGIS loop

**Additional Engines:**
- **Houdini**: Iterative weakening for conjunctive invariant inference
- **Abduction**: Abductive inference for invariant generation
- **Quantifier Instantiation (QI)**: Direct solving via quantifier instantiation
- **Quantifier Elimination (QE)**: Quantifier elimination based verification
- **Predicate Abstraction**: Finite-state abstraction using predicates
- **Symbolic Abstraction**: Symbolic abstraction-based verification
- **BDD**: Binary Decision Diagram based verification
- **K-Safety**: Relational verification for hyperproperties (programmatic API)

**Specialized Engines:**
- **PolyHorn**: Polynomial invariant generation for Horn clauses
- **Abstract Interpretation**: Abstract interpretation based analysis


Correctness
-----------

Consider the following program that computes the factorial of a number
.. code-block:: python

    def factorial(n):
        if n == 0:
            return 1
        else:
            return n * factorial(n-1)

The program computes the factorial of a non-negative integer `n`. The correctness of this program can be verified by proving that it satisfies the specification: for all non-negative integers `n`, the program returns `n!`, where `n!` is the factorial of `n`.

To verify the correctness of the program, we can use mathematical induction. The proof consists of two parts:
1. Base case: Show that the program is correct for the base case `n = 0`.
   
   When `n = 0`, the program returns `1`, which is `0!`. Therefore, the base case holds.

2. Inductive step: Assume that the program is correct for `n = k`, i.e., assume that `factorial(k)` returns `k!`. Show that the program is correct for `n = k + 1`.

   When `n = k + 1`, the program returns `(k + 1) * factorial(k)`. By the inductive hypothesis, `factorial(k)` returns `k!`. Therefore, the program returns `(k + 1) * k!`, which is `(k + 1)!`. Hence, the inductive step holds.

Since both the base case and the inductive step have been proven, by mathematical induction, the program is correct for all non-negative integers `n`.