# bwind

A bit-width independence solver, which is a prototype implementation of the translation presented in:

- Towards Satisfiability Modulo Parametric Bit-vectors, CADE 19.
https://theory.stanford.edu/~barrett/pubs/NPR+21c.pdf


## Usage

~~~~
./bwind.sh <path-to-solver> <path-to-benchmark>
~~~~

where <path-to-solver> points to an SMT solver that supports quantifiers, non-linear arithmetic, and uninterpreted fucntions.

## Example

~~~~
$ cat tests/test-pbv65.smt2
(set-logic ALL)
(set-option :produce-models true)
(set-option :incremental true)
(declare-const k Int)
(declare-const s (_ BitVec k))
(declare-const t (_ BitVec k))
(assert (distinct (bvsub (_ bv0 k) (bvshl (bvsub s t) (_ bv1 k))) (bvshl (bvsub t s) (_ bv1 k))))
(check-sat)
(exit)

$ ./bwind.sh ~/bin/cvc5 tests/test-pbv65.smt2
unsat
~~~~

## Limitations and warnings:

~~~~
- This is a prototype, general support for operators is limited
- Assumes a single bit-width, which must be named k.
- In particular: no extract and concat.
~~~~