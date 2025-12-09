# Ideas

## Meta-Portfolio: Treating Solvers as Tactics

### Concept

To add third-party solvers to our toolchain, we can regard them as a kind of "tactic".
For example, treat "cvc5" as a tactic (similar to Z3's "smt" tactic).
This enables chaining: apply Z3 tactics for preprocessing, then invoke an external solver.

**Note:** cvc5 also has a lightweight tactic system, using portfolios of strategies in its SMT-COMP scripts.

### Example Workflows

**Eager solver approach:**
```
(apply (simplify, solve-eqs, ackermann, bit-blast, minisat))
```

**Lazy solver approach with timeout and fallback:**
```
(apply (simplify, solve-eqs, try-for (timeout, qf-lra), cvc5))
```

**Multi-solver meta-portfolio:**
```
(apply (simplify, normalize, or-else (z3-smt, cvc5, yices2)))
```

### Benefits

- Leverage Z3's powerful preprocessing tactics before external solvers
- Create flexible solver portfolios mixing different backends
- Enable timeout-based strategy switching across solver boundaries
