# Tactic System in Z3

## Probe

- 'num-consts',
- 'num-exprs',
- 'size',
- 'depth',
- 'ackr-bound-probe',
- 'is-qfbv-eq',
- 'arith-max-deg',
- 'arith-avg-deg',
- 'arith-max-bw',
- 'arith-avg-bw',
- 'is-unbounded',

## Tactic

Example

~~~~
# coding: utf-8
from z3 import *

# Some example about bv solvers.

t1 = AndThen(With('simplify', blast_distinct=True, elim_and=False, flat=False, hoist_mul=True, local_ctx=False,
                  pull_cheap_ite=True, push_ite_bv=False, som=False),
             Tactic('elim-uncnstr'),
             Tactic('solve-eqs'),
             Tactic('max-bv-sharing'),
             Tactic('bit-blast'),
             With('propagate-values', push_ite_bv=False),
             Tactic('smt'))

t2 = AndThen(With('simplify', blast_distinct=False, elim_and=True, flat=False, hoist_mul=True, local_ctx=False,
                  pull_cheap_ite=True, push_ite_bv=False, som=False),
             Tactic('elim-uncnstr'),
             Tactic('purify-arith'),
             Tactic('smt'))

t3 = AndThen(With('simplify', blast_distinct=True, elim_and=False, flat=True, hoist_mul=True,
                  local_ctx=False, pull_cheap_ite=True, push_ite_bv=False, som=False),
             Tactic('max-bv-sharing'),
             Tactic('bit-blast'),
             Tactic('sat'))

x, y = BitVecs("x y", 8)
fml = And(x + y == 6, x - y == 2)
sol = t3.solver()
sol.add(fml)
print(sol.check())

"""
AndThen(With('simplify',blast_distinct=False,elim_and=False,flat=False,hoist_mul=True,local_ctx=False,pull_cheap_ite=True,push_ite_bv=True,som=False),With(propagate-values,push_ite_bv=True),Tactic(max-bv-sharing),Tactic(smt))
AndThen(With('simplify',blast_distinct=False,elim_and=True,flat=False,hoist_mul=False,local_ctx=True,pull_cheap_ite=False,push_ite_bv=False,som=False),With(propagate-values,push_ite_bv=False),Tactic(smt))
With('simplify',blast_distinct=True,elim_and=False,flat=True,hoist_mul=False,local_ctx=True,pull_cheap_ite=False,push_ite_bv=False,som=False
AndThen(With('simplify',blast_distinct=True,elim_and=False,flat=False,hoist_mul=True,local_ctx=False,pull_cheap_ite=True,push_ite_bv=True,som=False),With(aig,aig_per_assertion=False),Tactic(purify-arith),Tactic(max-bv-sharing),Tactic(bit-blast),Tactic(sat))
AndThen(With('simplify',blast_distinct=True,elim_and=False,flat=False,hoist_mul=True,local_ctx=False,pull_cheap_ite=True,push_ite_bv=True,som=False),Tactic(bit-blast),With('simplify',blast_distinct=True,elim_and=False,flat=False,hoist_mul=True,local_ctx=True,pull_cheap_ite=False,push_ite_bv=True,som=True),Tactic(sat))
AndThen(With('simplify',blast_distinct=False,elim_and=False,flat=False,hoist_mul=True,local_ctx=False,pull_cheap_ite=True,push_ite_bv=True,som=False),With(propagate-values,push_ite_bv=False),Tactic(smt))
"""

~~~~

#### Parameter for "sat" tactic

- gc: psm, glue, glue_psm, dyn_psm
- phase: always_false, always_true, caching, random
- restart: static, luby, geometric, ema
- branching.heuristic: vsids, chb, lrb
- burst_search: 100

#### Parameter for "smt" tactic

- phase_selection: 0, 1, 2, 3, 4, 5, 6 (default 3)
- restart_strategy: 0, 1, 2, 3, 4 (default 1)
- ..

#### Parameter for "simplify" tactic

Boolean typed

- elim_and
- blast_distinct
- push_ite_bv
- som
- pull_cheap_ite
- hoist_mul
- local_ctx
- flat