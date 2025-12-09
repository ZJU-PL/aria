# coding: utf-8
import z3

# Some example about bv solvers.

t1 = z3.AndThen(z3.With('simplify', blast_distinct=True, elim_and=False, flat=False, hoist_mul=True, local_ctx=False,
                        pull_cheap_ite=True, push_ite_bv=False, som=False),
                z3.Tactic('elim-uncnstr'),
                z3.Tactic('solve-eqs'),
                z3.Tactic('max-bv-sharing'),
                z3.Tactic('bit-blast'),
                z3.With('propagate-values', push_ite_bv=False),
                z3.Tactic('smt'))

t2 = z3.AndThen(z3.With('simplify', blast_distinct=False, elim_and=True, flat=False, hoist_mul=True, local_ctx=False,
                        pull_cheap_ite=True, push_ite_bv=False, som=False),
                z3.Tactic('elim-uncnstr'),
                z3.Tactic('purify-arith'),
                z3.Tactic('smt'))

t3 = z3.AndThen(z3.With('simplify', blast_distinct=True, elim_and=False, flat=True, hoist_mul=True, local_ctx=False,
                        pull_cheap_ite=True, push_ite_bv=False, som=False),
                z3.Tactic('max-bv-sharing'),
                z3.Tactic('bit-blast'),
                z3.Tactic('sat'))

x, y = z3.BitVecs("x y", 8)
fml = z3.And(x + y == 6, x - y == 2)
sol = t1.solver()
sol.add(fml)
