from aria.efmc.verifytools.boogie.z3_embed import And, stmt_to_z3, env_to_expr, satisfiable, Int,\
    maybeModel, expr_to_z3, AllIntTypeEnv, simplify, model
from aria.efmc.verifytools.boogie.ast import expr_read, AstAssume, AstAssert, replace, AstId,\
        AstNumber
from aria.efmc.verifytools.boogie.paths import get_path_vars, nd_bb_path_to_ssa, sp_nd_ssa_path, \
        extract_ssa_path_vars
from aria.efmc.verifytools.boogie.predicate_transformers import sp_stmts
from aria.efmc.verifytools.boogie.ssa import SSAEnv
from itertools import permutations
from copy import deepcopy

def _to_dict(vs, vals):
    return { vs[i]: vals[i] for i in range(0, len(vs)) }

def evalPred(boogie_expr, env):
    typeEnv = { x : Int for x in env }
    q = And(map(lambda stmt:    stmt_to_z3(stmt, typeEnv),
        [AstAssume(env_to_expr(env)),
         AstAssert(boogie_expr)]))
    res = satisfiable(q)
    return res

# Given an invariant template as a boogie expression where [x,y,z] are
# variables and [a,b,c] constants And a series of environments, find all
# instantiations of the template that holds for all elements of the series.
def instantiateAndEval(inv, vals,
        var_names = None,
        const_names = None):

    if (var_names == None):
        var_names = ["_sv_x", "_sv_y", "_sv_z"]

    if (const_names == None):
        const_names = ["_sc_a", "_sc_b", "_sc_c"]

    res = []
    symVs = [ x for x in expr_read(inv) if x in var_names ]
    symConsts = [ x for x in expr_read(inv) if x in const_names ]

    nonSymVs = set(expr_read(inv)).difference(set(symVs))\
            .difference(set(symConsts))
    traceVs = vals[0].keys()
    prms = permutations(range(len(traceVs)), len(symVs))

    typeEnv = { str(x) + str(i) : Int
                    for x in vals[0].keys()
                        for i in range(len(vals)) }
    typeEnv.update({ str(c) : Int for c in symConsts })

    for prm in prms:
        varM = { symVs[i]: traceVs[prm[i]] for i in range(len(symVs)) }
        varM.update({ nonSymV: nonSymV for nonSymV in nonSymVs })

        inst_inv = replace(inv, { AstId(x) : AstId(varM[x]) for x in symVs })
        p = [ AstAssume(env_to_expr(x, str(i))) for (i,x) in enumerate(vals) ]
        p += [ AstAssert(replace(inst_inv,
                                 { AstId(x) : AstId(x + str(i))
                                     for x in varM.values() }))
               for i in range(len(vals)) ]

        m = maybeModel(And(map(lambda s: stmt_to_z3(s, typeEnv), p)))

        if (m):
            const_vals = { AstId(x) : AstNumber(m[x]) for x in symConsts }
            res.append(replace(inst_inv, const_vals))

    return res

def execute(env, bb, bbs, limit):
    q = [ (expr_to_z3(env_to_expr(env), AllIntTypeEnv()),
           bb ,
           SSAEnv(None, ""),
           [ ],
           [ ]) ]

    def bb_sp(bb, initial_ssa_env, precond):
      nd_path, final_env = nd_bb_path_to_ssa([bb], bbs, initial_ssa_env)
      sp = sp_nd_ssa_path(nd_path, bbs, precond, AllIntTypeEnv())
      return simplify(sp), final_env, nd_path

    while len(q) > 0:
      precond, bb, ssa_env, curp, cur_ssap = q.pop()
      #print("Running ", bb, " with pre: ", precond, "env:", ssa_env.replm())
      postcond, after_env, ssaed_bb = bb_sp(bb, ssa_env, precond)

      if (not satisfiable(postcond)):
        continue

      newp = curp + [ bb ]
      new_ssap = cur_ssap + ssaed_bb

      if (len(bbs[bb].successors) == 0 or len(curp) + 1 >= limit):
        yield postcond, after_env, newp, new_ssap, \
              extract_ssa_path_vars(new_ssap, model(postcond))
        continue

      for s in bbs[bb].successors:
        q.append((postcond, s, deepcopy(after_env), newp, new_ssap))
