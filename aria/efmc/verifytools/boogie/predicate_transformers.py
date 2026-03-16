from aria.efmc.verifytools.boogie.ast import AstLabel, AstAssignment, AstAssert, AstAssume, \
        expr_read
from aria.efmc.verifytools.boogie.z3_embed import expr_to_z3, And, Implies, ids
from collections import namedtuple
import z3

def wp_stmt(stmt, pred, typeEnv):
    if (isinstance(stmt, AstLabel)):
        stmt = stmt.stmt

    if (isinstance(stmt, AstAssignment)):
        assignee = str(stmt.lhs)
        # Should already be SSA-ed
        assert(assignee not in expr_read(stmt.rhs))
        lhs = typeEnv[stmt.lhs](assignee)
        rhs = expr_to_z3(stmt.rhs, typeEnv)
        return z3.substitute(pred, (lhs, rhs))
    elif (isinstance(stmt, AstAssert)):
        return And(pred, expr_to_z3(stmt.expr, typeEnv))
    elif (isinstance(stmt, AstAssume)):
        return Implies(expr_to_z3(stmt.expr, typeEnv), pred)
    else:
        raise Exception("Cannot handle Boogie Statement: " + str(stmt))

def wp_stmts(stmts, pred, typeEnv):
    for s in reversed(stmts):
        #old_pred = pred
        pred = wp_stmt(s, pred, typeEnv)
        #print "WP of ", old_pred, " w.r.t. ", s, " is ", pred
    return pred

def sp_stmt(stmt, pred, typeEnv):
    if (isinstance(stmt, AstLabel)):
        stmt = stmt.stmt

    if (isinstance(stmt, AstAssignment)):
        assignee = str(stmt.lhs)
        # Should already be SSA-ed
        assert(assignee not in expr_read(stmt.rhs) and \
              (assignee not in map(str, ids(pred))))
        lhs = typeEnv[stmt.lhs](assignee)
        rhs = expr_to_z3(stmt.rhs, typeEnv)
        return And(lhs == rhs, pred)
    elif (isinstance(stmt, AstAssert)):
        return And(pred, expr_to_z3(stmt.expr, typeEnv))
    elif (isinstance(stmt, AstAssume)):
        return And(pred, expr_to_z3(stmt.expr, typeEnv))
    else:
        raise Exception("Cannot handle Boogie Statement: " + str(stmt))

def sp_stmts(stmts, pred, typeEnv):
    for s in stmts:
        pred = sp_stmt(s, pred, typeEnv)
    return pred
