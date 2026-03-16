

from aria.efmc.verifytools.boogie.ast import parseAst, AstImplementation, AstLabel, \
        AstAssert, AstAssume, AstHavoc, AstAssignment, AstGoto, \
        AstReturn;
from collections import namedtuple

def unique(iterable, msg=""):
  l = list(iterable)
  assert len(l) == 1, msg
  return l[0]


BB = namedtuple("BB", ["predecessors", "stmts", "successors"])

def get_bbs(filename):
    ast = parseAst(open(filename).read())
    fun = ast._children[0][0]
    assert (isinstance(fun, AstImplementation))
    # Step 1: Break statements into basic blocks
    bbs = {}
    curLbl = None
    for stmt in fun.body.stmts:
        # A BB starts with a labeled statment
        if (isinstance(stmt, AstLabel)):
            curLbl = str(stmt.label);
            bbs[curLbl] = BB([], [], [])
            stmt = stmt.stmt;

        if (isinstance(stmt, AstAssert) or
            isinstance(stmt, AstAssume) or
            isinstance(stmt, AstHavoc) or
            isinstance(stmt, AstAssignment)):
            bbs[curLbl].stmts.append(stmt)
        elif (isinstance(stmt, AstGoto)):
            bbs[curLbl].successors.extend(map(str, stmt.labels))
            curLbl = None;
        elif (isinstance(stmt, AstReturn)):
            curLbl = None;
        else:
            raise Exception("Unknown statement : " + str(stmt))

    for bb in bbs:
        for succ in bbs[bb].successors:
            bbs[succ].predecessors.append(bb)

    return bbs

def is_internal_bb(bb):
    return bb.startswith("_union_") or bb == "_tmp_header_pred_"

def bbEntry(bbs):
    e = [x for x in bbs
           if (not is_internal_bb(x) and len(bbs[x].predecessors) == 0)]
    assert (len(e) == 1)
    return e[0]

def bbExits(bbs):
    return [x for x in bbs
              if not is_internal_bb(x) and len(bbs[x].successors) == 0]

def bbExit(bbs):
    return unique(bbExits(bbs))

def bbpath_to_stmts(bb_path, bbs):
    r = []
    for b in bb_path:
        if (isinstance(b, BB)):
            r.extend(bbs[b].stmts)
        else:
            r.append([ bbpath_to_stmts(x, bbs) for x in b ])
    return r

def ensureSingleExit(bbs):
    e = bbExits(bbs);
    if (len(e) == 1):
      return;

    bbs["_exit_"] = BB(e, [ ], []);
    
    for bb_lbl in e:
      bbs[bb_lbl].successors.append("_exit_");
