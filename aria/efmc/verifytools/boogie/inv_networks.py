#pylint: disable=no-self-argument
from aria.efmc.verifytools.boogie.ast import ast_and, replace, AstBinExpr, AstAssert, AstAssume, \
        AstTrue
from aria.efmc.verifytools.common.util import split, nonempty, powerset
from aria.efmc.verifytools.boogie.z3_embed import expr_to_z3, AllIntTypeEnv, Unknown, counterex, \
        Implies, And, tautology, satisfiable, unsatisfiable, to_smt2
from aria.efmc.verifytools.boogie.paths import nd_bb_path_to_ssa, ssa_path_to_z3, _ssa_stmts
from aria.efmc.verifytools.boogie.ssa import SSAEnv, unssa_z3_model
from aria.efmc.verifytools.boogie.predicate_transformers import wp_stmts, sp_stmt
from copy import copy
from aria.efmc.verifytools.boogie.bb import bbEntry, bbExit

def filterCandidateInvariants(bbs, preCond, postCond, cutPoints, timeout=None):
    assert (len(cutPoints) == 1)
    entryBB = bbEntry(bbs);

    cps = { bb : set(cutPoints[bb]) for bb in cutPoints }
    cps[entryBB] = [ preCond ]
    overfitted = { bb : set([]) for bb in cps }
    nonind = { bb : set([]) for bb in cps }

    # The separation in overfitted and nonind is well defined only in the
    # Single loop case. So for now only handle these. Can probably extend later

    aiTyEnv = AllIntTypeEnv()
    cpWorkQ = set([ entryBB ] + cps.keys())

    while (len(cpWorkQ) > 0):
      cp = cpWorkQ.pop()
      cp_inv = expr_to_z3(ast_and(cps[cp]), aiTyEnv)

      initial_path, intial_ssa_env = nd_bb_path_to_ssa([cp], bbs, SSAEnv())
      pathWorkQ = [(initial_path, intial_ssa_env, cp_inv)]

      # Pass 1: Widdle down candidate invariants at cutpoints iteratively
      while len(pathWorkQ) > 0:
        path, curFinalSSAEnv, sp = pathWorkQ.pop(0);

        nextBB, nextReplMaps = path[-1]
        processedStmts = []
        ssa_stmts = _ssa_stmts(bbs[nextBB].stmts, nextReplMaps)

        for s in ssa_stmts:
          if (isinstance(s, AstAssert)):
            pass; # During the first pass we ignore safety violations. We just
                  # want to get an inductive invariant network
          elif (isinstance(s, AstAssume)):
            try:
              if (unsatisfiable(And(sp, expr_to_z3(s.expr, aiTyEnv)), timeout)):
                break;
            except Unknown:
              pass; # Conservatively assume path is possible on timeout
          processedStmts.append(s)
          new_sp = sp_stmt(s, sp, aiTyEnv)
          #print "SP: {", sp, "} ", s, " {", new_sp, "}"
          sp = new_sp

        if (len(processedStmts) != len(bbs[nextBB].stmts)):
          # Didn't make it to the end of the block - path must be unsat
          continue;

        if (len(bbs[nextBB].successors) == 0): # This is exit
          assert nextBB == bbExit(bbs)
          # During Pass 1 we don't check the postcondition is implied
        else:
          for succ in bbs[nextBB].successors:
            if succ in cps:
              # Check implication
              start = initial_path[0][0]

              candidate_invs = copy(cps[succ])
              for candidate in candidate_invs:
                candidateSSA = expr_to_z3(replace(candidate,
                                                  curFinalSSAEnv.replm()),
                                          aiTyEnv)
                try:
                  c = counterex(Implies(sp, candidateSSA), timeout,
                                "Candidate: " + str(candidate))
                except Unknown:
                  c = { } # On timeout conservatively assume fail

                if (c != None):
                  v = Violation("inductiveness",
                                path + [( succ, None )],
                                [],
                                Implies(sp, candidateSSA),
                                c)

                  if (start == entryBB):
                    overfitted[succ].add((candidate, v));
                  else:
                    nonind[succ].add((candidate, v));

                  cps[succ].remove(candidate);
              if (len(candidate_invs) != len(cps[succ])):
                cpWorkQ.add(succ);
            else:
              assert succ not in path; # We should have cutpoints at every loop
              succSSA, nextFinalSSAEnv = \
                      nd_bb_path_to_ssa([succ], bbs, SSAEnv(curFinalSSAEnv));
              pathWorkQ.append((path + succSSA, nextFinalSSAEnv, sp));

    sound = cps

    # Pass 2: Check for safety violations
    violations = checkInvNetwork(bbs, preCond, postCond, sound, timeout)
    for v in violations:
      if (not v.isSafety()):
        print(v)
      assert (v.isSafety()) # sound should be an inductive network

    return (overfitted, nonind, sound, violations)

def checkInvNetwork(bbs, preCond, postCond, cutPoints, timeout=None):
    cps = copy(cutPoints);
    entryBB = bbEntry(bbs);
    cps[entryBB] = [ preCond ];
    aiTyEnv = AllIntTypeEnv()
    violations = [ ]

    for cp in cps:
      initial_path, intial_ssa_env = nd_bb_path_to_ssa([cp], bbs, SSAEnv())
      workQ = [ (initial_path,
                 intial_ssa_env,
                 expr_to_z3(ast_and(cps[cp]), aiTyEnv)) ]
      while len(workQ) > 0:
        path, curFinalSSAEnv, sp = workQ.pop(0);

        nextBB, nextReplMaps = path[-1]
        processedStmts = []
        ssa_stmts = zip(_ssa_stmts(bbs[nextBB].stmts, nextReplMaps),
                        nextReplMaps)

        for (s, replM) in ssa_stmts:
          if (isinstance(s, AstAssert)):
            try:
              c = counterex(Implies(sp, expr_to_z3(s.expr, aiTyEnv)), timeout);
            except Unknown:
              c = { } # On timeout conservatively assume fail
            if (c != None):
              # Current path can violate assertion
              v = Violation("safety", path, processedStmts + [(s, replM)],
                            Implies(sp, expr_to_z3(s.expr, aiTyEnv)),
                            c)
              violations.append(v)
              break;
          elif (isinstance(s, AstAssume)):
            try:
              if (unsatisfiable(And(sp, expr_to_z3(s.expr, aiTyEnv)), timeout)):
                break;
            except Unknown:
              pass; # Conservatively assume path is possible on timeout
          processedStmts.append((s, replM))
          new_sp = sp_stmt(s, sp, aiTyEnv)
          #print "SP: {", sp, "} ", s, " {", new_sp, "}"
          sp = new_sp

        if (len(processedStmts) != len(bbs[nextBB].stmts)):
          # Didn't make it to the end of the block - path must be unsat or
          # violation found
          continue;

        if (len(bbs[nextBB].successors) == 0): # This is exit
          assert nextBB == bbExit(bbs)
          postSSA = expr_to_z3(replace(postCond, curFinalSSAEnv.replm()),
                               aiTyEnv)
          try:
            c = counterex(Implies(sp, postSSA), timeout)
          except Unknown:
            c = { } # On timeout conservatively assume fail
          if (c != None):
            v = Violation("safety",
                          path,
                          processedStmts,
                          Implies(sp, postSSA),
                          c)
            violations.append(v)
        else:
          for succ in bbs[nextBB].successors:
            if succ in cps:
              # Check implication
              post = ast_and(cps[succ])
              postSSA = replace(post, curFinalSSAEnv.replm())
              postSSAZ3 = expr_to_z3(postSSA, aiTyEnv)
              try:
                c = counterex(Implies(sp, postSSAZ3), timeout)
              except Unknown:
                try:
                  # Sometimes its easier to check each post-invariant
                  # individually rather than all of them at once (similarly
                  # to the filter case). Try this if the implication of the
                  # conjunction of all of them fails
                  for p in cps[succ]:
                    postSSA = replace(p, curFinalSSAEnv.replm())
                    postSSAZ3 = expr_to_z3(postSSA, aiTyEnv)
                    c = counterex(Implies(sp, postSSAZ3), timeout)
                    # If any of them doesn't hold, neither does their conj
                    if (c != None):
                        break;
                except Unknown:
                  c = { } # On timeout conservatively assume fail
              if (c != None):
                v = Violation("inductiveness",
                  path + [ (succ, None) ],
                  [],
                  Implies(sp, postSSAZ3),
                  c)
                violations.append(v)
            else:
              assert succ not in path; # We should have cutpoints at every loop
              succSSA, nextFinalSSAEnv = \
                      nd_bb_path_to_ssa([succ], bbs, SSAEnv(curFinalSSAEnv));
              workQ.append((path + succSSA, nextFinalSSAEnv, sp));

    return violations

class Violation:
  def __init__(s, typ, path, lastBBCompletedStmts, query, ctrex):
    s._typ = typ;
    s._path = path;
    s._lastBBCompletedStmts = lastBBCompletedStmts;
    s._query = query;
    s._ctrex = ctrex;

  def isInductive(s):
      return s._typ == "inductiveness"
  def isSafety(s):
      return s._typ == "safety"
  def startBB(s):
      return s._path[0][0]
  def endBB(s):
      return s._path[-1][0]
  def startReplM(s):
      return s._path[0][1][0]
  def endReplM(s):
    if (len(s._lastBBCompletedStmts) > 0):
      return s._lastBBCompletedStmts[-1][1]
    return s._path[-2][1][-1]

  def startEnv(s):
    assert {} == s.startReplM()
    return unssa_z3_model(s._ctrex, s.startReplM())

  def endEnv(s):
    return unssa_z3_model(s._ctrex, s.endReplM())

  def __str__(s):
    if (s.isSafety()):
      return "Safety@" + str(s.endBB()) + ":" + str(s.endEnv())
    else:
      return "Inductiveness@" + str([x[0] for x in s._path]) + ":" + \
              str(s.startEnv()) + "->" + str(s.endEnv())

  def __repr__(s):
    return s.__str__()
