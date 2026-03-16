from aria.efmc.verifytools.boogie.ast import (
    ast_and,
    replace,
    AstBinExpr,
    AstTrue,
    AstNumber,
    AstId,
)
from aria.efmc.verifytools.common.util import nonempty, powerset, flattenSet
from aria.efmc.verifytools.boogie.z3_embed import (
    expr_to_z3,
    AllIntTypeEnv,
    Unknown,
    And,
    tautology,
)
from aria.efmc.verifytools.boogie.analysis import propagate_sp
from aria.efmc.verifytools.boogie.bb import bbEntry
from aria.efmc.verifytools.boogie.inv_networks import (
    checkInvNetwork,
    filterCandidateInvariants,
)


def conservative_tautology(q):
    try:
        return tautology(q)
    except Unknown:
        return False


def _from_dict(vs, vals, missing=None):
    if type(vals) == tuple:
        return (_from_dict(vs, vals[0]), _from_dict(vs, vals[1]))
    else:
        return [(vals[vs[i]] if vs[i] in vals else missing) for i in range(0, len(vs))]


def traceConstantVars(lvl):
    vs = lvl["variables"]
    table = lvl["data"][0]
    cInds = [
        (x[0], list(x[1])[0])
        for x in enumerate(
            [
                set([table[row][col] for row in range(len(table))])
                for col in range(len(vs))
            ]
        )
        if len(x[1]) == 1
    ]
    return [(vs[x[0]], x[1]) for x in cInds]


def substitutions(expr, replMs):
    family = set([expr])
    for replM in replMs:
        family.add(replace(expr, replM))

    return family


def generalizeConstTraceVars(lvl):
    """Given an expression and a set of pairs (Var, Num) where Var is always
    equal to Num in the traces presented to the user, add all variations
    of expr with Num substituded for Var and vice-versa. Note we don't do
    combinations of multiple substitutions where that is possible.
    """
    varNums = traceConstantVars(lvl)
    replMs = [{AstId(v[0]): AstNumber(int(v[1])) for v in varNums}] + [
        {AstNumber(int(v[1])): AstId(v[0]) for v in varNums}
    ]
    return replMs


def tryAndVerify(
    bbs, loop, splitterPreds, partialInvs, userInvs, otherInvs, replMaps, timeout=None
):
    userInvs = flattenSet([substitutions(x, replMaps) for x in userInvs])
    invs = userInvs.union(otherInvs)
    invs = invs.union(partialInvs)

    if len(splitterPreds) == 0:
        assert len(partialInvs) == 0
        (ovrefitted, nonind, sound, violations) = tryAndVerify_impl(
            bbs, loop, set(), invs, timeout
        )
        return ((ovrefitted, []), (nonind, []), sound, violations)
    else:
        return tryAndVerifyWithSplitterPreds(
            bbs, loop, set(), invs, splitterPreds, partialInvs, timeout
        )


def tryAndVerifyLvl(
    lvl,
    userInvs,
    otherInvs,
    timeout=None,
    useSplitters=True,
    addSPs=True,
    generalizeUserInvs=False,
):
    """Try and verify a given Lvl.

    lvl - level to verify
    userInvs - invariant candidate from the user
    otherInvs - invariant candidates from other sources (e.g. previous
                users).
                Note: We will keep userInvs and otherInvs separate in the
                results.
    timeout - if specified, the z3 timeout for each query
    useSplitters - if the level supports splitter predicates, whether to
                   use them or not.
    addSPs - if true, try to propagte SP through the loop and add the
             results as invariants. For example if before the loop we
             have "assert n>0;" and n is not modified in the loop, we
             can determine that"n>0" holds through the loop and add
             that to our cnadidate invariants.
    generalizeUserInvs - if true, for any invariant I involving a
             constant C, where one of the variables V shown to the user
             was always equal to C in the traces, try substituting C with V.
             For example if in the level always n=4, and the user
             entered i<=4, the generalizaition would also try i<=n.
    """
    bbs = lvl["program"]
    loop = lvl["loop"]
    partialInvs = [lvl["partialInv"]] if ("partialInv" in lvl) and useSplitters else []
    splitterPreds = (
        lvl["splitterPreds"] if ("splitterPreds" in lvl) and useSplitters else []
    )
    if generalizeUserInvs:
        replMaps = generalizeConstTraceVars(lvl)
    else:
        replMaps = []

    # Push any SPs that are syntactically unmodified
    if addSPs:
        loop_header = loop.loop_paths[0][0]
        sps = propagate_sp(bbs)[loop_header]
        otherInvs = otherInvs.union(sps)

    return tryAndVerify(
        bbs, loop, splitterPreds, partialInvs, userInvs, otherInvs, replMaps, timeout
    )


def tryAndVerify_impl(bbs, loop, old_sound_invs, invs, timeout=None):
    """Wrapper around checkInvNetwork for the case of a function
    with a single loop and no pre- post- conditions.
    Returns a tuple (overfitted, nonind, sound, violations) where
      overfitted, nonind are each a list of tuples (inv, Violation)
      sound is a set of sound invariants
      violations is a (potentially empty) list of safety Violations
        for the sound invariants.
    """
    assert isinstance(old_sound_invs, set)
    assert isinstance(invs, set)
    loopHdr = loop.loop_paths[0][0]
    cps = {loopHdr: set(old_sound_invs).union(set(invs))}

    (overfitted, nonind, sound, violations) = filterCandidateInvariants(
        bbs, AstTrue(), AstTrue(), cps, timeout
    )

    overfitted = list(overfitted[loopHdr])
    nonind_invs = list(nonind[loopHdr])
    sound = sound[loopHdr]

    return overfitted, nonind_invs, sound, violations


def tryAndVerifyWithSplitterPreds(
    bbs, loop, old_sound_invs, boogie_invs, splitterPreds, partialInvs, timeout=None
):
    """Wrapper around tryAndVerify_impl that adds implication with
    the splitter predicates to all candidate invariants. Returns
      ((p1_overfit, p2_overfit), (p1_nonindg, p2_nonind), sound, violations)

    Where

      p1_overfit, p2_ovefit are lists of pairs of overfittted invariants
                and their respective counterexamples from passes 1 and 2
      p1_nonind, p2_ovefit are lists of pairs of noninductive invariants
                and their respective counterexamples from passes 1 and 2
      sound is a set of sound invariants
      violations is a list of any safety violations permitted by the sound
      invariants
    """
    assert isinstance(old_sound_invs, set)
    assert isinstance(boogie_invs, set)
    assert isinstance(partialInvs, list)
    assert isinstance(splitterPreds, list)

    initial_sound = old_sound_invs.union(partialInvs)

    # First lets find the invariants that are sound without implication
    p1_overfitted, p1_nonind, p1_sound, violations = tryAndVerify_impl(
        bbs, loop, initial_sound, boogie_invs, timeout
    )
    p1_sound = set(
        [
            x
            for x in p1_sound
            if not conservative_tautology(expr_to_z3(x, AllIntTypeEnv()))
        ]
    )

    # Next lets add implication  to all unsound invariants from first pass
    # Also add manually specified partialInvs
    unsound = [inv_ctr_pair[0] for inv_ctr_pair in p1_overfitted + p1_nonind]
    candidate_precedents = [ast_and(pSet) for pSet in nonempty(powerset(splitterPreds))]
    p2_invs = [
        AstBinExpr(precc, "==>", inv)
        for precc in candidate_precedents
        for inv in unsound
    ] + partialInvs

    p2_invs = set(
        [
            x
            for x in p2_invs
            if not conservative_tautology(expr_to_z3(x, AllIntTypeEnv()))
        ]
    )

    # And look for any new sound invariants
    p2_overfitted, p2_nonind, p2_sound, violations = tryAndVerify_impl(
        bbs, loop, p1_sound.union(set(partialInvs)), p2_invs, timeout
    )
    sound = p1_sound.union(p2_sound)

    return ((p1_overfitted, p2_overfitted), (p1_nonind, p2_nonind), sound, violations)


def loopInvOverfittedCtrex(loop, invs, bbs, timeout=None):
    """Given a candidate loop invariant inv find 'overfittedness'
    counterexamples.  I.e. find counterexamples to "precondition ==> inv".
    Returns a potentially empty set of environments (dicts) that the invariant
    should satisfy.
    """
    loopHdr = loop.loop_paths[0][0]
    cps = {loopHdr: set(invs)}
    violations = checkInvNetwork(bbs, AstTrue(), AstTrue(), cps, timeout)
    entryBB = bbEntry(bbs)

    return [
        x.endEnv()
        for x in violations
        if x.isInductive()  # Implication fail
        and x.startBB() == entryBB  # From entry
        and x.endBB() == loopHdr
    ]  # To loop inv


def loopInvSafetyCtrex(loop, invs, bbs, timeout=None):
    """Given a candidate loop invariant inv find 'safety'
    counterexamples.  I.e. find counterexamples to "inv ==> post" or "inv ==>
    assert".  Returns a potentially empty set of environments (dicts) that
    the invariant should satisfy.
    """
    loopHdr = loop.loop_paths[0][0]
    cps = {loopHdr: set(invs)}
    violations = checkInvNetwork(bbs, AstTrue(), AstTrue(), cps, timeout)

    return [
        x.endEnv()
        for x in violations
        if x.isSafety() and x.startBB() == loopHdr  # Safety fail
    ]  # From Inv
