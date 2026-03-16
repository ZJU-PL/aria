"""Functions for analyzing and unrolling loops in Boogie programs."""

from collections import namedtuple

from aria.efmc.verifytools.boogie.ast import AstAssume, AstTrue, ast_or, parseExprAst
from aria.efmc.verifytools.boogie.bb import BB, bbEntry
from aria.efmc.verifytools.boogie.eval import env_to_expr
from aria.efmc.verifytools.boogie.paths import get_path_vars, is_nd_bb_path_possible
from aria.efmc.verifytools.common.util import unique

Loop = namedtuple("Loop", ["header", "loop_paths", "exit_paths", "entry_cond"])


# TODO(dimo): This code is fragile - too many potentially limiting
# assumptions about code shape. Should be removed when we
# moved to on-demand level generation as loops are
# discovered during dynamic exploration of the program.

# NOTE(dimo): The work to use lib/boogie/inv_networks.py to check
# if invariants are sufficient was half the work to obsolete this
# file.


# There are several implicit assumption in the loop code
# (these seem to hold for the desugared boogie code)
#
# 1) Each loop is identified by a unique header node
# 2) Each loop has a single exit point
# 3) All the relevant postcondition asserts are in the
#    immediate loop exit code
#
# 3) is particularly risky.
def _loops(bbs, curpath, loop_m):
    """Find loops in basic blocks."""
    if curpath == []:
        curpath.append(bbEntry(bbs))

    # TODO: Is the code resilient to random dead loops?
    # if (not is_nd_bb_path_possible(curpath, bbs)):
    #    return

    for s in bbs[curpath[-1]].successors:
        if s in curpath:
            prefix = tuple(curpath[: curpath.index(s)])
            body = curpath[curpath.index(s) :]

            if prefix not in loop_m:
                loop_m[prefix] = []

            loop_m[prefix].append(body)
            continue

        curpath.append(s)
        _loops(bbs, curpath, loop_m)
        curpath.pop()
    return loop_m


def loop_path_entry_cond(p, bbs):
    """Get the entry condition for a loop path."""
    # We are relying on boogie placing the entry condition in
    # the second bb for a given path as the first statement
    bb = bbs[p[1]]
    assert isinstance(bb.stmts[0], AstAssume)
    return bb.stmts[0].expr


def loops(bbs):
    """Find all loops in basic blocks and return Loop objects."""
    loop_m = _loops(bbs, [], {})
    return [
        Loop(
            k,
            v,
            [[v[0][0], loop_exit_bb(v, bbs)]],
            ast_or([loop_path_entry_cond(p, bbs) for p in v]),
        )
        for (k, v) in loop_m.items()
    ]


def loop_exit_bb(body_paths, bbs):
    """Find the exit basic block for a loop."""
    loop_header_succ = set(bbs[body_paths[0][0]].successors)
    bbs_in_loop = {p[1] for p in body_paths}
    assert len(loop_header_succ) == len(bbs_in_loop) + 1  # Singular exit node
    # TODO: Check that no other paths exit the loop (sanity)
    return unique(loop_header_succ.difference(bbs_in_loop))


# Assumes single loop at the end of the path
def unroll_loop(loop, nunrolls, extra_pred_bb=None, exact=False):
    """Unroll a loop a specified number of times."""
    return (
        list(loop.header)
        + ([extra_pred_bb] if extra_pred_bb else [])
        + nunrolls * [loop.loop_paths]
        + ([loop.exit_paths] if exact else [])
    )


def bad_envs_to_expr(bad_envs):
    """Convert bad environments to an expression."""
    s = "&&".join(
        [
            "!(" + "&&".join([f"({k}=={v})" for (k, v) in bad_env.items()]) + ")"
            for bad_env in bad_envs
        ]
    )
    if s == "":
        return AstTrue()
    return parseExprAst(s)


# get_loop_header_values tries to unroll the given loop between min_unrolls and
# max_unrolls.
#
#   loop - loop to unroll (specified in the Loop named tuple format)
#   bbs - the basic block representation of the function
#   min_unrolls - minimum number of unrolls to try
#   max_unrolls - max number of unrolls to try
#   forbidden_envs - an optional list of 'bad' initial assignments for the live
#     loop variables at the start of iteration. tries to find values that avoid
#     those. Useful for driving the search away from previous examples.
#   starting_env - an optional precise initial assignment for the live loop
#     variables to start with. Useful when you want to continue unrolling a
#     partially unrolled loop
#   exact - if true, look for loop unrolling that exit the loop afterwards
#
def get_loop_header_values(
    loop,
    bbs,
    min_unrolls=0,
    max_unrolls=5,
    forbidden_envs=None,
    start_env=None,
    exact=False,
):
    """Get loop header values by unrolling the loop."""
    # Try unrolling it up to to the limit times
    loop_header_bb = loop.loop_paths[0][0]
    nunrolls = min_unrolls

    extra_bb = None
    if forbidden_envs or start_env:
        assert not (forbidden_envs and start_env)
        extra_bb = "_tmp_header_pred_"
        if forbidden_envs:
            expr = bad_envs_to_expr(forbidden_envs)
        else:
            expr = env_to_expr(start_env)
        bbs[extra_bb] = BB([], [AstAssume(expr)], [])

    while (
        is_nd_bb_path_possible(unroll_loop(loop, nunrolls + 1, extra_bb, exact), bbs)
        and nunrolls < max_unrolls
    ):
        nunrolls += 1

    if not is_nd_bb_path_possible(unroll_loop(loop, nunrolls, extra_bb, exact), bbs):
        return []

    unrolled_path = unroll_loop(loop, nunrolls, extra_bb, exact)
    path_vars = get_path_vars(unrolled_path, bbs)
    return [bb[1][0] for bb in path_vars if bb[0] == loop_header_bb]
