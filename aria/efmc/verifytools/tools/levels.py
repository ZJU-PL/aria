"""Functions for loading and managing Boogie level sets."""

from collections import OrderedDict
from functools import reduce
from itertools import product
from json import load
from os import listdir
from os.path import abspath, dirname, join, realpath
from random import randint

from aria.efmc.verifytools.boogie.analysis import livevars
from aria.efmc.verifytools.boogie.ast import ast_and, ast_or, parseExprAst
from aria.efmc.verifytools.boogie.bb import bbEntry, ensureSingleExit, get_bbs
from aria.efmc.verifytools.boogie.eval import _to_dict, execute, evalPred, instantiateAndEval
from aria.efmc.verifytools.common.util import average, error, powerset, unique
from aria.efmc.verifytools.tools.boogie_loops import get_loop_header_values, loops
from aria.efmc.verifytools.tools.vc_check import loopInvOverfittedCtrex


def _try_unroll(loop, bbs, min_un, max_un, bad_envs, good_env):
    """Try to unroll a loop and return values and termination status."""
    # Lets first try to find a terminating loop between min and max iterations
    term_vals = get_loop_header_values(
        loop, bbs, min_un, max_un, bad_envs, good_env, True
    )
    if term_vals != []:
        return (term_vals, True)

    # Couldn't find a terminating loop between 0 and 6 iteration. Lets find
    # a loop that has at LEAST min iterations
    term_vals = get_loop_header_values(
        loop, bbs, min_un, max_un, bad_envs, good_env, False
    )
    return (term_vals, False)


def varproduct(vargens):
    """Generate all variable assignments from variable generators."""
    # Take var: gen and generate all var assignments (smallest first), i.e.
    # "A": count(), "B": count() yields
    #   (A, B) = (0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0), ...
    if len(vargens.items()) == 0:
        yield {}
    else:
        vars_, gens = zip(*vargens.items())
        for vals in product(*gens):
            yield dict(zip(vars_, vals))


def get_ensemble(
    loop,
    bbs,
    exec_limit,
    try_find=100,
    distr=lambda: randint(0, 5),
    include_bbhit=False,
    vargens=None,
):
    """Get an ensemble of traces for a loop."""
    loop_hdr = loop.loop_paths[0][0]
    trace_vs = list(livevars(bbs)[loop_hdr])
    if vargens is None:

        def candidatef():
            while True:
                yield {v: distr() for v in trace_vs}

        candidategen = candidatef()
    else:
        candidategen = varproduct(vargens)
    tried = set()
    # TODO: Not a smart way for generating random start values. Fix.
    s = 0
    print("Trying to find ", try_find, " traces of length up to ", exec_limit)
    while s < try_find:
        candidate = next(candidategen)
        hashable = tuple(candidate[v] for v in trace_vs)
        if hashable in tried:
            continue
        tried.add(hashable)

        found = False
        trace = execute(candidate, bbEntry(bbs), bbs, exec_limit)
        for _, _, _, ssap, vals in trace:
            vals = [envs[0] for (bb, envs) in vals if bb == loop_hdr]
            if include_bbhit:
                bbhit = {bbname for bbname, _ in ssap}
                yield (vals, bbhit)
            else:
                yield vals
            found = True
            s += 1
            if s >= try_find:
                break

        if not found:
            s += 1


def get_initial_data(loop, bbs, nunrolls, invs, inv_vars=None, inv_consts=None):
    """Get initial data for invariant generation."""
    if inv_consts is None:
        inv_consts = ["_sc_a", "_sc_b", "_sc_c"]
    trace_ensemble = list(get_ensemble(loop, bbs, nunrolls, 1))
    vals, _ = _try_unroll(loop, bbs, 0, nunrolls, None, None)
    if vals:
        trace_ensemble.append(vals)

    trace_vs = list(livevars(bbs)[loop.loop_paths[0][0]])
    trace_ensemble = [
        [{var_name: env[var_name] for var_name in trace_vs} for env in tr]
        for tr in trace_ensemble
    ]

    if inv_vars is None:
        inv_vars = trace_vs

    invs_lst = [
        reduce(
            lambda x, y: x + y,
            [instantiateAndEval(inv, trace, inv_vars, inv_consts) for inv in invs],
            [],
        )
        for trace in trace_ensemble
        if len(trace) > 0
    ]

    tmp_lst = [(len(invs), invs, tr) for (invs, tr) in zip(invs_lst, trace_ensemble)]

    tmp_lst.sort(key=lambda t: t[0])
    return (tmp_lst[0][2], False)


def find_negating_trace(loop, bbs, nunrolls, invs, inv_vrs=None):
    """Find a trace that negates the given invariants."""
    vals, terminates = _try_unroll(loop, bbs, 0, nunrolls, None, None)
    trace_vs = list(livevars(bbs)[loop.loop_paths[0][0]])
    vals = [{x: env[x] for x in trace_vs} for env in vals]
    hold_for_data = []

    if inv_vrs is None:
        inv_vrs = trace_vs

    def diversity(vals):
        lsts = [[vals[i][k] for i in range(len(vals))] for k in vals[0].keys()]
        return average([len(set(lst)) for lst in lsts])
        # return average([len(set(lst)) / 1.0 * len(lst) for lst in lsts])

    for inv in invs:
        hold_for_data.extend(
            instantiateAndEval(inv, vals, inv_vrs, ["_sc_a", "_sc_b", "_sc_c"])
        )

    print("The following invariants hold for initial trace: ", hold_for_data)
    hold_for_data = list(set(hold_for_data))
    print("The following remain after clearing duplicates: ", hold_for_data)
    res = []
    no_ctrex = set([])
    for s in powerset(hold_for_data):
        if s.issubset(no_ctrex):
            continue
        # print "Looking for ctrex for: ", s, " with no_ctrex: ", no_ctrex
        inv = ast_or(s)
        ctrexs = loopInvOverfittedCtrex(loop, inv, bbs)
        if len(ctrexs) > 0:
            for ctrex in ctrexs:
                trace, terminates = _try_unroll(loop, bbs, 0, nunrolls, None, ctrex)
                if len(trace) > 0:
                    print("Ctrexample for ", inv, " is ", trace)
                    res.append(
                        (diversity(trace), len(s), list(s), ctrex, (trace, terminates))
                    )
        else:
            no_ctrex = no_ctrex.union(s)

    res.sort(key=lambda x: x[0])
    if len(res) > 0:
        return res[-1][4]
    return (None, False)


def read_trace(fname):
    """Read a trace from a file."""
    with open(fname, "r", encoding="utf-8") as f:
        trace = f.read()
    lines = list(
        filter(lambda x: len(x) != 0, map(lambda x: x.strip(), trace.split("\n")))
    )
    if not lines:
        return ([], [])
    vs = list(filter(lambda x: len(x) != 0, lines[0].split(" ")))
    header_vals = []
    for l in lines[1:]:
        if l[0] == "#":
            continue

        env = {}
        for var, val in zip(vs, filter(lambda x: len(x) != 0, l.split(" "))):
            env[var] = val
        header_vals.append(env)
    return (vs, header_vals)


def write_trace(fname, header_vals):
    """Write a trace to a file."""
    with open(fname, "w", encoding="utf-8") as f:
        if len(header_vals) != 0:
            vs = list(header_vals[0].keys())
            f.write(" ".join(vs) + "\n")
            for env in header_vals:
                f.write(" ".join([str(env[v]) for v in vs]) + "\n")


# TODO: Remove multiround. Its crud.
def load_boogie_file(fname, multiround):
    """Load a Boogie file and return level data."""
    bbs = get_bbs(fname)
    ensureSingleExit(bbs)
    loop = unique(loops(bbs), "Cannot handle program with multiple loops:" + fname)

    # The variables to trace are all live variables at the loop header
    vs = list(livevars(bbs)[loop.loop_paths[0][0]])

    # Make sure variable names are different modulo case
    assert len(vs) == len(set([var.lower() for var in vs]))

    # See if there is a .trace or a .hint file
    hint = None
    header_vals = None
    terminates = False
    try:
        (vs, header_vals) = read_trace(fname[:-4] + ".trace")
        with open(fname[:-4] + ".hint", encoding="utf-8") as f:
            hint = load(f)
    except Exception:  # TODO (Dimo) IOError here instead?
        pass

    if not header_vals:
        header_vals, terminates = _try_unroll(loop, bbs, 0, 4, None, None)
        # Assume we have no tests with dead loops
        assert header_vals != []
        inv_templates = [
            "_sv_x<_sv_y",
            "_sv_x<=_sv_y",
            "_sv_x==_sc_c",
            "_sv_x==_sv_y",
            "_sv_x==0",
            "_sv_x<0",
        ]
        inv_templates = [parseExprAst(inv) for inv in inv_templates]

        new_header_vals, new_terminates = get_initial_data(
            loop, bbs, 4, inv_templates, ["_sv_x", "_sv_y"]
        )

        if new_header_vals is not None:
            header_vals = new_header_vals
            terminates = new_terminates
            write_trace(fname[:-4] + ".trace", new_header_vals)

    return {
        "variables": vs,
        "data": [[[str(row[v]) for v in vs] for row in header_vals], [], []],
        "exploration_state": [
            ([str(header_vals[0][v]) for v in vs], len(header_vals), terminates)
        ],
        "hint": hint,
        "goal": {"verify": True},
        "support_pos_ex": True,
        "support_neg_ex": True,
        "support_ind_ex": True,
        "multiround": multiround,
        "program": bbs,
        "loop": loop,
    }


def load_boogies(dir_n, multiround=False):
    """Load all Boogie files from a directory."""
    return {
        name[:-4]: load_boogie_file(dir_n + "/" + name, multiround)
        for name in listdir(dir_n)
        if name.endswith(".bpl")
    }


def read_trace_only_lvl(fname):
    """Read a trace-only level from a file."""
    rows = []
    first = True
    vs = []
    with open(fname, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if l == "":
                continue
            row = {}
            for n, v in [x.split("=") for x in l.split(" ")]:
                row[n] = v

            if first:
                vs = [x.split("=")[0] for x in l.split(" ")]
                first = False
            rows.append(row)

    hint = None
    goal = None
    try:
        with open(fname[:-4] + ".goal", encoding="utf-8") as f:
            goal = load(f)
        with open(fname[:-4] + ".hint", encoding="utf-8") as f:
            hint = f.read()
    except Exception:
        pass

    return {
        "variables": vs,
        "data": [[[row.get(n, None) for n in vs] for row in rows], [], []],
        "hint": hint,
        "goal": goal,
        "support_pos_ex": False,
        "support_neg_ex": False,
        "support_ind_ex": False,
        "multiround": False,
    }


def load_traces(dir_n):
    """Load all trace files from a directory."""
    return {
        name[:-4]: read_trace_only_lvl(dir_n + "/" + name)
        for name in listdir(dir_n)
        if name.endswith(".out")
    }


def load_boogie_lvl_set(lvl_set_file):
    """Load a Boogie level set from a file."""

    # Small helper funct to make sure we didn't
    # accidentally give two levels the same name
    def assert_unique_keys(kvs):
        keys = [x[0] for x in kvs]
        assert len(set(keys)) == len(keys)
        return dict(kvs)

    with open(lvl_set_file, "r", encoding="utf-8") as f:
        lvl_set = load(f, object_pairs_hook=assert_unique_keys)
    lvl_set_dir = dirname(abspath(realpath(lvl_set_file)))
    error("Loading level set " + lvl_set["name"] + " from " + lvl_set_file)
    lvls = OrderedDict()
    for t in lvl_set["levels"]:
        lvl_name = t[0]
        lvl_path = t[1]

        for i, path in enumerate(lvl_path):
            lvl_path[i] = join(lvl_set_dir, path)

        error("Loading level: ", lvl_path[0])
        lvl = load_boogie_file(lvl_path[0], False)
        lvl["path"] = lvl_path

        if len(t) > 2:
            splitter_preds = [parseExprAst(exp) for exp in t[2]]
            splitter_pred = ast_and(splitter_preds)
            remainder_inv = parseExprAst(t[3])

            lvl["data"][0] = list(
                filter(
                    lambda row: evalPred(
                        splitter_pred, _to_dict(lvl["variables"], row)
                    ),
                    lvl["data"][0],
                )
            )

            if len(lvl["data"][0]) == 0:
                error("SKIPPING : ", lvl_name, " due to no filtered rows.")
                continue

            lvl["partialInv"] = remainder_inv
            lvl["splitterPreds"] = splitter_preds

        lvls[lvl_name] = lvl

    return (lvl_set["name"], lvls)


# Backward compatibility aliases
_tryUnroll = _try_unroll  # pylint: disable=invalid-name
getEnsamble = get_ensemble  # pylint: disable=invalid-name
getInitialData = get_initial_data  # pylint: disable=invalid-name
findNegatingTrace = find_negating_trace  # pylint: disable=invalid-name
readTrace = read_trace  # pylint: disable=invalid-name
writeTrace = write_trace  # pylint: disable=invalid-name
loadBoogieFile = load_boogie_file  # pylint: disable=invalid-name
loadBoogies = load_boogies  # pylint: disable=invalid-name
readTraceOnlyLvl = read_trace_only_lvl  # pylint: disable=invalid-name
loadTraces = load_traces  # pylint: disable=invalid-name
loadBoogieLvlSet = load_boogie_lvl_set  # pylint: disable=invalid-name
