#! /usr/bin/env python
"""Try and verify invariants against a level set."""
import argparse
import sys

from aria.efmc.verifytools.boogie.ast import parseExprAst
from aria.efmc.verifytools.common.util import error
from aria.efmc.verifytools.tools.levels import loadBoogieLvlSet
from aria.efmc.verifytools.tools.vc_check import tryAndVerifyLvl

p = argparse.ArgumentParser(description="invariant gen game server")
p.add_argument(
    "--lvlset", type=str, default="desugared-boogie-benchmarks", help='Lvlset to use"'
)
p.add_argument(
    "--lvlid",
    type=str,
    default="desugared-boogie-benchmarks",
    help='Lvl-id in level set to try and verify"',
)
p.add_argument("invs", type=str, nargs="+", help="Invariants to try")
p.add_argument("--timeout", type=int, help="Optional z3 timeout", default=None)

args = p.parse_args()

_, lvls = loadBoogieLvlSet(args.lvlset)
lvl = lvls[args.lvlid]
boogie_invs = set([])

for inv in args.invs:
    try:
        b_inv = parseExprAst(inv)
        boogie_invs.add(b_inv)
    except Exception:
        error("Failed parsing invariant " + inv)
        sys.exit(-1)

((overfitted, overfitted_p2), (nonind, nonind_p2), sound, violations) = tryAndVerifyLvl(
    lvl, boogie_invs, set(), args.timeout
)

overfitted = set(overfitted).union(overfitted_p2)
nonind = set(nonind).union(nonind_p2)

print("Overfitted: ", overfitted)
print("Nonind: ", nonind)
print("Sound: ", sound)
print("Violations: ", violations)
print("Verified? ", len(violations) == 0)
