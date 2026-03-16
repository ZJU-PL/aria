#! /usr/bin/env python
from levels import loadBoogieLvlSet
from lib.daikon import toDaikonTrace, runDaikon
from conversions import daikonToBoogieExpr
import argparse
from os.path import exists
from os import listdir
from vc_check import tryAndVerifyLvl
import lib.boogie.ast as bast
from lib.boogie.eval import evalPred
import sys

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="run daikon on a levelset")
    p.add_argument("--lvlset", type=str, help="Path to lvlset file", required=True)
    p.add_argument(
        "--use-splitter-predicates",
        action="store_true",
        default=False,
        help="Wether to try inductive invariants with the splitter " + "predicates",
    )
    p.add_argument(
        "--no-suppression",
        action="store_true",
        default=False,
        help="Wether to have daikon suppress obvious invariants",
    )
    p.add_argument(
        "--check-solved",
        action="store_true",
        default=False,
        help="Wether to check for each level if it was solved",
    )
    p.add_argument(
        "--csv-table",
        action="store_true",
        default=False,
        help="Print results as a csv table",
    )
    p.add_argument(
        "--timeout", type=int, default=300, help="Timeout in seconds for each z3 query"
    )
    args = p.parse_args()

    name, lvls = loadBoogieLvlSet(args.lvlset)

    if args.csv_table:
        if args.check_solved:
            print(
                "Level, Num Invariants Found, Invariants Found, "
                + "Num Sound Invariants, Sound Invariants, IsSolved"
            )
        else:
            print(
                "Level, Num Invariants Found, Invariants Found, "
                + "Num Sound Invariants, Sound Invariants"
            )

    for lvlName, lvl in lvls.iteritems():
        (vs, header_vals) = (lvl["variables"], lvl["data"][0])
        fuzz_path = lvl["path"][0][: -len(".bpl")] + ".fuzz_traces"
        if exists(fuzz_path):
            for f in listdir(fuzz_path):
                t = eval(open(fuzz_path + "/" + f).read())
                t_loop_headers = [x for x in t if "LoopHead" in x[0]]
                assert len(set([x[0] for x in t_loop_headers])) == 1
                if "splitterPreds" in lvl:
                    splitterPred = bast.ast_and(lvl["splitterPreds"])
                    t_loop_headers = filter(
                        lambda row: evalPred(splitterPred, row[1][0]), t_loop_headers
                    )

                t_header_vals = [[row[1][0][v] for v in vs] for row in t_loop_headers]
                header_vals = header_vals + t_header_vals
        invs = runDaikon(vs, header_vals, args.no_suppression)

        binvs = []
        for inv in invs:
            try:
                binvs.append(daikonToBoogieExpr(inv))
            except Exception:
                if not args.csv_table:
                    sys.stderr.write("Can't translate " + str(inv) + "\n")

        ((overfitted, overfitted_p2), (nonind, nonind_p2), soundInvs, violations) = (
            tryAndVerifyLvl(
                lvl,
                binvs,
                set(),
                args.timeout,
                useSplitters=args.use_splitter_predicates,
            )
        )
        overfitted += overfitted_p2
        nonind += nonind_p2

        if args.check_solved:
            solved = "," + str(len(violations) == 0)
        else:
            solved = ""

        if args.csv_table:
            (
                print(
                    ",".join(
                        [
                            lvlName,
                            str(len(binvs)),
                            ";".join(map(str, binvs)),
                            str(len(soundInvs)),
                            ";".join(map(str, soundInvs)),
                        ]
                    )
                )
                + solved
            )
        else:
            print(
                "For lvl",
                lvlName,
                " daikon found",
                len(binvs),
                ":",
                binvs,
                "of which",
                len(soundInvs),
                " are sound: ",
                soundInvs,
            )
