#! /usr/bin/env python
"""Run InvGen on a level set."""
import argparse
import re
from os.path import exists

from aria.efmc.verifytools.boogie.ast import parseExprAst
from aria.efmc.verifytools.common.util import error
from aria.efmc.verifytools.invgen import convert_cpp_file_for_invgen, run_invgen
from aria.efmc.verifytools.tools.levels import loadBoogieLvlSet
from aria.efmc.verifytools.tools.vc_check import tryAndVerifyLvl


def parse_invgen_invariant(inv):
    """Parse an InvGen invariant string into a Boogie AST expression."""
    replace_eq = re.compile("([^<>=])=([^<>=])")
    inv = replace_eq.sub(r"\1==\2", inv)
    inv = inv.replace("=<", "<=")
    return parseExprAst(inv)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="run invgen on a levelset")
    p.add_argument("--lvlset", type=str, help="Path to lvlset file", required=True)
    p.add_argument(
        "--time-limit",
        type=int,
        help="Timeout for any operation involving InvGen or z3",
    )
    p.add_argument(
        "--csv-table",
        action="store_true",
        default=False,
        help="Print results as a csv table",
    )
    args = p.parse_args()

    lvl_set_name, lvls = loadBoogieLvlSet(args.lvlset)

    res = {}
    if args.csv_table:
        print("Level,Invariants,TranslatedTo,Solved,Confirmed")

    for lvl_name, lvl in lvls.items():
        cpp_file = lvl["path"][1]
        invgen_cpp_file = cpp_file + ".invgen.preprocessed"

        if not exists(invgen_cpp_file):
            optional_manual_modified_file = cpp_file + ".invgen.manual.pretranslation"
            if exists(optional_manual_modified_file):
                src = optional_manual_modified_file
            else:
                src = cpp_file

            convertCppFileForInvGen(src, invgen_cpp_file)

        error("Running ", lvl_name)

        main_routine = "main" if lvl_name != "i-cegar1" else "cegar1"
        res[lvl_name] = runInvGen(invgen_cpp_file, main_routine)

        solved, loop_invs, raw_output = res[lvl_name]
        conf_status = "n/a"

        if solved:
            error("z3 invs: ", len(loop_invs), loop_invs)
            boogie_invs = list(map(parse_invgen_invariant, loop_invs))
            bbs = lvl["program"]
            loop = lvl["loop"]
            try:
                ((overfitted, overfitted2), (nonind, nonind2), sound, violations) = (
                    tryAndVerifyLvl(
                        lvl,
                        set(boogie_invs),
                        set(),
                        args.time_limit,
                        addSPs=True,
                        useSplitters=False,
                        generalizeUserInvs=False,
                    )
                )
                assert len(overfitted2) == 0 and len(nonind2) == 0
                if len(violations) > 0:
                    error("Supposedly sound inv: ", loop_invs)
                    error("Results : ", overfitted, nonind, sound)
                    error("Level ", lvl_name, "false claimed to be sound!")
                    error("Raw output: ", raw_output)
                    conf_status = False
                else:
                    conf_status = True
            except Exception as exc:
                if str(exc) == "Unknown binary operator /":
                    conf_status = "unhandled division"
                else:
                    raise
        else:
            boogie_invs = []

        if args.csv_table:
            print(
                lvl_name,
                ",",
                ";".join(map(str, loop_invs)),
                ",",
                ";".join(map(str, boogie_invs)),
                ",",
                res[lvl_name][0],
                ",",
                conf_status,
            )
        else:
            print(
                "Level",
                lvl_name,
                "discovered:",
                loop_invs,
                "solved: ",
                solved,
                "confirmed?: ",
                conf_status,
            )
