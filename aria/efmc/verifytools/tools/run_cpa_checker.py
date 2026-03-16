#! /usr/bin/env python
"""Run CPAChecker on a level set."""
import argparse
import shutil
from os.path import exists
from threading import Timer

from aria.efmc.verifytools.boogie.z3_embed import to_smt2, z3_expr_to_boogie
from aria.efmc.verifytools.common.util import error
from aria.efmc.verifytools.cpachecker import convertCppFileForCPAChecker, runCPAChecker
from aria.efmc.verifytools.tools.levels import loadBoogieLvlSet
from aria.efmc.verifytools.tools.vc_check import tryAndVerifyLvl


def handler():
    """Handle timeout exception."""
    # def handler(signum):
    #  assert (signum == SIGALRM)
    raise TimeoutError("timeout")


# signal(SIGALRM, handler);


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="run CPAChecker on a levelset")
    p.add_argument("--lvlset", type=str, help="Path to lvlset file", required=True)
    p.add_argument(
        "--csv-table",
        action="store_true",
        default=True,
        help="Print results as a csv table",
    )
    p.add_argument(
        "--time-limit", type=int, default=300, help="Time limit for CPAChecker"
    )
    p.add_argument(
        "--waitEnter",
        action="store_true",
        default=False,
        help="Wait for user to perss Enter before continuing (great for debug)",
    )
    args = p.parse_args()
    if args.waitEnter:
        input("Press Enter to continue...")
    lvl_set_name, lvls = loadBoogieLvlSet(args.lvlset)

    res = {}
    conf = {}

    for lvl_name, lvl in lvls.items():
        cpp_file = lvl["path"][1]
        preprocessed_file = cpp_file + ".cpachecker.preprocessed"
        error("Running ", lvl_name)

        if not exists(preprocessed_file):
            convertCppFileForCPAChecker(cpp_file, preprocessed_file)

        res[lvl_name] = runCPAChecker(preprocessed_file, args.time_limit)

        shutil.move("output", "tmp_outputs/" + lvl_name + "")

        solved, loop_header_lbl, loop_invs, raw_output = res[lvl_name]
        conf_status = "N/A"

        if solved:
            error("z3 invs: ", len(loop_invs), loop_invs)
            try:
                t = Timer(args.time_limit, handler)
                t.start()
                # alarm(args.time_limit)
                # On lvl d-14 for example the invariants explode exponentially due to
                # inlining of lets. So add timeout. Seems to be the only level with
                # this problem
                invs = list(map(z3_expr_to_boogie, loop_invs))
            except Exception as exc:
                if str(exc) == "timeout":
                    invs = None
                    conf_status = "timeout"
                else:
                    for i in loop_invs:
                        error(to_smt2(i))
                    raise
            finally:
                # alarm(0)
                t.cancel()
            if invs is not None:
                try:
                    (overfitted, nonind, sound, violations) = tryAndVerifyLvl(
                        lvl, set(invs), set(), args.time_limit, addSPs=True
                    )

                    error("Out of ", invs, "sound: ", sound)

                    if len(violations) > 0:
                        error("Supposedly sound inv: ", invs)
                        error("Level ", lvl_name, "false claimed to be sound!")
                        error("Raw output: ", raw_output)
                        conf_status = False
                    else:
                        conf_status = True
                except Exception as exc:
                    conf_status = "verification error: " + str(exc)
                conf[lvl_name] = conf_status
        # if (args.csv_table):
        #   print lvlName, ",", res[lvlName][0], ",", conf_status
        # else:
        print("Level", lvl_name, "solved: ", solved, "confirmed?: ", conf_status)

    if args.csv_table:
        print("Level,Solved,Confirmed")
        for lvl_name in res:
            print(
                lvl_name,
                ",",
                res[lvl_name][0],
                ",",
                conf[lvl_name] if lvl_name in conf else "N/A",
            )
