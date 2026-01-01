#!/usr/bin/env python3
"""
Translated from
https://github.com/cvc5/cvc5/blob/c645d256b82f3391d83e96a910f9cf0573016fca/contrib/competitions/smt-comp/run-script-smtcomp2024

"""
import os
import re
import subprocess
import sys
from pathlib import Path


def get_cvc5_path():
    script_dir = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
    print(script_dir)
    return str(script_dir / "bin_solvers/cvc5")


def get_logic(bench_file):
    with open(bench_file) as f:
        for line in f:
            if "set-logic" in line:
                match = re.search(r"\(set-logic\s+([A-Z_]*)\s*\)", line)
                if match:
                    return match.group(1)
    return None


def run_with_timeout(cmd, timeout=None):
    try:
        if timeout:
            # Set resource limit
            import resource

            def set_limit():
                resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))

            result = subprocess.run(
                cmd, capture_output=True, text=True, preexec_fn=set_limit
            )
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)

        output = result.stdout + result.stderr
        return output
    except subprocess.SubprocessError as e:
        return str(e)


def try_with(timeout, bench_file, cvc5_path, out_file, *args):
    base_cmd = [
        cvc5_path,
        "-L",
        "smt2.6",
        "--no-incremental",
        "--no-type-checking",
        "--no-interactive",
    ]
    if "--fp-exp" not in args:
        cmd = base_cmd + list(args) + [bench_file]
    else:
        cmd = base_cmd + ["--fp-exp"] + list(args) + [bench_file]

    result = run_with_timeout(cmd, timeout)

    if result.strip() in ["sat", "unsat"]:
        print(result.strip())
        sys.exit(0)
    else:
        with open(out_file, "w") as f:
            f.write(result)


def finish_with(bench_file, cvc5_path, out_file, *args):
    base_cmd = [
        cvc5_path,
        "-L",
        "smt2.6",
        "--no-incremental",
        "--no-type-checking",
        "--no-interactive",
    ]
    cmd = base_cmd + list(args) + [bench_file]

    result = run_with_timeout(cmd)

    if result.strip() in ["sat", "unsat"]:
        print(result.strip())
        sys.exit(0)
    else:
        with open(out_file, "w") as f:
            f.write(result)


def main():
    if len(sys.argv) < 2:
        sys.exit("Benchmark file required")

    bench_file = sys.argv[1]
    cvc5_path = get_cvc5_path()

    # Determine output file
    out_file = "/dev/stderr"
    if os.environ.get("STAREXEC_WALLCLOCK_LIMIT"):
        out_file = "/dev/null"
    if len(sys.argv) > 2:
        out_file = os.path.join(sys.argv[2], "err.log")

    logic = get_logic(bench_file)

    if logic == "QF_LRA":
        try_with(
            200,
            bench_file,
            cvc5_path,
            out_file,
            "--miplib-trick",
            "--miplib-trick-subs=4",
            "--use-approx",
            "--lemmas-on-replay-failure",
            "--replay-early-close-depth=4",
            "--replay-lemma-reject-cut=128",
            "--replay-reject-cut=512",
            "--unconstrained-simp",
            "--use-soi",
        )
        finish_with(
            bench_file,
            cvc5_path,
            out_file,
            "--no-restrict-pivots",
            "--use-soi",
            "--new-prop",
            "--unconstrained-simp",
        )

    elif logic == "QF_LIA":
        finish_with(
            bench_file,
            cvc5_path,
            out_file,
            "--miplib-trick",
            "--miplib-trick-subs=4",
            "--use-approx",
            "--lemmas-on-replay-failure",
            "--replay-early-close-depth=4",
            "--replay-lemma-reject-cut=128",
            "--replay-reject-cut=512",
            "--unconstrained-simp",
            "--use-soi",
            "--pb-rewrites",
            "--ite-simp",
            "--simp-ite-compress",
        )

    elif logic == "QF_NIA":
        try_with(
            420,
            bench_file,
            cvc5_path,
            out_file,
            "--nl-ext-tplanes",
            "--decision=justification",
        )
        try_with(
            60,
            bench_file,
            cvc5_path,
            out_file,
            "--nl-ext-tplanes",
            "--decision=internal",
        )
        try_with(
            60,
            bench_file,
            cvc5_path,
            out_file,
            "--no-nl-ext-tplanes",
            "--decision=internal",
        )
        try_with(
            60,
            bench_file,
            cvc5_path,
            out_file,
            "--no-arith-brab",
            "--nl-ext-tplanes",
            "--decision=internal",
        )
        try_with(
            300,
            bench_file,
            cvc5_path,
            out_file,
            "--solve-int-as-bv=2",
            "--bitblast=eager",
        )
        try_with(
            300,
            bench_file,
            cvc5_path,
            out_file,
            "--solve-int-as-bv=4",
            "--bitblast=eager",
        )
        try_with(
            300,
            bench_file,
            cvc5_path,
            out_file,
            "--solve-int-as-bv=8",
            "--bitblast=eager",
        )
        try_with(
            300,
            bench_file,
            cvc5_path,
            out_file,
            "--solve-int-as-bv=16",
            "--bitblast=eager",
        )
        try_with(
            600,
            bench_file,
            cvc5_path,
            out_file,
            "--solve-int-as-bv=32",
            "--bitblast=eager",
        )
        finish_with(
            bench_file, cvc5_path, out_file, "--nl-ext-tplanes", "--decision=internal"
        )

    elif logic == "QF_NRA":
        try_with(600, bench_file, cvc5_path, out_file, "--decision=justification")
        try_with(
            300,
            bench_file,
            cvc5_path,
            out_file,
            "--decision=internal",
            "--no-nl-cov",
            "--nl-ext=full",
            "--nl-ext-tplanes",
        )
        finish_with(
            bench_file, cvc5_path, out_file, "--decision=internal", "--nl-ext=none"
        )

    elif logic in [
        "ALIA",
        "AUFLIA",
        "AUFLIRA",
        "AUFNIRA",
        "UF",
        "UFBVLIA",
        "UFBVFP",
        "UFIDL",
        "UFLIA",
        "UFLRA",
        "UFNIA",
        "UFDT",
        "UFDTLIA",
        "UFDTLIRA",
        "AUFDTLIA",
        "AUFDTLIRA",
        "AUFBV",
        "AUFBVDTLIA",
        "AUFBVFP",
        "AUFNIA",
        "UFFPDTLIRA",
        "UFFPDTNIRA",
    ]:
        # initial runs 1 min
        try_with(
            30, bench_file, cvc5_path, out_file, "--simplification=none", "--enum-inst"
        )
        try_with(30, bench_file, cvc5_path, out_file, "--no-e-matching", "--enum-inst")
        try_with(
            30,
            bench_file,
            cvc5_path,
            out_file,
            "--no-e-matching",
            "--enum-inst",
            "--enum-inst-sum",
        )

        # trigger selections 3 min
        try_with(
            30, bench_file, cvc5_path, out_file, "--relevant-triggers", "--enum-inst"
        )
        try_with(
            30, bench_file, cvc5_path, out_file, "--trigger-sel=max", "--enum-inst"
        )
        try_with(
            30,
            bench_file,
            cvc5_path,
            out_file,
            "--multi-trigger-when-single",
            "--enum-inst",
        )
        try_with(
            30,
            bench_file,
            cvc5_path,
            out_file,
            "--multi-trigger-when-single",
            "--multi-trigger-priority",
            "--enum-inst",
        )
        try_with(
            30, bench_file, cvc5_path, out_file, "--multi-trigger-cache", "--enum-inst"
        )
        try_with(
            30,
            bench_file,
            cvc5_path,
            out_file,
            "--no-multi-trigger-linear",
            "--enum-inst",
        )

        # other 4 min 30 sec
        try_with(
            30, bench_file, cvc5_path, out_file, "--pre-skolem-quant=on", "--enum-inst"
        )
        try_with(30, bench_file, cvc5_path, out_file, "--inst-when=full", "--enum-inst")
        try_with(
            30,
            bench_file,
            cvc5_path,
            out_file,
            "--no-e-matching",
            "--no-cbqi",
            "--enum-inst",
        )
        try_with(30, bench_file, cvc5_path, out_file, "--enum-inst", "--quant-ind")
        try_with(
            30,
            bench_file,
            cvc5_path,
            out_file,
            "--decision=internal",
            "--simplification=none",
            "--no-inst-no-entail",
            "--no-cbqi",
            "--enum-inst",
        )
        try_with(
            30,
            bench_file,
            cvc5_path,
            out_file,
            "--decision=internal",
            "--enum-inst",
            "--enum-inst-sum",
        )
        try_with(
            30,
            bench_file,
            cvc5_path,
            out_file,
            "--term-db-mode=relevant",
            "--enum-inst",
        )
        try_with(
            30, bench_file, cvc5_path, out_file, "--enum-inst-interleave", "--enum-inst"
        )
        try_with(
            30,
            bench_file,
            cvc5_path,
            out_file,
            "--preregister-mode=lazy",
            "--enum-inst",
        )

        # finite model find and mbqi 3 min 30 sec
        try_with(
            30,
            bench_file,
            cvc5_path,
            out_file,
            "--finite-model-find",
            "--fmf-mbqi=none",
        )
        try_with(
            30,
            bench_file,
            cvc5_path,
            out_file,
            "--finite-model-find",
            "--decision=internal",
        )
        try_with(
            30,
            bench_file,
            cvc5_path,
            out_file,
            "--finite-model-find",
            "--macros-quant",
            "--macros-quant-mode=all",
        )
        try_with(
            60, bench_file, cvc5_path, out_file, "--finite-model-find", "--e-matching"
        )
        try_with(60, bench_file, cvc5_path, out_file, "--mbqi")

        # long runs 3 min
        try_with(
            180,
            bench_file,
            cvc5_path,
            out_file,
            "--finite-model-find",
            "--decision=internal",
        )
        finish_with(bench_file, cvc5_path, out_file, "--enum-inst")

    elif logic == "UFBV":
        try_with(150, bench_file, cvc5_path, out_file, "--sygus-inst")
        try_with(
            150,
            bench_file,
            cvc5_path,
            out_file,
            "--mbqi",
            "--no-cegqi",
            "--no-sygus-inst",
        )
        try_with(
            300,
            bench_file,
            cvc5_path,
            out_file,
            "--enum-inst",
            "--cegqi-nested-qe",
            "--decision=internal",
        )
        try_with(
            300,
            bench_file,
            cvc5_path,
            out_file,
            "--mbqi-fast-sygus",
            "--no-cegqi",
            "--no-sygus-inst",
        )
        try_with(
            30,
            bench_file,
            cvc5_path,
            out_file,
            "--enum-inst",
            "--no-cegqi-innermost",
            "--global-negate",
        )
        finish_with(bench_file, cvc5_path, out_file, "--finite-model-find")

    elif logic in ["ABV", "BV"]:
        try_with(80, bench_file, cvc5_path, out_file, "--enum-inst")
        try_with(80, bench_file, cvc5_path, out_file, "--sygus-inst")
        try_with(
            80,
            bench_file,
            cvc5_path,
            out_file,
            "--mbqi",
            "--no-cegqi",
            "--no-sygus-inst",
        )
        try_with(
            300,
            bench_file,
            cvc5_path,
            out_file,
            "--mbqi-fast-sygus",
            "--no-cegqi",
            "--no-sygus-inst",
        )
        try_with(
            300,
            bench_file,
            cvc5_path,
            out_file,
            "--enum-inst",
            "--cegqi-nested-qe",
            "--decision=internal",
        )
        try_with(30, bench_file, cvc5_path, out_file, "--enum-inst", "--no-cegqi-bv")
        try_with(
            30,
            bench_file,
            cvc5_path,
            out_file,
            "--enum-inst",
            "--cegqi-bv-ineq=eq-slack",
        )
        finish_with(
            bench_file,
            cvc5_path,
            out_file,
            "--enum-inst",
            "--no-cegqi-innermost",
            "--global-negate",
        )

    elif logic in ["ABVFP", "ABVFPLRA", "BVFP", "FP", "NIA", "NRA", "BVFPLRA"]:
        try_with(
            300,
            bench_file,
            cvc5_path,
            out_file,
            "--mbqi-fast-sygus",
            "--no-cegqi",
            "--no-sygus-inst",
        )
        try_with(
            300, bench_file, cvc5_path, out_file, "--enum-inst", "--nl-ext-tplanes"
        )
        try_with(
            60,
            bench_file,
            cvc5_path,
            out_file,
            "--mbqi",
            "--no-cegqi",
            "--no-sygus-inst",
        )
        finish_with(bench_file, cvc5_path, out_file, "--sygus-inst")

    elif logic in ["LIA", "LRA"]:
        try_with(30, bench_file, cvc5_path, out_file, "--enum-inst")
        try_with(
            300, bench_file, cvc5_path, out_file, "--enum-inst", "--cegqi-nested-qe"
        )
        try_with(
            30,
            bench_file,
            cvc5_path,
            out_file,
            "--mbqi",
            "--no-cegqi",
            "--no-sygus-inst",
        )
        try_with(
            30,
            bench_file,
            cvc5_path,
            out_file,
            "--mbqi-fast-sygus",
            "--no-cegqi",
            "--no-sygus-inst",
        )
        finish_with(
            bench_file,
            cvc5_path,
            out_file,
            "--enum-inst",
            "--cegqi-nested-qe",
            "--decision=internal",
        )

    elif logic == "QF_AUFBV":
        try_with(600, bench_file, cvc5_path, out_file)
        finish_with(bench_file, cvc5_path, out_file, "--decision=stoponly")

    elif logic == "QF_ABV":
        try_with(
            50,
            bench_file,
            cvc5_path,
            out_file,
            "--ite-simp",
            "--simp-with-care",
            "--repeat-simp",
        )
        try_with(500, bench_file, cvc5_path, out_file)
        finish_with(
            bench_file,
            cvc5_path,
            out_file,
            "--ite-simp",
            "--simp-with-care",
            "--repeat-simp",
        )

    elif logic in ["QF_BV", "QF_UFBV"]:
        finish_with(
            bench_file, cvc5_path, out_file, "--bitblast=eager", "--bv-assert-input"
        )

    elif logic == "QF_AUFLIA":
        finish_with(
            bench_file,
            cvc5_path,
            out_file,
            "--no-arrays-eager-index",
            "--arrays-eager-lemmas",
            "--decision=justification",
        )

    elif logic == "QF_AX":
        finish_with(
            bench_file,
            cvc5_path,
            out_file,
            "--no-arrays-eager-index",
            "--arrays-eager-lemmas",
            "--decision=internal",
        )

    elif logic == "QF_AUFNIA":
        finish_with(
            bench_file,
            cvc5_path,
            out_file,
            "--decision=justification",
            "--no-arrays-eager-index",
            "--arrays-eager-lemmas",
        )

    elif logic == "QF_ALIA":
        try_with(140, bench_file, cvc5_path, out_file, "--decision=justification")
        finish_with(
            bench_file,
            cvc5_path,
            out_file,
            "--decision=stoponly",
            "--no-arrays-eager-index",
            "--arrays-eager-lemmas",
        )

    elif logic in ["QF_S", "QF_SLIA"]:
        try_with(
            300,
            bench_file,
            cvc5_path,
            out_file,
            "--strings-exp",
            "--strings-fmf",
            "--no-jh-rlv-order",
        )
        finish_with(
            bench_file, cvc5_path, out_file, "--strings-exp", "--no-jh-rlv-order"
        )

    else:
        # Default case - just run with default settings
        finish_with(bench_file, cvc5_path, out_file)


if __name__ == "__main__":
    main()
