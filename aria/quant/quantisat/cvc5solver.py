"""CVC5 solver interface for satisfiability checking."""

import subprocess
from typing import Dict, Tuple

import regex as re


def cvc5_call(formula: str, timeout: int) -> Tuple[bool, Dict[str, float]]:
    """
    Check if the formula is satisfiable using cvc5 with a timeout.

    Parameters
    ----------
    formula : str
        The formula in smt2 to be checked.
    timeout : int
        The timeout in seconds.

    Returns
    -------
    bool
        The result of the check. True if satisfiable, False if unsatisfiable,
        None if unknown.
    Dict[str, float]
        The model.
    """

    # Replace all negative numbers with (- x)
    formula = re.sub(r"(?<!\w)-(\d+(\.\d+)?)", r"(- \g<1>)", formula)
    print(formula)

    cmd = ["cvc5", "--lang=smt2", "--produce-models", f"--tlimit={timeout * 1000}"]
    with subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        process.stdin.write(formula.encode())
        process.stdin.close()

        try:
            process.wait(timeout + 1)
        except subprocess.TimeoutExpired:
            process.kill()
            return None, {}

        output = process.stdout.read().decode()

    if "unsat" in output:
        return False, {}
    if "sat" in output:
        model = {}
        for line in output.split("\n"):
            if line.startswith("(define-"):
                print(line)
                print(line.split(maxsplit=4))
                _, var, _, _, val = line.split(maxsplit=4)
                model[var] = val
        return True, model
    return None, {}
