"""
Converting CNF to Linear Programming
"""

import sys
from pysat.formula import CNF


def cnf2lp(inf=None, outf=None):
    """Convert a CNF file to Linear Programming format."""
    if inf is None:
        return
    f = CNF(inf)
    if outf is not None:
        with open(outf, "w", encoding="utf-8") as wf:
            for cls in f.clauses:
                head = " | ".join(["p" + str(x) for x in cls if x > 0])
                body = ", ".join(["p" + str(-x) for x in cls if x < 0])
                if body != "":
                    head = head + " :- "
                print(head + body + ".", file=wf)
    else:
        for cls in f.clauses:
            head = " | ".join(["p" + str(x) for x in cls if x > 0])
            body = ", ".join(["p" + str(-x) for x in cls if x < 0])
            if body != "":
                head = head + " :- "
            print(head + body + ".", file=sys.stdout)


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print(f"usage {sys.argv[0]} INPUT_CNF [OUTPUT_LP]")
        sys.exit()
    if len(sys.argv) == 2:
        cnf2lp(sys.argv[1])
    else:
        cnf2lp(sys.argv[1], sys.argv[2])
