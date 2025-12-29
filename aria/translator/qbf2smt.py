"""
Converting QDIMACS format to smtlib
This one is a bit tricky, as it uses bit-vector variables to "compactly" encode several Booleans.
"""

import sys
import os


def error(msg):
    """Print an error message and exit."""
    sys.stderr.write(f"{sys.argv[0]} : {msg}.{os.linesep}")
    sys.exit(1)


def spacesplit(string):
    """Split a string by spaces and filter out empty strings."""
    return list(filter(None, string.split(" ")))


def tointlist(lst):
    """
    Converts a list of strings to a list of integers, and checks that it's 0-terminated.

    Args:
        lst (List[str]): The list to convert.

    Returns:
        List[int]: The list with strings converted to integers and 0 removed.

    Raises:
        ValueError: If the list is not a 0-terminated list of integers.
    """
    try:
        ns = [int(x) for x in lst]
        if not ns[-1] == 0:
            error("expected 0-terminated number list")
        return ns[:-1]

    except ValueError:
        error(f"expected number list (got: {lst})")


def parse(filename):
    """
    Parses a QDIMACS file and outputs its equivalent in SMT-LIB2 format, using UFBV logic.
    """
    with open(filename, encoding='utf-8') as f:
        printed_comments = False
        seendesc = False
        overprefix = False
        mapping = {}
        level = 0

        for line in f.readlines():
            trimmed = line.strip()
            if trimmed.startswith("c"):
                # Comment
                printed_comments = True
                print(f"; {trimmed[1:].strip()}")
            elif trimmed.startswith("p"):
                # Problem definition
                if seendesc:
                    error("multiple problem description lines")
                else:
                    seendesc = True

                infoparts = spacesplit(trimmed[1:])

                if not len(infoparts) == 3:
                    error("unexpected problem description (not 3 parts?)")

                probformat = infoparts[0]
                probvcstr = infoparts[1]
                probccstr = infoparts[2]

                if not probformat == "cnf":
                    error(f"unexpected problem format ('{probformat}', not cnf?)")

                if not probvcstr.isdigit():
                    error(f"variable count is not a number ({probvcstr})")
                varcount = int(probvcstr)

                if not probccstr.isdigit():
                    error(f"clause count is not a number ({probccstr})")
                clausecount = int(probccstr)

                if printed_comments:
                    print(";")

                print(f"; QBF variable count : {varcount}")
                print(f"; QBF clause count   : {clausecount}")
                print("")
                print("(set-logic UFBV)")
                print("(assert")
            elif trimmed.startswith("a") or trimmed.startswith("e"):
                # Quantifier definition
                if overprefix:
                    error("unexpected quantifier declaration")
                isuniversal = trimmed.startswith("a")
                level = level + 1
                parts = spacesplit(trimmed[1:])

                vs = tointlist(parts)

                for i, v in enumerate(vs):
                    if v in mapping:
                        error(f"variable {v} bound multiple times")
                    mapping[v] = (level, i)

                quant = "forall" if isuniversal else "exists"

                print(f"  ({quant} ((vec{level} (_ BitVec {len(vs)})))")
            else:
                # Clause definition
                if not overprefix:
                    print("    (and")
                overprefix = True

                vs = tointlist(spacesplit(trimmed))

                sys.stdout.write("      (or")

                for v in vs:
                    a = abs(v)
                    (lvl, i) = mapping[a]
                    bit_val = 1 if a == v else 0
                    sys.stdout.write(
                        f" (= ((_ extract {i} {i}) vec{lvl}) #b{bit_val})"
                    )

                sys.stdout.write(f"){os.linesep}")

    print("    )")
    print(f"  {')' * level}")
    print(")")
    print("")
    print("(check-sat)")
    return 0


def main(argv):
    """Main function for command-line execution."""
    if len(argv) < 2:
        error("expected file argument")

    parse(argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
