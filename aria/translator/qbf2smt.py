"""
Converting QDIMACS format to smtlib
This one is a bit tricky, as it uses bit-vector variables to "compactly" encode several Booleans.
"""

import sys
import os


def error(msg):
    sys.stderr.write("%s : %s.%s" % (sys.argv[0], msg, os.linesep))
    sys.exit(1)


def spacesplit(string):
    return list(filter(lambda x: x, string.split(" ")))
    # return filter(lambda x: x, string.split(" "))


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
        # ns = map(lambda x: int(x), lst)
        ns = list(map(lambda x: int(x), lst))
        if not ns[-1] == 0:
            error("expected 0-terminated number list")
        return ns[:-1]

    except ValueError:
        error("expected number list (got: %s)" % str(lst))


def parse(filename):
    """
    Parses a QDIMACS file and outputs its equivalent in SMT-LIB2 format, using UFBV logic.
    """
    with open(filename) as f:
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
                print("; %s" % trimmed[1:].strip())
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
                    error("unexpected problem format ('%s', not cnf?)" % probformat)

                if not probvcstr.isdigit():
                    error("variable count is not a number (%s)" % probvcstr)
                else:
                    varcount = int(probvcstr)

                if not probccstr.isdigit():
                    error("clause count is not a number (%s)" % probccstr)
                else:
                    clausecount = int(probccstr)

                if printed_comments:
                    print(";")

                print("; QBF variable count : %d" % varcount)
                print("; QBF clause count   : %d" % clausecount)
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
                        error("variable %d bound multiple times" % v)
                    mapping[v] = (level, i)

                quant = "forall" if isuniversal else "exists"

                print("  (%s ((vec%d (_ BitVec %d)))" % (quant, level, len(vs)))
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
                    sys.stdout.write(" (= ((_ extract %d %d) vec%d) #b%d)" % (i, i, lvl, 1 if a == v else 0))

                sys.stdout.write(")%s" % os.linesep)

    print("    )")
    print("  %s" % (")" * level))
    print(")")
    print("")
    print("(check-sat)")
    return 0


def main(argv):
    if len(argv) < 2:
        error("expected file argument")
        return 1

    parse(argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
