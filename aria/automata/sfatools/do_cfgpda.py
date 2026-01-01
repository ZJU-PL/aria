from aria.automata.symautomata.cfgpda import CfgPDA as _CfgPDA
from sys import argv


def get_common_unicode():
    new_lines = [0, ord("\r"), ord("\n")]
    basic_latin = list(range(0x20, 0x7E + 1))
    return [chr(a) for a in basic_latin]


def get_char():
    # return [chr(a) for a in list(range(ord("a"), ord("z")+1))]
    return [chr(a) for a in list(range(ord("a"), ord("e") + 1))]


def main():
    """
    Testing function for Flex Regular Expressions to FST DFA
    """
    if len(argv) < 2:
        print("Usage: %s fst_file [optional: save_file]" % argv[0])
        return
    # cfgtopda = _CfgPDA(["a","b","c","d"])
    # print(get_common_unicode())
    # exit()
    # cfgtopda = _CfgPDA(get_common_unicode())
    cfgtopda = _CfgPDA(get_char())
    mma = cfgtopda.yyparse(argv[1], 1)
    # mma.minimize()
    # print mma
    # if len(argv) == 3:
    #    mma.save(argv[2])

    # mma.printer()
    # exit()

    print(mma.consume_input("ab"))
    exit()
    print(mma.consume_input("abcc"))
    print(mma.consume_input("aa"))
    exit()
    print(mma.consume_input("https://abc"))
    print(mma.consume_input("https"))
    print(mma.consume_input("aaaaa"))


if __name__ == "__main__":
    main()
