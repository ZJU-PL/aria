#!/usr/bin/env python
from sys import argv
import networkx as nx

from aria.automata.symautomata.flex2fst import Flexparser, mma_2_digraph
from aria.automata.symautomata.dfa import DFA


def main():
    """
    Testing function for Flex Regular Expressions to FST DFA
    """
    if len(argv) < 2:
        print('Usage: %s fst_file' % argv[0])
        return
    flex_a = Flexparser(["a","b","c","d", "/"])
    mma = flex_a.yyparse(argv[1])

    print("---")
    print(mma)
    model_name = "model_" + argv[1].replace('.', '_')
    mma.save(model_name + ".txt")

    graph = mma_2_digraph(mma)
    p = nx.nx_pydot.to_pydot(graph)
    p.write_png(model_name + ".png")

    print("---")
    print("F", mma.consume_input("aba"))
    print("T", mma.consume_input("/aba/"))
    print("F", mma.consume_input("cccc/"))
    print("T", mma.consume_input("/cccc/d"))
    print("F", mma.consume_input("/ccdc/d"))


if __name__ == '__main__':
    main()
