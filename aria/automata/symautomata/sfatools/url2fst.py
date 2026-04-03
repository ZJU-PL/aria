#!/usr/bin/env python
from sys import argv
import networkx as nx
import string

from aria.automata.symautomata.flex2fst import (
    Flexparser,
    mma_2_digraph,
    simplify_digraph,
)
from aria.automata.symautomata.flex2fst import mma_trace_2_digraph
from aria.automata.symautomata.dfa import DFA


# https://en.wikipedia.org/wiki/List_of_Unicode_characters
def get_common_unicode():
    # basic_latin = list(range(0x22, 0x7E+1))
    # return [chr(a) for a in basic_latin]
    return list(string.ascii_letters) + list(string.digits) + list(":/@#?=+-_.%&")

    # return basic_latin + new_lines


def main():
    """
    Testing function for Flex Regular Expressions to FST DFA
    """
    if len(argv) < 2:
        print("Usage: %s fst_file" % argv[0])
        return
    flex_a = Flexparser(get_common_unicode())
    mma = flex_a.yyparse(argv[1])

    print("---")
    print(mma)
    model_name = "model_" + argv[1].replace(".", "_")
    mma.save(model_name + ".txt")

    # graph = mma_2_digraph(mma)
    # graph = simplify_digraph(graph)

    # p = nx.nx_pydot.to_pydot(graph)
    # p.write_png(model_name + ".png")

    print("---")
    print("F", mma.consume_input("aba"))
    print("T", mma.consume_input("http://abcde.com/"))
    print("T", mma.consume_input("aaaa://abcde.com/"))
    print("F", mma.consume_input("9a://abcde.com/"))
    print("T", mma.consume_input("http://abcde.com:88/"))
    print("T", mma.consume_input("http://abcde.com:88/acf%123"))
    print("T", mma.consume_input("http://abcde.com/abc#ddd"))
    print("T", mma.consume_input("http://user:passwd@abcde.com/abc#ddd"))
    print("T", mma.consume_input("http://user@abcde.com/abc#ddd"))
    print("F", mma.consume_input("http://abcde.com/abc##ddd"))

    url_prefixes = [
        "http",
        "://",
        "user",
        ":",
        "passwd",
        "@",
        "domain.com",
        ":",
        "88",
        "/",
        "path",
        "#",
        "frag",
    ]
    traces = []
    for i in range(len(url_prefixes) - 1):
        trace = mma.trace_partial_input("".join(url_prefixes[: i + 1]))
        traces.append(trace)
        # print(trace)

    colors = [
        "aqua",
        "blue",
        "bisque",
        "brown",
        "chartreuse",
        "crimson",
        "darkorchid",
        "deeppink",
        "tomato",
        "lime",
        "orange",
        "cyan",
        "fuchsia",
        "gold",
    ]

    graph = mma_trace_2_digraph(mma, traces, colors)
    graph = simplify_digraph(graph, mma)
    p = nx.nx_pydot.to_pydot(graph)
    p.write_png(model_name + ".png")


if __name__ == "__main__":
    main()
