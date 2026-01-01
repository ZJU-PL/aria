"""
This module transforms a file containing regular expressions for flex parser
into a DFA object (pyfst automaton)
"""

import os
import random
import re
import string
import tempfile
import networkx as nx
from operator import attrgetter
from subprocess import call
from sys import argv

from aria.automata.symautomata.alphabet import createalphabet
from aria.automata.symautomata.dfa import DFA


class Flexparser:
    """
    This class parses compiles a file containing regular
    expressions into a flex compiled file, and then parses
    the flex DFA into a pyfst DFA
    """

    outfile = ""

    def __init__(self, alphabet=None):
        """
        Initialization function
        Args:
            alphabet (list): input alphabet
        Returns:
            None
        """
        if alphabet is not None:
            self.alphabet = alphabet
        else:
            self.alphabet = []

    def _create_automaton_from_regex(self, myfile):
        """
        Generates a flex compiled file using myfile
        as input
        Args:
            myfile (str): Flex file to be parsed
        Returns:
            None
        """
        call(["flex", "-o", self.outfile, "-f", myfile])

    def _read_transitions(self):
        """
        Read DFA transitions from flex compiled file
        Args:
            None
        Returns:
            list: The list of states and the destination for a character
        """
        states = []
        i = 0
        regex = re.compile("[ \t\n\r:,]+")
        found = 0  # For maintaining the state of yy_nxt declaration
        state = 0  # For maintaining the state of opening and closing tag of yy_nxt
        substate = 0  # For maintaining the state of opening and closing tag of each set in yy_nxt
        mapping = []  # For writing each set of yy_next
        cur_line = None
        with open(self.outfile) as flex_file:
            for cur_line in flex_file:
                if (
                    cur_line[0:35] == "static yyconst flex_int16_t yy_nxt["
                    or cur_line[0:33] == "static const flex_int16_t yy_nxt["
                ):
                    found = 1
                    # print 'Found yy_next declaration'
                    continue
                if found == 1:
                    if state == 0 and cur_line[0:5] == "    {":
                        state = 1
                        continue
                    if state == 1 and cur_line[0:7] == "    } ;":
                        state = 0
                        break

                    if substate == 0 and cur_line[0:5] == "    {":
                        mapping = []
                        substate = 1
                        continue
                    if substate == 1:
                        if cur_line[0:6] != "    },":
                            cur_line = "".join(cur_line.split())
                            if cur_line == "":
                                continue
                            if cur_line[cur_line.__len__() - 1] == ",":
                                splitted_line = regex.split(
                                    cur_line[: cur_line.__len__() - 1]
                                )
                            else:
                                splitted_line = regex.split(cur_line)
                            mapping = mapping + splitted_line
                            continue
                        else:
                            cleared = []
                            for j in mapping:
                                cleared.append(int(j))
                            states.append(cleared)
                            mapping = []
                            substate = 0

        return states

    def _read_accept_states(self):
        """
        Read DFA accepted states from flex compiled file
        Args:
            None
        Returns:
            list: The list of accepted states
        """
        states = []
        i = 0
        regex = re.compile("[ \t\n\r:,]+")
        found = 0  # For maintaining the state of yy_accept declaration
        state = 0  # For maintaining the state of opening and closing tag of yy_accept
        mapping = []  # For writing each set of yy_accept
        cur_line = None
        with open(self.outfile) as flex_file:
            for cur_line in flex_file:
                if (
                    cur_line[0:37] == "static yyconst flex_int16_t yy_accept"
                    or cur_line[0:35] == "static const flex_int16_t yy_accept"
                ):
                    found = 1
                    continue
                if found == 1:
                    # print x
                    if state == 0 and cur_line[0:5] == "    {":
                        mapping.append(0)  # there is always a zero there
                        state = 1
                        continue

                    if state == 1:
                        if cur_line[0:7] != "    } ;":
                            cur_line = "".join(cur_line.split())
                            if cur_line == "":
                                continue
                            if cur_line[cur_line.__len__() - 1] == ",":
                                splitted_line = regex.split(
                                    cur_line[: cur_line.__len__() - 1]
                                )
                            else:
                                splitted_line = regex.split(cur_line)
                            mapping = mapping + splitted_line
                            continue
                        else:
                            cleared = []
                            for j in mapping:
                                cleared.append(int(j))
                            return cleared
        return []

    def _x_read_accept_states(self):
        """
        Read DFA accepted states from flex compiled file
        Args:
            None
        Returns:
            list: The list of accepted states
        """
        states = []
        i = 0
        regex = re.compile("[ \t\n\r:,]+")
        found = 0  # For maintaining the state of yy_accept declaration
        state = 0  # For maintaining the state of opening and closing tag of yy_accept
        mapping = []  # For writing each set of yy_accept
        cur_line = None
        with open(self.outfile) as flex_file:
            for cur_line in flex_file:
                if (
                    cur_line[0:37] == "static yyconst flex_int16_t yy_accept"
                    or cur_line[0:35] == "static const flex_int16_t yy_accept"
                ):
                    found = 1
                    continue
                if found == 1:
                    # print x
                    if state == 0 and cur_line[0:5] == "    {":
                        mapping.append(0)  # there is always a zero there
                        state = 1
                        continue

                    if state == 1:
                        if cur_line[0:7] != "    } ;":
                            cur_line = "".join(cur_line.split())
                            if cur_line == "":
                                continue
                            if cur_line[cur_line.__len__() - 1] == ",":
                                splitted_line = regex.split(
                                    cur_line[: cur_line.__len__() - 1]
                                )
                            else:
                                splitted_line = regex.split(cur_line)
                            mapping = mapping + splitted_line
                            continue
                        else:
                            cleared = []
                            for j in mapping:
                                cleared.append(int(j))
                            max_value = max(cleared)
                            for i in range(0, len(cleared)):
                                if cleared[i] > 0 and cleared[i] < (max_value - 1):
                                    states.append(i)
                            return states
        return []

    def _read_null_transitions(self):
        """
        Read DFA'input_string NULL transitions from flex compiled file
        Args:
            None
        Returns:
            list: The list of state transitions for no character
        """
        states = []
        regex = re.compile("[ \t\n\r:,]+")
        found = 0  # For maintaining the state of yy_NUL_trans declaration
        state = (
            0  # For maintaining the state of opening and closing tag of yy_NUL_trans
        )
        mapping = []  # For writing each set of yy_NUL_trans
        cur_line = None
        with open(self.outfile) as flex_file:
            for cur_line in flex_file:
                if (
                    cur_line[0 : len("static yyconst yy_state_type yy_NUL_trans")]
                    == "static yyconst yy_state_type yy_NUL_trans"
                    or cur_line[0 : len("static const yy_state_type yy_NUL_trans")]
                    == "static const yy_state_type yy_NUL_trans"
                ):
                    found = 1
                    # print 'Found yy_next declaration'
                    continue
                if found == 1:
                    if state == 0 and cur_line[0:5] == "    {":
                        mapping.append(0)  # there is always a zero there
                        state = 1
                        continue
                    if state == 1:
                        if cur_line[0:7] != "    } ;":
                            cur_line = "".join(cur_line.split())
                            if cur_line == "":
                                continue
                            if cur_line[cur_line.__len__() - 1] == ",":
                                splitted_line = regex.split(
                                    cur_line[: cur_line.__len__() - 1]
                                )
                            else:
                                splitted_line = regex.split(cur_line)
                                #  print y
                            mapping = mapping + splitted_line
                            continue
                        else:
                            cleared = []
                            for j in mapping:
                                cleared.append(int(j))
                            states = cleared
                            mapping = []
                            state = 0

        return states

    def _create_states(self, states_num):
        """
        Args:
            states_num (int): Number of States
        Returns:
            list: An initialized list
        """
        states = []
        for i in range(0, states_num):
            states.append(i)
        return states

    def _add_sink_state(self, states):
        """
        This function adds a sing state in the total states
        Args:
            states (list): The current states
        Returns:
            None
        """
        cleared = []
        for i in range(0, 128):
            cleared.append(-1)
        states.append(cleared)

    def _create_delta(self):
        """
        This function creates the delta transition
        Args:
            startState (int): Initial state of automaton
        Results:
            int, func: A number indicating the total states, and the delta function
        """
        states = self._read_transitions()
        total_states = len(states)
        self._add_sink_state(states)
        nulltrans = self._read_null_transitions()

        def delta(current_state, character):
            """
            Sub function describing the transitions
            Args:
                current_state (str): The current state
                character (str): The input character
            Returns:
                str: The next state
            """
            if character != "":
                newstate = states[current_state][ord(character)]
                return newstate
                # if newstate > 0:
                #    return newstate
                # else:
                #    return total_states
            else:
                return nulltrans[current_state]

        return total_states + 1, delta

    def yyparse(self, lexfile):
        """
        Args:
            lexfile (str): Flex file to be parsed
        Returns:
            DFA: A dfa automaton
        """
        temp = tempfile.gettempdir()
        self.outfile = (
            temp
            + "/"
            + "".join(
                random.choice(string.ascii_uppercase + string.digits) for _ in range(5)
            )
            + "_lex.yy.c"
        )
        self._create_automaton_from_regex(lexfile)
        states_num, delta = self._create_delta()
        states = self._create_states(states_num)
        # print(states)
        accepted_states = self._read_accept_states()
        # print(accepted_states)
        if self.alphabet != []:
            alphabet = self.alphabet
        else:
            alphabet = createalphabet()
        mma = DFA(alphabet)
        mma.yy_accept = accepted_states
        for state in states:
            if state != 0:
                # print ""
                # print state in accepted_states
                for char in alphabet:
                    # TODO: yy_last_accepting_state impl
                    # Normally, if ( yy_accept[yy_current_state] ), (yy_last_accepting_state) = yy_current_state.
                    # When yy_act == 0, will return yy_accept[yy_last_accepting_state],
                    # But since we're looping here, don't know how to do
                    nextstate = delta(state, char)
                    if nextstate > 0:
                        mma.add_arc(state, nextstate, char)
                        # print("add", state, nextstate, char)
                    else:
                        nextstate = -nextstate
                        yy_act = accepted_states[nextstate]
                        if yy_act == 1:
                            # print("is fin", state, char)
                            mma.states[state].final = True
                        elif yy_act == 0:
                            # print("Lookback:", state, char, yy_act)
                            mma.add_arc(state, 0, char)
                        elif yy_act == 2:
                            # print("Dead:", state, nextstate, char, yy_act)
                            pass
                        else:
                            # print("TODO:", state, nextstate, char, yy_act)
                            pass
                # if state in accepted_states:
                #    mma[state - 1].final = True
        if os.path.exists(self.outfile):
            os.remove(self.outfile)
        mma.states[1].initial = True
        return mma


def mma_2_digraph(mma):
    G = nx.DiGraph()
    states = sorted(mma.states, key=attrgetter("initial"), reverse=True)
    # states = sorted(mma.states, reverse=True)

    for state in states:
        # Separate node creation so that can mark node correctly
        if state.stateid not in G:
            if state.final:
                G.add_node(state.stateid, shape="doublecircle")
            elif state.initial:
                G.add_node(state.stateid, shape="box")
            else:
                G.add_node(state.stateid)
    for state in states:
        for arc in state.arcs:
            if arc.nextstate not in G:
                G.add_node(arc.nextstate)
            itext = mma.isyms.find(arc.ilabel)
            # print(itext)
            label = itext
            # print(label)
            if G.has_edge(state.stateid, arc.nextstate):
                cur_l = G.get_edge_data(state.stateid, arc.nextstate)["label"]
                if label not in cur_l:
                    label += cur_l
            G.add_edge(state.stateid, arc.nextstate, label=label)
    return G


def mma_trace_2_digraph(mma, traces, colors):
    # Construct the base digraph
    G = mma_2_digraph(mma)

    # Start coloring with traces
    cur_union = traces[-1]
    for trace_id in range(len(traces) - 1, -1, -1):
        trace = traces[trace_id]
        color = colors[trace_id]
        new_union = []
        for t in trace:
            if t in cur_union:
                new_union.append(t)
        for t in new_union:
            G.nodes[t[0]]["color"] = color
            G.nodes[t[0]]["style"] = "filled"

        for edge in G.edges():
            for t, s in new_union:
                if t == edge[0] and s in G.edges[edge]["label"]:
                    G.edges[edge]["color"] = color

        cur_union = new_union

    return G


def simplify_digraph(G, mma):
    init_states = []
    for state in mma.states:
        if state.initial:
            init_states.append(state.stateid)

    # Drop nodes that are not starting states and no incoming edges
    self_loops = set()
    for edge in G.edges():
        if edge[0] == edge[1]:
            self_loops.add(edge[0])

    if init_states:
        useless_states = []
        for node_id, in_degree in G.in_degree():
            if (
                in_degree == 0 or (in_degree == 1 and node_id in self_loops)
            ) and node_id not in init_states:
                useless_states.append(node_id)
        G.remove_nodes_from(useless_states)

    # Shrink labels
    def do_replace(label, target, sub):
        if target in label:
            label = label.replace(target, sub)
        if target[::-1] in label:
            label = label.replace(target[::-1], sub)
        return label

    for n, nbrsdict in G.adjacency():
        for nbr, eattr in nbrsdict.items():
            label = eattr["label"]

            label = do_replace(label, string.ascii_uppercase, "{A-Z}")
            label = do_replace(label, string.ascii_lowercase, "{a-z}")
            label = do_replace(label, string.digits, "{0-9}")
            label = do_replace(label, "abcdef", "{a-f}")
            label = do_replace(label, "abcdef".upper(), "{a-f}".upper())
            label = do_replace(
                label, string.ascii_lowercase.replace("abcdef", ""), "{!a-f}"
            )
            label = do_replace(
                label, string.ascii_uppercase.replace("ABCDEF", ""), "{!A-F}"
            )

            eattr["label"] = label
    return G


def main():
    """
    Testing function for Flex Regular Expressions to FST DFA
    """
    if len(argv) < 2:
        print("Usage: %s fst_file [optional: save_file]" % argv[0])
        return
    flex_a = Flexparser(["a", "b", "c", "d", "/"])
    mma = flex_a.yyparse(argv[1])
    # mma.minimize()
    print(mma)
    if len(argv) == 3:
        mma.save(argv[2] + ".txt")

        graph = mma_2_digraph(mma)
        p = nx.nx_pydot.to_pydot(graph)
        # p.write_dot(args.output)
        p.write_png(argv[2] + ".png")

    print("F", mma.consume_input("aba"))
    print("T", mma.consume_input("/aba/"))
    print("F", mma.consume_input("cccc/"))
    print("T", mma.consume_input("/cccc/d"))
    print("F", mma.consume_input("/ccdc/d"))


if __name__ == "__main__":
    main()
