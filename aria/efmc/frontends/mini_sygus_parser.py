"""
Parser for SyGuS files based on a simple S-expression parser

NOTE: I only this file for inputs in the style of the SyGuS invariant track.
It could be used for SyGuS(PBE). To keep this file independent,
I do not important other files in this project.

This parser handles:
- S-expression parsing
- Grammar extraction
- Transition system conversion
- Support for multiple data types (Int, Real, Array, BitVec)
"""

import z3
from typing import Union, List

# Type definitions
Symbol = str
Number = Union[int, float]
Atom = Union[Symbol, Number]
Expr = Union[Atom, List]


class SyGuSParsingError(Exception):
    """Custom exception for SyGuS parsing errors"""

    pass


def input_to_list(string: str) -> List[str]:
    """
    Parse a .sl file into a list of S-Expressions.
    """
    n: int = 0
    result: List[str] = []
    s: str = ""
    for c in string:
        if c == "(":
            n += 1
        if c == ")":
            n -= 1
        if c != "\n":
            s += c
        if n == 0 and s != "":
            result.append(s)
            s = ""
    return result


def tokenize(chars: str) -> list:
    """Convert a string of characters into a list of tokens."""

    return chars.replace("(", " ( ").replace(")", " ) ").replace('" "', "space").split()


def parse_sexpression(program: str) -> Expr:
    """Read an S-expression from a string."""
    return read_from_tokens(tokenize(program))


def read_from_tokens(tokens: list) -> Expr:
    """Read an expression from a sequence of tokens."""
    if len(tokens) == 0:
        return
        # raise SyntaxError('unexpected EOF') # is this OK?
    token = tokens.pop(0)
    if token == "(":
        L = []
        while tokens and tokens[0] != ")":
            L.append(read_from_tokens(tokens))
        tokens.pop(0)  # pop off ')'
        return L
    elif token == ")":
        raise SyntaxError("unexpected )")
    else:
        return atom(token)


def atom(token: str) -> Atom:
    """Numbers become numbers; every other token is a symbol."""
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return Symbol(token)


def get_start(cmd) -> str:
    assert cmd[0] == "synth-fun"
    return cmd[4][0]


def get_nonterminals(cmd):
    nonterms = set()
    type_dict = {}
    assert cmd[0] == "synth-fun"
    for elem in cmd[4]:
        nt = elem[0]
        typ = elem[1]
        nonterms.add(nt)
        type_dict[nt] = typ
    return nonterms, type_dict


def get_terms_prods(cmd):
    terminals = set()
    productions = {}
    for nonterm_list in cmd[5]:
        non_term = nonterm_list[0]
        product = []
        for t in nonterm_list[2]:
            # Non-terminals that go with a production
            nts = []

            if isinstance(t, list):
                # gets a terminal from the input, e.g., `"+"`
                term = t[0]
                # List of potential non-terminal children, e.g, `["I", "I"]` for `"+"`
                nts += t[1:]
            else:
                term = t

            if isinstance(term, list):
                term = tuple(term)
            terminals.add(term)

            product.append((term, nts))
        productions[non_term] = product
    return terminals, productions


def get_grammar(lines: List[str]):
    s_exprs = []
    for line in lines:
        s_exprs.append(parse_sexpression(line))
    for s in s_exprs:
        if s[0] == "synth-fun":
            start_sym = get_start(s)
            nonterminals = get_nonterminals(s)
            terminals, productions = get_terms_prods(s)
            return nonterminals, terminals, productions, start_sym
    raise ValueError("No 'synth-fun' command found in the input.")


pre_func_names = ["pre_fun", "PreF", "pre-f"]
trans_func_names = ["trans_fun", "TransF", "trans-f"]
post_func_names = ["post_fun", "PostF", "post-f"]


class SyGusInVParser:
    """
    Parser for SyGuS Invariant Track problems.
    """

    def __init__(self, inputs: str, to_real: bool):
        # print("to real? ", to_real)
        self.to_real = to_real
        self.logic = None
        self.inv_vars = []
        self.all_vars = []
        self.inv_arity = 0
        self.pre_fun_body = ""
        self.trans_fun_body = ""
        self.post_fun_body = ""

        self.init_symbols(inputs)

    def to_sexpr_misc(self, lines: List[str]):
        """
        E.g.,
        ['and', ['=', 'x', 1], ['=', 'y', 1]]
        ['and', ['=', 'x!', ['+', 'x', 'y']], ['=', 'y!', ['+', 'x', 'y']]]
        """
        res = ["("]
        for element in lines:
            if isinstance(element, list):
                for e in self.to_sexpr_misc(element):
                    res.append(str(e))
            else:
                # TODO: what if element is real/float (just use str(element))?
                if isinstance(element, int):
                    if self.to_real:
                        element = str(element) + ".0"
                    else:
                        element = str(element)
                res.append(element)
        res.append(")")
        return res

    def to_sexpr(self, lines: List[str]):
        return " ".join(self.to_sexpr_misc(lines))

    def init_symbols(self, inputs: str):
        """
        TODO: this is not robust (as an sexpr can have several lines)
        """
        lines = input_to_list(inputs)
        for line in lines:
            slist = parse_sexpression(line)
            # print(slist)
            if isinstance(slist, List):
                cmd_name = slist[0]
                if cmd_name == "set-logic":
                    self.logic = slist[1]
                elif cmd_name == "synth-inv":
                    self.inv_arity = len(slist[2])
                elif cmd_name == "define-fun":
                    self.process_func(slist)
                elif cmd_name == "inv-constraint":
                    break
                else:
                    break
        # print(self.pre_fun_body)
        # print(self.trans_fun_body)
        # print(self.post_fun_body)

    def process_func(self, slist):
        """Process a 'define-fun' command to extract function bodies and variables."""
        # print(slist)
        assert len(slist) >= 5  # why?
        func_name = slist[1]
        if func_name in pre_func_names:
            self.pre_fun_body = self.to_sexpr(slist[4])
            self.inv_vars = slist[2]
        elif func_name in trans_func_names:
            self.trans_fun_body = self.to_sexpr(slist[4])
            self.all_vars = slist[2]
        elif func_name in post_func_names:
            self.post_fun_body = self.to_sexpr(slist[4])
        else:
            # print("Function name not recognized!:", func_name)
            raise AssertionError(f"Function name not recognized!: {func_name}")

    def get_transition_system(self):
        """
        Return the format of our trivial transition system
        """
        # from z3.z3util import get_vars
        init_constraints = []
        trans_constraints = []
        post_constraints = []
        for var in self.all_vars:
            var_name, var_type = var[0], var[1]
            # print(var_name, var_type)
            if isinstance(var_type, List):
                # E.g., ['Array', 'Int', 'Int']
                # E.g., ['_', 'BitVec', 32] (32 should be converted to str)
                # print(var_type)
                var_type_final = []
                for t in var_type:
                    if isinstance(t, str):
                        var_type_final.append(t)
                    else:
                        var_type_final.append(str(t))
                final_type = "({})".format(" ".join(var_type_final))
            else:
                # FIXME: to test abstract domains, we may cast Int types to Real
                # FIXME: but casting here is not very elegant
                if self.to_real and var_type != "Real":
                    final_type = "Real"
                else:
                    final_type = var_type

            init_constraints.append(f"(declare-const {var_name} {final_type})")
            trans_constraints.append(f"(declare-const {var_name} {final_type})")
            post_constraints.append(f"(declare-const {var_name} {final_type})")

        init_constraints.append("(assert {})".format(self.pre_fun_body))
        trans_constraints.append("(assert {})".format(self.trans_fun_body))
        post_constraints.append("(assert {})".format(self.post_fun_body))

        # print("\n".join(init_constraints))
        # print("\n".join(trans_constraints))
        # print("\n".join(post_constraints))

        init = z3.parse_smt2_string("\n".join(init_constraints))[0]
        trans = z3.parse_smt2_string("\n".join(trans_constraints))[0]
        post = z3.parse_smt2_string("\n".join(post_constraints))[0]

        # a dirty trick to keep order?
        # TODO: does the following functions preserve some order?
        #  Is the order necessary? (x, y, x!, y!, ..)
        """
        # it's possible that some variables in self.all_vars are not used
        z3_vars = get_vars(z3.And(init, trans, post))
        z3_vars_names = [str(var) for var in z3_vars]
        temp_cnts = []
        for var_sig in self.all_vars:  # ['i', 'Int']
            if var_sig[0] not in z3_vars_names:
                if self.to_real:
                    temp_cnts.append(z3.Real(var_sig[0]) == z3.RealVal(1.0))
                else:
                    temp_cnts.append(z3.Int(var_sig[0]) == z3.IntVal(1))
        all_vars = get_vars(z3.And(z3.And(temp_cnts), z3.And(init, trans, post)))
        """
        all_vars = []  # another way for collecting the signature
        for var_sig in self.all_vars:  # ['i', 'Int']
            if "Int" in var_sig:
                if self.to_real:
                    all_vars.append(z3.Real(var_sig[0]))
                else:
                    all_vars.append(z3.Int(var_sig[0]))
            elif "Real" in var_sig:
                all_vars.append(z3.Real(var_sig[0]))
            else:  # TODO: should we just assume the type is bv?
                # e.g., var_sig is ['x', ['_', 'BitVec', 32]]
                #  then, the z3 var is BitVec(x, 32)
                all_vars.append(z3.BitVec(var_sig[0], var_sig[1][2]))

        return all_vars, init, trans, post


def demo_parser():
    tt = [
        ";\n",
        "(set-logic LIA)",
        " (synth-inv inv_fun ((x Int) (y Int)))\n",
        "(define-fun pre_fun ((x Int) (y Int)) Bool (and (= x 1) (= y 1))) ",
        "(define-fun trans_fun ((x Int) (y Int) (x! Int) (y! Int)) "
        "Bool (and (= x! (+ x y)) (= y! (+ x y))))",
        "(define-fun post_fun ((x Int) (y Int)) Bool (>= y 1))",
        "(inv-constraint inv_fun pre_fun trans_fun post_fun)",
        "(check-synth)" "; xxx",
    ]

    # ss = SyGusInVParser("\n".join(tt_arr), to_real=False)
    ss = SyGusInVParser(tt, to_real=False)
    _, init, trans, post = ss.get_transition_system()
    print(init, "\n", trans, "\n", post)
    # print(input_to_list(tttt))


if __name__ == "__main__":
    demo_parser()
