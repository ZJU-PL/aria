"""
Parse EFSMT instances in SMT-LIB2 files
We provide two differnet implementations
1. Use a customized s-expression parser
2. Use z3's substitution facility
"""

import z3
from aria.utils.z3_expr_utils import get_variables

# Being explicit about Types
Symbol = str
NUMBER = (int, float)
Atom = (Symbol, NUMBER)
List = list
Expr = (Atom, List)


def input_to_list(string: str) -> [str]:
    """
    Parse a .sl file into a list of S-Expressions.
    """
    n: int = 0
    result: [str] = []
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
    return chars.replace('(', ' ( ').replace(')', ' ) ').replace('" "', 'space').split()


def parse(program: str) -> Expr:
    """Read an S-expression from a string."""
    return read_from_tokens(tokenize(program))


def read_from_tokens(tokens: List) -> Expr:
    """Read an expression from a sequence of tokens."""
    if len(tokens) == 0:
        raise SyntaxError('unexpected EOF')
    token = tokens.pop(0)
    if token == '(':
        result_list = []
        while tokens[0] != ')':
            result_list.append(read_from_tokens(tokens))
        tokens.pop(0)  # pop off ')'
        return result_list
    if token == ')':
        raise SyntaxError('unexpected )')
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


class EFSMTParser:
    """
    Motivation: the following implementation can be very slow

    def ground_quantifier(qexpr):
      body = qexpr.body()
      var_list = list()
      for i in range(qexpr.num_vars()):
        vi_name = qexpr.var_name(i)
        vi_sort = qexpr.var_sort(i)
        vi = z3.Const(vi_name, vi_sort)
        var_list.append(vi)
      # the following line can be slow
      body = z3.substitute_vars(body, *var_list)
      return var_list, body
    """

    def __init__(self):
        self.logic = None
        self.exist_vars = []  # e.g., [['y', 'Int'], ['z', 'Int']]
        self.forall_vars = []
        self.fml_body = ""

    def parse_smt2_string(self, inputs: str):
        """Parse an SMT2 string and return the EF system."""
        self.init_symbols(inputs)
        print("Finish internal parsing")
        return self.get_ef_system()

    def parse_smt2_file(self, filename: str):
        """Parse an SMT2 file and return the EF system."""
        with open(filename, "r", encoding="utf-8") as f:
            res = f.read()
            return self.parse_smt2_string(res)

    def to_sexpr_misc(self, lines: [str]):
        """
        Convert a list representation to S-expression format.
        E.g.,
        ['and', ['=', 'x', 1], ['=', 'y', 1]]
        ['and', ['=', 'x!', ['+', 'x', 'y']], ['=', 'y!', ['+', 'x', 'y']]]
        """
        res = ["("]
        for element in lines:
            if isinstance(element, list):
                for e in self.to_sexpr_misc(element):
                    res.append(e)
            else:
                res.append(str(element))
        res.append(")")
        return res

    def to_sexpr_string(self, lines: [str]):
        """Convert a list representation to S-expression string."""
        return " ".join(self.to_sexpr_misc(lines))

    def init_symbols(self, inputs: str) -> None:
        """Initialize symbols from input string."""
        lines = input_to_list(inputs)
        for line in lines:
            # Note: perhaps we should not parse the assertion (because it is
            #  converted back to sexpr string after we extract the forall vars
            # print(line)
            slist = parse(line)
            if isinstance(slist, List):
                cmd_name = slist[0]
                if cmd_name == "set-logic":
                    self.logic = slist[1]
                elif cmd_name == "set-info":
                    continue
                elif cmd_name == "declare-fun":
                    var_name = slist[1]
                    var_type = slist[3]
                    self.exist_vars.append([var_name, var_type])
                elif cmd_name == "assert":
                    self.process_assert(slist)
                else:
                    break

    def process_assert(self, slist) -> None:
        """
        Process an assert command.
        slist is of the form ['assert', ['forall', [['y', 'Int'], ['z', 'Int']], [...]]]
        """
        assertion = slist[1]
        # assertions[0] is "forall"
        for var_info in assertion[1]:
            self.forall_vars.append(var_info)

        fml_body_in_list = assertion[2]
        self.fml_body = self.to_sexpr_string(fml_body_in_list)

    def create_vars(self, var_info_list: List):
        """Create Z3 variables from variable info list."""
        z3var_list = []
        sig_str = []
        for var_info in var_info_list:
            # [['y', 'Int'], ['z', 'Int']]
            var_name, var_type = var_info[0], var_info[1]
            # print(var_name, var_type)
            if isinstance(var_type, List):
                # ['x', ['_', 'BitVec', 8]]
                type_str = self.to_sexpr_string(var_type)
                sig_str.append(f"(declare-fun {var_name} () {type_str})")
                z3var_list.append(z3.BitVec(var_name, int(var_type[2])))
            else:
                sig_str.append(f"(declare-fun {var_name} () {var_type})")
                if var_type.startswith("I"):  # Int
                    z3var_list.append(z3.Int(var_name))
                elif var_type.startswith("R"):  # Real
                    z3var_list.append(z3.Real(var_name))
                else:
                    print("Error: Unsupported variable type, ", var_type)
        return z3var_list, sig_str

    def get_ef_system(self):
        """
        Return the format of our trivial transition system.
        """
        exists_vars, exists_vars_sig = self.create_vars(self.exist_vars)
        forall_vars_local, forall_vars_sig = self.create_vars(self.forall_vars)
        fml_sig_str = exists_vars_sig + forall_vars_sig

        fml_str = (f"\n".join(fml_sig_str) +
                   f"\n (assert {self.fml_body} )\n" +
                   "(check-sat)\n")
        print("Finish building fml str")
        # print(fml_str)
        # We assume that there is only one assertion?
        # But for clarity, we check the size of the parsed vector
        fml_vec = z3.parse_smt2_string(fml_str)
        print("Finish building ef problem")
        if len(fml_vec) == 1:
            return exists_vars, forall_vars_local, fml_vec[0]
        return exists_vars, forall_vars_local, z3.And(fml_vec)


def ground_quantifier(qexpr):
    """
    Ground a quantifier expression.
    Seems this can only handle exists x . fml, or forall x.fml?
    FIXME: it seems that this can be very slow?
    """
    # from z3.z3util import get_vars
    body = qexpr.body()
    forall_vars_local = []
    for i in range(qexpr.num_vars()):
        vi_name = qexpr.var_name(i)
        vi_sort = qexpr.var_sort(i)
        vi = z3.Const(vi_name, vi_sort)
        forall_vars_local.append(vi)

    # Substitute the free variables in body with the expression in var_list.
    body = z3.substitute_vars(body, *forall_vars_local)
    exists_vars = [x for x in get_variables(body) if x not in forall_vars_local]
    return exists_vars, forall_vars_local, body


class EFSMTZ3Parser:
    """Parser using Z3's built-in parsing facilities."""

    def __init__(self):
        self.logic = None

    def parse_smt2_string(self, inputs: str):
        """Parse an SMT2 string using Z3."""
        fml_vec = z3.parse_smt2_string(inputs)
        if len(fml_vec) == 1:
            fml_local = fml_vec[0]
        else:
            fml_local = fml_vec
        print("Z3 finishes parsing")
        return ground_quantifier(fml_local)

    def parse_smt2_file(self, filename: str):
        """Parse an SMT2 file using Z3."""
        fml_vec = z3.parse_smt2_file(filename)
        if len(fml_vec) == 1:
            fml_local = fml_vec[0]
        else:
            fml_local = fml_vec
        print("Z3 finishes parsing")
        return ground_quantifier(fml_local)


def test_parser():
    """Test the parser with sample inputs."""
    bv = """
    ; benchmark generated from python API
(set-info :status unknown)
(declare-fun x () (_ BitVec 8))
(assert
 (forall ((y (_ BitVec 8)) (z (_ BitVec 8)) )(or (= x y) (= y z)))
 )
(check-sat)
    """
    ss = EFSMTParser()
    print(ss.parse_smt2_string(bv))

    ss2 = EFSMTZ3Parser()
    print(ss2.parse_smt2_string(bv))


if __name__ == '__main__':
    test_parser()
