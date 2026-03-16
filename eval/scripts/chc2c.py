import z3
from z3 import *
from collections import defaultdict
import argparse

from efmc.utils.z3_expr_utils import get_variables

TEMPLATE_BV_COUNTING = """
#include <stdio.h>
int mod (int a, int b)
{{
   if(b < 0) //you can check for b == 0 separately and do what you want
     return mod(a, -b);   
   int ret = a % b;
   if(ret < 0)
     ret+=b;
   return ret;
}}
int main() {{
unsigned long long SOLUTION_COUNT = 0;
//count
{count}
printf("%llu \\n",SOLUTION_COUNT);
return 0;
}}
"""

TEMPLATE_BV_SAT = """
#include <stdio.h>
int mod (int a, int b)
{{
   if(b < 0) //you can check for b == 0 separately and do what you want
     return mod(a, -b);   
   int ret = a % b;
   if(ret < 0)
     ret+=b;
   return ret;
}}
int main() {{
//sat
{count}
return 0;
}}
"""


def is_literal(e):
    return is_const(e) and e.decl().kind() == Z3_OP_UNINTERPRETED


def is_bvconst(e):
    return is_const(e) and e.decl().kind() == Z3_OP_BNUM


def is_int_real_const(e):
    return z3.is_const(e) and (z3.is_int(e) or z3.is_real(e))


def is_ite(e):
    return e.decl().kind() == Z3_OP_ITE


def is_iff(e):
    return e.decl().kind() == Z3_OP_IFF


def is_sge(c):
    return c.decl().kind() == Z3_OP_SGEQ


def is_ira_ge(c):
    """>= in integer or real"""
    return c.decl().kind() == Z3_OP_GE


def is_sgt(c):
    return c.decl().kind() == Z3_OP_SGT


def is_ira_gt(c):
    return c.decl().kind() == Z3_OP_GT


def is_sle(c):
    return c.decl().kind() == Z3_OP_SLEQ


def is_ira_le(c):
    return c.decl().kind() == Z3_OP_LE


def is_slt(c):
    return c.decl().kind() == Z3_OP_SLT


def is_ira_lt(c):
    return c.decl().kind() == Z3_OP_LT


def is_badd(c):
    return c.decl().kind() == Z3_OP_BADD


def is_ira_add(c):
    return c.decl().kind() == Z3_OP_ADD


def is_bsub(c):
    return c.decl().kind() == Z3_OP_BSUB


def is_ira_sub(c):
    return c.decl().kind() == Z3_OP_SUB


def is_bmul(c):
    return c.decl().kind() == Z3_OP_BMUL


def is_ira_mul(c):
    return c.decl().kind() == Z3_OP_MUL


def is_bsdiv(c):
    return c.decl().kind() == Z3_OP_BSDIV


def is_ira_div(c):
    return c.decl().kind() == Z3_OP_DIV


def is_bsmod(c):
    return c.decl().kind() == Z3_OP_BSMOD


def is_band(c):
    return c.decl().kind() == Z3_OP_BAND


def is_bor(c):
    return c.decl().kind() == Z3_OP_BOR


def is_blshift_r(c):
    return c.decl().kind() == Z3_OP_BLSHR


def is_bshift_l(c):
    return c.decl().kind() == Z3_OP_BSHL


def find_var_bounds(clauses):
    # name => (lo,hi)
    input_vars = defaultdict(lambda: [0, 0])
    to_process = []

    for clause in clauses:
        children = clause.children()
        """
        if is_sge(clause):
            input_vars[children[0]][0] = children[1].as_long()
        elif is_sgt(clause):
            input_vars[children[0]][0] = children[1].as_long() + 1
        elif is_sle(clause):
            input_vars[children[0]][1] = children[1].as_long()
        elif is_slt(clause):
            input_vars[children[0]][1] = children[1].as_long() - 1
        else:
            to_process.append(clause)
        """
        tt = children[1]
        if is_sge(clause):
            # BitVecVal(tt, tt.size())
            input_vars[children[0]][0] = tt.as_long()
        elif is_sgt(clause):
            input_vars[children[0]][0] = tt.as_long() + 1
        elif is_sle(clause):
            input_vars[children[0]][1] = tt.as_long()
        elif is_slt(clause):
            input_vars[children[0]][1] = tt.as_long() - 1
        else:
            to_process.append(clause)

    return input_vars, to_process


"""
Int/real related operations
"""


def compile_ira_to_c_sat(formula):
    """
    TODO: finish the implementation?
    """
    input_vars = get_variables(formula)
    sig = []
    for var in input_vars:
        line = "  int {} ;".format(var)
        sig.append(line)

    loop = []
    for expr in c_visitor(formula):
        loop.append(" {};\n".format(expr))

    # print("sig: ", sig)
    # print("loop: ", loop)
    return loop


def compile_bv_to_c_sat(formula):
    # don't go deep now, just find out the bounds
    input_vars, to_process = find_var_bounds(formula.children())
    loop = []
    domain_size = 1
    for var, bounds in input_vars.items():
        assert var.size() == 32
        line = "  for (int {name} = {lo}; {name} <= {hi}; {name}++) {{"
        loop.append(line.format(name=var.sexpr(), lo=bounds[0], hi=bounds[1]))
        domain_size *= bounds[1] - bounds[0]

    filters = []
    n_ops = 0
    #    print(formula, file=sys.stderr)
    #    print(formula.children(), file=sys.stderr)
    for clause in formula.children():
        for boolean_expr in c_visitor(clause):
            line = "    if (!({})) {{\n     continue;\n    }} \n".format(boolean_expr)
            filters.append(line)
            for oprt in [
                "+",
                "-",
                "*",
                "/",
                "%",
                " & ",
                " | ",
                " ^ ",
                " ~ ",
                ">>",
                "<<",
            ]:
                n_ops += line.count(oprt)

    full_loop = []
    full_loop.extend(loop)
    full_loop.extend(filters)
    full_loop.append('  printf("Find model!\\n");')
    full_loop.append("  return 0;")  # break or return??
    full_loop.extend(["}"] * len(loop))

    count_code = "\n".join(full_loop)
    return TEMPLATE_BV_SAT.format(count=count_code), domain_size, n_ops


def compile_bv_to_c_counting(formula):
    # don't go deep now, just find out the bounds
    input_vars, to_process = find_var_bounds(formula.children())
    loop = []
    domain_size = 1
    for var, bounds in input_vars.items():
        assert var.size() == 32
        line = "  for (int {name} = {lo}; {name} <= {hi}; {name}++) {{"
        loop.append(line.format(name=var.sexpr(), lo=bounds[0], hi=bounds[1]))
        domain_size *= bounds[1] - bounds[0]

    filters = []
    n_ops = 0
    #    print(formula, file=sys.stderr)
    #    print(formula.children(), file=sys.stderr)
    for clause in formula.children():
        for boolean_expr in c_visitor(clause):
            line = "    if (!({})) {{\n     continue;\n    }} \n".format(boolean_expr)
            filters.append(line)
            for oprt in [
                "+",
                "-",
                "*",
                "/",
                "%",
                " & ",
                " | ",
                " ^ ",
                " ~ ",
                ">>",
                "<<",
            ]:
                n_ops += line.count(oprt)

    full_loop = []
    full_loop.extend(loop)
    full_loop.extend(filters)
    full_loop.append("SOLUTION_COUNT++;")
    full_loop.extend(["}"] * len(loop))

    count_code = "\n".join(full_loop)
    return TEMPLATE_BV_COUNTING.format(count=count_code), domain_size, n_ops


def c_visitor(e):
    if is_literal(e):
        yield e.decl().name()
        return
    elif is_bvconst(e) or is_int_real_const(e):
        yield str(e.as_long())
        return
    elif is_not(e):
        assert len(e.children()) == 1
        ch = e.children()[0]
        for expr in c_visitor(ch):
            yield "!(" + expr + ")"
        return
    elif is_or(e):
        or_clauses = []
        for ch in e.children():
            for expr in c_visitor(ch):
                or_clauses.append(expr)
        yield "(" + " || ".join(or_clauses) + ")"
        return
    elif is_and(e):
        or_clauses = []
        for ch in e.children():
            for expr in c_visitor(ch):
                or_clauses.append(expr)
        yield "(" + " && ".join(or_clauses) + ")"
    elif is_eq(e) or is_iff(e):
        eq_clauses = []
        for ch in e.children():
            for expr in c_visitor(ch):
                eq_clauses.append(expr)
        yield "(" + " == ".join(eq_clauses) + ")"
        return
    elif is_sge(e) or is_ira_ge(e):
        clauses = []
        for ch in e.children():
            for expr in c_visitor(ch):
                clauses.append(expr)
        yield "(" + " >= ".join(clauses) + ")"
        return
    elif is_sgt(e) or is_ira_gt(e):
        clauses = []
        for ch in e.children():
            for expr in c_visitor(ch):
                clauses.append(expr)
        yield "(" + " > ".join(clauses) + ")"
        return
    elif is_sle(e) or is_ira_le(e):
        clauses = []
        for ch in e.children():
            for expr in c_visitor(ch):
                clauses.append(expr)
        yield "(" + " <= ".join(clauses) + ")"
        return
    elif is_slt(e) or is_ira_lt(e):
        clauses = []
        for ch in e.children():
            for expr in c_visitor(ch):
                clauses.append(expr)
        yield "(" + " < ".join(clauses) + ")"
        return
    elif is_badd(e) or is_ira_add(e):
        clauses = []
        for ch in e.children():
            for expr in c_visitor(ch):
                clauses.append(expr)
        yield "(" + " + ".join(clauses) + ")"
        return
    elif is_bsub(e) or is_ira_sub(e):
        clauses = []
        for ch in e.children():
            for expr in c_visitor(ch):
                clauses.append(expr)
        yield "(" + " - ".join(clauses) + ")"
        return
    elif is_bmul(e) or is_ira_mul(e):
        clauses = []
        for ch in e.children():
            for expr in c_visitor(ch):
                clauses.append(expr)
        yield "(" + " * ".join(clauses) + ")"
        return
    elif is_bsdiv(e) or is_ira_div(e):
        clauses = []
        for ch in e.children():
            for expr in c_visitor(ch):
                clauses.append(expr)
        yield "(" + " / ".join(clauses) + ")"
        return
    elif is_bsmod(e):
        clauses = []
        for ch in e.children():
            for expr in c_visitor(ch):
                clauses.append(expr)
        yield "(mod(" + ",".join(clauses) + "))"
        return
    elif is_bor(e):
        clauses = []
        for ch in e.children():
            for expr in c_visitor(ch):
                clauses.append(expr)
        yield "(" + " | ".join(clauses) + ")"
        return
    elif is_band(e):
        clauses = []
        for ch in e.children():
            for expr in c_visitor(ch):
                clauses.append(expr)
        yield "(" + " & ".join(clauses) + ")"
        return
    elif is_blshift_r(e):
        clauses = []
        for ch in e.children():
            for expr in c_visitor(ch):
                clauses.append(expr)
        yield "((unsigned int)" + " >> ".join(clauses) + ")"
        return
    elif is_bshift_l(e):
        clauses = []
        for ch in e.children():
            for expr in c_visitor(ch):
                clauses.append(expr)
        yield "(" + " << ".join(clauses) + ")"
        return

    else:
        raise Exception("Unhandled type: " + e.sexpr(), e.decl())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", type=str)
    parser.add_argument("--mode", dest="mode", default="counting", type=str)
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    args = parser.parse_args()
    # inputfile = sys.argv[1]
    inputfile = args.input
    mode = args.mode

    print("//Reading formula...", file=sys.stderr)
    formula = And(parse_smt2_file(inputfile))
    # print(formula)
    print("//Generating C source...", file=sys.stderr)
    if mode == "bv-counting":
        code, domain, nopts = compile_bv_to_c_counting(formula)
        print("{},{},{}".format(inputfile, domain, nopts), file=sys.stderr)
        print(code)
    elif mode == "bv-sat":
        code, domain, nopts = compile_bv_to_c_sat(formula)
        print("{},{},{}".format(inputfile, domain, nopts), file=sys.stderr)
        print(code)


if __name__ == "__main__":
    main()
