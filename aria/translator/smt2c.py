#!/usr/bin/env python3

# from https://github.com/MateusAraujoBorges/bvcount
# import sys
from z3 import *
from collections import defaultdict
import argparse

TEMPLATE = """
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

TEMPLATE_SAT = """
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


## TODO:
#  support more operations (especially unsigned)
#  support other mode (other than counting)
#    e.g., satisfiabiity checking; optimization
# Operation kind mappings for concise type checking
OP_KINDS = {
    Z3_OP_UNINTERPRETED: 'literal',
    Z3_OP_BNUM: 'bvconst',
    Z3_OP_ITE: 'ite',
    Z3_OP_IFF: 'iff',
    Z3_OP_SGEQ: 'sge',
    Z3_OP_SGT: 'sgt',
    Z3_OP_SLEQ: 'sle',
    Z3_OP_SLT: 'slt',
    Z3_OP_BADD: 'badd',
    Z3_OP_BSUB: 'bsub',
    Z3_OP_BMUL: 'bmul',
    Z3_OP_BSDIV: 'bsdiv',
    Z3_OP_BSMOD: 'bsmod',
    Z3_OP_BAND: 'band',
    Z3_OP_BOR: 'bor',
    Z3_OP_BLSHR: 'blshift_r',
    Z3_OP_BSHL: 'bshift_l'
}

def is_op_kind(e, op_kind):
    return e.decl().kind() == op_kind

def is_literal(e):
    return is_const(e) and is_op_kind(e, Z3_OP_UNINTERPRETED)

def is_bvconst(e):
    return is_const(e) and is_op_kind(e, Z3_OP_BNUM)

def is_ite(e):
    return is_op_kind(e, Z3_OP_ITE)

def is_iff(e):
    return is_op_kind(e, Z3_OP_IFF)

def is_sge(e):
    return is_op_kind(e, Z3_OP_SGEQ)

def is_sgt(e):
    return is_op_kind(e, Z3_OP_SGT)

def is_sle(e):
    return is_op_kind(e, Z3_OP_SLEQ)

def is_slt(e):
    return is_op_kind(e, Z3_OP_SLT)

def is_badd(e):
    return is_op_kind(e, Z3_OP_BADD)

def is_bsub(e):
    return is_op_kind(e, Z3_OP_BSUB)

def is_bmul(e):
    return is_op_kind(e, Z3_OP_BMUL)

def is_bsdiv(e):
    return is_op_kind(e, Z3_OP_BSDIV)

def is_bsmod(e):
    return is_op_kind(e, Z3_OP_BSMOD)

def is_band(e):
    return is_op_kind(e, Z3_OP_BAND)

def is_bor(e):
    return is_op_kind(e, Z3_OP_BOR)

def is_blshift_r(e):
    return is_op_kind(e, Z3_OP_BLSHR)

def is_bshift_l(e):
    return is_op_kind(e, Z3_OP_BSHL)


def find_var_bounds(clauses):
    # name => (lo,hi)
    input_vars = defaultdict(lambda: [0, 0])
    to_process = []

    for clause in clauses:
        children = clause.children()
        '''
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
        '''
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


def compile_to_c(formula, mode='counting'):
    """Compile formula to C code. Mode can be 'counting' or 'sat'."""
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
    for clause in formula.children():
        for boolean_expr in c_visitor(clause):
            line = "    if (!({})) {{\n     continue;\n    }} \n".format(boolean_expr)
            filters.append(line)
            for oprt in ['+', '-', '*', '/', '%', ' & ', ' | ', ' ^ ', ' ~ ', '>>', '<<']:
                n_ops += line.count(oprt)

    full_loop = []
    full_loop.extend(loop)
    full_loop.extend(filters)

    if mode == 'sat':
        full_loop.append("  printf(\"Find model!\\n\");")
        full_loop.append("  return 0;")
        template = TEMPLATE_SAT
    else:  # counting mode
        full_loop.append("SOLUTION_COUNT++;")
        template = TEMPLATE

    full_loop.extend(["}"] * len(loop))
    count_code = "\n".join(full_loop)
    return template.format(count=count_code), domain_size, n_ops


# Binary operation mappings: op_kind -> (operator, special_formatting)
BINARY_OPS = {
    Z3_OP_SGEQ: (">=", None),
    Z3_OP_SGT: (">", None),
    Z3_OP_SLEQ: ("<=", None),
    Z3_OP_SLT: ("<", None),
    Z3_OP_BADD: ("+", None),
    Z3_OP_BSUB: ("-", None),
    Z3_OP_BMUL: ("*", None),
    Z3_OP_BSDIV: ("/", None),
    Z3_OP_BSMOD: (",", "mod({})"),
    Z3_OP_BAND: ("&", None),
    Z3_OP_BOR: ("|", None),
    Z3_OP_BLSHR: (">>", "(unsigned int){}"),
    Z3_OP_BSHL: ("<<", None)
}

def handle_binary_op(e, operator, format_str=None):
    """Generic handler for binary operations"""
    clauses = []
    for ch in e.children():
        for expr in c_visitor(ch):
            clauses.append(expr)

    if format_str:
        return format_str.format(",".join(clauses))
    else:
        return "(" + f" {operator} ".join(clauses) + ")"

def c_visitor(e):
    if is_literal(e):
        yield e.decl().name()
        return
    elif is_bvconst(e):
        yield str(e.as_long())
        return
    elif is_not(e):
        assert len(e.children()) == 1
        ch = e.children()[0]
        for expr in c_visitor(ch):
            yield "!(" + expr + ")"
        return
    elif is_or(e):
        clauses = []
        for ch in e.children():
            for expr in c_visitor(ch):
                clauses.append(expr)
        yield "(" + " || ".join(clauses) + ")"
        return
    elif is_and(e):
        clauses = []
        for ch in e.children():
            for expr in c_visitor(ch):
                clauses.append(expr)
        yield "(" + " && ".join(clauses) + ")"
    elif is_eq(e) or is_iff(e):
        clauses = []
        for ch in e.children():
            for expr in c_visitor(ch):
                clauses.append(expr)
        yield "(" + " == ".join(clauses) + ")"
        return
    # Handle binary operations generically
    elif e.decl().kind() in BINARY_OPS:
        operator, format_str = BINARY_OPS[e.decl().kind()]
        yield handle_binary_op(e, operator, format_str)
        return
    else:
        raise Exception("Unhandled type: " + e.sexpr(), e.decl())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', type=str)
    parser.add_argument('--mode', dest='mode', default='counting', type=str)
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()
    # inputfile = sys.argv[1]
    inputfile = args.input
    mode = args.mode

    print("//Reading formula...", file=sys.stderr)
    formula = And(parse_smt2_file(inputfile))
    print("//Generating C source...", file=sys.stderr)
    code, domain, nopts = compile_to_c(formula, mode)
    print(f"{inputfile},{domain},{nopts}", file=sys.stderr)
    print(code)


if __name__ == '__main__':
    main()
