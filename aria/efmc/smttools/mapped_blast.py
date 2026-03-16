# coding: utf-8
"""
This module contains functions for translating SMT2 formulas to CNF and DIMACS format.

It also keeps track of the mapping between bit-vector variables and their corresponding Boolean variables.

TODO: it seems that this file can be slow. We need to profile and optimize it.
"""
import sys
from typing import List, Dict, Tuple

import z3


# p cnf nvar nclauses
# cr projected_var_ids


def is_literal(exp: z3.ExprRef):
    """Check if an expression is a literal (uninterpreted constant)."""
    return z3.is_const(exp) and exp.decl().kind() == z3.Z3_OP_UNINTERPRETED


def proj_id_last(var, n_proj_vars, n_vars):
    """
    Convert a projected variable ID to the original variable ID.

    Args:
        var: The projected variable ID to convert.
        n_proj_vars: The number of projected variables.
        n_vars: The total number of variables.

    """
    assert var != 0
    is_neg = var < 0
    if abs(var) <= n_proj_vars:
        new_var = abs(var) - n_proj_vars + n_vars
    else:
        new_var = abs(var) - n_proj_vars

    return new_var * (-1 if is_neg else 1)


def bitblast(formula: z3.ExprRef):
    """Bit-blast a formula to CNF."""
    input_vars = list(collect_vars(formula))  # this might be slow?
    # input_vars = get_vars(formula)  # FIXME: get_vars is from z3, but it can be VERY slow...
    # map bits in the bv input vars
    map_clauses, map_vars, bv2bool = map_bitvector(input_vars)
    # print(bv2bool)
    id_table = {}  # {varname: id}

    curr_id = 1
    for var in map_vars:
        name = var.decl().name()
        id_table[name] = curr_id
        curr_id = curr_id + 1
    # projection_scope = curr_id - 1
    g = z3.Goal()
    g.add(map_clauses)  # why adding these constraints?
    g.add(formula)
    t = z3.Then("simplify", "bit-blast", "tseitin-cnf")
    blasted = t(g)[0]
    return blasted, id_table, bv2bool


def to_dimacs(cnf, table, proj_last) -> Tuple[List[str], List[str]]:
    """
    Translate an SMT2 formula to a CNF.
    """
    cnf_clauses = []
    projection_scope = len(table)

    for clause_expr in cnf:
        assert (
            z3.is_or(clause_expr) or z3.is_not(clause_expr) or is_literal(clause_expr)
        )
        dimacs_clause = list(dimacs_visitor(clause_expr, table))
        # dimacs_clause.append('0') # TODO: why append 0???
        cnf_clauses.append(" ".join(dimacs_clause))

    if proj_last:
        n_vars = len(table)
        clauses = []
        for clause in cnf_clauses:
            int_clause = [int(x) for x in clause.split(" ")[:-1]]
            proj_clause = [
                proj_id_last(x, projection_scope, n_vars) for x in int_clause
            ]
            # proj_clause.append(0)  # TODO: why append 0???
            str_clause = " ".join([str(x) for x in proj_clause])
            clauses.append(str_clause)

        cnf_clauses = clauses
        cnf_header = [
            f"p cnf {len(table)} {len(cnf_clauses)}",
            f"cr {' '.join([str(x) for x in range(n_vars - projection_scope + 1, n_vars + 1)])}",
        ]
    else:
        cnf_header = [
            f"p cnf {len(table)} {len(cnf_clauses)}",
            f"cr {' '.join([str(x) for x in range(1, projection_scope + 1)])}",
        ]
    return cnf_header, cnf_clauses


def to_dimacs_numeric(cnf, table, proj_last):
    """
    Translate an SMT2 formula to numeric clauses.
    """
    cnf_clauses = []
    projection_scope = len(table)

    for clause_expr in cnf:
        assert (
            z3.is_or(clause_expr) or z3.is_not(clause_expr) or is_literal(clause_expr)
        )
        dimacs_clause_numeric = list(dimacs_visitor_numeric(clause_expr, table))
        cnf_clauses.append(dimacs_clause_numeric)

    if proj_last:
        n_vars = len(table)
        clauses = []
        for clause in cnf_clauses:
            int_clause = clause
            proj_clause = [
                proj_id_last(x, projection_scope, n_vars) for x in int_clause
            ]
            clauses.append(proj_clause)
        cnf_clauses = clauses
        cnf_header = ["p"]
    else:
        cnf_header = ["p"]
    return cnf_header, cnf_clauses


def map_bitvector(input_vars):
    """
    Map bit-vector variables to their corresponding Boolean variables.
    """
    # print("input vars...")
    # print(input_vars)
    clauses = []
    mapped_vars = []
    bv2bool = {}  # for tracking what Bools corresponding to a bv
    for var in input_vars:
        name = var.decl().name()
        size = var.size()
        bool_vars = []
        for x in range(size):
            extracted_bool = z3.Bool(f"{name}!{x}")
            # why adding this
            clause = extracted_bool == (z3.Extract(x, x, var) == z3.BitVecVal(1, 1))
            mapped_vars.append(extracted_bool)
            clauses.append(clause)
            bool_vars.append(name + "!" + str(x))

        bv2bool[str(name)] = bool_vars
    # print(clauses)
    return clauses, mapped_vars, bv2bool


def dimacs_visitor(exp, table):
    """
    Visit an SMT2 formula and return the DIMACS clauses.
    """
    if is_literal(exp):
        name = exp.decl().name()
        if name in table:
            id_var = table[name]
        else:
            id_var = len(table) + 1
            table[name] = id_var
        yield str(id_var)
        return
    if z3.is_not(exp):
        assert len(exp.children()) == 1
        ch = exp.children()[0]
        yield from (f"-{var}" for var in dimacs_visitor(ch, table))
        return
    if z3.is_or(exp):
        for ch in exp.children():
            yield from dimacs_visitor(ch, table)
        return
    if z3.is_true(exp):
        return  # correct?
    # elif is_false(e): return ??
    raise ValueError(f"Unhandled type: {exp}")


def dimacs_visitor_numeric(exp, table):
    """Visit an SMT2 formula and return numeric DIMACS clauses."""
    if is_literal(exp):
        name = exp.decl().name()
        if name in table:
            id_var = int(table[name])
        else:
            id_var = len(table) + 1
            table[name] = id_var
        yield id_var
        return
    if z3.is_not(exp):
        assert len(exp.children()) == 1
        ch = exp.children()[0]
        yield from (-var for var in dimacs_visitor_numeric(ch, table))
        return
    if z3.is_or(exp):
        for ch in exp.children():
            yield from dimacs_visitor_numeric(ch, table)
        return
    if z3.is_true(exp):
        return  # correct?
    # elif is_false(e): return ??
    raise ValueError(f"Unhandled type: {exp}")


def collect_vars(exp, seen=None):
    """
    Collect all variables in an SMT2 formula.
    TODO: this can be slow?
    """
    if seen is None:
        seen = {}
    if exp in seen:
        return
    seen[exp] = True

    # check if 'e' is a bitvector input variable
    if is_literal(exp) and z3.is_bv(exp):
        yield exp
    elif z3.is_app(exp):
        for ch in exp.children():
            yield from collect_vars(ch, seen)
        return
    elif z3.is_quantifier(exp):
        yield from collect_vars(exp.body(), seen)
        return


def translate_smt2formula_to_cnf(
    formula: z3.ExprRef,
) -> Tuple[Dict[str, list], Dict[str, int], List[str], List[str]]:
    """
    Translate an SMT2 formula to a CNF.
    """
    projection_last = ""
    projection_last = projection_last and projection_last.lower() != "false"
    # print("Generating DIMACS with projection...")
    blasted, id_table, bv2bool = bitblast(formula)
    # FIXME: if formula is very simple, the "simplify" tactic may convert it to "False",
    #   However it seems that cnf tactic needs "simplify" to perform some normalization
    #   Maybe we need to turn the parameters of "simplify"?
    header, clauses = to_dimacs(blasted, id_table, projection_last)
    return bv2bool, id_table, header, clauses


def translate_smt2formula_to_numeric_clauses(
    formula: z3.ExprRef,
) -> Tuple[Dict[str, list], Dict[str, int], List[str], List[int]]:
    """
    Translate an SMT2 formula to numeric clauses.
    """
    projection_last = ""
    projection_last = projection_last and projection_last.lower() != "false"
    # print("Generating DIMACS with projection...")
    blasted, id_table, bv2bool = bitblast(formula)
    header, clauses = to_dimacs_numeric(blasted, id_table, projection_last)
    return bv2bool, id_table, header, clauses


def translate_smt2formula_to_cnf_file(formula: z3.ExprRef, output_file: str):
    """
    Translate an SMT2 formula to a CNF file.
    """
    projection_last = ""
    projection_last = projection_last and projection_last.lower() != "false"
    blasted, id_table, bv2bool = bitblast(formula)
    header, clauses = to_dimacs(blasted, id_table, projection_last)
    saved_stdout = sys.stdout
    with open(output_file, "w+", encoding="utf-8") as file:
        sys.stdout = file
        print("\n".join(header))
        print("\n".join(clauses))
    sys.stdout = saved_stdout


# TODO: what is projection_last
def test_blast(input_file):
    projection_last = ""
    projection_last = projection_last and projection_last.lower() != "false"

    formula_vec = z3.parse_smt2_file(input_file)
    formula = z3.And(formula_vec)
    print("Generating DIMACS with projection...")
    blasted, id_table, _ = bitblast(formula)
    header, clauses = to_dimacs(blasted, id_table, projection_last)
    # print('\n'.join(header))
    print("\n".join(clauses))
    # print(id_table)
