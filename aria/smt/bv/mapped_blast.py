# coding: utf-8
"""
Perform bit-blasting and keep tracking of the relation of
bit-vector variables and Boolean variables.

Revision history:
- Removed projection logic: Eliminated the complex proj_last parameter and related projection code that was unused and added unnecessary complexity. (FIXME: in some context, we may need the projection logic...)
- Streamlined DIMACS conversion: Simplified both to_dimacs() and to_dimacs_numeric() functions by removing projection logic and unnecessary parameters.
"""
from typing import List, Dict, Tuple

import z3
from aria.utils.z3_expr_utils import get_variables


def is_literal(exp: z3.ExprRef) -> bool:
    """Check if expression is a literal."""
    return z3.is_const(exp) and exp.decl().kind() == z3.Z3_OP_UNINTERPRETED


def bitblast(formula: z3.ExprRef):
    """Bit-blast a formula and return the CNF, variable mapping, and bv2bool mapping."""
    input_vars = get_variables(formula)
    map_clauses, map_vars, bv2bool = map_bitvector(input_vars)

    # Create variable ID mapping
    id_table = {var.decl().name(): i + 1 for i, var in enumerate(map_vars)}

    g = z3.Goal()
    g.add(map_clauses)
    g.add(formula)

    t = z3.Then('simplify', 'bit-blast', 'simplify', 'aig', 'tseitin-cnf')
    blasted = t(g)[0]
    return blasted, id_table, bv2bool


def map_bitvector(input_vars):
    """Map bit-vector variables to Boolean variables."""
    clauses = []
    mapped_vars = []
    bv2bool = {}

    for var in input_vars:
        name = var.decl().name()
        if z3.is_bv(var):
            size = var.size()
            bool_vars = []
            for x in range(size):
                extracted_bool = z3.Bool(f"{name}!{x}")
                clause = extracted_bool == (z3.Extract(x, x, var) == z3.BitVecVal(1, 1))
                mapped_vars.append(extracted_bool)
                clauses.append(clause)
                bool_vars.append(f"{name}!{x}")
            bv2bool[name] = bool_vars
        elif z3.is_bool(var):
            mapped_vars.append(var)

    return clauses, mapped_vars, bv2bool


def dimacs_visitor(exp, table):
    """Visit a Z3 expression and yield DIMACS variables."""
    if is_literal(exp):
        name = exp.decl().name()
        if name not in table:
            table[name] = len(table) + 1
        yield str(table[name])
    elif z3.is_not(exp):
        for var in dimacs_visitor(exp.children()[0], table):
            yield f"-{var}"
    elif z3.is_or(exp):
        for ch in exp.children():
            yield from dimacs_visitor(ch, table)
    elif z3.is_true(exp):
        return
    else:
        raise ValueError(f"Unhandled type: {exp}")


def dimacs_visitor_numeric(exp, table):
    """Visit a Z3 expression and yield numeric DIMACS variables."""
    if is_literal(exp):
        name = exp.decl().name()
        if name not in table:
            table[name] = len(table) + 1
        yield table[name]
    elif z3.is_not(exp):
        for var in dimacs_visitor_numeric(exp.children()[0], table):
            yield -var
    elif z3.is_or(exp):
        for ch in exp.children():
            yield from dimacs_visitor_numeric(ch, table)
    else:
        raise ValueError(f"Unhandled type: {exp}")


def to_dimacs(cnf, table) -> Tuple[List[str], List[str]]:
    """Convert a Z3 CNF formula to DIMACS format."""
    cnf_clauses = []

    for clause_expr in cnf:
        # FIXME: should we add the following assertion?
        #  e.g., clause_expr could be False (or True)?
        #   If it is False, then the formula is unsatisfiable, what should we return?
        #   If it is True, perhaps we can skip the clause directly?
        if z3.is_false(clause_expr):
            cnf_clauses.append("1 -1")
            continue

        assert (z3.is_or(clause_expr) or z3.is_not(clause_expr) or
                is_literal(clause_expr))
        dimacs_clause = list(dimacs_visitor(clause_expr, table))
        cnf_clauses.append(" ".join(dimacs_clause))

    cnf_header = [f"p cnf {len(table)} {len(cnf_clauses)}"]
    return cnf_header, cnf_clauses


def to_dimacs_numeric(cnf, table):
    """Convert a Z3 CNF formula to numeric DIMACS format."""
    cnf_clauses = []

    for clause_expr in cnf:
        if z3.is_false(clause_expr):
            cnf_clauses.append([1, -1])
            continue

        assert (z3.is_or(clause_expr) or z3.is_not(clause_expr) or
                is_literal(clause_expr))
        dimacs_clause = list(dimacs_visitor_numeric(clause_expr, table))
        cnf_clauses.append(dimacs_clause)

    cnf_header = ["p"]
    return cnf_header, cnf_clauses


def translate_smt2formula_to_cnf(
    formula: z3.ExprRef
) -> Tuple[Dict[str, list], Dict[str, int], List[str], List[str]]:
    """Translate a SMT2 formula to CNF format."""
    blasted, id_table, bv2bool = bitblast(formula)
    header, clauses = to_dimacs(blasted, id_table)
    return bv2bool, id_table, header, clauses


def translate_smt2formula_to_numeric_clauses(
    formula: z3.ExprRef
) -> Tuple[Dict[str, list], Dict[str, int], List[str], List[int]]:
    """Translate a SMT2 formula to numeric CNF format."""
    blasted, id_table, bv2bool = bitblast(formula)
    header, clauses = to_dimacs_numeric(blasted, id_table)
    return bv2bool, id_table, header, clauses


def translate_smt2formula_to_cnf_file(formula: z3.ExprRef, output_file: str):
    """Translate a SMT2 formula to CNF format and save it to a file."""
    blasted, id_table, bv2bool = bitblast(formula)
    header, clauses = to_dimacs(blasted, id_table)

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(header) + '\n')
        file.write('\n'.join(clauses) + '\n')
