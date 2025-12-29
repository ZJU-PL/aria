"""
Utilities for abduction
"""

import re
from typing import Dict

import z3


def extract_variables_from_smt2(smt2_str: str) -> Dict[str, z3.ExprRef]:
    """
    Extract variables from SMT-LIB2 string.

    Args:
        smt2_str: SMT-LIB2 string with variable declarations

    Returns:
        Dictionary mapping variable names to Z3 variables
    """
    var_dict = {}

    # Extract basic variable declarations using regex
    var_decls = re.findall(
        r'\(declare-(?:fun|const)\s+(\w+)\s+\(\)\s+(\w+)\)', smt2_str
    )

    # Process basic types
    for var_name, var_type in var_decls:
        if var_type == 'Int':
            var_dict[var_name] = z3.Int(var_name)
        elif var_type == 'Real':
            var_dict[var_name] = z3.Real(var_name)
        elif var_type == 'Bool':
            var_dict[var_name] = z3.Bool(var_name)
        elif var_type == 'String':
            var_dict[var_name] = z3.String(var_name)
        else:
            # For other types, try later in the more complex patterns
            pass

    # Extract bit-vector declarations
    # Format: (declare-fun x () (_ BitVec N))
    bv_pattern = (
        r'\(declare-(?:fun|const)\s+(\w+)\s+\(\)\s+\(_\s+BitVec\s+(\d+)\)\)'
    )
    bv_decls = re.findall(bv_pattern, smt2_str)
    for var_name, bit_width in bv_decls:
        var_dict[var_name] = z3.BitVec(var_name, int(bit_width))

    # Extract array declarations - improved pattern
    # Format: (declare-fun arr () (Array Type Type))
    array_pattern = (
        r'\(declare-(?:fun|const)\s+(\w+)\s+\(\)\s+'
        r'\(Array\s+(\w+)\s+(\w+)\)\)'
    )
    array_decls = re.findall(array_pattern, smt2_str)
    for var_name, index_type, element_type in array_decls:
        # Create appropriate array sort based on types
        index_sort = None
        element_sort = None

        # Parse index type
        if index_type == 'Int':
            index_sort = z3.IntSort()
        elif index_type == 'Real':
            index_sort = z3.RealSort()
        elif index_type == 'Bool':
            index_sort = z3.BoolSort()
        else:
            # Default for unknown types
            index_sort = z3.IntSort()

        # Parse element type
        if element_type == 'Int':
            element_sort = z3.IntSort()
        elif element_type == 'Real':
            element_sort = z3.RealSort()
        elif element_type == 'Bool':
            element_sort = z3.BoolSort()
        else:
            # Default for unknown types
            element_sort = z3.IntSort()

        # Create array
        var_dict[var_name] = z3.Array(var_name, index_sort, element_sort)

    # Extract uninterpreted function declarations
    # Format: (declare-fun f (Type1 Type2 ...) ReturnType)
    func_pattern = r'\(declare-fun\s+(\w+)\s+\((.*?)\)\s+(\w+)\)'
    func_decls = re.findall(func_pattern, smt2_str)
    for func_name, arg_types_str, return_type in func_decls:
        # Skip if already processed as a simple variable
        if func_name in var_dict:
            continue

        # Parse argument types
        arg_types = arg_types_str.strip().split()
        if not arg_types or arg_types == ['']:
            # This is a constant, not a function
            continue

        z3_arg_types = []

        for arg_type in arg_types:
            if arg_type == 'Int':
                z3_arg_types.append(z3.IntSort())
            elif arg_type == 'Real':
                z3_arg_types.append(z3.RealSort())
            elif arg_type == 'Bool':
                z3_arg_types.append(z3.BoolSort())
            elif arg_type == 'String':
                z3_arg_types.append(z3.StringSort())
            else:
                # Default to integer sort for unknown types
                z3_arg_types.append(z3.IntSort())

        # Parse return type
        if return_type == 'Int':
            z3_return_type = z3.IntSort()
        elif return_type == 'Real':
            z3_return_type = z3.RealSort()
        elif return_type == 'Bool':
            z3_return_type = z3.BoolSort()
        elif return_type == 'String':
            z3_return_type = z3.StringSort()
        else:
            # Default to integer sort for unknown types
            z3_return_type = z3.IntSort()

        # Create the function
        if z3_arg_types:  # If it has arguments, it's an actual function
            func = z3.Function(func_name, *z3_arg_types, z3_return_type)
            var_dict[func_name] = func

    return var_dict



def contains_only_linear_arithmetic(formula: z3.ExprRef) -> bool:
    """Check if formula contains only linear integer arithmetic."""
    has_non_lia = [False]

    def check(f):
        if z3.is_const(f) and not f.sort() == z3.IntSort():
            has_non_lia[0] = True
        elif z3.is_app(f):
            if f.decl().kind() in [
                z3.Z3_OP_MUL, z3.Z3_OP_DIV, z3.Z3_OP_IDIV, z3.Z3_OP_MOD
            ]:
                children = f.children()
                if len(children) >= 2 and all(
                    not z3.is_int_value(c) for c in children[:2]
                ):
                    has_non_lia[0] = True
        for child in f.children():
            check(child)

    check(formula)
    return not has_non_lia[0]
