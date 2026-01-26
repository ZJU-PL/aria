"""Parser for SMT-LIB format with oracle declarations.

Supports the extended SMT-LIB syntax with declare-nl statements
for closed-box functions with natural language descriptions.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import z3

from aria.ml.llm.smto.oracles import OracleInfo


def parse_smtlib_file(file_path: str) -> Tuple[List[OracleInfo], str]:
    """Parse an SMT-LIB file with oracle declarations and extract oracles and constraints.
    
    Args:
        file_path: Path to the SMT-LIB file
        
    Returns:
        Tuple of (list of OracleInfo objects, remaining SMT-LIB content)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return parse_smtlib_string(content)


def parse_smtlib_string(content: str) -> Tuple[List[OracleInfo], str]:
    """Parse SMT-LIB string with oracle declarations and extract oracles and constraints.
    
    Args:
        content: SMT-LIB file content as string
        
    Returns:
        Tuple of (list of OracleInfo objects, remaining SMT-LIB content)
    """
    oracles = []
    remaining_content = content
    
    # Find all declare-nl statements
    declare_nl_pattern = r'\(declare-nl\s+([^)]+)\)'
    
    for match in re.finditer(declare_nl_pattern, content, re.MULTILINE | re.DOTALL):
        full_match = match.group(0)
        inner_content = match.group(1)
        
        # Parse the declare-nl statement
        oracle_info = _parse_declare_nl(inner_content)
        if oracle_info:
            oracles.append(oracle_info)
            # Remove the declare-nl statement from remaining content
            remaining_content = remaining_content.replace(full_match, '', 1)
    
    return oracles, remaining_content


def _parse_declare_nl(inner_content: str) -> Optional[OracleInfo]:
    """Parse the inner content of a declare-nl statement.
    
    Format: name ((type1) (type2) ...) return_type (nldesc "...") (examples [...]) (library "...")
    Example: abs ((Int)) Int (nldesc "...") (examples [(10), (-5)]) (library libfun.so)
    """
    try:
        # Extract function name - first word
        name_match = re.match(r'(\w+)', inner_content)
        if not name_match:
            return None
        
        name = name_match.group(1)
        remaining = inner_content[len(name):].strip()
        
        # Extract argument types - format: ((Int) (Int) ...) or ((Int))
        # Find the first complete parenthesized group
        arg_types = []
        if remaining.startswith('('):
            # Find matching closing paren
            depth = 0
            end_idx = 0
            for i, char in enumerate(remaining):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break
            
            arg_types_str = remaining[1:end_idx-1]  # Remove outer parens
            remaining = remaining[end_idx:].strip()
            
            # Parse individual argument types - each is (Type)
            if arg_types_str.strip():
                arg_parts = _split_respecting_parens(arg_types_str)
                for arg_part in arg_parts:
                    # Remove outer parentheses if present
                    arg_part = arg_part.strip()
                    if arg_part.startswith('(') and arg_part.endswith(')'):
                        arg_part = arg_part[1:-1].strip()
                    arg_type = _parse_sort(arg_part)
                    if arg_type:
                        arg_types.append(arg_type)
        
        # Extract return type - next token
        return_type_match = re.match(r'(\S+)', remaining)
        if not return_type_match:
            return None
        
        return_type_str = return_type_match.group(1)
        remaining = remaining[len(return_type_str):].strip()
        
        # Parse return type
        return_type = _parse_sort(return_type_str)
        if not return_type:
            return None
        
        # Extract nldesc - can have colon or not: (nldesc "...") or (nldesc: "...")
        nldesc_pattern = r'\(nldesc:?\s+"([^"]+)"\)'
        nldesc_match = re.search(nldesc_pattern, remaining)
        description = nldesc_match.group(1) if nldesc_match else ""
        
        # Extract examples - format: (examples [...] or (examples: [...]
        examples_pattern = r'\(examples:?\s+\[([^\]]+)\]\)'
        examples_match = re.search(examples_pattern, remaining, re.DOTALL)
        examples = []
        if examples_match:
            examples_str = examples_match.group(1)
            examples = _parse_examples(examples_str, len(arg_types))
        
        # Extract library (optional)
        library_pattern = r'\(library\s+([^\s)]+)\)'
        library_match = re.search(library_pattern, remaining)
        library = library_match.group(1) if library_match else None
        
        return OracleInfo(
            name=name,
            input_types=arg_types,
            output_type=return_type,
            description=description,
            examples=examples,
        )
    
    except Exception:
        return None


def _split_respecting_parens(s: str) -> List[str]:
    """Split string by spaces while respecting parentheses."""
    parts = []
    current = []
    depth = 0
    
    for char in s:
        if char == '(':
            depth += 1
            current.append(char)
        elif char == ')':
            depth -= 1
            current.append(char)
        elif char == ' ' and depth == 0:
            if current:
                parts.append(''.join(current))
                current = []
        else:
            current.append(char)
    
    if current:
        parts.append(''.join(current))
    
    return parts


def _parse_sort(sort_str: str) -> Optional[z3.SortRef]:
    """Parse a sort string to Z3 sort."""
    sort_str = sort_str.strip()
    
    # Remove outer parentheses if present
    if sort_str.startswith('(') and sort_str.endswith(')'):
        sort_str = sort_str[1:-1].strip()
    
    # Basic types
    if sort_str == 'Int':
        return z3.IntSort()
    elif sort_str == 'Real':
        return z3.RealSort()
    elif sort_str == 'Bool':
        return z3.BoolSort()
    elif sort_str == 'String':
        return z3.StringSort()
    
    # Bit-vector: (_ BitVec n)
    bv_pattern = r'\(_?\s*BitVec\s+(\d+)\)'
    bv_match = re.match(bv_pattern, sort_str)
    if bv_match:
        width = int(bv_match.group(1))
        return z3.BitVecSort(width)
    
    # Floating point: (_ FloatingPoint e s)
    fp_pattern = r'\(_?\s*FloatingPoint\s+(\d+)\s+(\d+)\)'
    fp_match = re.match(fp_pattern, sort_str)
    if fp_match:
        exp_bits = int(fp_match.group(1))
        sig_bits = int(fp_match.group(2))
        return z3.FPSort(exp_bits, sig_bits)
    
    # Array: (Array IndexSort ElemSort)
    array_pattern = r'\(Array\s+(.+)\s+(.+)\)'
    array_match = re.match(array_pattern, sort_str)
    if array_match:
        index_sort_str = array_match.group(1).strip()
        elem_sort_str = array_match.group(2).strip()
        index_sort = _parse_sort(index_sort_str)
        elem_sort = _parse_sort(elem_sort_str)
        if index_sort and elem_sort:
            return z3.ArraySort(index_sort, elem_sort)
    
    return None


def _parse_examples(examples_str: str, num_args: int) -> List[Dict[str, Any]]:
    """Parse examples from examples string.
    
    Format: [(val1), (val2), ...] - just inputs
    Or: [(val1 val2 ...), ...] - inputs only
    Or: [[val1, val2, ...], ...] - with square brackets
    For PS_SMTO, we create examples with input only (output will be None or queried from oracle)
    """
    examples = []
    
    # Find all example tuples - format: (val1), (val2), or (val1 val2 ...)
    # Also handle [val1, val2, ...] format
    # Pattern to match parenthesized or bracketed groups
    tuple_pattern = r'[\[(]([^)\]]+)[\])]'
    
    for match in re.finditer(tuple_pattern, examples_str):
        tuple_str = match.group(1)
        # Split by comma or space, but be careful with commas
        # First try splitting by comma, then by space
        if ',' in tuple_str:
            values = [v.strip() for v in tuple_str.split(',') if v.strip()]
        else:
            values = re.split(r'\s+', tuple_str)
            values = [v.strip() for v in values if v.strip()]
        
        # Examples are typically just inputs
        # If we have exactly num_args values, treat as inputs only
        if len(values) == num_args:
            example_dict = {
                "input": {f"arg{i}": _parse_value(val) for i, val in enumerate(values)},
                "output": None  # Will need to be filled by querying oracle or user
            }
            examples.append(example_dict)
        elif len(values) > num_args:
            # If more values, assume last is output (for compatibility)
            inputs = values[:num_args]
            output = values[num_args]
            example_dict = {
                "input": {f"arg{i}": _parse_value(val) for i, val in enumerate(inputs)},
                "output": _parse_value(output)
            }
            examples.append(example_dict)
        elif len(values) == 1 and num_args == 1:
            # Single value for single-arg function
            example_dict = {
                "input": {"arg0": _parse_value(values[0])},
                "output": None
            }
            examples.append(example_dict)
    
    return examples


def _parse_value(val_str: str) -> Any:
    """Parse a value string to Python value."""
    val_str = val_str.strip()
    
    # Try integer
    try:
        return int(val_str)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(val_str)
    except ValueError:
        pass
    
    # Boolean
    if val_str.lower() in ['true', 'false']:
        return val_str.lower() == 'true'
    
    # String (remove quotes if present)
    if val_str.startswith('"') and val_str.endswith('"'):
        return val_str[1:-1]
    
    # Bit-vector literal: #b... or #x...
    if val_str.startswith('#b'):
        return int(val_str[2:], 2)
    if val_str.startswith('#x'):
        return int(val_str[2:], 16)
    
    return val_str
