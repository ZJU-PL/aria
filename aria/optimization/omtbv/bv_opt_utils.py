"""
Utility functions for bit-vector optimization.

This module provides helper functions for converting between different
representations used in bit-vector optimization problems.
"""
import logging
import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def cnt(result_list: List[int]) -> int:
    """Convert a list of binary digits to an integer.

    The list is interpreted as little-endian (LSB first after reversal).

    Args:
        result_list: List of integers representing binary digits (will be reversed)

    Returns:
        Integer value represented by the binary digits
    """
    result_list.reverse()
    total = 0
    for i, bit in enumerate(result_list):
        if bit > 0:
            total += 2 ** i
    return total


def list_to_int(result_list: List[List[int]], obj_type: List[int]) -> List[int]:
    """Convert lists of binary results to integers based on objective type.

    Args:
        result_list: List of binary result lists
        obj_type: List indicating objective type (0 for minimize, 1 for maximize)

    Returns:
        List of integer values, converted based on objective type
    """
    res: List[int] = []
    for i, binary_result in enumerate(result_list):
        score = cnt(binary_result)
        if obj_type[i] == 1:
            # Maximization: use score directly
            res.append(score)
        else:
            # Minimization: invert the score
            max_value = 2 ** len(binary_result) - 1
            res.append(max_value - score)
    return res


def assum_in_m(assum: List[int], m: List[int]) -> bool:
    """Check if all assumptions are in the model.

    Args:
        assum: List of assumption literals
        m: List of model literals

    Returns:
        True if all assumptions are in the model, False otherwise
    """
    return all(lit in m for lit in assum)


def cnf_from_z3(constraint_file: str) -> Optional[str]:
    """Generate CNF from Z3 constraint file.

    Args:
        constraint_file: Path to the constraint file

    Returns:
        Z3 output as string, or None if an error occurred
    """
    path = os.getcwd()
    path = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    z3_path = os.path.join(path, 'z3', 'build', 'z3')
    try:
        command = [z3_path, "opt.priority=box", constraint_file]
        process_result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        return process_result.stdout
    except subprocess.CalledProcessError as e:
        logger.error("Error running Z3: %s", e)
        return None


def read_cnf(  # noqa: PLR0912
    data: str
) -> Optional[Tuple[List[List[int]], List[List[int]], List[int]]]:
    """Parse CNF data from Z3 output.

    Args:
        data: String containing CNF data from Z3

    Returns:
        Tuple of (clauses, soft_clauses, obj_type) where:
        - clauses: List of hard clauses
        - soft_clauses: List of soft clause groups
        - obj_type: List indicating objective type (0 for minimize, 1 for maximize)
        Returns None if parsing fails
    """
    lines = data.splitlines()

    clauses: List[List[int]] = []
    obj_type: List[int] = []  # 0 for minimize, 1 for maximize
    soft: List[List[int]] = []
    constraint_type = '0'
    soft_temp: List[int] = []

    # Parse first line to get number of clauses
    first_line = lines[0].strip()
    parts = first_line.split()
    if len(parts) < 4:
        logger.error("Invalid CNF header: %s", first_line)
        return None
    num_clauses = int(parts[3])

    # Parse clauses
    i = 1
    while i <= num_clauses:
        clause = list(map(int, lines[i].split()))
        clause.pop()  # Remove trailing 0
        clauses.append(clause)
        i += 1

    # Parse comment lines for soft clauses
    j = i
    comment_dict: dict[int, str] = {}
    min_index = 10 ** 10
    while j < len(lines) and lines[j].startswith('c'):
        line_parts = lines[j].split()
        if len(line_parts) < 6:
            j += 1
            continue
        split_by_excl = lines[j].split('!')
        try:
            index = int(split_by_excl[-1])
            comment_dict[index] = lines[j]
            min_index = min(min_index, index)
        except (ValueError, IndexError) as e:
            logger.error("Error parsing comment line %s: %s", lines[j], e)
            return None
        j += 1

    # Reorder comment lines
    for k, line in comment_dict.items():
        lines[i + k - min_index] = line

    # Parse soft clauses
    num_comments = len(comment_dict)
    for k in range(num_comments):
        parts = lines[i + k].split()
        if len(parts) < 6:
            break

        if parts[4].endswith(':0]'):
            # End of current soft clause group
            if soft_temp:
                soft.append(soft_temp)
                soft_temp = []
                obj_type.append(int(constraint_type))
        constraint_type = parts[3][3]
        soft_temp.append(int(parts[1]))

    # Handle remaining soft clauses
    if soft_temp:
        soft.append(soft_temp)
        obj_type.append(int(constraint_type))

    return clauses, soft, obj_type


def res_z3_trans(r_z3: str, objective_order: Optional[List[str]] = None) -> List[int]:
    """Extract objective values from Z3 optimize output or models.

    Handles both:
    - Optimize objectives block: (objectives (x 10) (y 5) ...)
    - Model definitions: (define-fun x () (_ BitVec 8) #xff)
    """
    objective_values: Dict[str, int] = {}

    model_re = re.compile(
        r"\(define-fun\s+(?P<name>\S+)\s+\(\)\s+\(_\s*BitVec\s+\d+\)\s+"
        r"#x(?P<value>[0-9A-Fa-f]+)\)"
    )
    objective_re = re.compile(r"\(\s*(?P<name>[^\s\)]+)\s+(?P<value>\d+)\s*\)")

    pending_define: Optional[str] = None
    in_objectives = False

    for line in r_z3.splitlines():
        line = line.strip()
        if not line:
            continue

        # Detect start/end of objectives block
        if line.startswith("(objectives"):
            in_objectives = True
            continue
        if in_objectives and line.startswith(")"):
            in_objectives = False
            continue

        # Parse objectives block entries first
        if in_objectives:
            objective_match = objective_re.search(line)
            if objective_match:
                name = objective_match.group("name")
                try:
                    objective_values[name] = int(objective_match.group("value"))
                except ValueError:
                    logger.warning("Could not parse objective value from line: %s", line)
            continue

        model_match = model_re.search(line)
        if model_match:
            name = model_match.group("name")
            if name not in objective_values:
                value_hex = model_match.group("value")
                objective_values[name] = int(value_hex, 16)
            continue

        # Multi-line (define-fun ...) blocks: capture the name first,
        # then parse the value line.
        if line.startswith("(define-fun"):
            pattern = (
                r"\(define-fun\s+(?P<name>\S+)\s+\(\)\s+\(_\s*BitVec\s+\d+\)"
            )
            pending_match = re.match(pattern, line)
            if pending_match:
                pending_define = pending_match.group("name")
            continue

        if pending_define:
            hex_match = re.search(r"#x([0-9A-Fa-f]+)", line)
            bin_match = re.search(r"#b([01]+)", line)
            if hex_match:
                if pending_define not in objective_values:
                    objective_values[pending_define] = int(hex_match.group(1), 16)
                pending_define = None
                continue
            if bin_match:
                if pending_define not in objective_values:
                    objective_values[pending_define] = int(bin_match.group(1), 2)
                pending_define = None
                continue
            # If we reach a closing parenthesis before finding a value, reset.
            if line.startswith(")"):
                pending_define = None
            continue

        objective_match = objective_re.search(line)
        if objective_match and line.startswith("("):
            name = objective_match.group("name")
            value_dec = objective_match.group("value")
            try:
                objective_values[name] = int(value_dec)
            except ValueError:
                logger.warning("Could not parse objective value from line: %s", line)

    # Order results deterministically
    if objective_order:
        ordered = [
            objective_values[name]
            for name in objective_order
            if name in objective_values
        ]
    else:
        def _sort_key(var_name: str):
            match = re.search(r"(\d+)", var_name)
            return int(match.group(1)) if match else var_name

        ordered = [
            objective_values[name]
            for name in sorted(objective_values.keys(), key=_sort_key)
        ]

    return ordered


if __name__ == '__main__':
    BENCHMARK_PATH = '/aria/benchmarks/omt/'
    result = subprocess.run(
        ['z3', 'opt.priority=box', BENCHMARK_PATH],
        capture_output=True,
        text=True,
        check=True
    )
    print(res_z3_trans(result.stdout))
