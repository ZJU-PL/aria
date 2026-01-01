"""Parallel Quantifier Elimination via Lazy Model Enumeration (LME-QE)"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import tempfile
import os
import re
import json
import logging
import z3

from aria.utils.z3_expr_utils import negate  # get_atoms
from aria.global_params import global_config

# Set up logging
logger = logging.getLogger(__name__)

# Path to Z3 executable
Z3_PATH = global_config.get_solver_path("z3")

# SMT-LIB templates
SMT_HEADER = """
(set-option :produce-models true)
(set-option :interactive-mode true)
(set-logic ALL)
{declarations}
(assert {formula})
"""

QE_TEMPLATE = """
(set-option :produce-models true)
(set-option :interactive-mode true)
(set-logic ALL)
{declarations}
(assert (exists ({qvars}) {formula}))
(apply qe)
(get-assertions)
"""


def to_smtlib(expr):
    """Convert Z3 expression to SMT-LIB string format"""
    if hasattr(expr, "sexpr"):
        return expr.sexpr()
    return str(expr)


def get_declarations(expr):
    """Extract variable declarations from expression"""
    decls = set()

    try:
        if hasattr(expr, "children"):
            variables = set()

            def collect_vars(e):
                if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                    variables.add(e)
                elif hasattr(e, "children"):
                    for child in e.children():
                        collect_vars(child)

            collect_vars(expr)

            for var in variables:
                sort = var.sort()
                sort_name = sort.name()
                if sort_name == "Int":
                    decls.add(f"(declare-const {var} Int)")
                elif sort_name == "Real":
                    decls.add(f"(declare-const {var} Real)")
                elif sort_name == "Bool":
                    decls.add(f"(declare-const {var} Bool)")
                else:
                    decls.add(f"(declare-const {var} {sort_name})")

        if not decls:
            var_pattern = r"([a-zA-Z][a-zA-Z0-9_]*)"
            expr_str = str(expr)

            for match in re.finditer(var_pattern, expr_str):
                var_name = match.group(1)
                if var_name not in [
                    "and",
                    "or",
                    "not",
                    "true",
                    "false",
                    "exists",
                    "forall",
                ]:
                    decls.add(f"(declare-const {var_name} Real)")

    except (AttributeError, TypeError, ValueError) as e:
        print(f"Warning: Error parsing declarations: {e}")

    return "\n".join(sorted(list(decls)))


def parse_model(z3_output):
    """Parse Z3 model output to a dictionary of variable assignments"""
    model_dict = {}

    try:
        if "sat" not in z3_output:
            return model_dict

        model_match = re.search(r"sat\s*\((.*)\)\s*$", z3_output, re.DOTALL)
        if not model_match:
            return model_dict

        model_content = model_match.group(1).strip()

        define_fun_blocks = []
        current_block = ""
        depth = 0

        for char in model_content:
            if char == "(":
                depth += 1
                if depth == 1:
                    current_block = "("
                else:
                    current_block += char
            elif char == ")":
                depth -= 1
                current_block += char
                if depth == 0:
                    define_fun_blocks.append(current_block)
                    current_block = ""
            elif depth > 0:
                current_block += char

        for block in define_fun_blocks:
            if block.strip().startswith("(define-fun"):
                parts = block.strip()[11:].strip().split(None, 3)
                if len(parts) >= 3:
                    var_name = parts[0].strip()
                    var_type = parts[2].strip()

                    value_start_idx = block.find(var_type) + len(var_type)
                    var_value = block[value_start_idx:].strip()

                    model_dict[var_name] = {"type": var_type, "value": var_value}
    except (AttributeError, ValueError, KeyError) as e:
        print(f"Error parsing model: {e}")

    return model_dict


def create_blocking_clause(model_dict):
    """Create an SMT-LIB blocking clause from a model dictionary"""
    clauses = []

    for var_name, var_info in model_dict.items():
        if var_info["type"] == "Bool":
            if var_info["value"] == "true":
                clauses.append(f"(not {var_name})")
            else:
                clauses.append(var_name)
        else:
            clauses.append(f"(not (= {var_name} {var_info['value']}))")

    if not clauses:
        return "true"
    if len(clauses) == 1:
        return clauses[0]
    return f"(or {' '.join(clauses)})"


def extract_models(formula, num_models=10, blocked_models=None):
    """Extract models from a formula using Z3 via IPC"""
    if blocked_models is None:
        blocked_models = []

    models = []

    try:
        formula_smtlib = to_smtlib(formula)

        if blocked_models:
            blocking_clauses = [
                to_smtlib(negate(model_expr)) for model_expr in blocked_models
            ]
            formula_smtlib = f"(and {formula_smtlib} {' '.join(blocking_clauses)})"

        declarations = get_declarations(formula)

        for _ in range(num_models):
            smt_script = SMT_HEADER.format(
                declarations=declarations, formula=formula_smtlib
            )
            smt_script += "\n(check-sat)"
            smt_script += "\n(get-model)"

            with tempfile.NamedTemporaryFile(
                suffix=".smt2", mode="w+", delete=False
            ) as temp_file:
                temp_path = temp_file.name
                temp_file.write(smt_script)

            result = subprocess.run(
                [Z3_PATH, temp_path],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            os.unlink(temp_path)

            if "sat" in result.stdout:
                model_dict = parse_model(result.stdout)
                if model_dict:
                    models.append(model_dict)

                    blocking_clause = create_blocking_clause(model_dict)
                    formula_smtlib = f"(and {formula_smtlib} {blocking_clause})"
                else:
                    break
            else:
                break

    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
        print(f"Error in extract_models: {e}")

    return models


def parse_qe_result(z3_output):
    """Parse quantifier elimination result from Z3 output"""
    double_paren_match = re.search(r"\(\((and[^)]+)\)\)", z3_output, re.DOTALL)
    if double_paren_match:
        return double_paren_match.group(1)

    direct_and_match = re.search(r"\(and\s+([^)]+)\)", z3_output, re.DOTALL)
    if direct_and_match:
        return f"(and {direct_and_match.group(1)})"

    get_assertions = re.search(r"\(get-assertions\)\s*(.+)", z3_output, re.DOTALL)
    if get_assertions:
        assertions_text = get_assertions.group(1).strip()
        if assertions_text.startswith("(("):
            depth = 0
            for i, char in enumerate(assertions_text):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        return assertions_text[: i + 1]

    return "false"


def build_minterm_from_model(model):
    """Build a minterm from a model's evaluation of predicates"""
    constraints = []

    for var_name, var_info in model.items():
        var_type = var_info["type"]
        var_value = var_info["value"].strip()

        if var_type == "Bool":
            if var_value == "true":
                constraints.append(var_name)
            elif var_value == "false":
                constraints.append(f"(not {var_name})")
        else:
            if var_value.startswith("(") and var_value.endswith(")"):
                constraints.append(f"(= {var_name} {var_value})")
            else:
                constraints.append(f"(= {var_name} {var_value})")

    if not constraints:
        return "true"
    if len(constraints) == 1:
        return constraints[0]
    return f"(and {' '.join(constraints)})"


def process_model(model_json, qvars_json):
    """Process a single model for QE"""
    try:
        model = json.loads(model_json)
        qvars = json.loads(qvars_json)

        minterm_smtlib = build_minterm_from_model(model)

        free_vars = [var_name for var_name in model.keys() if var_name not in qvars]

        if not free_vars:
            return "true"

        free_var_constraints = []
        for var_name in free_vars:
            var_info = model[var_name]
            var_type = var_info["type"]
            var_value = var_info["value"].strip()

            if var_type == "Bool":
                if var_value == "true":
                    free_var_constraints.append(var_name)
                elif var_value == "false":
                    free_var_constraints.append(f"(not {var_name})")
            else:
                free_var_constraints.append(f"(= {var_name} {var_value})")

        if not free_var_constraints:
            return "true"
        if len(free_var_constraints) == 1:
            projection = free_var_constraints[0]
        else:
            projection = f"(and {' '.join(free_var_constraints)})"

        # Verify projection with Z3
        verify_smt = """
(set-option :produce-models true)
(set-option :interactive-mode true)
(set-logic ALL)
"""
        for var_name, var_info in model.items():
            var_type = var_info["type"]
            verify_smt += f"(declare-const {var_name} {var_type})\n"

        verify_smt += f"(assert {minterm_smtlib})\n"
        verify_smt += f"(assert {projection})\n"
        verify_smt += "(check-sat)\n"

        with tempfile.NamedTemporaryFile(
            suffix=".smt2", mode="w+", delete=False
        ) as temp_file:
            temp_path = temp_file.name
            temp_file.write(verify_smt)

        result = subprocess.run(
            [Z3_PATH, temp_path],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        os.unlink(temp_path)

        if "sat" in result.stdout:
            return projection

        return "false"

    except (
        json.JSONDecodeError,
        KeyError,
        ValueError,
        subprocess.TimeoutExpired,
        subprocess.SubprocessError,
        OSError,
    ) as e:
        print(f"Error in process_model: {e}")
        return "false"


def simplify_result(result):
    """Simplify the result formula by removing duplicate clauses from disjunctions"""
    if not result or result == "true" or result == "false":
        return result

    if result.startswith("(or ") and result.endswith(")"):
        inner_content = result[4:-1].strip()

        constraints = []
        pattern = r"\(=\s+z\s+(.+?)\)"
        for match in re.finditer(pattern, inner_content):
            constraint = match.group(0)
            if constraint not in constraints:
                constraints.append(constraint)

        if constraints:
            return "(or " + " ".join(constraints) + ")"

    return result


def qelim_exists_lme_parallel(phi, qvars, num_workers=None, batch_size=4):
    """
    Parallel Existential Quantifier Elimination using Lazy Model Enumeration with IPC

    Args:
        phi: Formula to eliminate quantifiers from (Z3 expression or SMT-LIB string)
        qvars: List of variables to eliminate (Z3 variables)
        num_workers: Number of parallel workers (default: CPU count)
        batch_size: Number of models to sample in each iteration
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    try:
        # Get atomic predicates
        # predicates = [to_smtlib(pred) for pred in get_atoms(phi)]

        # Convert formula to SMT-LIB format
        phi_smtlib = to_smtlib(phi)

        # Serialize variables for IPC
        qvars_json = json.dumps([str(var) for var in qvars])

        # Track projections and blocking clauses
        projections = []
        blocking_clauses = []

        # Main loop for model enumeration
        max_iterations = 5  # Limit iterations for testing
        for _ in range(max_iterations):
            # Extract models from the formula, blocking existing projections
            formula_with_blocking = phi_smtlib
            for clause in blocking_clauses:
                formula_with_blocking = f"(and {formula_with_blocking} (not {clause}))"

            models = extract_models(formula_with_blocking, num_models=batch_size)

            # If no more models, break
            if not models:
                break

            # Process models in parallel
            new_projections = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for model in models:
                    # Serialize model for IPC
                    model_json = json.dumps(model)

                    # Submit job to process this model
                    future = executor.submit(process_model, model_json, qvars_json)
                    futures.append(future)

                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        projection = future.result()
                        if projection and projection != "false":
                            new_projections.append(projection)
                    except (ValueError, KeyError) as e:
                        print(f"Error in parallel processing: {e}")

            # Update projections and blocking clauses
            projections.extend(new_projections)
            blocking_clauses.extend(new_projections)

            if not new_projections:
                break

        # Combine all projections with OR
        if not projections:
            return "false"

        if len(projections) == 1:
            result = projections[0]
        else:
            result = f"(or {' '.join(projections)})"

        # Simplify the result by removing duplicates
        return simplify_result(result)

    except (
        AttributeError,
        TypeError,
        ValueError,
        subprocess.TimeoutExpired,
        subprocess.SubprocessError,
        OSError,
    ) as e:
        print(f"Error in qelim_exists_lme_parallel: {e}")
        return "false"
