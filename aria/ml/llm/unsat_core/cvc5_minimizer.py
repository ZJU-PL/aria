"""Contains utilities for cvc5.
"""
import re
import time

import cvc5


class OutputFormatError(Exception):
    """Exception raised when GPT-x's response does not conform to prompt instructions.
    """
    pass


def extract_output_constraints(response):
    """Extracts constraints from GPT-x's response.

    Expected input format: <OUTPUT><c>C1</c><c>C2</c></OUTPUT>
    Expected output: [C1, C2]

    Arguments:
        response (str): GPT-x's response.
    
    Returns:
        output_constraints (list): Output list of constraints from GPT-x in string format.
    """
    response = response.strip()
    response = response.replace("\n", "<nl>")
    all_outputs = re.findall("<OUTPUT>(.*?)</OUTPUT>", response)
    if len(all_outputs) != 1:
        raise OutputFormatError(f"Only 1 output should be returned, {len(all_outputs)} were returned instead.")
    output = all_outputs[0]
    output_constraints = re.findall("<c>(.*?)</c>", output)
    if len(output_constraints) == 0:
        raise OutputFormatError("No constraints found in output.")
    return output_constraints


def build_smt2_formula_from_string_constraints(
        constraints, all_constraints, all_smt2_constraints, placeholder
    ):
    """Builds SMT2-Lib formula from constraints' subset, retrieved from GPT-x's response.

    Arguments:
        constraints (list): Constraint subset from GPT-x's response in string format.
        all_constraints (list): Complete list of constraints in string format.
        all_smt2_constraints (list): Complete list of constraints in SMT2-Lib format.
            There is one-to-one correspondence with constraints in ``all_constraints``.
        placeholder (str): SMT2-Lib input file string placeholder, where all assertions
            are represented by "<ASSERT>" keyword.
    
    Returns:
        smt2_formula (str): SMT2-Lib input format string, where assertions corresponding
            to ``constraints`` are inserted in ``placeholder``.
    """
    constraints_idx = [all_constraints.index(constraint) for constraint in constraints]
    smt2_assertions = [f"(assert {all_smt2_constraints[idx]})" for idx in constraints_idx]
    assertions = "\n".join(smt2_assertions)
    smt2_formula = placeholder.replace("<ASSERT>", assertions, 1)
    return smt2_formula


def build_smt2_formula_from_cvc5_constraints(
        cvc5_constraints, all_smt2_constraints, all_cvc5_constraints, placeholder
    ):
    """Builds SMT2-Lib formula from 

    Arguments:
        cvc5_constraints (list): Constraint subset in cvc5 format.
        all_smt2_constraints (list): Complete list of constraints in SMT2-Lib format.
        all_cvc5_constraints (list): Complete list of constraints in cvc5 format.
        placeholder (str): SMT2-Lib input file string placeholder, where all assertions
            are represented by "<ASSERT>" keyword.
    
    Returns:
        smt2_formula (str): SMT2-Lib input format string, where assertions corresponding
            to ``constraints`` are inserted in ``placeholder``.
    """
    constraints_idx = [all_cvc5_constraints.index(constraint) for constraint in cvc5_constraints]
    smt2_assertions = [f"(assert {all_smt2_constraints[idx]})" for idx in constraints_idx]
    assertions = "\n".join(smt2_assertions)
    smt2_formula = placeholder.replace("<ASSERT>", assertions, 1)
    return smt2_formula


def build_smt2_formula_from_smt2_constraints(constraints, placeholder):
    """Builds SMT2-Lib formula from constraints' subset, retrieved from GPT-x's response.

    Arguments:
        constraints (list): Constraint subset in string format.
        placeholder (str): SMT2-Lib input file string placeholder, where all assertions
            are represented by "<ASSERT>" keyword.
    
    Returns:
        smt2_formula (str): SMT2-Lib input format string, where assertions corresponding
            to ``constraints`` are inserted in ``placeholder``.
    """
    smt2_assertions = [f"(assert {constraint})" for constraint in constraints]
    assertions = "\n".join(smt2_assertions)
    smt2_formula = placeholder.replace("<ASSERT>", assertions, 1)
    return smt2_formula


def parse_input_formula(solver, formula, formula_name):
    """Parse input formula and load into solver memory.

    Arguments:
        solver (cvc5.Solver): SMT solver.
        formula (str): Input formula in string format.
        formula_name (str): Key for input formula.
    """
    parser = cvc5.InputParser(solver)
    parser.setStringInput(
        cvc5.InputLanguage.SMT_LIB_2_6, formula, formula_name,
    )
    symbol_manager = parser.getSymbolManager()

    while True:
        cmd = parser.nextCommand()
        if cmd.isNull():
            break
        # Invoke the command on the solver and the symbol manager
        cmd.invoke(solver, symbol_manager)


def set_cvc5_options_for_mus(solver, optimize_for_mus):
    """Initialize configuration for ``cvc5.Solver``.
    If ``optimize_for_mus`` is ``True``, option ``minimal-unsat-cores`` is 
    set to ``true``. This gets the solver to look for MUSes.

    Arguments:
        solver (cvc5.Solver): SMT solver.
        optimize_for_mus (bool): If ``True``, initialize for solving MUSes.
    """
    solver.setOption("print-success", "true")
    solver.setOption("produce-models", "true")
    solver.setOption("produce-unsat-cores", "true")
    solver.setOption("unsat-cores-mode", "assumptions")
    solver.setOption("minimal-unsat-cores", str(optimize_for_mus).lower())
    return


def set_cvc5_options_for_unsat(solver):
    """Initialize configuration for ``cvc5.Solver``, to prove unsatisfiability.

    Arguments:
        solver (cvc5.Solver): SMT solver.
    """    
    solver.setOption("print-success", "true")
    solver.setOption("produce-models", "true")
    return


def compute_time_for_mus(constraints, placeholder):
    """Compute time taken for extracting the Minimal Unsatisfiability Subset.

    Arguments:
        constraints (list): Constraint subset in string format.
        placeholder (str): SMT2-Lib input file string placeholder, where all assertions
            are represented by "<ASSERT>" keyword.

    Returns:
        mus_time (float): Time for MUS computation.        
    """
    solver = cvc5.Solver()
    set_cvc5_options_for_mus(solver, True)
    solver.setLogic("QF_SLIA")

    import random
    random.seed(42)
    random.shuffle(constraints)
    smt2_formula = build_smt2_formula_from_smt2_constraints(constraints, placeholder)
    parse_input_formula(solver, smt2_formula, "smt_formula")

    unsat_check_start_time = time.time()
    result = solver.checkSat()

    if not result.isUnsat(): 
        return None

    unsat_core = solver.getUnsatCore()
    unsat_core_end_time = time.time()
    mus_time = "{:.3f}".format((unsat_core_end_time - unsat_check_start_time)*1000)
    return mus_time
