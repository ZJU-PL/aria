import json
import random
import string
import uuid
import os
from typing import Tuple, Union

from pysmt.fnode import FNode
from pysmt.operators import op_to_str
from pysmt.solvers.solver import Solver as PysmtSolver

from aria.quant.polyhorn.Parser import Parser
from aria.quant.polyhorn.PositiveModel import PositiveModel


def add_default_config(config: dict) -> dict:
    """
    Add default values to the config dictionary

    Parameters
    ----------
    config : dict
        The config dictionary

    Returns
    -------
    dict
        The config dictionary with default values
    """
    default_config = {
        "SAT_heuristic": False,
        "degree_of_sat": 0,
        "degree_of_nonstrict_unsat": 0,
        "degree_of_strict_unsat": 0,
        "max_d_of_strict": 0,
        "unsat_core_heuristic": False,
        "integer_arithmetic": False,
    }

    for key, value in default_config.items():
        config[key] = config.get(key, value)

    return config


def load_config(config_path: str) -> dict:
    """
    Load and parse the config file with default values

    Parameters
    ----------
    config_path : str
        The path to the config file

    Returns
    -------
    dict
        The parsed config file with default values
    """
    with open(config_path, "r") as file:
        config: dict = json.load(file)

    return add_default_config(config)


def pysmt_to_smt2(solver: PysmtSolver) -> str:
    """
    Convert a pysmt solver to a smt2 string

    Parameters
    ----------
    solver : pysmt.solvers.solver.Solver
        The pysmt solver

    Returns
    -------
    str
        The smt2 string
    """
    free_vars = set()
    for formula in solver.assertions:
        if formula.is_forall() and not formula.arg(0).is_implies():
            # Check for constraint pairs
            raise ValueError(
                f"PolyHorn expects universally quantified formulas to consist of constraint pairs in the form `forall x, y, ...: (constraint) => (constraint)`"
            )

        free_vars |= formula.get_free_variables()

    def to_smt2(formula: FNode) -> str:

        if formula.is_forall():
            assert (
                len(formula.args()) == 1
            ), f"`Forall` expected 1 argument, got {len(formula.args())}"
            return f'(forall ({" ".join(f"({str(v)} {v.symbol_type()})" for v in formula.quantifier_vars())}) {to_smt2(formula.arg(0))})'
        elif formula.is_implies():
            assert (
                len(formula.args()) == 2
            ), f"`Implies` expected 2 arguments, got {len(formula.args())}"
            return f"(=> {to_smt2(formula.arg(0))} {to_smt2(formula.arg(1))})"
        elif formula.is_and():
            assert (
                len(formula.args()) >= 2
            ), f"`And` expected at least 2 arguments, got {len(formula.args())}"
            return f'(and {" ".join(to_smt2(arg) for arg in formula.args())})'
        elif formula.is_or():
            assert (
                len(formula.args()) >= 2
            ), f"`Or` expected at least 2 arguments, got {len(formula.args())}"
            return f'(or {" ".join(to_smt2(arg) for arg in formula.args())})'
        elif formula.is_le():
            assert (
                len(formula.args()) == 2
            ), f"`Le` expected 2 arguments, got {len(formula.args())}"
            return f"(<= {to_smt2(formula.arg(0))} {to_smt2(formula.arg(1))})"
        elif formula.is_lt():
            assert (
                len(formula.args()) == 2
            ), f"`Lt` expected 2 arguments, got {len(formula.args())}"
            return f"(< {to_smt2(formula.arg(0))} {to_smt2(formula.arg(1))})"
        elif formula.is_equals():
            assert (
                len(formula.args()) == 2
            ), f"`Equals` expected 2 arguments, got {len(formula.args())}"
            return f"(= {to_smt2(formula.arg(0))} {to_smt2(formula.arg(1))})"
        elif formula.is_plus():
            assert (
                len(formula.args()) >= 2
            ), f"`Plus` expected at least 2 arguments, got {len(formula.args())}"
            return f'(+ {" ".join(to_smt2(arg) for arg in formula.args())})'
        elif formula.is_minus():
            assert (
                len(formula.args()) == 2
            ), f"`Minus` expected 2 arguments, got {len(formula.args())}"
            return f"(- {to_smt2(formula.arg(0))} {to_smt2(formula.arg(1))})"
        elif formula.is_times():
            assert (
                len(formula.args()) >= 2
            ), f"`Times` expected at least 2 arguments, got {len(formula.args())}"
            return f'(* {" ".join(to_smt2(arg) for arg in formula.args())})'
        elif formula.is_real_constant():
            return str(formula.constant_value())
        elif formula.is_int_constant():
            return str(formula.constant_value())
        elif formula.is_symbol():
            return str(formula)
        else:
            raise ValueError(
                f"PolyHorn does not support '{op_to_str(formula.node_type())}' operator in (sub-)formula {formula}"
            )

    constraints = []
    for formula in solver.assertions:
        constraints.append(to_smt2(formula))

    smt2 = "\n".join(
        [f"(declare-const {var} {var.symbol_type()})" for var in free_vars]
    )
    smt2 += "\n"
    smt2 += "\n".join([f"(assert {constraint})" for constraint in constraints])
    smt2 += "\n(check-sat)\n(get-model)"
    print(smt2)
    return smt2


def execute(
    formula: Union[str, PysmtSolver], config: Union[str, dict]
) -> Tuple[str, dict]:
    """
    Execute PolyHorn on the formula with the given configuration

    Parameters
    ----------
    formula : Union[str, pysmt.solvers.solver.Solver]
        The formula to execute PolyHorn on. Either a string to a `.smt2` file
        or a pysmt.Solver object with the constraints already added
    config : Union[str, dict]
        The path to the config file or the parsed config file

    Returns
    -------
    str
        The satisfiability of the formula (sat, unsat, unknown)
    dict
        The model of the formula (if it is satisfiable)
    """
    if isinstance(config, str):
        config = load_config(config)
    elif isinstance(config, dict):
        config = add_default_config(config)
    else:
        raise ValueError(
            "Config must be either a path to a config file or a dictionary"
        )

    if isinstance(formula, str):
        with open(formula, "r") as file:
            formula = file.read()
    elif isinstance(formula, PysmtSolver):
        formula = pysmt_to_smt2(formula)
    else:
        raise ValueError(
            "Formula must be either a path to a smt2 file or a pysmt.Solver object"
        )

    return __execute(config, formula, Parser.parse_smt_file)


def execute_smt2(smt2: str, config_path: str) -> None:
    """
    Execute PolyHorn on the smt2 system

    Parameters
    ----------
    smt2 : str
        The smt2 system
    config_path : str
        The path to the config file

    Returns
    -------
    Tuple[bool, dict]
        A tuple with the first element being the satisfiability of the system and the second element being the model
    """
    config = load_config(config_path)
    return __execute(config, smt2, Parser.parse_smt_file)


def execute_readable(readable: str, config_path: str):
    """
    Execute PolyHorn on the readable system

    Parameters
    ----------
    readable : str
        The readable system
    config_path : str
        The path to the config file

    Returns
    -------
    Tuple[bool, dict]
        A tuple with the first element being the satisfiability of the system and the second element being the model
    """
    config = load_config(config_path)
    return __execute(config, readable, Parser.parse_readable_file)


def __execute(config: dict, input: str, parser_method):
    """
    Execute PolyHorn on the input system

    Parameters
    ----------
    config : dict
        The dictionary containing configuration information
    input : str
        The input system
    parser_method : Callable
        The method to parse the input

    Returns
    -------
    Tuple[bool, dict]
        A tuple with the first element being the satisfiability of the system and the second element being the model
    """

    parser = Parser(
        PositiveModel(
            [],
            config["theorem_name"],
            True,
            not config["SAT_heuristic"],
            not config["SAT_heuristic"],
            config["degree_of_sat"],
            config["degree_of_nonstrict_unsat"],
            config["degree_of_strict_unsat"],
            config["max_d_of_strict"],
            preconditions=[],
        )
    )

    parser_method(parser, input)
    output_path_exists = True
    try:
        if "output_path" not in config:
            output_path_exists = False
            config["output_path"] = (
                "./POLYHORN_delme_"
                + str(uuid.uuid4())
                + "".join(random.choices(string.ascii_uppercase + string.digits, k=9))
            )
            with open(config["output_path"], "x") as file:
                file.write("")
        print("Running solver...")
        sat, model = parser.model.run_on_solver(
            output_path=config["output_path"],
            solver_name=config["solver_name"],
            core_iteration_heuristic=config["unsat_core_heuristic"],
            constant_heuristic=False,
            real_values=not config["integer_arithmetic"],
        )
    finally:
        if not output_path_exists:
            os.remove(config["output_path"])
    return sat, model


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--smt2", type=str, help="Path to the smt2 file")
    parser.add_argument("--readable", type=str, help="Path to the readable file")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    if args.smt2:
        with open(args.smt2, "r") as file:
            smt2 = file.read()
        is_sat, model = execute_smt2(smt2, args.config)
    elif args.readable:
        with open(args.readable, "r") as file:
            readable = file.read()
        is_sat, model = execute_readable(readable, args.config)
    else:
        raise ValueError("Either --smt2 or --readable must be provided")

    print(f"The system is {is_sat}")
    if is_sat == "sat":
        print("Model:")
        for var, value in model.items():
            print(f"{var}: {value}")
