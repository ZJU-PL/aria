"""
This module implements a compact boxed optimization algorithm for bit-vector
optimization problems, parsing SMT-LIB2 files with maximize/minimize commands.

Be caraful: we use pysmt in this file...
"""
import os
import subprocess
import time
from typing import List, Tuple

from aria.optimization.omtbv.bv_opt_utils import cnt, res_z3_trans
from aria.optimization.omtbv.boxed.bv_boxed_z3 import solve_boxed_z3
from pysmt.shortcuts import *
from pysmt.smtlib.parser import SmtLibParser
from pysmt.smtlib.script import SmtLibCommand


class TSSmtLibParser(SmtLibParser):
    """Extended SMT-LIB parser that supports maximize/minimize commands.

    This parser adds support for maximize, minimize, and get-objectives
    commands while removing incompatible commands like get-value.
    """
    def __init__(self, env=None, interactive=False):
        SmtLibParser.__init__(self, env, interactive)

        # Add new commands
        #
        # The mapping function takes care of consuming the command
        # name from the input stream, e.g., '(init' . Therefore,
        # _cmd_init will receive the rest of the stream, in our
        # example, '(and A B)) ...'
        self.commands["maximize"] = self._cmd_maximize
        self.commands["minimize"] = self._cmd_minimize
        self.commands["get-objectives"] = self._cmd_get_objs

        # Remove unused commands
        #
        # If some commands are not compatible with the extension, they
        # can be removed from the parser. If found, they will cause
        # the raising of the exception UnknownSmtLibCommandError
        del self.commands["get-value"]

    def _cmd_maximize(self, current, tokens):
        """Parse a maximize command: (maximize <expr>)."""
        expr = self.get_expression(tokens)
        self.consume_closing(tokens, current)
        return SmtLibCommand(name="maximize", args=(expr,))

    def _cmd_minimize(self, current, tokens):
        """Parse a minimize command: (minimize <expr>)."""
        expr = self.get_expression(tokens)
        self.consume_closing(tokens, current)
        return SmtLibCommand(name="minimize", args=(expr,))

    def _cmd_get_objs(self, current, tokens):
        """Parse a get-objectives command: (get-objectives)."""
        self.consume_closing(tokens, current)
        return SmtLibCommand(name="get-objectives", args=())


def get_input(file: str) -> Tuple:
    """Parse SMT-LIB2 file and extract constraints and objectives.

    Args:
        file: Path to SMT-LIB2 file with maximize/minimize commands

    Returns:
        Tuple of (formula, objectives) where:
        - formula: Combined formula from all assert commands
        - objectives: List of [expression, direction] pairs where
          direction is 1 for maximize, 0 for minimize
    """
    ts_parser = TSSmtLibParser()
    script = ts_parser.get_script_fname(file)
    stack = []
    objs = []
    _And = get_env().formula_manager.And

    for cmd in script:
        if cmd.name == 'assert':
            stack.append(cmd.args[0])
        if cmd.name == 'maximize':
            objs.append([cmd.args[0], 1])
        if cmd.name == 'minimize':
            objs.append([cmd.args[0], 0])
    ori_formula = _And(stack)
    return ori_formula, objs


def map_bitvector(input_vars: List) -> List[List]:
    """Convert bit-vector objectives to boolean variable lists.

    For each bit-vector objective, creates boolean variables representing
    each bit position. For maximize objectives, checks if bits equal 1.
    For minimize objectives, checks if bits equal 0.

    Args:
        input_vars: List of [variable, direction] pairs where direction
                    is 1 for maximize, 0 for minimize

    Returns:
        List of lists, where each inner list contains boolean expressions
        representing bit conditions for one objective (MSB to LSB order)
    """
    bv2bool = {}  # Track boolean variables corresponding to each BV
    for var in input_vars:
        if var[0].get_type() != BOOL:
            name = var[0]
            size = var[0].symbol_type().width
            bool_vars = []
            if var[1] == 1:  # Maximize: check if bits equal 1
                for i in range(size):
                    x = size - 1 - i  # MSB to LSB order
                    v = Equals(BVExtract(var[0], x, x), BV(1, 1))
                    bool_vars.append(v)
                bv2bool[str(name)] = bool_vars
            else:  # Minimize: check if bits equal 0
                for i in range(size):
                    x = size - 1 - i
                    v = Equals(BVExtract(var[0], x, x), BV(0, 1))
                    bool_vars.append(v)
                bv2bool['-' + str(name)] = bool_vars
    objectives = []
    for key, value in bv2bool.items():
        objectives.append(value)
    return objectives


def check_assum(model, assums_obj: List[List[int]], unsol: List[int],
                objectives: List[List]) -> List[int]:
    """Check which assumptions are satisfied by the model.

    Args:
        model: Model from solver
        assums_obj: List of assumption lists for each objective
        unsol: List of indices of unsolved objectives
        objectives: List of objective boolean variable lists

    Returns:
        List of indices of objectives whose assumptions are satisfied
    """
    ass_index = []
    for i in unsol:
        sat = True
        assums = assums_obj[i] + [1]
        for j in range(len(assums)):
            if assums[j] and model[objectives[i][j]].is_false():
                sat = False
                break
        if sat:
            ass_index.append(i)
    return ass_index


def solve(formula, objectives: List[List]) -> List[List[int]]:
    """Compact boxed optimization using incremental SAT over bit slices.

    Strategy (pysmt-based):
    - Each bit-vector objective is mapped to a list of boolean literals
      (MSB→LSB). For maximize, the literal is “bit==1”; for minimize, “bit==0”.
    - We iteratively decide the next bit for every still-unsolved objective:
      * Build assumptions that require the currently known prefix plus the
        next bit == 1 for each objective.
      * Solve a disjunction of these assumptions; inspect the model to see
        which assumptions are simultaneously satisfiable.
      * For satisfiable objectives, commit the bit to 1 and extend their
        accumulated clause; otherwise commit the bit to 0.
    - Loop until all bits of all objectives are decided. This mimics Z3’s boxed
      priority: maximize the most significant bit first, then proceed downward.

    Args:
        formula: pysmt BoolRef of the hard constraints.
        objectives: List of per-objective boolean literals (one per bit, MSB→LSB).

    Returns:
        Per-objective bit lists (0/1) representing the optimal value for each
        objective in the same order as ``objectives``.
    """
    s = Solver()
    s.add_assertion(formula)
    unsol = list(range(len(objectives)))
    result = list([list() for _ in range(len(objectives))])
    res_clause = list([list() for _ in range(len(objectives))])  # Store results for each objective
    while len(unsol):  # While there are unsolved objectives
        assumption = {}  # Store assumptions for unsolved objectives and next bit
        for i in unsol:
            obj = objectives[i][len(result[i])]  # Get next bit to check
            if not res_clause[i]:
                assum = obj
            else:
                assum = And(res_clause[i], obj)
            assumption[i] = assum
        while len(assumption):
            a = 1
            for key, value in assumption.items():
                if a == 1:
                    a = value
                else:
                    a = Or(a, value)
            s.push()
            s.add_assertion(a)
            if s.solve():
                m = s.get_model()
                # Check which assumptions are satisfiable
                assum_index = check_assum(m, result, unsol, objectives)
                s.pop()
                for i in assum_index:  # Update satisfied assumptions: set bit to 1, move to next
                    obj = objectives[i][len(result[i])]
                    result[i].append(1)
                    if not res_clause[i]:
                        res_clause[i] = obj
                    else:
                        res_clause[i] = And(res_clause[i], obj)
                    if len(result[i]) == len(objectives[i]):
                        unsol.remove(i)
                        assumption.pop(i)
                    else:
                        obj = objectives[i][len(result[i])]
                        assumption[i] = And(assumption[i], obj)
            else:
                s.pop()
                finish = []
                for i in unsol:
                    result[i].append(0)
                    if len(result[i]) == len(objectives[i]):
                        finish.append(i)
                for i in finish:
                    unsol.remove(i)
                    assumption.pop(i)
                break
    return result


def res_2int(result: List[List[int]], objectives: List[List]) -> List[int]:
    """Convert binary result lists to integer values.

    Args:
        result: List of binary result lists (0/1 values)
        objectives: Original objectives list with [expression, direction] pairs
                    where direction is 1 for maximize, 0 for minimize

    Returns:
        List of integer values, converted based on objective direction
    """
    res_int = []
    for i in range(len(objectives)):
        score = cnt(result[i])
        if objectives[i][1] == 1:  # Maximize
            res_int.append(score)
        else:  # Minimize: invert the score
            l = len(result[i])
            score = 2 ** l - 1 - score
            res_int.append(score)
    return res_int


if __name__ == '__main__':
    # Get the project root directory (4 levels up from this file)
    # File is at: aria/optimization/omtbv/boxed/bv_boxed_compact.py
    # Project root is: aria/
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_dir))))

    # Construct the benchmark file path
    benchmark_file = os.path.join(project_root, 'benchmarks', 'smtlib2', 'omt', 'bv', 'box1.smt2')
    filename = os.path.normpath(benchmark_file)

    formu, objec = get_input(filename)
    objs = map_bitvector(objec)
    t = time.time()
    r = solve(formu, objs)
    r = res_2int(r, objec)
    print(r)
    # solve(formula, objs, res, res_cla, unsolved)
    print('t:', time.time() - t)

    obj_names = [str(obj[0]) for obj in objec]

    # Direct Z3 Optimize call to obtain objective values.
    t = time.time()
    z3_res = solve_boxed_z3(filename, objective_order=obj_names)
    t = time.time() - t
    print('t_z3:', t)
    print(z3_res)
    print(z3_res == r)
