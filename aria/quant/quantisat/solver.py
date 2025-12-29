"""Solver classes for quantifier elimination and satisfiability checking."""
import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Tuple

from aria.quant.quantisat.converter import convert
from aria.quant.quantisat.cvc5solver import cvc5_call
from aria.quant.polyhorn.main import execute_smt2
from aria.quant.quantisat.quantifier import Quantifier
from aria.quant.quantisat.util import Result, set_timeout
from aria.quant.quantisat.z3solver import z3_call


class SolverBackend(Enum):
    """
    Enum class for the solver backend. This is only relevant for template based
    skolemization.

    The following backends are supported:
    - Z3
    - CVC5
    """
    Z3 = 1
    CVC5 = 2


class Solver(ABC):
    """
    Abstract class for a solver. This class defines the basic structure of a solver.
    It unifies the organisational overhead of all the solvers, such as the conversion
    and the call to the solver.
    """

    def __init__(self, args: dict):
        """
        Initialize the solver with the given arguments.

        Parameters
        ----------
        args : dict
            Dictionary with the following
            - degree: The degree of the polynomial to be used for the conversion.
            - output: The output file for the converted formula.
            - timeout: The timeout for the solver.
        """
        self.degree = args['degree']  # Only relevant for template based skolemization
        self.output = args['output']
        self.timeout = args['timeout']

        self.sat = True  # Solver specific representation of SAT
        self.unsat = False  # Solver specific representation of UNSAT
        self.unknown = None  # Solver specific representation of UNKNOWN

        self.smt2_pos = None  # The converted positive formula
        self.smt2_negs = None  # The converted negated formula

    @abstractmethod
    def _solve(self, smt2: str) -> Tuple[Result, Dict]:
        """
        Solve the given SMT2 formula. This method has to be implemented by the
        concrete solver.

        Parameters
        ----------
        smt2 : str
            The SMT2 formula to be solved.

        Returns
        -------
        Tuple[Result, Dict]
            A tuple with the result of the solver and the model (if the formula is correct).
        """

    def convert(self, pos_system: List[Quantifier], neg_system: List[Quantifier]):
        """
        Convert the given positive and negated system to an SMT2 formula. The
        result is stored in the attributes smt2_pos and smt2_negs.

        Parameters
        ----------
        pos_system : List[Quantifier]
            The positive system to be converted.
        neg_system : List[Quantifier]
            The negated system to be converted.
        """
        self.smt2_pos = set_timeout(
            convert, self.timeout, pos_system, degree=self.degree)
        self.smt2_negs = set_timeout(
            convert, self.timeout, neg_system, degree=self.degree)

        if self.output is not None:
            with open(self.output, 'w', encoding='utf-8') as f:
                f.write(self.smt2_pos)

            neg_path = os.path.splitext(self.output)[0] + '_neg.smt2'
            with open(neg_path, 'w', encoding='utf-8') as f:
                f.write(self.smt2_negs)

    def solve(self) -> Tuple[Result, Dict, float]:
        """
        Solve the converted formula. If the formula is correct, the model is returned.

        Returns
        -------
        Tuple[Result, Dict, float]
            A tuple with the result of the solver and the model (if the formula is correct)
            and the time the solver needed.
        """
        if self.smt2_pos is None:
            raise ValueError('The formula has not been converted yet.')

        start = time.time()
        try:
            result_pos = self._solve(self.smt2_pos)
        except TimeoutError:
            result_pos = None
        solver_time = time.time() - start

        # Check if the solver timed out
        if result_pos is not None:
            timeout = result_pos[0] == self.unknown
        else:
            timeout = True

        if not timeout:
            sat, model = result_pos
            if sat == self.sat:
                return Result.CORRECT, model, solver_time

        start = time.time()
        try:
            result_neg = self._solve(self.smt2_negs)
        except TimeoutError:
            result_neg = None
        solver_time = time.time() - start

        if result_neg is not None:
            sat_neg, model_neg = result_neg
            if sat_neg == self.unknown or (sat_neg == self.unsat and timeout):
                return Result.SOLVER_TIMEOUT, {}, 0.0
            if sat_neg == self.unsat:
                return Result.INCORRECT, {}, 0.0
            if sat_neg == self.sat:
                return Result.CORRECT, model_neg, solver_time
        return Result.SOLVER_TIMEOUT, {}, 0.0

    def print(self):
        """
        Print the converted formulas.
        """
        print("Converted formula:")
        print(self.smt2_pos)
        print("Converted Negated formula:")
        print(self.smt2_negs)


class MultiQuantiSAT(Solver):
    """
    Solver for multiple PolyHorn configurations. This solver is used to test
    multiple configurations of PolyHorn with different degrees.
    """

    def __init__(self, args: dict):
        """
        Initialize the MultiQuantiSAT solver with the given arguments.

        Parameters
        ----------
        args : dict
            Dictionary with the arguments needed by the QuantiSAT solver (except for config).
        """
        self.configs = {'configs/farkas-z3.json': [0, 1],
                        'configs/handelman-z3.json': [0, 1, 2],
                        'configs/putinar-z3.json': [0, 1, 2]}

        config_args = []
        for ph_config, degrees in self.configs.items():
            for degree in degrees:
                config_args.append({'config': ph_config, 'degree': degree})

        self.solvers = [QuantiSAT({**args, **config})
                        for config in config_args]

    def convert(self, pos_system: List[Quantifier], neg_system: List[Quantifier]):
        """
        Simply store the positive and negated system for later use.

        For performance reasons, the conversion is done only when needed, as
        the conversion is only needed for the solver that is actually used and
        most of the time, only one or two solver are needed.


        Parameters
        ----------
        pos_system : List[Quantifier]
            The positive system to be converted.
        neg_system : List[Quantifier]
            The negated system to be converted.
        """
        self.pos_system = pos_system
        self.neg_system = neg_system

    def solve(self):
        """
        Solve the given system with the different configurations and degrees.

        As soon as a correct model is found, the function returns. If any of the
        solvers times out, the function returns a timeout.

        Returns
        -------
        Tuple[Result, Dict, float]
            A tuple with the result of the solver and the model (if the formula is correct)
            and the time the solver needed.
        """
        results = []
        for solver in self.solvers:
            try:
                solver.convert(self.pos_system, self.neg_system)
                print("Solving with", solver.polyhorn_config, solver.degree)
                result, model, solve_time = solver.solve()
            except TimeoutError:
                results.append(Result.SOLVER_TIMEOUT)
                continue
            if result == Result.CORRECT:
                print("Solved with", solver.polyhorn_config, solver.degree)
                return Result.CORRECT, model, solve_time
            results.append(result)

        if all(result == Result.INCORRECT for result in results):
            return Result.INCORRECT, {}, 0.0
        return Result.SOLVER_TIMEOUT, {}, 0.0

    def _solve(self, smt2: str) -> Tuple[Result, Dict]:
        """
        Not implemented, as this solver implements the solve method itself.
        """
        raise NotImplementedError("This solver implements solve() directly")

    def print(self):
        """
        Not implemented, as the conversion is done only when needed.
        """
        raise NotImplementedError("Conversion is done only when needed")


class QuantiSAT(Solver):
    """
    Solver for PolyHorn. This solver is used to call PolyHorn with a specific
    configuration and formula.
    """

    def __init__(self, args: dict):
        """
        Initialize the QuantiSAT solver with the given arguments.

        Parameters
        ----------
        args : dict
            Dictionary with the following
            - degree: The degree of the polynomial to be used for the conversion.
            - output: The output file for the converted formula.
            - config: The configuration file for PolyHorn.
            - timeout: The timeout for the solver.
        """
        super().__init__(args)

        # execute_smt2 returns (bool, dict), so we use boolean values
        self.sat = True
        self.unsat = False
        self.unknown = None

        self.degree = args['degree']
        self.output = args['output']
        self.polyhorn_config = args['config']
        self.timeout = args['timeout']

    def _solve(self, smt2: str) -> Tuple[Result, Dict]:
        """
        Call PolyHorn with the given SMT2 formula.

        Parameters
        ----------
        smt2 : str
            The SMT2 formula to be solved.

        Returns
        -------
        Tuple[Result, Dict]
            A tuple with the result of the solver and the model (if the formula is correct).

        Raises
        ------
        TimeoutError
            If the solver times out.
        """
        return set_timeout(execute_smt2, self.timeout, self.polyhorn_config, smt2)


class Skolem(Solver):
    """
    Solver for Skolemization. This solver is used to call the Z3 or CVC5 solver
    with a skolemized formula.
    """

    def __init__(self, args: dict):
        """
        Initialize the Skolem solver with the given arguments.

        Parameters
        ----------
        args : dict
            Dictionary with the following
            - degree: The degree of the polynomial to be used for the conversion.
            - output: The output file for the converted formula.
            - timeout: The timeout for the solver.
            - use_template: Whether to use a template for skolemization.
            - backend: The backend to be used for skolemization.
        """
        super().__init__(args)

        self.sat = True
        self.unsat = False
        self.unknown = None

        self.use_template = args['use_template']
        self.degree = args['degree']
        if not self.use_template:
            self.degree = None
        self.timeout = args['timeout']
        self.backend = args['backend']

    def _solve(self, smt2: str) -> Tuple[Result, Dict]:
        """
        Call the solver with the given SMT2 formula.

        Parameters
        ----------
        smt2 : str
            The SMT2 formula to be solved.

        Returns
        -------
        Tuple[Result, Dict]
            A tuple with the result of the solver and the model (if the formula is correct).
        """
        if self.backend == SolverBackend.Z3:
            return z3_call(smt2, self.timeout)
        if self.backend == SolverBackend.CVC5:
            return cvc5_call(smt2, self.timeout)
        raise ValueError('Invalid backend.')
