"""Main entry point for the QuantiSAT solver."""
import os
import time
from argparse import ArgumentParser
from enum import Enum
from warnings import warn

from aria.quant.quantisat.parser.formula_parser import parse_file

from aria.quant.quantisat.solver import MultiQuantiSAT, QuantiSAT, Skolem, SolverBackend
from aria.quant.quantisat.util import Result


class Verbosity(Enum):
    """
    Enum for the verbosity level of the tool.

    RESULT_ONLY: Only print the result of the formula (correct or not).
    RESULT_FORMULA: Print the result of the formula and the converted formula.
    RESULT_FORMULA_MODEL: Print the result of the formula, the converted formula and the model (if correct).
    """
    RESULT_ONLY = 1
    RESULT_FORMULA = 2
    RESULT_FORMULA_MODEL = 3

    def __str__(self):
        return self.name


class SolverType(Enum):
    """
    Enum for the solver type to be used.

    QUANTISAT: Use the QuantiSAT solver.
    MULTIQUANTISAT: Use the MultiQuantiSAT solver.
    SKOLEM: Use the Skolem solver.
    """
    QUANTISAT = QuantiSAT
    MULTIQUANTISAT = MultiQuantiSAT
    SKOLEM = Skolem

    def __str__(self):
        return self.name


class Experiment:
    """
    Class to run an experiment with a given solver type and arguments.

    The following steps are executed:
    1. Parse the input files.
    2. Convert the formula (replace quantifiers with Skolem functions).
    3. Solve the formula.

    A formula is solved correctly, if either the input formula or the negated
    formula is satisfiable.
    """

    def __init__(self, solver_type, experiment_args):
        """
        Initialize the experiment with the given solver type and arguments.

        Parameters
        ----------
        solver_type : SolverType
            The solver type to be used.
        experiment_args : argparse.Namespace
            The arguments of the experiment.
        """
        self.solver_type = solver_type
        self.file_pos = experiment_args.file_pos
        self.file_neg = experiment_args.file_neg
        self.verbose = experiment_args.verbose
        self.timeout = experiment_args.timeout
        self.args = experiment_args
        self.current_result = None
        self.current_conversion_time = None
        self.current_solving_time = None
        self.num_forall_vars = None
        self.num_exists_vars = None
        self.num_switches = None

    def run(self):
        """
        Run the experiment and print the result.

        The result is printed in the following order:
        - Status message (correct, incorrect, timeout)
        - Model (optional)
        - CSV line (for analysis) with the following columns:
            - Name of the file
            - Status message
            - Conversion time
            - Solving time
            - Number of universal variables
            - Number of existential variables
            - Number of quantifier
        """
        result = self.__main()

        print('Result:')
        print(self.__get_result_message())
        if result is not None:
            is_correct, model = result

            if self.verbose.value >= Verbosity.RESULT_FORMULA_MODEL.value and is_correct:
                print('Model:')
                for var, value in model.items():
                    print(f"{var}: {value}")

        print('CSV:')
        csv_line = (
            f'{os.path.basename(self.file_pos)},'
            f'{self.current_result.name},'
            f'{self.current_conversion_time},'
            f'{self.current_solving_time},'
            f'{self.num_forall_vars},'
            f'{self.num_exists_vars},'
            f'{self.num_switches}')
        print(csv_line)

    def __main(self):
        """
        Helper function to run the experiment.
        """
        constraint_system = parse_file(self.file_pos)
        neg_constraint_system = parse_file(self.file_neg)

        if constraint_system is None or neg_constraint_system is None:
            self.current_result = Result.PARSING_ERROR
            warn("Parsing error")
            return None

        per_formula_quantifier_counts = [
            quantified_formula.count_quantified_vars()
            for quantified_formula in constraint_system]
        per_formula_switches = [
            quantified_formula.count_quantifier_depth()
            for quantified_formula in constraint_system]

        self.num_forall_vars = max(
            num_forall_vars
            for num_forall_vars, _ in per_formula_quantifier_counts)
        self.num_exists_vars = max(
            num_exists_vars
            for _, num_exists_vars in per_formula_quantifier_counts)
        self.num_switches = max(per_formula_switches)

        args_dict = vars(self.args)
        solver = self.solver_type(args_dict)

        try:
            conversion_time = time.time()
            solver.convert(constraint_system, neg_constraint_system)
            self.current_conversion_time = time.time() - conversion_time
        except TimeoutError:
            self.current_result = Result.CONVERSION_TIMEOUT
            self.current_conversion_time = None
            return None

        if self.verbose.value >= Verbosity.RESULT_FORMULA.value:
            solver.print()

        try:
            self.current_result, model, self.current_solving_time = (
                solver.solve())
            if self.current_result == Result.CORRECT:
                return True, model
            if self.current_result == Result.INCORRECT:
                return False, None
            if self.current_result == Result.SOLVER_TIMEOUT:
                return None
        except TimeoutError:
            self.current_result = Result.SOLVER_TIMEOUT
            self.current_solving_time = None
        return None

    def __get_result_message(self):
        """
        Get a verbose result message of the experiment.
        """
        match self.current_result:
            case Result.CORRECT:
                return 'The formula was solved CORRECTLY.'
            case Result.INCORRECT:
                return 'The formula was solved INCORRECTLY.'
            case Result.CONVERSION_TIMEOUT:
                return 'The formula could not be converted due to timeout.'
            case Result.SOLVER_TIMEOUT:
                return 'The formula could not be solved due to timeout.'
            case Result.PARSING_ERROR:
                return 'The formula could not be parsed.'
            case _:
                raise ValueError(
                    'The experiment needs to be run before getting the '
                    'result message.')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--file_pos', type=str, required=True,
                        help='The file containing the formula to be solved.')
    parser.add_argument('--file_neg', type=str, required=True,
                        help='The file containing the negated formula to be solved.')
    parser.add_argument('--output', type=str,
                        help='The file to save the SMT2 formula.')
    parser.add_argument('--config', type=str, default='polyhorn_config.json.example',
                        help='The configuration file for PolyHorn.')
    parser.add_argument('--degree', type=int, default=1,
                        help='The degree of the templates to be used.')
    parser.add_argument(
        '--use-template', action='store_true',
        help='Use the template for the Skolem function. '
        '(Only used for Skolem)')
    parser.add_argument(
        '--solver', type=lambda v: SolverType[v],
        default=SolverType.QUANTISAT, choices=list(SolverType),
        help='The solver to be used.')
    parser.add_argument(
        '--verbose', type=lambda v: Verbosity[v],
        default=Verbosity.RESULT_ONLY, choices=list(Verbosity),
        help='The verbosity level of the output.')
    parser.add_argument(
        '--backend', type=lambda v: SolverBackend[v],
        default=SolverBackend.Z3, choices=list(SolverBackend),
        help='The backend to be used for the Skolem solver.')
    parser.add_argument('--timeout', type=int, default=15,
                        help='The timeout for the SMT solver. Default is 15s.')
    args = parser.parse_args()

    experiment = Experiment(args.solver.value, args)
    experiment.run()
