import argparse
import os
import sys

from .apis import dump_smtlib, solveFile
from .config import Config

DESCRIPTION = "pyPMT Driver Script"

def _is_valid_file(arg: str) -> str:
    """
    Checks whether input PDDL files exist and are valid
    Args: arg (str): The path to the PDDL file.
    Returns: str: The validated path to the PDDL file.
    Raises:
        argparse.ArgumentTypeError: If the file does not exist or is not a PDDL file.
    """
    if not os.path.exists(arg):
        raise argparse.ArgumentTypeError('{} not found!'.format(arg))
    elif not os.path.splitext(arg)[1] == ".pddl":
        raise argparse.ArgumentTypeError('{} is not a PDDL file'.format(arg))
    else:
        return arg

def can_create_file(arg: str) -> str:
    """
    Check if a file can be created at the specified path.
    
    This function determines whether a file can be created at the given full
    path by checking the existence and write permissions of the directory
    containing the file.

    Args:
        arg (str): The full path where the file is to be created. 
                   This can be an absolute or relative path.

    Returns:
        str: The validated path to the file.

    Raises:
        argparse.ArgumentTypeError: If the file cannot be created at the specified path.
    """
    # Extract the directory path from the full path
    full_path = os.path.abspath(arg)  # if relative, transform to absolute
    directory = os.path.dirname(full_path)

    # Check if the file can be created at the specified path
    if os.access(directory, os.F_OK) and os.access(directory, os.W_OK):
        return arg
    else:
        raise argparse.ArgumentTypeError(f"Cannot create file at {arg}")

def add_shared_arguments(parser):
    """
    Adds shared command-line arguments to the provided argument parser.
    Parameters:
    parser (argparse.ArgumentParser): The argument parser to which the arguments will be added.
    """
    parser.add_argument('-p', '--problem', required=True, metavar='problem.pddl',
                         help='Path to PDDL problem file', type=_is_valid_file)
    parser.add_argument('-d', '--domain', required=True, metavar='domain.pddl',
                         help='Path to PDDL domain file', type=_is_valid_file)

    conf = Config() # temporarily create a Config object to get the valid encodings
    valid_encodings = conf.get_valid_configs()
    parser.add_argument('-e', '--encoding',
                        choices=valid_encodings.keys(),
                        required=True,
                        help='Specify the encoding to use. ' +
                             ''.join([f'{key}: {desc}, ' for key, desc in valid_encodings.items()]))

    parser.add_argument('-v', '--verbose', type=int, choices=range(0, 5),
                         default=1, help='Verbosity level (integer between 0 - 4).')

def create_parser():
    """
    Specifies valid arguments for the pyPMT CLI
    """
    parser = argparse.ArgumentParser(description = DESCRIPTION,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Subparser for the "dump" command
    parser_dump = subparsers.add_parser('dump', help='Dump the SMT encoding of a single step to a file')
    add_shared_arguments(parser_dump)
    parser_dump.add_argument('--output_file', required=True,
                             help='Dump the encoding to the specified file.', 
                             type=can_create_file)
    parser_dump.add_argument('--step', type=int, required=True,
                             help='Specify the step number that is going to be encoded.')

    # Subparser for the "solve" command
    parser_solve = subparsers.add_parser('solve', help='Try to solve a given planning problem')
    add_shared_arguments(parser_solve)
    parser_solve.add_argument('--bound', type=int, default=100, help='Upper bound.')

    return parser

def process_arguments(args, conf):
    """ gets the parsed command line arguments and sets the config object """
    if args.verbose:  # set the verbosity level
        conf.set('verbose', args.verbose)

    # decide on the correct encoding
    conf.set_config(args.encoding)

    if args.command == 'solve' and args.bound:  # parse the bound
        conf.set('ub', args.bound)

    if args.command == 'dump' and args.output_file and args.step: 
        # set the output file for the dump action
        conf.set('output_file', args.output_file)
        conf.set('encoded_step', args.step)

def dump_encoding(domain, problem, conf):
    """!  Given a domain, problem and config, dump a SMTLIB2 file """
    dump_smtlib(domain, problem, conf)

def solve_problem(domain, problem, conf):
    """!  Given a domain, problem and config, try to solve the planning problem """
    solveFile(domain, problem, conf)

def main(args=None):
    """ Entry point """
    if args is None:
        args = sys.argv[1:]

    # A new empty configuration object
    conf = Config()

    # Parse planner args
    parser = create_parser()
    args = parser.parse_args(args)

    # Ensure a command is provided
    if args.command is None:
        parser.print_help()
        return 1

    # now parse the arguments and set the configuration
    process_arguments(args, conf)

    if args.command == 'dump':
        dump_encoding(args.domain, args.problem, conf)
    elif args.command == 'solve':
        solve_problem(args.domain, args.problem, conf)
    else:
        parser.print_help()
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
