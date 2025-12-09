"""
A simple wrapper around the FlatZinc interface of OptiMathSAT, and
a compiler from FlatZinc to SMT-LIBv2 enriched with OptiMathSAT extensions.
"""

import argparse
import io
import mmap
import os
import shutil
import subprocess
import sys
import tempfile
import common


###################
###################
###################
### OPTIMATHSAT ###
###################
###################
###################

def binary_filename():
    """Returns the expected filename for OptiMathSAT's binary."""
    return "optimathsat.exe" if os.name == 'nt' else \
        "optimathsat"


def optimathsat(config, solver_config=None):
    """Runs OptiMathSAT with the given configuration."""

    common.exit_if_binary_not_in_path(binary_filename())

    if config.smt2:
        optimathsat_compile(config, solver_config)
    else:
        optimathsat_solve(config, solver_config)


########################
########################
### FLATZINC SOLVING ###
########################
########################

def optimathsat_solve(config, solver_config=None):
    """Solves a FlatZinc model with OptiMathSAT."""
    assert common.is_binary_in_path(binary_filename())
    assert not config.smt2

    args = optimathsat_solve_cmdline_args(config)
    if solver_config:
        args.extend(solver_config)

    subprocess.run(args, universal_newlines=True, check=True)


def optimathsat_solve_cmdline_args(config):
    """Determines the command-line arguments for the optimathsat call."""
    assert not config.smt2

    if config.all_solutions:
        if common.is_optimization_problem(config.model):
            config.partial_solutions = True
        else:
            config.all_solutions_opt = True

    args = [binary_filename(),
            "-input=fzn",
            config.model,
            "-opt.fzn.asoft_encoding={}".format(config.cardinality_networks),
            "-opt.fzn.bv_integers={}".format(not config.int_enc),
            "-opt.fzn.max_solutions={}".format(config.max_solutions),
            "-opt.fzn.all_solutions={}".format(config.all_solutions_opt),
            "-opt.fzn.partial_solutions={}".format(config.partial_solutions)]

    if not config.infinite_precision:
        args.append("-opt.fzn.finite_precision_model=True")
        args.append("-opt.fzn.finite_precision={}".format(config.finite_precision))

    if config.random_seed:
        args.append("-random_seed={}".format(config.random_seed))

    return args


###########################
###########################
### SMT-LIB COMPILATION ###
###########################
###########################

def optimathsat_compile(config, solver_config=None):
    """Compiles a FlatZinc model in SMT-LIB format."""
    assert common.is_binary_in_path(binary_filename())
    assert config.smt2

    args = optimathsat_compile_cmdline_args(config)
    if solver_config:
        args.extend(solver_config)

    result = subprocess.run(args, capture_output=True, universal_newlines=True, check=True)

    if result.returncode:
        print(result.stdout, end='')
        if result.stderr:
            print(result.stderr, end='')
        common.eprint("error: failed to generate SMT-LIB formula. ")
        sys.exit(1)

    if common.is_file_empty(config.smt2):
        common.eprint("error: failed to generate SMT-LIB formula. ")
        sys.exit(1)

    if config.ovars and common.is_file_empty(config.ovars):
        common.eprint(("error: failed to generate mapping between "
                       "SMT-LIB and FlatZinc output variables. "
                       "Please report this issue."))
        sys.exit(1)

    if not config.compile_raw:
        make_smtlib_compatible_with_optimathsat(config, solver_config)


def optimathsat_compile_cmdline_args(config):
    """Determines the command-line arguments for the optimathsat call."""
    assert config.smt2
    assert config.int_enc or config.bv_enc
    assert not config.int_enc or not config.bv_enc

    args = [binary_filename(),
            "-input=fzn",
            config.model,
            "-debug.api_call_trace=1",
            "-debug.api_call_trace_dump_config=False",
            "-debug.api_call_trace_filename={}".format(config.smt2),
            "-debug.solver_enabled=False",
            "-opt.fzn.asoft_encoding={}".format(config.cardinality_networks),
            "-opt.fzn.bv_integers={}".format(not config.int_enc)]

    if config.ovars:
        args.append("-opt.fzn.output_var_file={}".format(config.ovars))

    return args


def make_smtlib_compatible_with_optimathsat(config, solver_config):
    """Modifies SMT-LIB file with OptiMathSAT-specific syntax."""
    tmp_file_name = None

    with io.open(config.smt2, 'rt', newline=None) as in_f:
        with mmap.mmap(in_f.fileno(), 0, access=mmap.ACCESS_READ) as formula:

            with tempfile.NamedTemporaryFile(mode="w+t", delete=False, newline='') as out_f:
                tmp_file_name = out_f.name

                # Consume first two lines
                out_f.write(formula.readline().decode("utf-8"))
                out_f.write(formula.readline().decode("utf-8"))

                # Print header
                for header_line in common.get_smtlib_header_lines(config, "optimathsat"):
                    out_f.write(header_line)

                # Print configuration
                is_opt_problem = common.is_optimization_problem(config.model)
                if is_opt_problem:
                    priority_cfg = list(filter(lambda h: len(h) == 3,
                                               map(lambda g: g[14:],
                                                   filter(lambda f: "-opt.priority="
                                                                    in f, solver_config))))
                    if priority_cfg:
                        priority = priority_cfg[-1]
                        out_f.write("(set-option :opt-priority {})\n".format(priority))
                        out_f.write("(set-option :config opt.par.mode=callback)\n")

                # copy formula
                for line in iter(formula.readline, b""):
                    out_f.write(line.decode("utf-8"))

                # footer
                if is_opt_problem:
                    out_f.write("(get-objectives)\n")
                    out_f.write("(load-objective-model -1)\n")
                out_f.write("(get-model)\n")
                out_f.write("(exit)\n")

    # overwrite raw SMT-LIB file with OptiMathSAT-specific file
    if tmp_file_name:
        shutil.copy(tmp_file_name, config.smt2)


#####################
#####################
### I/O INTERFACE ###
#####################
#####################

def optimathsat_parse_cmdline_options():
    """parses and returns input parameters."""
    parser = argparse.ArgumentParser(description=("A tool for solving FlatZinc "
                                                  "problems with OptiMathSAT."),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    main_group = parser.add_argument_group("Main Options")

    main_group.add_argument("model", metavar="<model>.fzn",
                            type=argparse.FileType('r'),
                            help="The FlatZinc model",
                            action=common.check_file_ext('fzn'))

    main_group.add_argument("--smt2", "--output-smt2-to-file",
                            metavar="<filename>.smt2",
                            type=argparse.FileType('w'),
                            action=common.check_file_ext('smt2'),
                            help="Filename for the generated SMT-LIB output",
                            default=None)

    main_group.add_argument("--ovars", "--output-vars-file", metavar="<filename>",
                            type=argparse.FileType('w'), action=common.check_file_ext(),
                            help=("Filename for the generated mapping between "
                                  "internal SMT-LIB variables and the original "
                                  "FlatZinc variables that must be printed."),
                            default=None)

    main_group.add_argument("--compile-raw",
                            help=("Generate a raw SMT-LIB formula instead of one "
                                  "suitable for OptiMathSAT."),
                            action="store_true", default=False)

    ####################
    # ENCODING config #
    ####################

    enc_group = parser.add_argument_group("Encoding Options")

    # opt.fzn.bv_integers (false)
    group = enc_group.add_mutually_exclusive_group()
    group.add_argument("--bv-enc", help="Encode ints with the SMT-LIB Bit-Vector type.",
                       action="store_true", default=False)
    group.add_argument("--int-enc", help="Encode ints with the SMT-LIB Int type.",
                       action="store_true", default=True)

    # opt.fzn.asoft_encoding (true)
    enc_group.add_argument("--cardinality-networks",
                           help="Enable cardinality networks (when applicable).",
                           action="store_true", default=False)

    ##################
    # SEARCH config #
    ##################

    search_group = parser.add_argument_group("Search Options")

    # Random Seed
    search_group.add_argument("--random-seed", "-r",
                              help=("Set seed for pseudo-random number generators."),
                              metavar="N", type=int, default=None)

    # opt.fzn.partial_solutions (false)
    search_group.add_argument("--partial-solutions",
                              help=("Print any sub-optimal solution satisfying "
                                    "the input model."),
                              action="store_true", default=False)

    # opt.fzn.all_solutions (false)
    search_group.add_argument("--all-solutions-opt",
                              help=("Print all solutions of the input problem. "
                                    "If this is an optimization problem, it prints "
                                    "all solutions with the same optimal value."),
                              action="store_true", default=False)

    # MiniZinc style all solutions
    search_group.add_argument("--all-solutions", "-a",
                              help=("Print all solutions of the input problem. "
                                    "With satisfaction problems, it enables "
                                    "'--all-solutions-opt'. With optimization "
                                    "problems, it enables '--partial-solutions'."),
                              action="store_true", default=False)

    # opt.fzn.max_solutions (0)
    search_group.add_argument("--max-solutions", "-n",
                              help="Maximum number of solutions printed.",
                              metavar="N", type=int, default=0)

    ##################
    # MODEL PRINTING #
    ##################

    model_group = parser.add_argument_group("Model Options")

    # opt.fzn.finite_precision_model (false)
    # opt.fzn.finite_precision (32)
    model_group.add_argument("--infinite-precision",
                             help=("Print infinite-precision rational numbers "
                                   "as fractions. Overrides --finite-precision."),
                             action="store_true", default=False)
    model_group.add_argument("--finite-precision",
                             help=("Print infinite-precision rational numbers "
                                   "as finite precision decimals using the specified "
                                   "precision level. Must be larger or equal 2."),
                             action=common.check_finite_precision(),
                             metavar="prec", type=int, default=32)

    ###################
    # IGNORED config #
    ###################

    ignore_group = parser.add_argument_group("Ignored Options")

    # Ignored config
    ignore_group.add_argument("--free-search", "-f",
                              help=("No need to follow search specification. "
                                    "(OptiMathSAT ignores all search "
                                    "specifications)"),
                              action="store_true", default=True)
    ignore_group.add_argument("--num-threads", "-p",
                              help=("Number of threads. "
                                    "(OptiMathSAT is single threaded)"),
                              metavar="N", type=int, default=1)

    # parse
    config, solver_config = parser.parse_known_args()

    config.int_enc = not config.bv_enc

    return config, solver_config


############
############
### MAIN ###
############
############

def main():
    """The main entry point of this program."""
    min_python = (3, 7)
    if sys.version_info < min_python:
        sys.exit("Python %s.%s or later is required.\n" % min_python)

    config, solver_config = optimathsat_parse_cmdline_options()

    optimathsat(config, solver_config)


if __name__ == "__main__":
    main()
