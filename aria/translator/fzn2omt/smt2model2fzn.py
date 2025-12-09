"""
This tool transforms a SMT2 model into a FZN (MZN) model.

Notice that the transformation preserves only those variables
that have been indicated as being part of the output model
in the original FZN (MZN) file. See
https://www.minizinc.org/doc-2.4.3/en/spec.html#output-items
for further information.
"""

import argparse
import io
import logging
import mmap
import os
import re
import sys
import tempfile
import common
import fzn2optimathsat


########################
########################
### FLATZINC SOLVING ###
########################
########################

def decode(config, solver_config=None):
    """Decodes a SMT2 model."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        config.smt2 = os.path.join(tmp_dir, "formula.smt2")
        config.ovars = os.path.join(tmp_dir, "output_vars.txt")
        output_trace = config.trace
        config.get_model = False

        # 1. generate SMT-LIB + OVARS files
        compile(config)

        # 2. parse OVARS skeleton
        try:
            oskeleton = common.extract_solution_skeleton(config.ovars)
        except FileNotFoundError:
            # This may happen if the original FZN file does not contain any output var.
            logging.critical("No output variables found.")
            oskeleton = None

        # 3. extract status
        status = extract_search_status(output_trace)

        # 4. extract model(s)
        models = extract_models(output_trace)

        # 5. print status + model(s)
        is_opt_problem = common.is_optimization_problem(config.model)
        common.print_search_status(config, status, models, oskeleton, is_opt_problem)


def extract_search_status(tracefile):
    """Extracts and returns the search status from the
    given tracefile."""

    ret = common.UNKNOWN

    if not common.is_file_empty(tracefile):

        uns_regex = re.compile(rb"^unsat\r?$", re.MULTILINE)
        sat_regex = re.compile(rb"^sat\r?$", re.MULTILINE)

        with io.open(tracefile, 'r', newline=None) as fd_trace:
            with mmap.mmap(fd_trace.fileno(), 0, access=mmap.ACCESS_READ) as output:

                uns_match = re.search(uns_regex, output)
                sat_match = re.search(sat_regex, output)

                assert sum([bool(x) for x in (uns_match, sat_match)]) <= 1

                if sat_match:
                    ret = common.SAT
                elif uns_match:
                    ret = common.UNSAT
                else:
                    ret = common.UNKNOWN

    return ret


def extract_models(tracefile):
    """Extract and returns all models contained
    in the given tracefile."""

    regex = re.compile((rb"\(define-fun (.*) \(\) (Int|Bool|Real|\(_ BitVec [0-9]+\))"
                        rb"\r?\n +(.*)\)"))

    models = []

    if not common.is_file_empty(tracefile):
        with io.open(tracefile, 'r', newline=None) as fd_trace:
            with mmap.mmap(fd_trace.fileno(), 0, access=mmap.ACCESS_READ) as output:
                model = {}
                for match in re.finditer(regex, output):
                    var, stype, value = (x.decode("utf-8") for x in match.groups())

                    if "BitVec" in stype:
                        stype = "BitVec"

                    if var in model:
                        models.append(model)
                        model = {}

                    model[var] = common.smtlib_to_python_type(stype, value)

                if model:
                    models.append(model)

    return models


###########################
###########################
### SMT-LIB COMPILATION ###
###########################
###########################

def compile(config):
    """Compiles a FlatZinc model in SMT-LIB format."""
    assert config.smt2

    optimathsat_config = argparse.Namespace(**vars(config))
    optimathsat_config.infinite_precision = True  # always override!
    optimathsat_config.compile_raw = True

    if not hasattr(optimathsat_config, 'ovars'):
        optimathsat_config.ovars = None

    fzn2optimathsat.optimathsat(optimathsat_config)


#####################
#####################
### I/O INTERFACE ###
#####################
#####################

def parse_cmdline_options():
    """parses and returns input parameters."""
    parser = argparse.ArgumentParser(description=("A tool for converting SMT2 models "
                                                  "to the FlatZinc (MiniZinc) format."),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    main_group = parser.add_argument_group("Main Options")

    main_group.add_argument("model", metavar="<model>.fzn",
                            type=argparse.FileType('r'),
                            help="The FlatZinc model",
                            action=common.check_file_ext('fzn'))

    main_group.add_argument("--trace", "--output-trace",
                            metavar="<filename>",
                            type=str,
                            help="Filename for the generated SMT-LIB output",
                            default=None)

    ###################
    # ENCODING config #
    ###################

    enc_group = parser.add_argument_group("Encoding Options")

    group = enc_group.add_mutually_exclusive_group()
    group.add_argument("--bv-enc", help="Encode ints with the SMT-LIB Bit-Vector type.",
                       action="store_true", default=False)
    group.add_argument("--int-enc", help="Encode ints with the SMT-LIB Int type.",
                       action="store_true", default=True)

    enc_group.add_argument("--cardinality-networks",
                           help="Enable cardinality networks (when applicable).",
                           action="store_true", default=False)

    ##################
    # MODEL PRINTING #
    ##################

    model_group = parser.add_argument_group("Model Options")

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

    # parse
    config = parser.parse_args()

    config.int_enc = not config.bv_enc
    config.get_model = True

    if config.finite_precision and \
            config.finite_precision >= 2:
        config.float_fmt = "%.{}g".format(config.finite_precision)
    else:
        config.float_fmt = "%g"

    return config


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

    config = parse_cmdline_options()

    decode(config)


if __name__ == "__main__":
    main()
