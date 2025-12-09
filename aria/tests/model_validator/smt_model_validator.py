"""
From SMT-COMP
https://github.com/SMT-COMP/postprocessors/tree/master/model-validation-track
"""
import argparse
import sys
from os import path

from pysmt.exceptions import PysmtSyntaxError
from pysmt.shortcuts import get_env
from pysmt.smtlib.parser import SmtLibParser
from pysmt.smtlib.utils import SmtLibModelValidationSimplifier

get_env().allow_empty_var_names = True


def readModel(parser, model_file, input_file):
    """Read a model file and return the status and the model."""
    with open(model_file) as script:
        lino = 0
        for line in script:
            lino += 1
            read_status = line.strip()
            if read_status != 'success':
                break

        if read_status == "unknown":
            print("model_validator_pysmt_status=UNKNOWN")
            print("model_validator_pysmt_error=solver_returned_unknown")
            sys.exit(0)
        if read_status == "unsat":
            status = None
            with open(input_file, 'r') as infile:
                for line in infile:
                    if ":status" in line:
                        if "unsat" in line:
                            status = "unsat"
                            print("model_validator_pysmt_status=UNKNOWN")
                            print("model_validator_pysmt_error=the_problem_status_is_unsatisfiable")
                        elif "sat" in line:
                            status = "sat"
                            print("model_validator_pysmt_status=INVALID")
                            print("model_validator_pysmt_error=the_problem_status_is_satisfiable")
                        elif "unknown" in line:
                            print("model_validator_pysmt_status=UNKNOWN")
                            print("model_validator_pysmt_error=the_problem_status_is_unknown")
                            status = "unknown"
                        break
            # the benchmark scrambler removes the status line, in case of a
            # benchmark without status line we assume satisfiability
            if not status:
                print("model_validator_pysmt_status=INVALID")
                print("model_validator_pysmt_error=solver_returned_unsat")
            sys.exit(0)
        if read_status != "sat":
            raise PysmtSyntaxError("'sat' expected at line %d" % lino)
        # Return UNKNOWN if the output is only "sat" and does not contain a model

        model, interpretation = parser.parse_model(script)
        return model, interpretation


def readSmtFile(parser, smt_file):
    """Read a SMT file and return the formula and the symbols."""
    with open(smt_file) as stream:
        script = parser.get_script(stream)
        formula = script.get_strict_formula()
        return formula, script.get_declared_symbols()


def checkFullModel(model, interpretation, symbols):
    """Check if the model is full."""
    if len(model) + len(interpretation) > len(symbols):
        print("model_validator_pysmt_status=UNKNOWN")
        print("model_validator_pysmt_error=more_variables_in_model_than_input")
        sys.exit(0)

    for symbol in symbols:
        if symbol not in model and symbol not in interpretation:
            print("model_validator_pysmt_status=UNKNOWN")
            print("model_validator_pysmt_error=missing_model_value")
            sys.exit(0)


def validateModel(smt_file, model_file, input_file):
    """Validate a model."""
    try:
        if not path.exists(smt_file):
            raise Exception("File not found: {}".format(smt_file))

        if not path.exists(model_file):
            raise Exception("File not found: {}".format(model_file))

        if path.getsize(model_file) == 0:
            print("model_validator_pysmt_status=UNKNOWN")
            print("model_validator_pysmt_error=no_output")
            sys.exit(0)

        parser = SmtLibParser()

        (formula, symbols) = readSmtFile(parser, smt_file)
        model, interpretation = readModel(parser, model_file, input_file)

        checkFullModel(model, interpretation, symbols)
        simplifier = SmtLibModelValidationSimplifier()
        result = simplifier.simplify(formula.substitute(model, interpretation))

        if result.is_false():
            print("model_validator_pysmt_status=INVALID")
            print("model_validator_pysmt_error=model_evaluates_to_false")
        elif result.is_true():
            print("model_validator_pysmt_status=VALID")
            print("model_validator_pysmt_error=none")
        else:
            print("model_validator_pysmt_status=UNKNOWN")
            print("model_validator_pysmt_error=not_full_model")
    except Exception as exp:
        print("model_validator_pysmt_status=UNKNOWN")
        print("model_validator_pysmt_error=unhandled_exception")
        # starexec cut the last = instead of the first
        print("model_validator_pysmt_exception=\"{}\"".format(
            str(exp).replace("'", "\\'").replace('"', '\\"').replace('\n', ' ').replace('=', '⩵')))
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Model validator for QF logics with bit-vectors and linear arithemetic.')
    parser.add_argument('--smt2', type=str,
                        help='SMT-LIB v2 benchmark',
                        required=True)
    parser.add_argument('--model', type=str,
                        help='The full model returned by the SMT solver',
                        required=True)

    args = parser.parse_args()
    validateModel(args.smt2, args.model, args.smt2)


try:
    main()
except Exception as e:
    print("model_validator_pysmt_status=UNKNOWN")
    print("model_validator_pysmt_error=toplevel_unhandled_exception")
    print("model_validator_pysmt_exception=\"{}\"".format(
        str(e).replace("'", "\\'").replace('"', '\\"').replace('\n', ' ')))
    sys.exit(0)
