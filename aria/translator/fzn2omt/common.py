"""
A library of common functions shared by the various executables.
"""

import argparse
import io
import mmap
import os
import re
import shutil
import sys
from fractions import Fraction
from functools import partial


###########
# GENERIC #
###########

def eprint(*args, **kwargs):
    """Prints an error message to stderr."""
    print(*args, file=sys.stderr, **kwargs)


#################
# BINARY LOOKUP #
#################

def is_binary_in_path(name):
    """Returns True if binary found in path."""
    return shutil.which(name) is not None


def exit_if_binary_not_in_path(name):
    """Quits if binary not found in path."""
    if not is_binary_in_path(name):
        sys.exit("error: `{}' not found.".format(name))


def is_file_empty(file_path):
    """Check if file is empty by confirming if its size is 0 bytes"""
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0


###########################
# ARGPARSE HELP FUNCTIONS #
###########################

def check_file_ext(extension=None):
    """Checks that the argument extension matches the given extension."""

    class Act(argparse.Action):  # pylint: disable=too-few-public-methods
        """Act."""

        def __call__(self, parser, namespace, file, option_string=None):
            """Check that the argument extension matches the given extension"""
            ext = os.path.splitext(file.name)[1][1:]
            if extension and ext != extension:
                option_string = '({})'.format(option_string) if option_string else ''
                parser.error("file doesn't end with `{}' {}".format(extension, option_string))
            else:
                setattr(namespace, self.dest, file.name)

    return Act


def check_finite_precision():
    """Checks that the argument finite precision matches the given restriction."""

    class Act(argparse.Action):  # pylint: disable=too-few-public-methods
        """Act."""

        def __call__(self, parser, namespace, value, option_string=None):
            """Check that the argument finite precision matches the given restriction."""
            if value < 2:
                option_string = '({})'.format(option_string) if option_string else ''
                parser.error("minimum finite precision is 2 {}".format(option_string))
            else:
                setattr(namespace, self.dest, value)

    return Act


#############################
# FLATZINC MODEL INSPECTION #
#############################

def is_optimization_problem(fzn_file):
    """Checks whether the FlatZinc problem contains an optimization goal."""
    with io.open(fzn_file, 'r', newline=None) as fd_fzn:
        with mmap.mmap(fd_fzn.fileno(), 0, access=mmap.ACCESS_READ) as fd_mmap:
            return re.search(b"solve .*satisfy;", fd_mmap) is None


#############################
# SMTLIB FORMULA INSPECTION #
#############################

def match_check_sat(bline):
    """Matches (check-sat) in the given argument."""
    regex = re.compile(br"^\(check-sat\)\r?$")
    return re.match(regex, bline)


def match_objective(bline):
    """Matches an optimization goal declaration
    in the given argument."""
    regex = re.compile(br"^\((minimize|maximize) ([^:]*?) ?(:signed)?\)\r?$")
    match = re.match(regex, bline)
    if match:
        obj = argparse.Namespace()
        obj.minimize = match[1] == b'minimize'
        obj.term = match[2].decode("utf-8")
        obj.bv_width = 0
        match = obj
    return match


def replace_sbv_to_int(config, bline):
    """Matches (sbv_to_int <term>) in the
    given argument and replaces with a
    SMT-LIB compatible encoding"""
    assert config.bv_enc

    regex_sbv = re.compile(br"\(sbv_to_int ([\w\_\-.]*?)\)")

    while True:

        match = re.search(regex_sbv, bline)
        if not match:
            break

        bv_var = match.group(1)
        bv_width = extract_bv_var_width(config, bv_var)

        var = bv_var.decode("utf-8")
        msb = bv_width - 1
        dsp = 2 ** bv_width
        expr = (f"(ite (= ((_ extract {msb} {msb}) {var}) #b0) "
                f"     (bv2nat {var}) "
                f"     (- (bv2nat {var}) {dsp}))")

        bline = bline.replace(b"(sbv_to_int %b)" % bv_var,
                              expr.encode("utf-8"))

    return bline


def extract_bv_var_width(config, bv_var):
    """Extracts and returns the width of a BitVec variable
    `bv_var` from the SMT-LIB formula in `config.smt2`"""
    assert config.bv_enc
    assert not is_file_empty(config.smt2)

    width = -1
    with io.open(config.smt2, 'rt', newline=None) as in_f:
        with mmap.mmap(in_f.fileno(), 0, access=mmap.ACCESS_READ) as formula:

            regex_decl = re.compile((br"\((?:define|declare)-fun +%b +"
                                     br"\(\) \(_ BitVec ([0-9]+)\)") % bv_var)

            match = re.search(regex_decl, formula)
            if match:
                width = int(match.group(1))
            else:
                raise Exception(("error: failed to determine "
                                 "BitVec width of `{}`").format(bv_var.decode("utf-8")))

    return width


###################
# SMT-LIB FORMULA #
###################

def get_smtlib_header_lines(config, solver):
    """Yields the header for the SMT-LIB file."""
    yield ";;\n"
    yield ";; fzn-optimathsat configuration:\n"
    yield ";;    --int-enc={}\n".format(config.int_enc)
    yield ";;    --bv-enc={}\n".format(config.bv_enc)
    yield ";;    --cardinality-networks={}\n".format(config.cardinality_networks)
    yield ";; target solver: {}\n".format(solver)
    yield ";;\n"


#############################
# SMT-LIB OUTPUT CONVERSION #
#############################

def smtlib_to_python(value):
    """Converts a constant SMT-LIB value to
    a Python type."""
    funcs = [smtlib_bool_to_python_bool,
             smtlib_int_to_python_int,
             smtlib_real_to_python_fraction]
    ret = None
    counter = 0
    for func in funcs:
        conv = func(value)
        if conv is not None:
            ret = conv
            break
        counter += 1
    if ret is None:
        ret = value
    return ret


def smtlib_to_python_type(stype, value):
    """Converts a constant SMT-LIB value to
    a Python type."""
    switch = {
        "Bool": smtlib_bool_to_python_bool,
        "Int": smtlib_int_to_python_int,
        "Real": smtlib_real_to_python_fraction,
        "BitVec": smtlib_bv_to_python_int,
    }
    ret = switch[stype](value)

    if ret is None:
        eprint("error: unable to convert `{}` of type `{}`.".format(value, stype))
        raise Exception("error: smtlib_to_python_type. please report this exception.")

    return ret


def smtlib_bool_to_python_bool(value):
    """Converts a constant SMT-LIB Bool value
    to bool."""
    ret = None
    if value in ("true", "false"):
        ret = (value == "true")
    return ret


def smtlib_int_to_python_int(value):
    """Converts a constant SMT-LIB Int value to int."""
    regex_min = re.compile(r"\( ?- (.*) ?\)")
    ret = None
    try:
        match = re.match(regex_min, value)
        if match:
            ret = - int(match.group(1))
        else:
            ret = int(value)
    except ValueError:
        pass
    return ret


def smtlib_real_to_python_fraction(value):
    """Converts a constant SMT-LIB Real value
    to Fraction."""
    ret = None

    regex_frac = re.compile((r"(?:\(- )?\( ?/ ([^\.]+)(?:\.0)? "
                             r"([^\.]+)(?:\.0)? ?\)(?:\)?)|([^\(]*)/(.*)"))
    regex_min = re.compile(r"\( ?- (.*) ?\)")

    match = re.match(regex_frac, value)
    if match:
        num, den = filter(None, match.groups())

        match = re.match(regex_min, num)
        if match:  # barcelogic
            num = - int(match.group(1))
        elif re.match(regex_min, value):  # z3
            num = - int(num)

        ret = Fraction(int(num), int(den))
    else:
        try:
            ret = Fraction(value)
        except ValueError:
            pass

    return ret


def smtlib_bv_to_python_int(value):
    """Converts a constant SMT-LIB BitVec value to int."""
    ret = None
    regex = re.compile(r"\(_ bv([0-9]+) ([0-9]+)\)")
    match = re.match(regex, value)
    if match:
        num, bits = map(int, match.groups())
        if num >= (2 ** (bits - 1)):
            num -= (2 ** bits)
        ret = num
    return ret


##################
# MODEL PRINTING #
##################

UNSAT = -1
UNKNOWN = 0
SAT = 1
OPTIMAL = 2


def print_search_status(config, status, models, oskeleton, is_opt_problem):
    """Prints the search status in a format
    compatible with FlatZinc."""
    switch = {
        UNSAT: "=====UNSATISFIABLE=====",
        UNKNOWN: "=====UNKNOWN=====",
        SAT: "----------",
        OPTIMAL: "==========",
    }
    if status in [UNSAT, UNKNOWN]:
        print(switch[status])
    else:
        # assign value to eliminated variables
        autocomplete = {}
        if models:
            autocomplete = extract_model_autocomplete(config, models[0], oskeleton)

        for model in models:
            model = {**model, **autocomplete}
            print_model(config, model, oskeleton)
            print(switch[status])
            if is_opt_problem:
                print(switch[OPTIMAL])

        if not models:
            print(switch[status])
            if is_opt_problem:
                print(switch[OPTIMAL])


def print_model(config, model, oskeleton):
    """Prints a model in FlatZic format."""
    for oinfo in oskeleton:
        mvalue = model_value(model, oinfo.term)
        svalue = model_value_to_str(config, mvalue)
        print(oinfo.str(svalue))


def model_value_to_str(config, term):  # pylint: disable=R0912
    """Returns the string representation of a model value."""
    ret = None
    if isinstance(term, bool):
        ret = "true" if term else "false"
    elif isinstance(term, int):
        ret = str(term)
    elif isinstance(term, Fraction):
        if config.infinite_precision:
            if term.denominator != 1:
                ret = "{}/{}".format(term.numerator, term.denominator)
            else:
                ret = str(term.numerator)
        else:
            ret = config.float_fmt % term
    elif isinstance(term, set):
        if term:
            ret = str(term)  # can only contain int
        else:
            ret = "{}"
    elif isinstance(term, list):
        ret = []  # may contain bool/float
        for sub_term in term:
            ret.append(model_value_to_str(config, sub_term))
        # throw away quotes
        ret = "[{}]".format(', '.join(ret))

    if ret is None:
        raise Exception("error: model_value_to_str; please report this exception.")

    return ret


def model_value(model, term):
    """Returns the model value of term."""
    ret = None
    if isinstance(term, (bool, int)):
        ret = term
    elif isinstance(term, Fraction):
        ret = term
    elif isinstance(term, str):
        mvalue = model.get(term, None)
        ret = mvalue
    elif isinstance(term, set):
        ret = set()
        for sub_term in term:
            val = model_value(model, sub_term)

            if isinstance(val, bool):
                if val:
                    val = int(sub_term[sub_term.rindex('.') + 1:])
                    ret.add(val)
            elif isinstance(val, int):
                ret.add(val)
            else:
                raise Exception("error: model_value; please report this exception.")

    elif isinstance(term, list):
        ret = []
        for sub_term in term:
            val = model_value(model, sub_term)
            ret.append(val)

    if ret is None:
        raise Exception("error: model_value; please report this exception.")

    return ret


#########################
# MODEL AUTO-COMPLETION #
#########################

def extract_model_autocomplete(config, model, oskeleton):  # pylint: disable=R0914,R1721
    """Returns a dictionary assigning a value to
    every variable required by the solution skeleton
    that is not assigned in the model."""
    autocomplete = {}

    # 1. extract required SMT-LIB variables from skeleton
    var2decl = {}
    for entry in oskeleton:
        n_vars = -1
        for n_vars, var in enumerate(extract_variables(entry.term)):
            if var not in var2decl:
                var2decl[var] = set()
            var2decl[var].add(entry.decl)
        if n_vars < 0:
            autocomplete[entry.decl] = entry.term

    # 2. find variables without a model value
    orphans = set(var2decl.keys()).difference(set(model.keys()))

    # 3. assign a default value
    if orphans and not is_file_empty(config.model):

        cache = {}

        with io.open(config.model, 'rt', newline=None) as in_f:
            with mmap.mmap(in_f.fileno(), 0, access=mmap.ACCESS_READ) as flatzinc:

                for orphan in orphans:
                    # search for a match in the cache first
                    for decl in var2decl[orphan]:
                        if decl in cache:
                            autocomplete[orphan] = cache[decl]
                            break
                    else:
                        # inspect flatzinc file
                        for decl in var2decl[orphan]:
                            regex = re.compile((rb'(?:array +\[.*\] of +)?(?:var)? *'
                                                rb'(set +of)? *(bool|int|float|.*)?'
                                                rb' *: *%b *(?:::|=|;)')
                                               % decl.encode("utf-8"))

                            match = re.search(regex, flatzinc)
                            if match:
                                value = False if match.group(1) is not None \
                                    else autocomplete_value(match.group(2))
                                autocomplete[orphan] = value
                                cache[decl] = value
                                break

    return autocomplete


def autocomplete_value(match):
    """Returns the default value for the given argument.
    It expects either b'bool', b'int' or b'float'."""
    ret = 0
    if match == b"bool":
        ret = False
    return ret


def extract_variables(term):
    """Yields all variables contained in term."""
    if isinstance(term, str):
        yield term
    elif isinstance(term, (set, list)):
        for sub_term in term:
            yield from extract_variables(sub_term)


#####################
# SOLUTION SKELETON #
#####################

def extract_solution_skeleton(ovars_file):  # pylint: disable=R0914
    """Extracts the required solution skeleton,
    and also the mapping between FlatZinc and
    SMT-LIB variables."""
    assert not is_file_empty(ovars_file)
    ret = []

    regex_bsc = re.compile(r"% (.*) = (.*);")
    regex_arr = re.compile(r"array(.*)d\((.*) (\[.*\])\)")
    regex_raw = re.compile(r"\{.+?\}|[^, \[\]]+")

    with io.open(ovars_file, 'rt', newline=None) as in_f:
        with mmap.mmap(in_f.fileno(), 0, access=mmap.ACCESS_READ) as skeleton:

            # skip first line
            skeleton.readline()

            for line in iter(skeleton.readline, b""):
                line = line.decode("utf-8")

                out = argparse.Namespace()

                # extract 'var = expr'
                out.decl, expr = re.match(regex_bsc, line).groups()

                # extract array(..., [])
                match = re.match(regex_arr, expr)
                if match:
                    dim, decl, raw_arr = match.groups()
                    out.str = partial("{0} = array{1}d({2} {3});".format,
                                      out.decl, dim, decl)
                    out.term = []

                    terms = re.findall(regex_raw, raw_arr)
                    for term in terms:
                        out.term.append(skeleton_term_to_python(term))

                else:
                    out.str = partial("{0} = {1};".format, out.decl)
                    out.term = skeleton_term_to_python(expr)

                ret.append(out)

    return ret


def skeleton_term_to_python(term):
    """Converts a skeleton term/value to a Python type."""
    regex_set = re.compile(r"\{.*\}")
    regex_ext = re.compile(r"[\w\d\.\:\-]+")

    ret = None
    if re.match(regex_set, term):
        ret = set()
        for match in re.findall(regex_ext, term):
            val = smtlib_to_python(match)
            ret.add(val)
    else:
        ret = smtlib_to_python(term)
    return ret
