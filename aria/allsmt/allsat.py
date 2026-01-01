#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
## models.py
##
##  Created on: Mar 4, 2018
##      Author: Alexey S. Ignatiev
##      E-mail: aignatiev@ciencias.ulisboa.pt
##

"""
===============
List of classes
===============

.. autosummary::
    :nosignatures:

    enumerate_models

==================
Module description
==================

The module implements a simple iterative enumeration of a given number of
models of :class:`.CNF` or :class:`CNFPlus` formula. In the latter case,
only :class:`.Minicard` can be used as a SAT solver. The module aims at
illustrating how one can work with model computation and enumeration.

The implementation facilitates the simplest use of a SAT oracle from the
*command line*. If one deals with the enumeration task from a Python
script, it is more convenient to exploit the internal model enumeration of
the :mod:`pysat.solvers` module. Concretely, see
:meth:`pysat.solvers.Solver.enum_models()`.

::

    $ cat formula.cnf
    p cnf 4 4
    -1 2 0
    -1 3 0
    -2 4 0
    3 -4 0

    $ models.py -e all -s glucose3 formula.cnf
    v -1 -2 +3 -4 0
    v +1 +2 -3 +4 0
    c nof models: 2
    c accum time: 0.00s
    c mean  time: 0.00s

==============
Module details
==============
"""

#
# ==============================================================================
from __future__ import print_function
import getopt
import os
import sys
from typing import Union, List, Tuple
from pysat.formula import CNFPlus
from pysat.solvers import Solver, SolverNames


#
# ==============================================================================
def enumerate_models(  # pylint: disable=too-many-arguments
    cnf_formula: CNFPlus,
    num_to_enum: Union[int, str],
    solver_name: str,  # pylint: disable=redefined-outer-name
    warm_start: bool = False,
) -> None:  # pylint: disable=redefined-outer-name
    """
    Enumeration procedure. It represents a loop iterating over satisfying
    assignment for a given formula until either all or a given number of
    them is enumerated.

    :param formula: input WCNF formula
    :param to_enum: number of models to compute
    :param solver: name of SAT solver
    :param warm: warm start flag

    :type formula: :class:`.CNFPlus`
    :type to_enum: int or 'all'
    :type solver: str
    :type warm: bool
    """

    with Solver(
        name=solver_name,
        bootstrap_with=cnf_formula.clauses,
        use_timer=True,
        warm_start=warm_start,
    ) as s:
        # adding native cardinality constraints if needed
        if cnf_formula.atmosts:
            # we are using CaDiCaL195 and it can use external linear engine
            if (
                hasattr(SolverNames, "cadical195")
                and solver_name in SolverNames.cadical195
            ):  # pylint: disable=no-member
                s.activate_atmost()  # pylint: disable=no-member

            assert (
                s.supports_atmost()
            ), f"{solver_name} does not support native cardinality constraints"

            for atm in cnf_formula.atmosts:
                s.add_atmost(*atm)

        # model enumeration and printing is done here
        computed = 0
        for i, model in enumerate(s.enum_models(), 1):
            model_str = " ".join([f'{"" if v > 0 else ""}{v}' for v in model])
            print(f"v {model_str} 0")

            computed = i
            if i == num_to_enum:
                break

        # some final statistics
        print(f"c nof models: {computed}")
        print(f"c accum time: {s.time_accum():.2f}s")

        if computed:
            print(f"c mean  time: {s.time_accum() / computed:.2f}s")


#
# ==============================================================================
def parse_options() -> Tuple[Union[int, str], str, bool, List[str]]:
    """
    Parses command-line options:

    Returns:
        Tuple containing (to_enum, solver, warm, args)
    """

    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "e:h:s:w", ["enum=", "help", "solver=", "warm"]
        )
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        usage()
        sys.exit(1)

    to_enum_t: Union[int, str] = 1
    solver_t: str = "g3"
    warm_t: bool = False

    for opt, arg in opts:
        if opt in ("-e", "--enum"):
            to_enum_t = str(arg)
            if to_enum_t == "all":
                to_enum_t = -1
            else:
                to_enum_t = int(to_enum_t)
        elif opt in ("-h", "--help"):
            usage()
            sys.exit(0)
        elif opt in ("-s", "--solver"):
            solver_t = str(arg)
        elif opt in ("-w", "--warm"):
            warm_t = True
        else:
            assert False, f"Unhandled option: {opt} {arg}"

    return to_enum_t, solver_t, warm_t, args


#
# ==============================================================================
def usage() -> None:
    """
    Prints usage message.
    """

    print("Usage:", os.path.basename(sys.argv[0]), "[options] dimacs-file")
    print("Options:")
    print("        -e, --enum=<int>         Compute at most this number of models")
    print(
        "                                 Available values: [1 .. INT_MAX], all (default: 1)"
    )
    print("        -h, --help               Show this message")
    print("        -s, --solver=<string>    SAT solver to use")
    print(
        "                                 Available values: cd, g3, g4, lgl, "
        "mcb, mcm, mpl, m22, mc, mgh (default = g3)"
    )
    print("        -w, --warm               Use solver's warm start mode")


#
# ==============================================================================
if __name__ == "__main__":
    # parsing command-line options
    to_enum, solver_name, warm_start, files = parse_options()

    # reading an input formula either from a file or from stdin
    if files:
        formula = CNFPlus(from_file=files[0])
    else:
        formula = CNFPlus(from_fp=sys.stdin)

    # pylint: disable=redefined-outer-name
    # Function parameters intentionally shadow outer scope variables
    enumerate_models(formula, to_enum, solver_name, warm_start)
