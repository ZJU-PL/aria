#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
## musx.py
##
##  Created on: Jan 25, 2018
##      Author: Antonio Morgado, Alexey Ignatiev
##      E-mail: {ajmorgado, aignatiev}@ciencias.ulisboa.pt
##

"""
===============
List of classes
===============

.. autosummary::
    :nosignatures:

    MUSX

==================
Module description
==================

This module implements a deletion-based algorithm [1]_ for extracting a
*minimal unsatisfiable subset* (*MUS*) of a given (unsafistiable) CNF
formula. This simplistic implementation can deal with *plain* and *partial*
CNF formulas, e.g. formulas in the DIMACS CNF and WCNF formats.

.. [1] Joao Marques-Silva. *Minimal Unsatisfiability: Models, Algorithms
    and Applications*. ISMVL 2010. pp. 9-14

The following extraction procedure is implemented:

.. code-block:: python

    # oracle: SAT solver (initialized)
    # assump: full set of assumptions

    i = 0

    while i < len(assump):
        to_test = assump[:i] + assump[(i + 1):]
        if oracle.solve(assumptions=to_test):
            i += 1
        else:
            assump = to_test

    return assump

The implementation can be used as an executable (the list of available
command-line options can be shown using ``musx.py -h``) in the following
way:

::

    $ cat formula.wcnf
    p wcnf 3 6 4
    1 1 0
    1 2 0
    1 3 0
    4 -1 -2 0
    4 -1 -3 0
    4 -2 -3 0

    $ musx.py -s glucose3 -vv formula.wcnf
    c MUS approx: 1 2 0
    c testing clid: 0 -> sat (keeping 0)
    c testing clid: 1 -> sat (keeping 1)
    c nof soft: 3
    c MUS size: 2
    v 1 2 0
    c oracle time: 0.0001

Alternatively, the algorithm can be accessed and invoked through the
standard ``import`` interface of Python, e.g.

.. code-block:: python

    >>> from pysat.examples.musx import MUSX
    >>> from pysat.formula import WCNF
    >>>
    >>> wcnf = WCNF(from_file='formula.wcnf')
    >>>
    >>> musx = MUSX(wcnf, verbosity=0)
    >>> musx.compute()  # compute a minimally unsatisfiable set of clauses
    [1, 2]

Note that the implementation is able to compute only one MUS (MUS
enumeration is not supported).

==============
Module details
==============
"""

#
# ==============================================================================
from __future__ import print_function
import getopt
import os
import re
import sys
from pysat.formula import CNFPlus, WCNFPlus, CNF
from pysat.solvers import Solver, SolverNames


#
# ==============================================================================
class MUSX:
    """
    MUS eXtractor using the deletion-based algorithm. The algorithm is
    described in [1]_ (also see the module description above). Essentially,
    the algorithm can be seen as an iterative process, which tries to
    remove one soft clause at a time and check whether the remaining set of
    soft clauses is still unsatisfiable together with the hard clauses.

    The constructor of :class:`MUSX` objects receives a target
    :class:`.CNF` or `.WCNF` formula, a SAT solver name, and a verbosity level. Note
    that the default SAT solver is MiniSat22 (referred to as ``'m22'``, see
    :class:`.SolverNames` for details). The default verbosity level is
    ``1``.

    :param formula: input WCNF formula
    :param solver: name of SAT solver
    :param verbosity: verbosity level

    :type formula: :class:`.WCNF`
    :type solver: str
    :type verbosity: int
    """

    def __init__(self, formula_obj, solver="m22", verbosity=1):
        """
        Constructor.
        """

        topv, self.verbose = formula_obj.nv, verbosity

        # clause selectors and a mapping from selectors to clause ids
        self.sels, self.vmap = [], {}

        # to deal with a CNF* formula, we create its weighted version
        if isinstance(formula_obj, CNF):
            formula_obj = formula_obj.weighted()

        # constructing the oracle
        self.oracle = Solver(
            name=solver, bootstrap_with=formula_obj.hard, use_timer=True
        )

        if isinstance(formula_obj, WCNFPlus) and formula_obj.atms:
            # we are using CaDiCaL195 and it can use external linear engine
            cadical195 = getattr(SolverNames, "cadical195", [])
            if hasattr(SolverNames, "cadical195") and solver in cadical195:
                # CaDiCaL195 supports activate_atmost
                if hasattr(self.oracle, "activate_atmost"):
                    self.oracle.activate_atmost()

            error_msg = (
                f"{solver} does not support native cardinality "
                "constraints. Make sure you use the right type of "
                "formula."
            )
            assert self.oracle.supports_atmost(), error_msg

            for atm in formula_obj.atms:
                self.oracle.add_atmost(*atm)

        # relaxing soft clauses and adding them to the oracle
        for i, cl in enumerate(formula_obj.soft):
            topv += 1

            self.sels.append(topv)
            self.vmap[topv] = i

            self.oracle.add_clause(cl + [-topv])

    def __enter__(self):
        """
        'with' constructor.
        """

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        'with' destructor.
        """

        self.oracle.delete()
        self.oracle = None

    def delete(self):
        """
        Explicit destructor of the internal SAT oracle.
        """

        if self.oracle:
            self.oracle.delete()
            self.oracle = None

    def compute(self):
        """
        This is the main method of the :class:`MUSX` class. It computes a
        set of soft clauses belonging to an MUS of the input formula.
        First, the method checks whether the formula is satisfiable. If it
        is, nothing else is done. Otherwise, an *unsatisfiable core* of the
        formula is extracted, which is later used as an over-approximation
        of an MUS refined in :func:`_compute`.
        """

        # cheking whether or not the formula is unsatisfiable
        if not self.oracle.solve(assumptions=self.sels):
            # get an overapproximation of an MUS
            approx = sorted(self.oracle.get_core())

            if self.verbose:
                approx_str = " ".join([str(self.vmap[sel] + 1) for sel in approx])
                print(f"c MUS approx: {approx_str} 0")

            # iterate over clauses in the approximation and try to delete them
            mus = self._compute(approx)

            # return an MUS
            return list(map(lambda x: self.vmap[x] + 1, mus))
        return None

    def _compute(self, approx):
        """
        Deletion-based MUS extraction. Given an over-approximation of an
        MUS, i.e. an unsatisfiable core previously returned by a SAT
        oracle, the method represents a loop, which at each iteration
        removes a clause from the core and checks whether the remaining
        clauses of the approximation are unsatisfiable together with the
        hard clauses.

        Soft clauses are (de)activated using the standard MiniSat-like
        assumptions interface [2]_. Each soft clause :math:`c` is augmented
        with a selector literal :math:`s`, e.g. :math:`(c) \\gets (c \\vee
        \\neg{s})`. As a result, clause :math:`c` can be activated by
        assuming literal :math:`s`. The over-approximation provided as an
        input is specified as a list of selector literals for clauses in
        the unsatisfiable core.

        .. [2] Niklas Eén, Niklas Sörensson. *Temporal induction by
            incremental SAT solving*. Electr. Notes Theor. Comput. Sci.
            89(4). 2003. pp. 543-560

        :param approx: an over-approximation of an MUS
        :type approx: list(int)

        Note that the method does not return. Instead, after its execution,
        the input over-approximation is refined and contains an MUS.
        """

        i = 0

        while i < len(approx):
            to_test = approx[:i] + approx[(i + 1) :]
            sel, clid = approx[i], self.vmap[approx[i]]

            if self.verbose > 1:
                print(f"c testing clid: {clid}", end="")

            if self.oracle.solve(assumptions=to_test):
                if self.verbose > 1:
                    print(f" -> sat (keeping {clid})")

                i += 1
            else:
                if self.verbose > 1:
                    print(f" -> unsat (removing {clid})")

                approx = to_test

        return approx

    def oracle_time(self):
        """
        Method for calculating and reporting the total SAT solving time.
        """

        return self.oracle.time_accum()


#
# ==============================================================================
def parse_options():
    """
    Parses command-line option
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hs:v", ["help", "solver=", "verbose"])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        usage()
        sys.exit(1)

    solver = "m22"
    verbose = 0

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit(0)
        elif opt in ("-s", "--solver"):
            solver = str(arg)
        elif opt in ("-v", "--verbose"):
            verbose += 1
        else:
            assert False, f"Unhandled option: {opt} {arg}"

    return solver, verbose, args


#
# ==============================================================================
def usage():
    """
    Prints usage message.
    """

    print("Usage:", os.path.basename(sys.argv[0]), "[options] dimacs-file")
    print("Options:")
    print("        -h, --help")
    print("        -s, --solver     SAT solver to use")
    solver_list = "cd15, cd19, g3, lgl, mcb, mcm, mpl, m22, mc, mgh"
    print(f"                         Available values: {solver_list} " "(default: m22)")
    print("        -v, --verbose    Be verbose")


#
# ==============================================================================
if __name__ == "__main__":
    solver_name, verbosity, files = parse_options()

    if files:
        # parsing the input formula
        if re.search(r"\.wcnf[p|+]?(\.(gz|bz2|lzma|xz))?$", files[0]):
            formula = WCNFPlus(from_file=files[0])
        else:  # expecting '*.cnf[,p,+].*'
            formula = CNFPlus(from_file=files[0])

        with MUSX(formula, solver=solver_name, verbosity=verbosity) as musx:
            mus_result = musx.compute()

            if mus_result:
                if verbosity:
                    print(f"c nof soft: {len(formula.soft)}")
                    print(f"c MUS size: {len(mus_result)}")

                print("v", " ".join([str(clid) for clid in mus_result]), "0")
                print(f"c oracle time: {musx.oracle_time():.4f}")
