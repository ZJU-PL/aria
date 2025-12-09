#!/usr/bin/env python3
"""
The entrance of the SMT solving engines for the following logics
QF_BV, QF_UFBV, QF_AUFBV, QF_ABV
QF_FP, QF_BVFP, QF_UFFP, QF_AUFBVFP

This version includes automatic logic detection from SMT-LIB files.
"""
import os
import signal
import logging
import psutil
import re
import sys
from aria.smt.bv import QFBVSolver
from aria.smt.bv import QFUFBVSolver
from aria.smt.bv import QFAUFBVSolver
from aria.smt.fp import QFFPSolver
from aria.smt.fp import QFAUFBVFPSolver
from aria.utils import SolverResult

G_ARGS = None


def signal_handler(sig: int, frame) -> None:
    parent = psutil.Process(os.getpid())
    for child in parent.children(recursive=True):
        child.kill()


def detect_logic_from_file(file_path):
    """Extract logic type from SMT-LIB file"""
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if 'set-logic' in line and not line.strip().startswith(';'):
                    match = re.search(r'\(set-logic\s+([A-Z_]+)\s*\)', line)
                    if match:
                        return match.group(1)
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    return None


def process_file(filename: str):
    """Process one file"""
    # If logic wasn't specified, try to detect it from the file
    if G_ARGS.logic == "AUTO":
        detected_logic = detect_logic_from_file(filename)
        if detected_logic:
            if G_ARGS.verbosity > 0:
                print(f"Detected logic: {detected_logic}")
            logic = detected_logic
        else:
            # Use default logic if detection fails
            logic = "QF_BV"
            if G_ARGS.verbosity > 0:
                print(f"Could not detect logic from file, using default: {logic}")
    else:
        # Use the specified logic
        logic = G_ARGS.logic
    logic2solver = {'QF_BV': QFBVSolver,
                    'QF_UFBV': QFUFBVSolver,
                    'QF_AUFBV': QFAUFBVSolver,
                    'QF_ABV': QFAUFBVSolver,
                    'QF_FP': QFFPSolver,
                    'QF_BVFP': QFFPSolver,
                    'QF_UFFP': QFFPSolver,
                    "QF_ABVFP": QFAUFBVFPSolver,
                    "QF_AUFBVFP": QFAUFBVFPSolver
                    }

    if logic in logic2solver:
        solver_class = logic2solver[logic]
        solver_class.sat_engine = G_ARGS.sat_engine
        sol = solver_class()
        res = sol.solve_smt_file(filename)
        if res == SolverResult.SAT:
            print("sat")
        elif res == SolverResult.UNSAT:
            print("unsat")
        else:
            print("unknown")
    else:
        raise NotImplementedError(f'Unsupported logic: {logic}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="SMT solver for BV and FP logics")
    parser.add_argument(
        '--timeout',
        dest='timeout',
        default=8,
        type=int,
        help='timeout in seconds')
    parser.add_argument(
        '--workers',
        dest='workers',
        default=1,
        type=int,
        help='number of threads/processes')
    parser.add_argument(
        '--logic',
        dest='logic',
        default='AUTO',  # Changed default to AUTO for automatic detection
        type=str,
        help='logic of the formula (AUTO for automatic detection)')
    parser.add_argument('--model', dest='model', default=False, action='store_true',
                        help='enable model generation or not')
    parser.add_argument('--verbose', dest='verbosity', default=0, type=int, help="verbosity")
    parser.add_argument('--unsat_core', dest='unsat_core', default=False, action='store_true',
                        help='enable core generation or not')
    parser.add_argument('--incremental', dest='incremental', default=False, action='store_true',
                        help='enable incremental solving or not')
    parser.add_argument('--sat_engine', dest='sat_engine', default="mgh", type=str,
                        help='set the SAT backend: z3, cd(cadical103), cd15(cadical153),'
                             'gc3(gluecard3), gc4(glucard4), g3(glucose3), g4(glucose4),'
                             'lgl(lingeling), mcb(maplechrono), mcm(maplecm), mpl(maplesat)'
                             'mg3(mergesat3), mc(minicard), m22(minisat22, mgh(minsatgh)')
    parser.add_argument('infile', help='the input file (in SMT-LIB v2 format)')

    G_ARGS = parser.parse_args()

    if G_ARGS.verbosity > 0:
        if G_ARGS.verbosity == 1:
            logging.basicConfig(level=logging.INFO)
        elif G_ARGS.verbosity == 2:
            logging.basicConfig(level=logging.WARNING)
        else:
            logging.basicConfig(level=logging.DEBUG)

    # Which signals should we?
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    # Registers signal handler, so we can kill all of our child processes.
    process_file(G_ARGS.infile)
