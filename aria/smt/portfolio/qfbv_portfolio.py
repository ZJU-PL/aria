#!/usr/bin/env python3
"""
Multi-solver SMT formula solver using Z3 and PySAT.
Processes SMT2 files using various solving strategies in parallel.
"""

import argparse
import json
import logging
import multiprocessing
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import z3
from pysat.formula import CNF
from pysat.solvers import Solver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class SolverResult(Enum):
    """Possible results from solvers."""
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"

    @property
    def return_code(self) -> int:
        """Maps solver results to return codes."""
        return {
            self.SAT: 10,
            self.UNSAT: 20,
            self.UNKNOWN: 0
        }[self]

@dataclass
class SolverConfig:
    """Configuration for solver execution."""
    SAT_SOLVERS = [
        'cd', 'cd15', 'gc3', 'gc4', 'g3', 'g4',
        'lgl', 'mcb', 'mpl', 'mg3', 'mc', 'm22'
    ]
    
    Z3_PREAMBLES = [
        z3.AndThen(
            z3.With('simplify', flat_and_or=False),
            z3.With('propagate-values', flat_and_or=False),
            z3.Tactic('elim-uncnstr'),
            z3.With('solve-eqs', solve_eqs_max_occs=2),
            z3.Tactic('reduce-bv-size'),
            z3.With('simplify', 
                   som=True, 
                   pull_cheap_ite=True,
                   push_ite_bv=False,
                   local_ctx=True,
                   local_ctx_limit=10000000,
                   flat=True,
                   hoist_mul=False,
                   flat_and_or=False),
            z3.With('simplify', hoist_mul=False, som=False, flat_and_or=False),
            'max-bv-sharing',
            'ackermannize_bv',
            'bit-blast',
            z3.With('simplify', local_ctx=True, flat=False, flat_and_or=False),
            z3.With('solve-eqs', solve_eqs_max_occs=2),
            'aig',
            'tseitin-cnf'
        ),
        z3.AndThen(
            z3.With('simplify', flat_and_or=False),
            z3.With('propagate-values', flat_and_or=False),
            z3.With('solve-eqs', solve_eqs_max_occs=2),
            z3.Tactic('elim-uncnstr'),
            z3.With('simplify',
                   som=True,
                   pull_cheap_ite=True,
                   push_ite_bv=False,
                   local_ctx=True,
                   local_ctx_limit=10000000,
                   flat=True,
                   hoist_mul=False,
                   flat_and_or=False),
            z3.Tactic('max-bv-sharing'),
            z3.Tactic('bit-blast'),
            z3.With('simplify', local_ctx=True, flat=False, flat_and_or=False),
            'aig',
            'tseitin-cnf'
        )
    ]

class SATSolver:
    """Handles SAT solving operations."""
    
    @staticmethod
    def solve_sat(solver_name: str, cnf: CNF, result_queue: multiprocessing.Queue) -> None:
        """Solves SAT problem using specified solver."""
        try:
            with Solver(name=solver_name, bootstrap_with=cnf) as solver:
                logger.info(f"Solving with {solver_name}")
                result = SolverResult.SAT if solver.solve() else SolverResult.UNSAT
                result_queue.put(result)
        except Exception as e:
            logger.error(f"Error in {solver_name}: {str(e)}")
            result_queue.put(SolverResult.UNKNOWN)

    @staticmethod
    def preprocess_and_solve_sat(fml, qfbv_preamble, result_queue: multiprocessing.Queue) -> None:
        """Preprocesses and solves SAT problem."""
        try:
            qfbv_tactic = z3.With(qfbv_preamble, 
                                 elim_and=True,
                                 push_ite_bv=True,
                                 blast_distinct=True)
            after_simp = qfbv_tactic(fml).as_expr()

            if z3.is_true(after_simp):
                result_queue.put(SolverResult.SAT)
                return
            elif z3.is_false(after_simp):
                result_queue.put(SolverResult.UNSAT)
                return

            g = z3.Goal()
            g.add(after_simp)
            cnf = CNF(from_string=g.dimacs())

            with multiprocessing.Manager() as manager:
                queue = multiprocessing.Queue()
                processes = []
                
                for solver in SolverConfig.SAT_SOLVERS:
                    p = multiprocessing.Process(
                        target=SATSolver.solve_sat,
                        args=(solver, cnf, queue)
                    )
                    processes.append(p)
                    p.start()

                try:
                    result = queue.get(timeout=300)  # 5 minute timeout
                    result_queue.put(result)
                except Exception:
                    result_queue.put(SolverResult.UNKNOWN)
                finally:
                    for p in processes:
                        p.terminate()

        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            result_queue.put(SolverResult.UNKNOWN)

class FormulaParser:
    """Handles SMT formula parsing and solving."""
    
    @staticmethod
    def solve(file_name: str) -> SolverResult:
        """Main solving routine."""
        try:
            fml_vec = z3.parse_smt2_file(file_name)
            fml = fml_vec[0] if len(fml_vec) == 1 else z3.And(fml_vec)
            print(fml)

            with multiprocessing.Manager() as manager:
                result_queue = multiprocessing.Queue()
                processes = []

                # Start preprocessing processes
                for preamble in SolverConfig.Z3_PREAMBLES:
                    p = multiprocessing.Process(
                        target=SATSolver.preprocess_and_solve_sat,
                        args=(fml, preamble, result_queue)
                    )
                    processes.append(p)
                    p.start()

                # Start Z3 solver process
                p = multiprocessing.Process(
                    target=lambda: result_queue.put(
                        SolverResult(z3.Solver().check(fml).__str__())
                    )
                )
                processes.append(p)
                p.start()

                try:
                    result = result_queue.get(timeout=600)  # 10 minute timeout
                except Exception as ex:
                    print(ex)
                    result = SolverResult.UNKNOWN
                finally:
                    for p in processes:
                        p.terminate()

                return result

        except Exception as e:
            logger.error(f"Error solving formula: {str(e)}")
            return SolverResult.UNKNOWN

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SMT formula solver")
    parser.add_argument("request_directory", help="Directory containing input files")
    args = parser.parse_args()

    try:
        with open(os.path.join(args.request_directory, "input.json")) as f:
            input_json = json.load(f)

        formula_file = input_json.get("formula_file")
        if not formula_file:
            raise ValueError("No formula file specified in input.json")

        result = FormulaParser.solve(formula_file)
        
        solver_output = {
            "return_code": result.return_code,
            "artifacts": {
                "stdout_path": os.path.join(args.request_directory, "stdout.log"),
                "stderr_path": os.path.join(args.request_directory, "stderr.log")
            }
        }

        output_path = os.path.join(args.request_directory, "solver_out.json")
        with open(output_path, "w") as f:
            json.dump(solver_output, f, indent=2)

        logger.info(f"Result: {result.value} (code: {result.return_code})")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()