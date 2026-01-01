"""
Angr Query Collector

This script collects SMT queries from binary analysis using angr.
It symbolically executes the target binary and dumps the path constraints as SMT2 format.

Usage:
    python collect_angr_query.py <binary_path> [output_dir]

Arguments:
    binary_path: Path to the target binary
    output_dir: Optional directory to store SMT2 files

To dump OMT queries from symbolic memory model, use the following variant of claripy
https://github.com/notch1p/claripy
"""

import os
import sys
import logging
import angr
import claripy
from z3 import *
from claripy.backends.backend_z3 import BackendZ3
from typing import Optional, List

# Constants
MAX_QUERIES = 5000
SYMBOLIC_ARG_SIZE = 8  # Size in bits for each symbolic argument


class QueryCollector:
    def __init__(self, binary_path: str, output_dir: Optional[str] = None):
        self.binary_path = binary_path
        self.binary_name = os.path.basename(binary_path)
        self.output_dir = output_dir
        self.num_queries = 0

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def setup_symbolic_args(self) -> List[claripy.BV]:
        """Create symbolic arguments for binary execution"""
        return [claripy.BVS(f"arg{i}", SYMBOLIC_ARG_SIZE) for i in range(4)]

    def save_constraint(self, constraint: str, level: int, count: int) -> None:
        """Save constraint to file or print to stdout"""
        if self.output_dir:
            filename = f"{self.binary_name}-{level}-{count}.smt2"
            output_path = os.path.join(self.output_dir, filename)
            with open(output_path, "w") as f:
                f.write(constraint)
            logging.debug(f"Saved constraint to: {output_path}")
        else:
            print(constraint)

    def collect_queries(self) -> None:
        """Main collection logic"""
        try:
            # Initialize angr project
            proj = angr.Project(self.binary_path)
            init_state = proj.factory.entry_state(args=self.setup_symbolic_args())
            simgr = proj.factory.simgr(init_state)

            level = 0
            while simgr.active and self.num_queries <= MAX_QUERIES:
                logging.info(f"Processing level: {level}")

                for count, state in enumerate(simgr.active):
                    constraints = state.solver.constraints
                    if not constraints:
                        continue

                    solver = Solver()
                    backend = BackendZ3()
                    z3_constraints = backend.convert_list(constraints)
                    solver.add(z3_constraints)

                    self.save_constraint(solver.to_smt2(), level, count)
                    self.num_queries += 1

                    if self.num_queries > MAX_QUERIES:
                        logging.info(
                            f"Reached maximum number of queries ({MAX_QUERIES})"
                        )
                        return

                simgr.step()
                level += 1

        except Exception as e:
            logging.error(f"Error during collection: {str(e)}")
            raise


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    binary_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        collector = QueryCollector(binary_path, output_dir)
        collector.collect_queries()
    except Exception as e:
        logging.error(f"Failed to collect queries: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
