"""Configuration for the CDCL(T) SMT solver"""

import os

# Solver configuration
MAX_SOLVER_TIME_SEC = 300
NUM_SAMPLES_PER_ROUND = 10  # Boolean models to check per iteration
MAX_T_CHECKING_PROCESSES = 0  # 0 means use number of CPUs

# Process management
# Workers are given WORKER_SHUTDOWN_TIMEOUT seconds to exit gracefully
# before being forcefully terminated to prevent process leaks
WORKER_SHUTDOWN_TIMEOUT = 2.0  # seconds

# Preprocessing
SIMPLIFY_CLAUSES = True  # Simplify blocking clauses from theory solver

# SAT solver
SAT_SOLVER_ENGINE = 'glucose4'

# SMT query logging (optional, for debugging)
ENABLE_QUERY_LOGGING = False
QUERY_LOG_DIR = os.path.join(os.getcwd(), "smt_query_logs")
