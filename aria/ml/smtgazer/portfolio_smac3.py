"""
SMAC3 Integration for SMTgazer Portfolio Optimization

This module integrates SMAC3 (Sequential Model-based Algorithm Configuration) with SMTgazer
to optimize solver timeout configurations within each problem cluster. SMAC3 uses Bayesian
optimization to find optimal timeout distributions for solver portfolios.

Key Features:
- ConfigSpace for defining search space of timeout configurations
- SMAC3 for Bayesian optimization of timeout parameters
- Cross-validation support for robust performance evaluation
- Integration with SMTgazer's clustering and solver selection

The optimization searches for optimal timeout distributions (t1, t2, t3, t4) that sum to 1200s
and are allocated across the selected solvers in the portfolio.

Author: SMTgazer Team
Publication: ASE 2025
"""

import json
import sys

from ConfigSpace import Configuration, ConfigurationSpace, Float, Categorical
from smac import HyperparameterOptimizationFacade, Scenario

# Global configuration variables (parsed from command line arguments)
x1 = 0      # Timeout for solver 1 (seconds)
x2 = 0      # Timeout for solver 2 (seconds)
x3 = 0      # Timeout for solver 3 (seconds)
s1 = -1     # Index of solver 1 in portfolio
s2 = -1     # Index of solver 2 in portfolio
s3 = -1     # Index of solver 3 in portfolio
s4 = -1     # Index of solver 4 in portfolio
dataset = "" # SMT dataset name (e.g., "Equality+LinearArith")
seed = ""   # Random seed for reproducible optimization
w1 = 0.5    # SMAC3 weight parameter 1
w2 = 0.5    # SMAC3 weight parameter 2
start_idx = 0  # Cross-validation fold start index
cluster = 0    # Cluster ID for optimization (default value)

# Parse command line arguments for SMAC3 configuration
for i in range(len(sys.argv) - 1):
    if sys.argv[i] == '-t1':
        x1 = float(sys.argv[i+1])      # Timeout for first solver
    elif sys.argv[i] == '-t2':
        x2 = float(sys.argv[i+1])     # Timeout for second solver
    elif sys.argv[i] == '-t3':
        x3 = float(sys.argv[i+1])     # Timeout for third solver
    elif sys.argv[i] == '-s1':
        s1 = int(sys.argv[i+1])       # Index of first solver
    elif sys.argv[i] == '-s2':
        s2 = int(sys.argv[i+1])       # Index of second solver
    elif sys.argv[i] == '-s3':
        s3 = int(sys.argv[i+1])       # Index of third solver
    elif sys.argv[i] == '-s4':
        s4 = int(sys.argv[i+1])       # Index of fourth solver
    elif sys.argv[i] == '-cluster':
        cluster = int(sys.argv[i+1])   # Cluster ID for optimization
    elif sys.argv[i] == '-dataset':
        dataset = str(sys.argv[i+1])   # Dataset name
    elif sys.argv[i] == '-seed':
        seed = str(sys.argv[i+1])     # Random seed
    elif sys.argv[i] == '-w1':
        w1 = float(sys.argv[i+1])     # SMAC3 weight parameter
    elif sys.argv[i] == '-si':
        start_idx = int(sys.argv[i+1]) # Cross-validation fold index

# Calculate fourth timeout to ensure total = 1200 seconds
x4 = 1200 - x1 - x2 - x3
w2 = 1 - w1  # Complementary weight parameter
def train(config: Configuration, seed_val: int = 0) -> float:
    """
    SMAC3 objective function for optimizing solver timeout configurations.

    This function evaluates a given timeout configuration by simulating solver
    execution on training problems within a cluster. It implements a sequential
    solver strategy where each solver gets a timeout allocation and runs until
    either it solves the problem or its timeout expires.

    Args:
        config (Configuration): SMAC3 configuration containing timeout and
            solver parameters
        seed_val (int): Random seed for reproducible evaluation

    Returns:
        float: Total PAR2 time (lower is better) for the given configuration

    The evaluation process:
    1. Extract solver indices and timeout configuration from SMAC3 config
    2. Load training data for the specified cluster
    3. For each problem in the cluster:
       - Try each solver in sequence with its allocated timeout
       - If a solver succeeds within its timeout, add its runtime
       - If all solvers fail, add penalty (2400s)
    4. Return total PAR2 time across all problems
    """
    import random
    # Extract parameters from SMAC3 configuration
    dataset_name = config['dataset']
    cluster_id = config['cluster']
    random_seed = 1
    random.seed(random_seed)
    # Handle dataset name variations
    if dataset_name == "ELA":
        dataset_name = "Equality+LinearArith"

    # Map dataset names to data directory names
    if dataset_name == "Equality+LinearArith":
        dataplace = "ELA"
    elif dataset_name == "QF_Bitvec":
        dataplace = "QFBV"
    elif dataset_name == "QF_NonLinearIntArith":
        dataplace = "QFNIA"
    else:
        dataplace = dataset_name

    # Construct file paths for cluster assignments and training data
    tc = (f"tmp/machfea_infer_result_{dataset_name}_train_feature_train_"
          f"{config['seed']}.json")
    td = f"data/{dataset_name}Labels.json"

    # Load training data and cluster assignments
    with open(td, 'r', encoding='UTF-8') as f:
        par2_dict = json.load(f)
    with open(tc, 'r', encoding='UTF-8') as f:
        train_cluster_dict = json.load(f)

    train_set = par2_dict['train']
    print(f"Training set size: {len(train_set)} problems")

    # Extract solver indices for this portfolio configuration
    output_idx = []
    if config['s1'] != -1:
        output_idx.append(config['s1'])
    if config['s2'] != -1:
        output_idx.append(config['s2'])
    if config['s3'] != -1:
        output_idx.append(config['s3'])
    if config['s4'] != -1:
        output_idx.append(config['s4'])

    # Initialize evaluation metrics
    total_time = 0  # Total PAR2 time across all problems
    fail = 0       # Number of unsolved problems

    # Normalize timeout configuration to sum to 1200 seconds
    tmp = [config['t1'], config['t2'], config['t3'], config['t4']]
    total_timeout = sum(tmp[i] for i in range(len(output_idx)))
    final_config = [tmp[i]/total_timeout*1200 for i in range(len(output_idx))]

    # Filter training problems to only include those in the target cluster
    key_set = list(train_set.keys())
    full_key_set = list(train_set.keys())
    key_set = []

    for problem_name in full_key_set:
        # Check if problem belongs to target cluster using multiple formats
        if (problem_name in train_cluster_dict.keys() and
                train_cluster_dict[problem_name] == cluster_id):
            key_set.append(problem_name)
        key_alt1 = (f"./infer_result/{dataset_name}/_data_sibly_sibyl_data_"
                    f"{dataset_name}_{dataset_name}_" +
                    problem_name.replace("/", "_") + ".json")
        if (key_alt1 in train_cluster_dict.keys() and
                train_cluster_dict[key_alt1] == cluster_id):
            key_set.append(problem_name)
        key_alt2 = (f"./infer_result/{dataplace}/_data_sibly_sibyl_data_"
                    f"Comp_non-incremental_" +
                    problem_name.replace("/", "_") + ".json")
        if (key_alt2 in train_cluster_dict.keys() and
                train_cluster_dict[key_alt2] == cluster_id):
            key_set.append(problem_name)

    print(f"Problems in cluster {cluster_id}: {len(key_set)}")
    # Shuffle problems for random evaluation order
    random.shuffle(key_set)

    # Set up cross-validation folds (5-fold by default)
    idxs = list(range(len(key_set)))
    if config['si'] != -1:
        si = int(config['si'])
        # Use 80% of data for training, 20% for validation
        idxs = []
        for idx in range(len(key_set)):
            start_bound = int(len(key_set) * (0.2 * si))
            end_bound = int(len(key_set) * (0.2 * (si + 1)))
            if idx < start_bound or idx >= end_bound:
                idxs.append(idx)
    print(f"Evaluation set size: {len(idxs)} problems")
    # Evaluate portfolio configuration on each problem in evaluation set
    for problem_idx in idxs:
        tmp_time = 0
        par2list = train_set[key_set[problem_idx]]  # PAR2 scores for all solvers
        solved = False

        # Try each solver in sequence until one solves the problem
        for solver_pos, solver_idx in enumerate(output_idx):
            solver_timeout = final_config[solver_pos]

            # Check if this solver can solve the problem within its timeout
            if float(par2list[solver_idx]) <= solver_timeout:
                # Solver succeeded - add its actual runtime
                tmp_time += par2list[solver_idx]
                # Add time spent waiting for previous solvers
                for prev_solver_pos in range(solver_pos):
                    tmp_time += final_config[prev_solver_pos]
                total_time += tmp_time
                solved = True
                break

        # If no solver succeeded, add penalty time
        if not solved:
            total_time += 2400  # PAR2 penalty for unsolved problems
            fail += 1

    print(f"Evaluation result: SUCCESS, 0, 0, {total_time}, 0")
    return total_time


# Define SMAC3 configuration space for timeout optimization
cs = ConfigurationSpace(seed=int(seed))

# Timeout hyperparameters (0-1200 seconds each)
t_1 = Float("t1", (0, 1200), default=1200)  # Timeout for solver 1
t_2 = Float("t2", (0, 1200), default=0)    # Timeout for solver 2
t_3 = Float("t3", (0, 1200), default=0)    # Timeout for solver 3
t_4 = Float("t4", (0, 1200), default=0)    # Timeout for solver 4

# Categorical hyperparameters (fixed for this optimization run)
cluster_ = Categorical("cluster", [cluster], default=cluster)
dataset_ = Categorical("dataset", [dataset], default=dataset)
seed_ = Categorical("seed", [seed], default=seed)
s1_ = Categorical("s1", [s1], default=s1)
s2_ = Categorical("s2", [s2], default=s2)
s3_ = Categorical("s3", [s3], default=s3)
s4_ = Categorical("s4", [s4], default=s4)
si_ = Categorical("si", [start_idx], default=start_idx)

# Add all hyperparameters to configuration space
cs.add([t_1, t_2, t_3, t_4, cluster_, dataset_, seed_, s1_, s2_, s3_, s4_, si_])

# Configure SMAC3 scenario (200 trials, deterministic for reproducibility)
scenario = Scenario(cs, deterministic=True, n_trials=200)

# Initialize and run SMAC3 optimization
smac = HyperparameterOptimizationFacade(scenario, train, overwrite=True, w1=w1, w2=w2)
incumbent = smac.optimize()


# Extract and display the optimal timeout configuration found by SMAC3
output_idx = []
if incumbent['s1'] != -1:
    output_idx.append(int(incumbent['s1']))
if incumbent['s2'] != -1:
    output_idx.append(int(incumbent['s2']))
if incumbent['s3'] != -1:
    output_idx.append(int(incumbent['s3']))
if incumbent['s4'] != -1:
    output_idx.append(int(incumbent['s4']))

# Normalize final timeout configuration to sum to 1200 seconds
tmp = [incumbent['t1'], incumbent['t2'], incumbent['t3'], incumbent['t4']]
total_timeout = sum(tmp[i] for i in range(len(output_idx)))

# Output final normalized timeouts
optimal_timeouts = [str(tmp[i]/total_timeout*1200) for i in range(len(output_idx))]
print(",".join(optimal_timeouts))
