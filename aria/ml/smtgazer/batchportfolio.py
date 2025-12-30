"""
SMTgazer Batch Portfolio Runner

This script runs SMTgazer portfolio training and testing across multiple SMT logic
categories using multiprocessing for parallel execution.

SMTgazer is an effective algorithm scheduling method for SMT solving that uses machine
learning to select optimal solver portfolios for different problem categories.

Author: SMTgazer Team
Publication: ASE 2025
"""

from functools import partial
from multiprocessing import Pool
from os import popen

# Seeds for reproducible experiments across different SMT categories
seed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def run_seed(seed_val, category_key):
    """
    Execute SMTgazer for a specific seed and SMT category.

    Args:
        seed_val (int): Random seed for reproducible results
        category_key (str): SMT logic category name
            (e.g., "Equality+LinearArith", "QF_Bitvec")

    Note:
        Different categories use different numbers of clusters based on complexity:
        - Equality+LinearArith: 2 clusters (simpler logic)
        - SyGuS: 3 clusters (synthesis problems)
        - Others: 20 clusters (default)
    """
    c_num = 20  # Default number of clusters

    # Adjust cluster count based on SMT category complexity
    if category_key == "Equality+LinearArith":
        c_num = 2  # Simpler linear arithmetic problems need fewer clusters
    if category_key == "SyGuS":
        c_num = 3  # Synthesis problems have different characteristics

    command = ""
    # Training Phase (currently commented out)
    # Uncomment to run training for each category and seed
    # command = (f"python -u SMTportfolio.py train -dataset {category_key} "
    #            f"-solverdict machfea/{category_key}_solver.json -seed "
    #            f"{seed_val} -cluster_num {c_num}")

    # Testing Phase (currently commented out)
    # Uncomment to run inference using trained portfolios
    # command = (f"python -u SMTportfolio.py infer -clusterPortfolio "
    #            f"output/train_result_{category_key}_4_{c_num}_{seed_val}.json "
    #            f"-dataset {category_key} -solverdict "
    #            f"machfea/{category_key}_solver.json -seed {seed_val}")

    print(f"Running command: {command}")
    with popen(command) as process:
        output = process.read()
    print(output)

if __name__ == '__main__':
    """
    Main execution: Run SMTgazer across multiple SMT logic categories in parallel.

    This script processes the following SMT categories:
    - Equality+LinearArith: Equality and linear arithmetic problems
    - QF_NonLinearIntArith: Quantifier-free nonlinear integer arithmetic
    - QF_Bitvec: Quantifier-free bit-vector problems
    - SyGuS: Syntax-guided synthesis problems
    - BMC: Bounded model checking problems
    - SymEx: Symbolic execution problems

    Uses multiprocessing with 10 parallel processes for efficiency.
    """
    # Define the SMT logic categories to process
    key_set = [
        'Equality+LinearArith',    # Simple arithmetic and equality logic
        'QF_NonLinearIntArith',    # Nonlinear integer arithmetic
        'QF_Bitvec',               # Bit-vector operations
        'SyGuS',                   # Program synthesis problems
        'BMC',                     # Bounded model checking
        'SymEx'                    # Symbolic execution problems
    ]

    # Create process pool for parallel execution
    p = Pool(processes=10)

    # Process each SMT category in parallel
    for key in key_set:
        print(f"Processing SMT category: {key}")
        # Create partial function with fixed category parameter
        partial_run_seed = partial(run_seed, category_key=key)
        # Map function across all seeds for this category
        p.map(partial_run_seed, seed)

    # Clean up multiprocessing resources
    p.close()
    p.join()
