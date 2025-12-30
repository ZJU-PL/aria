"""
SMTgazer: Machine Learning-Based SMT Solver Portfolio System

This module implements SMTgazer, an effective algorithm scheduling method for SMT solving.
SMTgazer uses machine learning techniques to automatically select optimal combinations
of SMT solvers for different problem categories and instances.

Key Components:
- Feature normalization and preprocessing
- Unsupervised clustering using X-means algorithm
- SMAC3-based portfolio optimization
- Parallel solver execution and evaluation

The system works in two phases:
1. Training: Extract features, cluster problems, optimize solver portfolios per cluster
2. Inference: Classify new problems and apply learned portfolios

Author: SMTgazer Team
Publication: ASE 2025
"""
# pylint: disable=invalid-name

import json
import os
import sys
from functools import partial
from multiprocessing import Pool
from os import popen

import numpy as np
from pyclustering.cluster.center_initializer import (
    kmeans_plusplus_initializer
)
from pyclustering.cluster.xmeans import xmeans

def normalize(tf, seed_val):
    """
    Normalize feature vectors to [0,1] range for better clustering performance.

    This function performs min-max normalization on SMT problem features to ensure
    all features contribute equally to clustering and distance calculations.

    Args:
        tf (str): Path to input JSON file containing feature vectors
        seed_val (int): Random seed for reproducible normalization

    The normalization formula is: (x - min) / (max - min) where division by zero
    is handled by setting the denominator to 1.

    Output files:
        - _norm{seed_val}.json: Normalized feature vectors
        - _lim{seed_val}.json: Min/max limits used for normalization
    """
    with open(tf, 'r', encoding='UTF-8') as f:
        fea_dict_input = json.load(f)  # Load feature dictionary

    # Extract problem names and feature vectors
    pro_dict = []  # Problem names
    fea_dict_list = []  # Feature vectors
    for problem_name in fea_dict_input.keys():
        pro_dict.append(problem_name)
        fea_dict_list.append(fea_dict_input[problem_name])
    fea_dict_array = np.array(fea_dict_list)

    # Calculate min and max values for each feature dimension
    max_vals = fea_dict_array.max(axis=0)
    min_vals = fea_dict_array.min(axis=0)

    # Calculate normalization ranges (avoid division by zero)
    sub_vals = max_vals - min_vals
    for idx in range(len(sub_vals)):
        if sub_vals[idx] == 0:
            sub_vals[idx] = 1  # Set to 1 if max == min for this feature

    # Apply min-max normalization: (x - min) / (max - min)
    new_fea_dict = (fea_dict_array - min_vals) / sub_vals

    # Store normalization parameters for later use in inference
    lim = {"min": list(min_vals), "sub": list(sub_vals)}

    # Prepare normalized feature dictionary
    dict_output = {}
    for idx, problem_name in enumerate(pro_dict):
        dict_output[problem_name] = new_fea_dict[idx].tolist()

    # Clean up file path for output naming
    clean_tf = tf.replace("../", "").replace("./", "").replace("/", "_")

    # Save normalized features and normalization limits
    norm_file = f"tmp/{clean_tf.replace('.json', f'_norm{seed_val}.json')}"
    with open(norm_file, 'w', encoding='UTF-8') as f:
        json.dump(dict_output, f)

    lim_file = f"tmp/{clean_tf.replace('.json', f'_lim{seed_val}.json')}"
    with open(lim_file, 'w', encoding='UTF-8') as f:
        json.dump(lim, f)

def cluster(tfnorm, seed_val=0, cluster_num=20):
    """
    Perform unsupervised clustering of SMT problems using X-means algorithm.

    This function clusters SMT problems based on their normalized feature vectors
    using the X-means algorithm, which automatically determines the optimal number
    of clusters (up to cluster_num).

    Args:
        tfnorm (str): Path to normalized feature file
        seed_val (int): Random seed for reproducible clustering results
        cluster_num (int): Maximum number of clusters to consider

    The clustering process:
    1. Uses K-means++ initialization with 3 initial centers (or cluster_num if
       smaller)
    2. Applies X-means algorithm which can split and merge clusters
    3. Assigns each problem instance to its closest cluster

    Output files:
        - _train_{seed_val}.json: Cluster assignments for each problem
        - _cluster_center_{seed_val}.json: Final cluster centers
    """
    # Determine number of initial centers (minimum of 3 or cluster_num)
    amount_initial_centers = min(3, cluster_num)

    # Load normalized feature vectors
    with open(tfnorm, 'r', encoding='UTF-8') as f:
        fea_dict = json.load(f)

    # Prepare feature matrix and problem names
    feature_mat = []
    key_set = []
    for key in fea_dict.keys():
        key_set.append(key)
        feature_mat.append(fea_dict[key])

    print("Feature loading complete")
    feature_mat_array = np.array(feature_mat)
    print(f"Feature matrix shape: {feature_mat_array.shape}")

    train_dict = {}  # Will store cluster assignments

    x_train = feature_mat_array

    # Initialize X-means with K-means++ centers
    initial_centers = kmeans_plusplus_initializer(
        x_train, amount_initial_centers, random_state=seed_val
    ).initialize()

    # Run X-means clustering (automatically determines optimal cluster count)
    xmeans_instance = xmeans(
        x_train,
        initial_centers=initial_centers,
        kmax=cluster_num,  # Maximum clusters to consider
        ccore=False,       # Use Python implementation
        random_state=seed_val  # For reproducibility
    )
    xmeans_instance.process()

    # Extract clustering results
    clusters = xmeans_instance.get_clusters()  # List of cluster assignments
    centers = xmeans_instance.get_centers()   # Cluster centers

    cluster_center = {"center": list(centers)}

    # Assign each problem to its cluster
    for cluster_id in range(len(centers)):
        for problem_idx in clusters[cluster_id]:
            train_dict[key_set[problem_idx]] = cluster_id

    # Save cluster assignments and centers
    train_file = tfnorm.replace(f"_norm{seed_val}.json",
                                 f"_train_{seed_val}.json")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_dict, f)
    center_file = tfnorm.replace(f"_norm{seed_val}.json",
                                  f"_cluster_center_{seed_val}.json")
    with open(center_file, 'w', encoding='utf-8') as f:
        json.dump(cluster_center, f)

def get_test_portfolio(tfnorm, cluster_portfolio, solver_list, dataset_name,
                       seed_val, outputfile=""):
    """
    Generate test portfolios by classifying new problems into learned clusters.

    This function takes normalized features of test problems and assigns each problem
    to the closest cluster based on Euclidean distance to cluster centers. It then
    applies the learned solver portfolio for that cluster.

    Args:
        tfnorm (str): Path to normalized test feature file
        cluster_portfolio (str): Path to trained portfolio configuration file
        solver_list (list): List of available SMT solvers
        dataset_name (str): Name of the dataset being processed
        seed_val (int): Random seed for reproducibility
        outputfile (str): Output file path (auto-generated if empty)

    The process:
    1. Load test problem features and trained cluster centers
    2. Calculate Euclidean distance from each test problem to all cluster centers
    3. Assign each problem to the closest cluster
    4. Apply the learned solver portfolio for that cluster

    Output:
        JSON file containing solver portfolios for each test problem
    """
    # Load test problem features
    with open(tfnorm, 'r', encoding='UTF-8') as f:
        fea_dict = json.load(f)

    # Load trained portfolio configuration
    with open(cluster_portfolio, 'r', encoding='UTF-8') as f:
        output_dict = json.load(f)
    portfolio_dict = output_dict['portfolio']
    center_dict = output_dict['center']

    # Prepare portfolio mapping: cluster_id -> [solver_indices, timeout]
    portfolio_map = {}
    time_map = {}
    for cluster_id in portfolio_dict.keys():
        cluster_config = portfolio_dict[cluster_id]
        time_map[cluster_id] = cluster_config[1]  # Timeout configuration
        solver_indices = cluster_config[0]     # Solver indices for this cluster

        # Convert solver indices to actual solver names
        solver_names = []
        for solver_idx in solver_indices:
            solver_names.append(solver_list[solver_idx])
        portfolio_map[cluster_id] = solver_names

    # Prepare test feature matrix
    feature_mat = []
    key_set = []
    for problem_name in fea_dict.keys():
        key_set.append(problem_name)
        feature_mat.append(fea_dict[problem_name])

    # Get cluster centers
    centers = center_dict['center']
    x_test = np.array(feature_mat)

    # Assign each test problem to closest cluster
    test_dict = {}
    for problem_idx, _ in enumerate(x_test):
        # Calculate distances to all cluster centers
        distances = []
        for center in centers:
            # Euclidean distance calculation
            dist = np.sqrt(np.sum((x_test[problem_idx] - np.array(center))**2))
            distances.append(dist)

        # Find closest cluster
        closest_cluster_idx = np.argmin(distances)

        # Apply portfolio for this cluster
        cluster_id = str(closest_cluster_idx)
        test_dict[key_set[problem_idx]] = [
            portfolio_map[cluster_id],  # Solver names for this cluster
            time_map[cluster_id]        # Timeout configuration
        ]

    # Ensure output directory exists
    dirpath = 'output'
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)

    # Generate output filename if not provided
    if outputfile == "":
        outputfile = (f"output/test_result_{dataset_name}_{seed_val}_"
                      f"{len(centers)}.json")

    # Save test portfolio assignments
    with open(outputfile, 'w', encoding='utf-8') as f:
        json.dump(test_dict, f)

def run_seed3(sf, seed_val, start_idx):
    """
    Execute SMAC3 optimization for a specific solver configuration and seed.

    This function constructs and runs a SMAC3 command for optimizing solver
    portfolios using the SMAC3 algorithm with a hybrid model.

    Args:
        sf (list): [config_dict, solver_index] where config_dict contains SMAC3
            parameters
        seed_val (int): Random seed for SMAC3 run
        start_idx (int): Start index for cross-validation fold

    Returns:
        list: [output_lines, solver_index] from SMAC3 execution
    """
    # Build SMAC3 command line
    command = f"python -u portfolio_smac3.py -seed {seed_val}"

    # Add configuration parameters from the config dictionary
    for key in sf[0].keys():
        command = f"{command} -{key} {sf[0][key]}"

    # Add cross-validation start index
    command = f"{command} -si {start_idx}"

    print(f"Running SMAC3: {command}")
    with popen(command) as process:
        output = process.read()
    output_lines = output.split('\n')
    return [output_lines, sf[1]]

def get_portfolio_3(solver_list, td, tc, tlim, tcenter, dataset_name,
                    outputfile="", portfolio_size=4, cluster_num=20,
                    seed_val=0, timelimit=1200):
    """
    Optimize solver portfolios for each cluster using SMAC3 algorithm.

    This is the core portfolio optimization function that uses SMAC3 (Sequential
    Model-based Algorithm Configuration) to find optimal solver configurations
    for each problem cluster. It performs cross-validation and evaluates different
    solver combinations to minimize PAR2 (Penalized Average Runtime) scores.

    Args:
        solver_list (list): Available SMT solvers
        td (str): Path to training data (PAR2 scores)
        tc (str): Path to cluster assignments
        tlim (str): Path to normalization limits
        tcenter (str): Path to cluster centers
        dataset_name (str): Dataset name for configuration
        outputfile (str): Output portfolio file path
        portfolio_size (int): Number of solvers in each portfolio (default: 4)
        cluster_num (int): Number of clusters to optimize
        seed_val (int): Random seed for reproducibility
        timelimit (int): SMAC3 time limit in seconds (default: 1200)

    The optimization process:
    1. For each cluster, evaluate different solver combinations
    2. Use 5-fold cross-validation to assess performance
    3. Select best solver for each position in the portfolio
    4. Optimize timeout configurations using SMAC3
    """
    # Load all necessary data files
    with open(td, 'r', encoding='UTF-8') as td_file:
        par2_dict = json.load(td_file)  # PAR2 scores for all problems
    with open(tc, 'r', encoding='UTF-8') as tc_file:
        train_cluster_dict = json.load(tc_file)  # Cluster assignments

    with open(tlim, 'r', encoding='UTF-8') as tlim_file:
        lim_dict = json.load(tlim_file)  # Normalization limits
    with open(tcenter, 'r', encoding='UTF-8') as tcenter_file:
        center_dict = json.load(tcenter_file)  # Cluster centers

    # Validate portfolio size against available solvers
    if portfolio_size > len(solver_list):
        print("warning: PortfolioSize is bigger than the number of solvers!")
        portfolio_size = len(solver_list)

    # Ensure output directory exists and generate output filename
    output_dir = 'output'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if outputfile == "":
        outputfile = (f"output/train_result_{dataset_name}_{portfolio_size}_"
                      f"{cluster_num}_{seed_val}.json")

    # Initialize portfolio storage for each cluster
    final_portfolio = {}

    # Optimize portfolio for each cluster independently
    for cluster_id in range(cluster_num):
        print(f"Optimizing portfolio for cluster {cluster_id}")
        print(f"Available solvers: {solver_list}")

        # Track which solvers have been selected (coverage vector)
        cov = [0 for _ in range(len(solver_list))]

        # Store selected solver indices and timeout configuration
        output_idx = []
        final_config = []

        # Get training data for this cluster
        train_set = par2_dict['train']

        # Build portfolio by selecting best solver for each position
        for portfolio_position in range(portfolio_size):
            min_time = float('inf')  # Track best (minimum) PAR2 time
            min_idx = -1             # Track best solver index
            sf = []  # SMAC3 configurations to evaluate
            for j in range(len(solver_list)):
                tmpdict = {}
                if cov[j] == 1:
                    continue
                tmp = list(output_idx)
                tmp.append(j)
                tmpdict['t1'] = 1200
                tmpdict['t2'] = 0
                tmpdict['t3'] = 0
                for solver_pos, solver_idx in enumerate(tmp):
                    tmpdict[f"s{solver_pos+1}"] = str(solver_idx)
                tmpdict["cluster"] = cluster_id
                if dataset_name == 'Equality+LinearArith':
                    tmpdict["dataset"] = "ELA"
                else:
                    tmpdict["dataset"] = str(dataset_name)

                sf.append([tmpdict, j])
            valid_scores = [0 for _ in range(len(sf))]
            for si in range(0, 5):
                with Pool(processes=10) as p:
                    partial_run_seed = partial(run_seed3, seed_val=seed_val,
                                               start_idx=si)
                    ret = p.map(partial_run_seed, sf)

                ret_seed = []
                for idx, result in enumerate(ret):
                    ret_seed.append(result[1])
                    k = result[0][-2].split(",")
                    ret[idx] = k
                print(ret)
                configs = []
                for result_item in ret:
                    tmp_config = [float(val) for val in result_item]
                    configs.append(tmp_config)

                full_key_set = list(train_set.keys())
                key_set = []

                if dataset_name == "Equality+LinearArith":
                    dataplace = "ELA"
                elif dataset_name == "QF_Bitvec":
                    dataplace = "QFBV"
                elif dataset_name == "QF_NonLinearIntArith":
                    dataplace = "QFNIA"
                else:
                    dataplace = dataset_name
                for j in full_key_set:
                    if (j in train_cluster_dict.keys() and
                            train_cluster_dict[j] == cluster_id):
                        key_set.append(j)
                    key_alt1 = (f"./infer_result/{dataset_name}/"
                                f"_data_sibly_sibyl_data_{dataset_name}_"
                                f"{dataset_name}_" +
                                j.replace("/", "_") + ".json")
                    if (key_alt1 in train_cluster_dict.keys() and
                            train_cluster_dict[key_alt1] == cluster_id):
                        key_set.append(j)
                    key_alt2 = (f"./infer_result/{dataplace}/"
                                f"_data_sibly_sibyl_data_Comp_non-incremental_"
                                + j.replace("/", "_") + ".json")
                    if (key_alt2 in train_cluster_dict.keys() and
                            train_cluster_dict[key_alt2] == cluster_id):
                        key_set.append(j)

                print("configs len:", len(configs))
                print(output_idx)

                for config_idx, config in enumerate(configs):
                    x1 = config[0]
                    x2 = config[1]
                    x3 = config[2]
                    x4 = 1200 - x1 - x2 - x3
                    tmp_config = [x1, x2, x3, x4]

                    total_time = 0

                    ri = int(len(key_set) * (0.2 * (si + 1)))
                    ri = min(ri, int(len(key_set)))
                    for problem_idx in range(int(len(key_set) * (0.2 * si)),
                                             ri):
                        tmp_time = 0
                        flag = 0
                        par2list = train_set[key_set[problem_idx]]
                        tmplist_ = list(output_idx)
                        tmplist_.append(ret_seed[config_idx])
                        for solver_pos, solver_idx in enumerate(tmplist_):
                            if float(par2list[solver_idx]) <= tmp_config[solver_pos]:
                                tmp_time += par2list[solver_idx]
                                for k in range(solver_pos):
                                    tmp_time += tmp_config[k]
                                total_time += tmp_time
                                flag = 1
                                break
                        if flag == 0:
                            total_time += 2400
                    valid_scores[config_idx] += total_time
            print(valid_scores)
            chosen_idx = np.argmin(valid_scores)
            final_config = configs[chosen_idx]
            output_idx.append(ret_seed[chosen_idx])
            cov[ret_seed[chosen_idx]] = 1

        tmpdict = {}
        tmpdict['t1'] = 1200
        tmpdict['t2'] = 0
        tmpdict['t3'] = 0
        for solver_pos, solver_idx in enumerate(output_idx):
            tmpdict[f"s{solver_pos+1}"] = str(solver_idx)
        tmpdict["cluster"] = cluster_id
        if dataset_name == 'Equality+LinearArith':
            tmpdict["dataset"] = "ELA"
        else:
            tmpdict["dataset"] = str(dataset_name)
        sf = [[tmpdict, -1]]
        with Pool(processes=1) as p:
            partial_run_seed = partial(run_seed3, seed_val=seed_val,
                                       start_idx=-1)
            ret = p.map(partial_run_seed, sf)
        for result_idx, result in enumerate(ret):
            k = result[0][-2].split(",")
            ret[result_idx] = k
        print(ret)
        configs = []
        for result_item in ret:
            tmp_config = [float(val) for val in result_item]
            configs.append(tmp_config)
        final_config = configs[0]
        final_portfolio[cluster_id] = [output_idx, final_config]
    output_dict = {"portfolio": final_portfolio, "lim": lim_dict,
                   "center": center_dict}
    with open(outputfile, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f)

if __name__ == '__main__':
    """
    Main execution entry point for SMTgazer training and inference.

    Usage:
        python SMTportfolio.py train [options]  - Train portfolios for dataset
        python SMTportfolio.py infer [options]  - Apply trained portfolios

    Command line arguments:
        -train_features: Path to training feature file
        -train_data: Path to training PAR2 data file
        -seed: Random seed for reproducibility
        -cluster_num: Maximum number of clusters
        -solverdict: Path to solver configuration file
        -dataset: Dataset name (e.g., "Equality+LinearArith")
        -clusterPortfolio: Path to trained portfolio file (for inference)
    """
    work_type = 'infer'  # Default to inference mode

    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ('train', 'infer'):
        work_type = sys.argv[1]

    # Initialize default values
    tf = ""           # Training features file
    td = ""           # Training data file
    seed = 0          # Random seed
    cluster_num = 20  # Maximum clusters
    solverdict = ""   # Solver configuration file
    dataset = ""      # Dataset name
    cluster_portfolio = ""  # Trained portfolio file
    for i in range(len(sys.argv) - 1):
        if sys.argv[i] == '-train_features':
            tf = sys.argv[i+1]
        if sys.argv[i] == '-train_data':
            td = sys.argv[i+1]

        if sys.argv[i] == '-seed':
            seed = int(sys.argv[i+1])
        if sys.argv[i] == '-cluster_num':
            cluster_num = int(sys.argv[i+1])

        if sys.argv[i] == '-solverdict':
            solverdict = sys.argv[i+1]

        if sys.argv[i] == '-dataset':
            dataset = sys.argv[i+1]

        if sys.argv[i] == '-clusterPortfolio':
            cluster_portfolio = sys.argv[i+1]

    tf = "./machfea/infer_result/" + str(dataset) + "_train_feature.json"
    td = "./data/" + str(dataset) + "Labels.json"
    dirpath = 'tmp'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    # Execute training or inference based on work_type
    if work_type == "train":
        print(f"Starting SMTgazer training for dataset: {dataset}")

        # Step 1: Normalize feature vectors
        print("Step 1: Normalizing features...")
        normalize(tf, seed)

        # Step 2: Perform clustering
        print("Step 2: Clustering problems...")
        # Clean up file path for consistent naming
        tmp = tf.replace("../", "").replace("./", "").replace("/", "_")
        tfnorm = f"tmp/{tmp.replace('.json', f'_norm{seed}.json')}"
        tflim = tfnorm.replace("_norm", "_lim")
        tcenter = tfnorm.replace(f"_norm{seed}.json",
                                  f"_cluster_center_{seed}.json")

        cluster(tfnorm, seed, cluster_num)
        tc = tfnorm.replace(f"_norm{seed}.json", f"_train_{seed}.json")

        # Step 3: Optimize solver portfolios
        print("Step 3: Optimizing solver portfolios...")
        with open(solverdict, 'r', encoding='UTF-8') as f:
            solver_dict = json.load(f)
        solver_list = solver_dict["solver_list"]

        get_portfolio_3(
            solver_list, td, tc, tflim, tcenter, dataset,
            outputfile="", portfolio_size=4, cluster_num=cluster_num,
            seed_val=seed, timelimit=1200
        )
        print("Training completed!")

    elif work_type == "infer":
        print(f"Starting SMTgazer inference for dataset: {dataset}")

        # Load trained portfolio configuration
        with open(cluster_portfolio, 'r', encoding='UTF-8') as portfolio_file:
            output_dict_main = json.load(portfolio_file)

        # Extract normalization parameters used during training
        lim = output_dict_main['lim']
        min_vals = lim['min']
        sub_vals = lim['sub']

        # Load test features and apply same normalization as training
        testf = f"./machfea/infer_result/{dataset}_test_feature.json"
        with open(testf, 'r', encoding='UTF-8') as test_file:
            fea_dict_input = json.load(test_file)

        # Prepare feature matrix for normalization
        pro_dict = []
        fea_dict_list = []
        for problem_name in fea_dict_input.keys():
            pro_dict.append(problem_name)
            fea_dict_list.append(fea_dict_input[problem_name])
        fea_dict_array = np.array(fea_dict_list)
        print(f"Test feature matrix shape: {fea_dict_array.shape}")

        # Apply same normalization as training data
        new_fea_dict = (fea_dict_array - np.array(min_vals)) / np.array(sub_vals)

        print(f"Normalized test features: {len(new_fea_dict)} problems")
        print(f"Problem names: {len(pro_dict)}")

        # Prepare normalized test feature dictionary
        dict_output = {}
        for problem_idx, problem_name in enumerate(pro_dict):
            dict_output[problem_name] = new_fea_dict[problem_idx].tolist()

        # Clean up file path for output naming
        clean_testf = testf.replace("../", "").replace("./", "").replace("/", "_")
        testnorm = f"tmp/{clean_testf.replace('.json', f'_norm{seed}.json')}"

        # Save normalized test features
        with open(testnorm, 'w', encoding='UTF-8') as testnorm_file:
            json.dump(dict_output, testnorm_file)

        # Load solver configuration and run inference
        with open(solverdict, 'r', encoding='UTF-8') as solver_file:
            solver_dict_main = json.load(solver_file)
        solver_list = solver_dict_main["solver_list"]

        print("Running portfolio inference...")
        get_test_portfolio(testnorm, cluster_portfolio, solver_list, dataset,
                           seed)
        print("Inference completed!")
