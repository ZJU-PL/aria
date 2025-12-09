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

from os import system
import time as __time
import os
import sys
from functools import partial
from multiprocessing import Pool
from os import popen
import shutil
import random
import logging

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import json
import numpy as np

def normalize(tf, seed):
    """
    Normalize feature vectors to [0,1] range for better clustering performance.

    This function performs min-max normalization on SMT problem features to ensure
    all features contribute equally to clustering and distance calculations.

    Args:
        tf (str): Path to input JSON file containing feature vectors
        seed (int): Random seed for reproducible normalization

    The normalization formula is: (x - min) / (max - min) where division by zero
    is handled by setting the denominator to 1.

    Output files:
        - _norm{seed}.json: Normalized feature vectors
        - _lim{seed}.json: Min/max limits used for normalization
    """
    with open(tf, 'r', encoding='UTF-8') as f:
        fea_dict_ = json.load(f)  # Load feature dictionary

    # Extract problem names and feature vectors
    pro_dict = []  # Problem names
    fea_dict = []  # Feature vectors
    for problem_name in fea_dict_.keys():
        pro_dict.append(problem_name)
        fea_dict.append(fea_dict_[problem_name])
    fea_dict = np.array(fea_dict)

    # Calculate min and max values for each feature dimension
    max_ = fea_dict.max(axis=0)
    min_ = fea_dict.min(axis=0)

    # Calculate normalization ranges (avoid division by zero)
    sub = max_ - min_
    for i in range(len(sub)):
        if sub[i] == 0:
            sub[i] = 1  # Set to 1 if max == min for this feature

    # Apply min-max normalization: (x - min) / (max - min)
    new_fea_dict = (fea_dict - min_) / sub
    new_pro_dict = pro_dict

    # Store normalization parameters for later use in inference
    lim = {"min": list(min_), "sub": list(sub)}

    # Prepare normalized feature dictionary
    dict_output = {}
    for i in range(len(pro_dict)):
        dict_output[pro_dict[i]] = new_fea_dict[i].tolist()

    # Clean up file path for output naming
    clean_tf = tf.replace("../", "").replace("./", "").replace("/", "_")

    # Save normalized features and normalization limits
    with open("tmp/" + clean_tf.replace(".json", f"_norm{seed}.json"), 'w', encoding='UTF-8') as f:
        json.dump(dict_output, f)

    with open("tmp/" + clean_tf.replace(".json", f"_lim{seed}.json"), 'w', encoding='UTF-8') as f:
        json.dump(lim, f)

def cluster(tfnorm, seed=0, cluster_num=20):
    """
    Perform unsupervised clustering of SMT problems using X-means algorithm.

    This function clusters SMT problems based on their normalized feature vectors
    using the X-means algorithm, which automatically determines the optimal number
    of clusters (up to cluster_num).

    Args:
        tfnorm (str): Path to normalized feature file
        seed (int): Random seed for reproducible clustering results
        cluster_num (int): Maximum number of clusters to consider

    The clustering process:
    1. Uses K-means++ initialization with 3 initial centers (or cluster_num if smaller)
    2. Applies X-means algorithm which can split and merge clusters
    3. Assigns each problem instance to its closest cluster

    Output files:
        - _train_{seed}.json: Cluster assignments for each problem
        - _cluster_center_{seed}.json: Final cluster centers
    """
    # Determine number of initial centers (minimum of 3 or cluster_num)
    amount_initial_centers = 3
    if amount_initial_centers > cluster_num:
        amount_initial_centers = cluster_num

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
    feature_mat = np.array(feature_mat)
    print(f"Feature matrix shape: {feature_mat.shape}")

    train_dict = {}  # Will store cluster assignments

    X_train = feature_mat

    # Initialize X-means with K-means++ centers
    initial_centers = kmeans_plusplus_initializer(
        X_train, amount_initial_centers, random_state=seed
    ).initialize()

    # Run X-means clustering (automatically determines optimal cluster count)
    xmeans_instance = xmeans(
        X_train,
        initial_centers=initial_centers,
        kmax=cluster_num,  # Maximum clusters to consider
        ccore=False,       # Use Python implementation
        random_state=seed  # For reproducibility
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
    with open(tfnorm.replace(f"_norm{seed}.json", f"_train_{seed}.json"), 'w', encoding='utf-8') as f:
        json.dump(train_dict, f)
    with open(tfnorm.replace(f"_norm{seed}.json", f"_cluster_center_{seed}.json"), 'w', encoding='utf-8') as f:
        json.dump(cluster_center, f)

def getTestPortfoliio(tfnorm, clusterPortfolio, solverlist, dataset, seed, outputfile=""):
    """
    Generate test portfolios by classifying new problems into learned clusters.

    This function takes normalized features of test problems and assigns each problem
    to the closest cluster based on Euclidean distance to cluster centers. It then
    applies the learned solver portfolio for that cluster.

    Args:
        tfnorm (str): Path to normalized test feature file
        clusterPortfolio (str): Path to trained portfolio configuration file
        solverlist (list): List of available SMT solvers
        dataset (str): Name of the dataset being processed
        seed (int): Random seed for reproducibility
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
    with open(clusterPortfolio, 'r', encoding='UTF-8') as f:
        output_dict = json.load(f)
    portfolio_dict = output_dict['portfolio']
    center_dict = output_dict['center']

    # Prepare portfolio mapping: cluster_id -> [solver_indices, timeout]
    portfolio_ = {}
    time_ = {}
    for cluster_id in portfolio_dict.keys():
        tmp = portfolio_dict[cluster_id]
        time_[cluster_id] = tmp[1]  # Timeout configuration
        solver_indices = tmp[0]     # Solver indices for this cluster

        # Convert solver indices to actual solver names
        solver_names = []
        for solver_idx in solver_indices:
            solver_names.append(solverlist[solver_idx])
        portfolio_[cluster_id] = solver_names

    # Prepare test feature matrix
    feature_mat = []
    key_set = []
    for problem_name in fea_dict.keys():
        key_set.append(problem_name)
        feature_mat.append(fea_dict[problem_name])

    # Get cluster centers
    centers = center_dict['center']
    X_test = np.array(feature_mat)

    # Assign each test problem to closest cluster
    test_dict = {}
    for problem_idx in range(len(X_test)):
        # Calculate distances to all cluster centers
        distances = []
        for center in centers:
            # Euclidean distance calculation
            dist = np.sqrt(np.sum((X_test[problem_idx] - np.array(center))**2))
            distances.append(dist)

        # Find closest cluster
        closest_cluster_idx = np.argmin(distances)

        # Apply portfolio for this cluster
        cluster_id = str(closest_cluster_idx)
        test_dict[key_set[problem_idx]] = [
            portfolio_[cluster_id],  # Solver names for this cluster
            time_[cluster_id]        # Timeout configuration
        ]

    # Ensure output directory exists
    dirpath = 'output'
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)

    # Generate output filename if not provided
    if outputfile == "":
        outputfile = f"output/test_result_{dataset}_{seed}_{len(centers)}.json"

    # Save test portfolio assignments
    with open(outputfile, 'w', encoding='utf-8') as f:
        json.dump(test_dict, f)

def RunSeed3(sf, seed, start_idx):
    """
    Execute SMAC3 optimization for a specific solver configuration and seed.

    This function constructs and runs a SMAC3 command for optimizing solver
    portfolios using the SMAC3 algorithm with a hybrid model.

    Args:
        sf (list): [config_dict, solver_index] where config_dict contains SMAC3 parameters
        seed (int): Random seed for SMAC3 run
        start_idx (int): Start index for cross-validation fold

    Returns:
        list: [output_lines, solver_index] from SMAC3 execution
    """
    # Build SMAC3 command line
    command = "python -u portfolio_smac3.py -seed " + str(seed)

    # Add configuration parameters from the config dictionary
    for key in sf[0].keys():
        command = command + " -" + str(key) + " " + str(sf[0][key])

    # Add cross-validation start index
    command = command + " -si " + str(start_idx)

    print(f"Running SMAC3: {command}")
    output = popen(command).read()
    tmp = output.split('\n')
    return [tmp, sf[1]]

def get_portfolio_3(solverlist, td, tc, tlim, tcenter, dataset, outputfile="", portfolioSize=4, cluster_num=20, seed=0, timelimit=1200):
    """
    Optimize solver portfolios for each cluster using SMAC3 algorithm.

    This is the core portfolio optimization function that uses SMAC3 (Sequential
    Model-based Algorithm Configuration) to find optimal solver configurations
    for each problem cluster. It performs cross-validation and evaluates different
    solver combinations to minimize PAR2 (Penalized Average Runtime) scores.

    Args:
        solverlist (list): Available SMT solvers
        td (str): Path to training data (PAR2 scores)
        tc (str): Path to cluster assignments
        tlim (str): Path to normalization limits
        tcenter (str): Path to cluster centers
        dataset (str): Dataset name for configuration
        outputfile (str): Output portfolio file path
        portfolioSize (int): Number of solvers in each portfolio (default: 4)
        cluster_num (int): Number of clusters to optimize
        seed (int): Random seed for reproducibility
        timelimit (int): SMAC3 time limit in seconds (default: 1200)

    The optimization process:
    1. For each cluster, evaluate different solver combinations
    2. Use 5-fold cross-validation to assess performance
    3. Select best solver for each position in the portfolio
    4. Optimize timeout configurations using SMAC3
    """
    # Load all necessary data files
    with open(td, 'r', encoding='UTF-8') as f:
        par2_dict = json.load(f)  # PAR2 scores for all problems
    with open(tc, 'r', encoding='UTF-8') as f:
        train_cluster_dict = json.load(f)  # Cluster assignments

    with open(tlim, 'r', encoding='UTF-8') as f:
        lim_dict = json.load(f)  # Normalization limits
    with open(tcenter, 'r', encoding='UTF-8') as f:
        center_dict = json.load(f)  # Cluster centers

    # Validate portfolio size against available solvers
    if portfolioSize > len(solverlist):
        print("warning: PortfolioSize is bigger than the number of solvers!")
        portfolioSize = len(solverlist)

    # Ensure output directory exists and generate output filename
    dirpath = 'output'
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    if outputfile == "":
        outputfile = f"output/train_result_{dataset}_{portfolioSize}_{cluster_num}_{seed}.json"

    # Initialize portfolio storage for each cluster
    cluster = 0
    final_portfolio = {}

    # Optimize portfolio for each cluster independently
    for cluster in range(cluster_num):
        print(f"Optimizing portfolio for cluster {cluster}")
        print(f"Available solvers: {solverlist}")

        # Track which solvers have been selected (coverage vector)
        cov = [0 for i in range(len(solverlist))]

        # Store selected solver indices and timeout configuration
        output_idx = []
        final_config = []

        # Get training data for this cluster
        train_set = par2_dict['train']

        # Build portfolio by selecting best solver for each position
        for portfolio_position in range(portfolioSize):
            min_time = float('inf')  # Track best (minimum) PAR2 time
            min_idx = -1             # Track best solver index
            sf = []  # SMAC3 configurations to evaluate
            pf = []
            for j in range(len(solverlist)):
                tmpdict = {}
                if cov[j] == 1:
                    continue
                tmp = []
                for l in output_idx:
                    tmp.append(l)
                tmp.append(j)
                tmpdict['t1'] = 1200
                tmpdict['t2'] = 0
                tmpdict['t3'] = 0
                for l in range(len(tmp)):
                    tmpdict["s"+str(l+1)] = str(tmp[l])
                tmpdict["cluster"] = cluster
                if dataset == 'Equality+LinearArith':
                    tmpdict["dataset"] = "ELA"
                else:
                    tmpdict["dataset"] = str(dataset)

                sf.append([tmpdict,j])
            valid_scores = [0 for i in range(len(sf))]
            for si in range(0,5):
                p = Pool(processes=10)

                partial_RunSeed = partial(RunSeed3, seed = seed,start_idx=si)
                ret = p.map(partial_RunSeed, sf)

                p.close()
                p.join()

                ret_seed = []
                for l in range(len(ret)):
                    ret_seed.append(ret[l][1])
                    k = ret[l][0][-2].split(",")
                    ret[l] = k
                print(ret)
                configs = [[] for i in range(len(ret))]
                for l in range(len(ret)):
                    tmp_config = []
                    for _ in range(len(ret[l])):
                        tmp_config.append(float(ret[l][_]))
                    configs[l] = tmp_config

                key_set = list(train_set.keys())

                full_key_set = list(train_set.keys())
                key_set = []

                if dataset == "Equality+LinearArith":
                    dataplace = "ELA"
                elif dataset == "QF_Bitvec":
                    dataplace = "QFBV"
                elif dataset == "QF_NonLinearIntArith":
                    dataplace = "QFNIA"
                else:
                    dataplace = dataset
                for j in full_key_set:
                    if j in train_cluster_dict.keys() and train_cluster_dict[j] == cluster:
                        key_set.append(j)
                    if  "./infer_result/"+str(dataset)+"/_data_sibly_sibyl_data_"+str(dataset)+"_"+str(dataset)+"_" + j.replace("/","_") + ".json" in train_cluster_dict.keys() and train_cluster_dict["./infer_result/"+str(dataset)+"/_data_sibly_sibyl_data_"+str(dataset)+"_"+str(dataset)+"_" + j.replace("/","_") + ".json"] == cluster:
                        key_set.append(j)
                    if  "./infer_result/"+str(dataplace)+"/_data_sibly_sibyl_data_Comp_non-incremental_" + j.replace("/","_") + ".json" in train_cluster_dict.keys() and train_cluster_dict["./infer_result/"+str(dataplace)+"/_data_sibly_sibyl_data_Comp_non-incremental_" + j.replace("/","_") + ".json"] == cluster:
                        key_set.append(j)

                print("configs len:",len(configs))
                print(output_idx)

                for config_idx in range(len(configs)):
                    config = configs[config_idx]
                    x1 = config[0]
                    x2 = config[1]
                    x3 = config[2]
                    x4 = 1200-x1-x2-x3
                    tmp_config = [x1,x2,x3,x4]

                    time = 0

                    ri = int(len(key_set)*(0.2*(si+1)))
                    ri = min(ri,int(len(key_set)))
                    for _ in range(int(len(key_set)*(0.2*si)),ri):
                        tmp_time = 0
                        flag=0
                        par2list = train_set[key_set[_]]
                        tmplist_ = []
                        for idx in output_idx:
                            tmplist_.append(idx)
                        tmplist_.append(ret_seed[config_idx])
                        for l in range(len(tmplist_)):
                            if float(par2list[tmplist_[l]]) <= tmp_config[l]:
                                tmp_time+=par2list[tmplist_[l]]
                                for k in range(l):
                                    tmp_time+=tmp_config[k]
                                time += tmp_time
                                flag=1
                                break
                        if flag == 0:
                            time += 2400
                    valid_scores[config_idx] += time
            print(valid_scores)
            chosen_idx = np.argmin(valid_scores)
            final_config = configs[chosen_idx]
            output_idx.append(ret_seed[chosen_idx])
            cov[ret_seed[chosen_idx]] = 1

        tmpdict = {}
        tmpdict['t1'] = 1200
        tmpdict['t2'] = 0
        tmpdict['t3'] = 0
        for l in range(len(output_idx)):
            tmpdict["s"+str(l+1)] = str(output_idx[l])
        tmpdict["cluster"] = cluster
        if dataset == 'Equality+LinearArith':
            tmpdict["dataset"] = "ELA"
        else:
            tmpdict["dataset"] = str(dataset)
        sf = []
        sf.append([tmpdict,-1])
        p = Pool(processes=1)
        partial_RunSeed = partial(RunSeed3, seed = seed,start_idx=-1)
        ret = p.map(partial_RunSeed, sf)
        p.close()
        p.join()
        for l in range(len(ret)):
            k = ret[l][0][-2].split(",")
            ret[l] = k
        print(ret)
        configs = [[] for i in range(len(ret))]
        for l in range(len(ret)):
            tmp_config = []
            for _ in range(len(ret[l])):
                tmp_config.append(float(ret[l][_]))
            configs[l] = tmp_config
        final_config = configs[0]
        final_portfolio[cluster] = [output_idx,final_config]
    output_dict = {"portfolio":final_portfolio,"lim":lim_dict,"center":center_dict}
    with open(outputfile,'w',encoding='utf-8') as f:
        json.dump(output_dict, f)

if __name__ == '__main__':
    """
    Main execution entry point for SMTgazer training and inference.

    Usage:
        python SMTportfolio.py train [options]  - Train portfolios for a dataset
        python SMTportfolio.py infer [options]  - Apply trained portfolios to test data

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
    if len(sys.argv) > 1 and (sys.argv[1] == 'train' or sys.argv[1] == 'infer'):
        work_type = sys.argv[1]

    # Initialize default values
    tf = ""           # Training features file
    td = ""           # Training data file
    seed = 0          # Random seed
    cluster_num = 20  # Maximum clusters
    solverdict = ""   # Solver configuration file
    dataset = ""      # Dataset name
    clusterPortfolio = ""  # Trained portfolio file
    for i in range(len(sys.argv)-1):
        if (sys.argv[i] == '-train_features'):
            tf = sys.argv[i+1]
        if (sys.argv[i] == '-train_data'):
            td = sys.argv[i+1]

        if (sys.argv[i] == '-seed'):
            seed = int(sys.argv[i+1])
        if (sys.argv[i] == '-cluster_num'):
            cluster_num = int(sys.argv[i+1])

        if (sys.argv[i] == '-solverdict'):
            solverdict = sys.argv[i+1]

        if (sys.argv[i] == '-dataset'):
            dataset = sys.argv[i+1]


        if (sys.argv[i] == '-clusterPortfolio'):
            clusterPortfolio = sys.argv[i+1]

    tf = "./machfea/infer_result/" + str(dataset) + "_train_feature.json"
    td = "./data/" + str(dataset) + "Labels.json"
    dirpath = 'tmp'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    # Execute training or inference based on work_type
    if work_type == "train":
        print(f"Starting SMTgazer training for dataset: {dataset}")

        ### Step 1: Normalize feature vectors ###
        print("Step 1: Normalizing features...")
        normalize(tf, seed)

        ### Step 2: Perform clustering ###
        print("Step 2: Clustering problems...")
        # Clean up file path for consistent naming
        tmp = tf.replace("../", "").replace("./", "").replace("/", "_")
        tfnorm = f"tmp/{tmp.replace('.json', f'_norm{seed}.json')}"
        tflim = tfnorm.replace("_norm", "_lim")
        tcenter = tfnorm.replace(f"_norm{seed}.json", f"_cluster_center_{seed}.json")

        cluster(tfnorm, seed, cluster_num)
        tc = tfnorm.replace(f"_norm{seed}.json", f"_train_{seed}.json")

        ### Step 3: Optimize solver portfolios ###
        print("Step 3: Optimizing solver portfolios...")
        with open(solverdict, 'r', encoding='UTF-8') as f:
            solver_dict = json.load(f)
        solverlist = solver_dict["solver_list"]

        get_portfolio_3(
            solverlist, td, tc, tflim, tcenter, dataset,
            outputfile="", portfolioSize=4, cluster_num=cluster_num,
            seed=seed, timelimit=1200
        )
        print("Training completed!")

    elif work_type == "infer":
        print(f"Starting SMTgazer inference for dataset: {dataset}")

        # Load trained portfolio configuration
        with open(clusterPortfolio, 'r', encoding='UTF-8') as f:
            output_dict = json.load(f)

        # Extract normalization parameters used during training
        lim = output_dict['lim']
        min_ = lim['min']
        sub = lim['sub']

        # Load test features and apply same normalization as training
        testf = f"./machfea/infer_result/{dataset}_test_feature.json"
        with open(testf, 'r', encoding='UTF-8') as f:
            fea_dict_ = json.load(f)

        # Prepare feature matrix for normalization
        pro_dict = []
        fea_dict = []
        for problem_name in fea_dict_.keys():
            pro_dict.append(problem_name)
            fea_dict.append(fea_dict_[problem_name])
        fea_dict = np.array(fea_dict)
        print(f"Test feature matrix shape: {fea_dict.shape}")

        # Apply same normalization as training data
        new_fea_dict = (fea_dict - np.array(min_)) / np.array(sub)
        new_pro_dict = pro_dict

        print(f"Normalized test features: {len(new_fea_dict)} problems")
        print(f"Problem names: {len(new_pro_dict)}")

        # Prepare normalized test feature dictionary
        dict_output = {}
        for i in range(len(pro_dict)):
            dict_output[pro_dict[i]] = new_fea_dict[i].tolist()

        # Clean up file path for output naming
        clean_testf = testf.replace("../", "").replace("./", "").replace("/", "_")
        testnorm = f"tmp/{clean_testf.replace('.json', f'_norm{seed}.json')}"

        # Save normalized test features
        with open(testnorm, 'w', encoding='UTF-8') as f:
            json.dump(dict_output, f)

        # Load solver configuration and run inference
        with open(solverdict, 'r', encoding='UTF-8') as f:
            solver_dict = json.load(f)
        solverlist = solver_dict["solver_list"]

        print("Running portfolio inference...")
        getTestPortfoliio(testnorm, clusterPortfolio, solverlist, dataset, seed)
        print("Inference completed!")
