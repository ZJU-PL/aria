"""
Machine Learning Feature Extraction Runner for SMT Problems

This module runs feature extraction for SMT problems using the Sibyl feature extractor.
It processes both training and test datasets in parallel to generate feature vectors
that will be used for clustering and portfolio optimization in SMTgazer.

The feature extraction uses the Sibyl tool to analyze SMT problem files and extract
statistical features that characterize the problem structure and complexity.

Author: SMTgazer Team
Publication: ASE 2025
"""

import json
import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
from os import popen  # pylint: disable=no-name-in-module

# Seed for reproducible feature extraction
seed = [0]

def run_seed(data_index, seed_val, test_set_param, par2_dict_param, dataset_param):
    """
    Extract features for a single SMT problem instance using Sibyl.

    This function runs the Sibyl feature extractor on an individual SMT problem
    file to generate statistical features that characterize the problem.

    Args:
        data_index (str): Problem identifier (file path within dataset)
        seed_val (int): Random seed for reproducible feature extraction
        test_set_param (dict): Test set performance data
        par2_dict_param (dict): Training set performance data
        dataset_param (str): Dataset name

    Returns:
        list: Feature extraction output lines from Sibyl, or None if failed

    The process:
    1. Check if problem has valid solver performance data
    2. Locate the SMT problem file in the Sibyl dataset
    3. Run Sibyl feature extractor with appropriate parameters
    4. Return parsed feature vector
    """
    # Get PAR2 performance data for this problem (from test or training set)
    if data_index in test_set_param.keys():
        par2list = test_set_param[data_index]  # Test set performance data
    else:
        par2list = par2_dict_param['train'][data_index]  # Training set performance data

    # Skip problems that no solver could solve (all timeouts)
    if np.min(par2list) == 2400:
        return None

    # Construct path to SMT problem file in Sibyl dataset
    target_file = f"~/sibly/sibyl/data/Comp/non-incremental/{data_index}"
    if not os.path.exists(target_file):
        print(f"No instance file found: {data_index}")
        return None

    # Build Sibyl feature extraction command
    command = f"python get_feature.py {target_file} --dataset {dataset_param}"

    print(f"Extracting features: {command}")
    output = popen(command).read()
    tmp = output.split('\n')
    return tmp


if __name__ == '__main__':
    # Default dataset configuration
    DATASET = "ELA"  # Short name for Equality+LinearArith

    # Load performance data for all SMT-COMP datasets
    with open("../data/SMTCompLabels.json", 'r', encoding='UTF-8') as f:
        par2_dict = json.load(f)

    # Get random seed from command line arguments
    seed_ = int(sys.argv[1])

    # Set target dataset name
    DATASET_NAME = 'Equality+LinearArith'

    # Extract performance data for the target dataset
    par2_dict = par2_dict[DATASET_NAME]

    # Combine test and training problem sets
    test_set = par2_dict['test']
    key_set = list(test_set.keys()) + list(par2_dict['train'].keys())
    print(f"Total problems to process: {len(key_set)}")

    # Set up parallel processing (20 worker processes)
    with Pool(processes=20) as p:
        # Create partial function with fixed seed parameter
        partial_run_seed = partial(
            run_seed,
            seed_val=seed_,
            test_set_param=test_set,
            par2_dict_param=par2_dict,
            dataset_param=DATASET
        )

        # Extract features for all problems in parallel
        ret = p.map(partial_run_seed, key_set)

        # Parse and organize feature extraction results
        fea_dict = {}
        for result in ret:
            if result is not None:
                # Parse feature vector from Sibyl output
                # get_feature.py outputs: feature_vector_string, benchmark_name
                feature_line = result[-2]  # Second to last line contains feature vector
                problem_name = result[-1]  # Last line contains benchmark name

                # Parse feature vector string into list of floats
                feature_str = feature_line.replace('[', '').replace(']', '').replace(' ', '')
                feature_vector = list(map(float, feature_str.split(',')))
                fea_dict[problem_name] = feature_vector

        # Save extracted features to JSON file
        OUTPUT_FILE = f'infer_result/{DATASET_NAME}_feature.json'
        with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
            json.dump(fea_dict, f)

        print(f"Feature extraction completed. Saved to: {OUTPUT_FILE}")
