"""
SMT Problem Feature Extraction using MachSMT

This script uses the MachSMT library to extract statistical features from
individual SMT problem files. These features characterize the problem
structure and are used for machine learning-based solver selection.

The MachSMT library analyzes SMT formulas and extracts various statistical
features such as:
- Symbol and term counts
- Expression depths and sizes
- Theory-specific characteristics
- Problem complexity metrics

Author: SMTgazer Team
Publication: ASE 2025
"""

from machsmt import Benchmark, args
import json

if __name__ == '__main__':
    """
    Extract features from a single SMT benchmark file.

    Usage:
        python get_feature.py <benchmark_file> --dataset <dataset_name>

    This script loads an SMT problem file, parses it using MachSMT,
    extracts statistical features, and outputs them in a format
    suitable for machine learning processing.
    """
    print(f"Processing benchmark: {args.benchmark}")

    # Load and parse the SMT benchmark file
    benchmark = Benchmark(args.benchmark)
    benchmark.parse()

    # Extract statistical features using MachSMT
    feature = benchmark.get_features()

    # Clean benchmark name for output
    dataset = args.dataset
    benchmarkname = args.benchmark.replace("../data/", '')
    benchmarkname = benchmarkname.replace("/", "_")

    # Prepare feature dictionary
    fea = {}
    fea[benchmarkname] = feature

    # Output feature vector and benchmark name
    print(feature)
    print(f"{benchmarkname}.json")
