"""
Collect queries from CBMC

This script processes C source files using CBMC to generate SMT2 queries.
It walks through a source directory, processes each .c file, and outputs
the corresponding SMT2 files to a specified output directory.

Usage:
    python collect_cbmc_queries.py --source-dir /path/to/source --output-dir /path/to/output [--unwind N] [--timeout SECONDS]

Arguments:
    --source-dir: Directory containing C source files to process
    --output-dir: Directory where SMT2 files will be saved
    --unwind: Maximum number of loop unwindings (default: 10)
    --timeout: Timeout in seconds for each file processing (default: 600)
"""

import argparse
import os
import subprocess
from z3 import *


def process_file(source_file, output_file_path, unwind, timeout):
    """Process a single C file using CBMC and save the SMT2 output.

    Args:
        source_file (str): Path to the source C file
        output_file_path (str): Path where the SMT2 output should be saved
        unwind (int): Maximum number of loop unwindings
        timeout (int): Timeout in seconds
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct CBMC command
    cbmc_command = [
        "timeout",
        str(timeout),
        "cbmc",
        "--smt2",
        source_file,
        "--outfile",
        output_file_path,
        "--z3",
        "--unwind",
        str(unwind),
    ]

    # Execute CBMC command
    try:
        result = subprocess.run(cbmc_command, check=True)
        if result.returncode == 124:
            print(f"Timeout: {source_file}")
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
            return
        print(f"Successfully processed: {source_file}")

        # Check if output file contains "Array" string
        if os.path.exists(output_file_path):
            with open(output_file_path, "r") as f:
                content = f.read()
                if "Array" in content:
                    os.remove(output_file_path)
                    print(f"Deleted file containing Array: {output_file_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to process {source_file}: {e}")
        if os.path.exists(output_file_path):
            os.remove(output_file_path)


def main():
    parser = argparse.ArgumentParser(
        description="Collect CBMC queries from C source files"
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing C source files to process",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory where SMT2 files will be saved"
    )
    parser.add_argument(
        "--unwind",
        type=int,
        default=10,
        help="Maximum number of loop unwindings (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each file processing (default: 600)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Process all .c files in the source directory
    for root, dirs, files in os.walk(args.source_dir):
        for file in files:
            if file.endswith(".c"):
                # Construct full path for source file
                source_file = os.path.join(root, file)

                # Construct output file path
                relative_path = os.path.relpath(root, args.source_dir)
                output_file_path = os.path.join(
                    args.output_dir, relative_path, file.replace(".c", ".smt2")
                )

                process_file(source_file, output_file_path, args.unwind, args.timeout)


if __name__ == "__main__":
    main()
