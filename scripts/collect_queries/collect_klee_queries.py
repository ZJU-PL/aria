"""Dump SMT-LIB queries from KLEE traces.

Usage:
    python collect_klee_queries.py [--source-dir SOURCE_DIR] [--output-dir OUTPUT_DIR] [--size-limit SIZE_LIMIT]

Options:
    --source-dir SOURCE_DIR    Directory containing KLEE raw data [default: data/klee/raw]
    --output-dir OUTPUT_DIR    Directory to store separated queries [default: data/klee/single_test]
    --size-limit SIZE_LIMIT    Size limit in MB for files to be deleted [default: 512]

KLEE SMT scripts are collected in one file, separate them into single files
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from tqdm import tqdm


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process KLEE SMT queries")
    parser.add_argument(
        "--source-dir",
        default="data/klee/raw",
        help="Directory containing KLEE raw data",
    )
    parser.add_argument(
        "--output-dir",
        default="data/klee/single_test",
        help="Directory to store separated queries",
    )
    parser.add_argument(
        "--size-limit",
        type=int,
        default=512,
        help="Size limit in MB for files to be deleted",
    )
    return parser.parse_args()


def ensure_directory_exists(directory):
    """Ensure the specified directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def process_queries(source_dir, output_dir, size_limit_mb):
    """Process KLEE query files and extract individual queries."""
    size_limit_bytes = size_limit_mb * 1024 * 1024

    # Count total files for progress bar
    total_query_files = 0
    for root, _, files in os.walk(source_dir):
        if "solver-queries.smt2" in files:
            total_query_files += 1

    processed_files = 0
    total_queries = 0
    large_files_removed = 0

    with tqdm(total=total_query_files, desc="Processing files") as pbar:
        for root, _, files in os.walk(source_dir):
            if "solver-queries.smt2" in files:
                try:
                    filename = root.split("/")[-1].split("-")[0]
                    query_file_path = os.path.join(root, "solver-queries.smt2")

                    queries_extracted = process_single_file(
                        query_file_path, filename, output_dir
                    )
                    total_queries += queries_extracted

                    # Check file size and remove if too large
                    if os.path.getsize(query_file_path) > size_limit_bytes:
                        logging.info(f"Removing large file: {query_file_path}")
                        os.remove(query_file_path)
                        large_files_removed += 1

                    processed_files += 1
                    pbar.update(1)

                except Exception as e:
                    logging.error(f"Error processing {root}: {str(e)}")
                    logging.debug(traceback.format_exc())

    logging.info(
        f"Processing complete: {processed_files} files processed, {total_queries} queries extracted"
    )
    if large_files_removed > 0:
        logging.info(
            f"Removed {large_files_removed} large files exceeding {size_limit_mb}MB"
        )


def process_single_file(query_file_path, filename, output_dir):
    """Process a single KLEE query file and extract individual queries."""
    queries_extracted = 0

    with open(query_file_path, "r", errors="replace") as f:
        next_is_time = False
        start = False
        script = ""

        for line in f:
            try:
                if "(set-logic QF_AUFBV )" in line:
                    start = True

                if next_is_time:
                    origin_time = float(line.split(": ")[-1][:-2])
                    output_file = os.path.join(
                        output_dir, f"{filename}{queries_extracted}"
                    )

                    with open(output_file, "w") as f1:
                        json.dump(
                            {
                                "filename": filename,
                                "smt_script": script,
                                "time": origin_time,
                            },
                            f1,
                            indent=4,
                        )

                    start = False
                    next_is_time = False
                    script = ""
                    queries_extracted += 1

                if start:
                    script += line

                if "(exit)" in line:
                    next_is_time = True

            except Exception as e:
                logging.warning(f"Error processing line in {query_file_path}: {str(e)}")
                continue

    return queries_extracted


def main():
    """Main function to run the script."""
    setup_logging()
    args = parse_arguments()

    logging.info(
        f"Starting KLEE query collection from {args.source_dir} to {args.output_dir}"
    )

    # Ensure output directory exists
    ensure_directory_exists(args.output_dir)

    # Process the queries
    process_queries(args.source_dir, args.output_dir, args.size_limit)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Unhandled exception: {str(e)}")
        logging.debug(traceback.format_exc())
        sys.exit(1)
