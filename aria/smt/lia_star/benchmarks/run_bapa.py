#!/usr/bin/env python3
"""Run lia_star_solver.py on all bapa benchmarks and write runtime statistics."""

import argparse
import ast
import csv
import subprocess
import time


def main():
    """
    Run lia_star_solver.py on all bapa benchmarks with given options.

    Writes runtime statistics to txt and csv files.
    """
    # Initialize arg parser
    prog_desc = (
        'Runs lia_star_solver.py on all bapa benchmarks, with the given options, '
        'and writes runtime statistics to txt and csv files'
    )  # noqa: E501
    p = argparse.ArgumentParser(description=prog_desc)
    p.add_argument('outfile', metavar='FILENAME_NO_EXT', type=str,
                   help='name of the output file to write statistics to (omit extension)')
    p.add_argument('timeout', metavar='TIMEOUT', type=int,
                   help='timeout for each benchmark in seconds')
    p.add_argument('-m', '--mapa', action='store_true',
                   help='treat the BAPA benchmark as a MAPA problem '
                         '(interpret the variables as multisets, not sets)')
    p.add_argument('--no-interp', action='store_true',
                   help='turn off interpolation')
    p.add_argument('--unfold', metavar='N', type=int, default=0,
                   help='number of unfoldings to use when interpolating (default: 0)')

    # Read args
    args = p.parse_args()
    filename = args.outfile
    mapa = args.mapa
    unfold = args.unfold
    timeout = args.timeout
    no_interp = args.no_interp

    # Collect filenames
    dirs = ["bapa"]
    benchmarks = [f"fol_{str(i).zfill(7)}.smt2" for i in range(1, 121)]
    txtfile = f"{filename}.txt"
    csvfile = f"{filename}.csv"

    # Run each benchmark, storing the output in a file
    print(f"\ncheck {txtfile} to see test results\n")
    with open(txtfile, 'w', encoding='utf-8') as outfile:
        with open(csvfile, 'w', encoding='utf-8') as outcsv:

            # Set up csv file
            fieldnames = [
                'name',
                'sat',
                'problem_size',
                'sls_size',
                'z3_calls',
                'interpolants_generated',
                'merges',
                'shiftdowns',
                'offsets',
                'total_time',
                'reduction_time',
                'augment_time',
                'interpolation_time',
                'solution_time'
            ]
            writer = csv.DictWriter(outcsv, fieldnames=fieldnames)
            writer.writeheader()

            # Iterate over directories and filenames of bapa benchmarks
            for d in dirs:
                for f in benchmarks:

                    # Print the current file
                    print(f"{d}/{f}...")

                    # Set up command line arguments to lia_star_solver.py
                    cmd = [
                        "python3",
                        "../lia_star_solver.py",
                        f"{d}/{f}",
                        f"--unfold={unfold}",
                        "-i"
                    ]
                    if mapa:
                        cmd.append("--mapa")
                    if no_interp:
                        cmd.append("--no-interp")

                    # Run solver, catching exceptions and timing the execution
                    start = time.time()
                    try:

                        # Attempt to run command
                        res = subprocess.check_output(
                            cmd, stderr=subprocess.STDOUT, timeout=timeout
                        )
                        output_lines = res.decode("utf-8").split("\n")
                        output = output_lines[3]

                        # Collect statistics
                        end = time.time()
                        stats = ast.literal_eval(output_lines[2])
                        stats['total_time'] = end - start

                    # Solver throws an exception
                    except subprocess.CalledProcessError as exc:
                        end = time.time()
                        stats = {}
                        output = f"ERROR {exc.output.decode('utf-8')}"

                    # Solver times out
                    except subprocess.TimeoutExpired as exc:
                        end = time.time()
                        output_lines = exc.output.decode("utf-8").split('\n')
                        stats = {'sat': 2, 'problem_size': int(output_lines[1])}
                        output = "timeout"

                    # Write sat or unsat and time taken to file
                    file_line = f"{d}/{f}".ljust(27)
                    result_line = f" : {output.rjust(7)} : {end - start}\n"
                    outfile.write(file_line + result_line)
                    outfile.flush()

                    # Write stats to csv file
                    stats['name'] = f"{d}/{f}"
                    writer.writerow(stats)
                    outcsv.flush()

    # Reminder
    print(f"\ncheck {txtfile} to see test results\n")


# Entry point
if __name__ == "__main__":
    main()
