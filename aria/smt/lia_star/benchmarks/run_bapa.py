#!/usr/bin/env python3

import sys
import subprocess
import argparse
import time
import ast
import csv

def main():

    # Initialize arg parser
    prog_desc = ('Runs lia_star_solver.py on all bapa benchmarks, with the given options, '
                 'and writes runtime statistics to txt and csv files')
    p = argparse.ArgumentParser(description=prog_desc)
    p.add_argument('outfile', metavar='FILENAME_NO_EXT', type=str,
                   help='name of the output file to write statistics to (omit extension)')
    p.add_argument('timeout', metavar='TIMEOUT', type=int,
                   help='timeout for each benchmark in seconds')
    p.add_argument('-m', '--mapa', action='store_true',
                   help='treat the BAPA benchmark as a MAPA problem (interpret the variables as multisets, not sets)')
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
    benchmarks = ["fol_{}.smt2".format(str(i).zfill(7)) for i in range(1,121)]
    txtfile = "{}.txt".format(filename)
    csvfile = "{}.csv".format(filename)

    # Run each benchmark, storing the output in a file
    print("\ncheck {} to see test results\n".format(txtfile))
    with open(txtfile, 'w') as outfile:
        with open(csvfile, 'w') as outcsv:

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
                    print("{}/{}...".format(d, f))

                    # Set up command line arguments to lia_star_solver.py
                    cmd = ["python3", "../lia_star_solver.py", "{}/{}".format(d, f), "--unfold={}".format(unfold), "-i"]
                    if mapa:
                        cmd.append("--mapa")
                    if no_interp:
                        cmd.append("--no-interp")

                    # Run solver, catching exceptions and timing the execution
                    start = time.time()
                    try:

                        # Attempt to run command
                        res = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout)
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
                        output = "ERROR {}".format(exc.output.decode("utf-8"))

                    # Solver times out
                    except subprocess.TimeoutExpired as exc:
                        end = time.time()
                        output_lines = exc.output.decode("utf-8").split('\n')
                        stats = {'sat': 2, 'problem_size': int(output_lines[1])}
                        output = "timeout"

                    # Write sat or unsat and time taken to file
                    outfile.write("{}/{}".format(d, f).ljust(27) + " : {} : {}\n".format(output.rjust(7), end - start))
                    outfile.flush()

                    # Write stats to csv file
                    stats['name'] = "{}/{}".format(d, f)
                    writer.writerow(stats)
                    outcsv.flush()

    # Reminder
    print("\ncheck {} to see test results\n".format(txtfile))


# Entry point
if __name__ == "__main__":
    main()