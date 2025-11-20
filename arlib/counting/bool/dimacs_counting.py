"""
Model counting for DIMACS files
"""
import copy
import os
import subprocess
from typing import List
from threading import Timer
import logging
import uuid
import time
from pyapproxmc import Counter

import multiprocessing
from multiprocessing import cpu_count

from pysat.formula import CNF
from pysat.solvers import Solver

from arlib.bool.pysat_cnf import gen_cubes

from arlib.global_params import global_config

sharp_sat_timeout = 600  # seconds

logger = logging.getLogger(__name__)


def terminate(process, is_timeout: List):
    """
    Terminate a given process and set the is_timeout flag to True.

    Args:
        process (subprocess.Popen): The process to be terminated.
        is_timeout (List): A list containing a single boolean value indicating if the process has timed out.
    """
    if process.poll() is None:
        try:
            process.terminate()
            is_timeout[0] = True
        except Exception as ex:
            print("error for interrupting")
            print(ex)


def write_dimacs_to_file(header: List[str], clauses: List[str], output_file: str):
    """
    Write the header and clauses of a DIMACS CNF formula to a file.

    Args:
        header (List[str]): A list containing the header information of the DIMACS CNF file.
        clauses (List[str]): A list of strings representing the clauses of the DIMACS CNF file.
        output_file (str): The path to the output file where the DIMACS CNF formula will be written.
    """
    # print("header: ", header)
    # print("clauses: ", len(clauses), clauses)
    with open(output_file, 'w+') as file:
        for info in header:
            file.write(info + "\n")
        for cls in clauses:
            file.write(cls + " 0\n")


def call_approxmc(clauses, timeout=600):
    """
    Run the ApproxMC solver on a given DIMACS CNF file and return the number of solutions.

    Args:
        clauses: List of clauses to count
        timeout: Maximum time in seconds to run the solver (default: 300 seconds)

    Returns:
        int: Number of solutions, or -1 if timeout or error occurs
    """
    try:
        counter = Counter()
        for clause in clauses:
            clause_list = [int(x) for x in clause.split(" ")]
            if 0 in clause_list:
                clause_list.remove(0)
            counter.add_clause(clause_list)

        # 设置超时
        timer = Timer(timeout, lambda: counter.interrupt())
        timer.start()

        try:
            c = counter.count()
            print("approxmc result: ", c)
            return c[0] * 2 ** (c[1])
        except Exception as ex:
            print("Exception in approxmc counting:", ex)
            return -1
        finally:
            timer.cancel()

    except Exception as ex:
        print("Error in approxmc setup:", ex)
        return -1


def call_sharp_sat(cnf_filename: str):
    """
    Call the sharpSAT solver on a given DIMACS CNF file and return the number of solutions and solving time.

    Args:
        cnf_filename (str): The path to the DIMACS CNF file.

    Returns:
        tuple: (solutions, solving_time) where solutions is the number of solutions and solving_time is the time taken.
    """

    solutions = -1
    start_time = time.time()
    if not global_config.is_solver_available("sharp_sat"):
        logger.error("sharpSAT is not available")
        return -1, -1
    cmd = [global_config.get_solver_path("sharp_sat"), cnf_filename]
    print("Calling sharpSAT")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    is_timeout = [False]
    timer = Timer(sharp_sat_timeout, terminate, args=[p, is_timeout])
    timer.start()
    try:
        find_sol_line = False
        for line in iter(p.stdout.readline, b''):
            if not line:
                break
            decode_line = line.decode('UTF-8')
            if find_sol_line:
                # print("sharpSAT res: ", decode_line)
                solutions = int(decode_line)
                break
            if "solutions" in decode_line:
                find_sol_line = True
    except Exception as ex:
        # print(ex)
        print("exception when running sharpSAT, will return false")
    finally:
        timer.cancel()
        p.stdout.close()

        # Make sure the process is terminated
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)  # Wait up to 5 seconds for graceful termination
            except subprocess.TimeoutExpired:
                p.kill()  # Kill if termination doesn't complete
                p.wait()

        # Clean up the temp file
        if os.path.isfile(cnf_filename):
            os.remove(cnf_filename)

    if is_timeout[0]:
        logging.debug("sharpSAT timeout")

    solving_time = time.time() - start_time
    return solutions, solving_time


def count_dimacs_solutions(header: List, str_clauses: List):
    """
    Count the number of solutions for a given DIMACS CNF formula.

    Args:
        header (List): A list containing the header information of the DIMACS CNF file.
        str_clauses (List): A list of strings representing the clauses of the DIMACS CNF file.

    Returns:
        int: The number of solutions for the given DIMACS CNF formula.
    """
    output_file = '/tmp/{}.cnf'.format(str(uuid.uuid1()))
    write_dimacs_to_file(header, str_clauses, output_file)
    return call_sharp_sat(output_file)


def check_sat(clauses, assumptions):
    """
    Check the satisfiability of a CNF formula with given assumptions.

    Args:
        clauses (List): A list of clauses representing the CNF formula.
        assumptions (List): A list of literals representing the assumptions.

    Returns:
        bool: True if the CNF formula is satisfiable under the given assumptions, False otherwise.
    """
    solver = Solver(bootstrap_with=clauses)
    ans = solver.solve(assumptions=assumptions)
    return ans


def count_dimacs_solutions_parallel(header: List[str], clauses: List[str]) -> int:
    """
    Count the number of solutions for a given DIMACS CNF formula in parallel.
       1. Generate a set of disjoint cubes
        C1: a, b
        C2: Not(a), b
        C3: a, Not b
        C4: Not(a), Not(b)
      2. Count the models subject to each cube in parallel, and sum the results

    Args:
        header (List[str]): A list containing the header information of the DIMACS CNF file.
        clauses (List[str]): A list of strings representing the clauses of the DIMACS CNF file.

    Returns:
        int: The number of solutions for the given DIMACS CNF formula.
    """

    dimacs_str = ""
    for info in header:
        dimacs_str += "{}\n".format(info)
    for cls in clauses:
        dimacs_str += "{} 0\n".format(cls)

    cnf = CNF(from_string=dimacs_str)
    cubes = gen_cubes(cnf, min(2, cnf.nv))
    satisfaible_cubes = []

    # Prune unsatisfiable assumptions
    for cube in cubes:
        if check_sat(cnf, assumptions=cube):
            satisfaible_cubes.append(cube)

    cnf_tasks = []
    for cube in satisfaible_cubes:
        output_file = '/tmp/{}.cnf'.format(str(uuid.uuid1()))
        new_clauses = copy.deepcopy(clauses)
        # every list in the cube should be a unit clause?
        for lit in cube:
            new_clauses.append(str(lit))
        new_header = ["p cnf {0} {1}".format(cnf.nv, len(clauses) + len(cube))]
        write_dimacs_to_file(new_header, new_clauses, output_file)
        cnf_tasks.append(output_file)

    results = []
    pool = multiprocessing.Pool(processes=cpu_count())  # process pool
    try:
        for i in range(len(cnf_tasks)):
            result = pool.apply_async(call_sharp_sat,
                                    (cnf_tasks[i],))
            results.append(result)

        raw_solutions: List[int] = []
        sharp_sat_times: List[float] = [] 
        for i in range(len(cnf_tasks)):
            result = results[i].get()
            raw_solutions.append(int(result[0]))
            sharp_sat_times.append(result[1])

        print("results: ", raw_solutions)
        print("sharpSAT times: ", sharp_sat_times)
        # if -1 in raw_solutions:
        #     print("sharpSAT failed, calling approxmc")
        #     start_time = time.time()
        #     result = call_approxmc(clauses)
        #     approxmc_time = time.time() - start_time
        #     print("approxmc result: ", result)
        #     with open('temp.log', 'a') as f:
        #         f.write(f"{cnf.nv} {len(clauses)} {[-1 if x > 60 else x for x in sharp_sat_times]} {approxmc_time if result != -1 else -1}\n")
        #     return result
        with open('temp.log', 'a') as f:
            f.write(f"{cnf.nv} {len(clauses)} {sharp_sat_times} 0\n")
        return sum(raw_solutions)
    finally:
        # Ensure the pool is properly closed
        pool.close()
        pool.join()

        # Remove any temporary files that might be left
        for file_path in cnf_tasks:
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
