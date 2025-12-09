"""
Generate complex formulas using satfuzz.py, smtfuzz.py, and qbfuzz.py
"""

import logging
import random
import subprocess
from pathlib import Path
from threading import Timer
from typing import List

CNF_GENERATOR = str(Path(__file__).parent) + "/satfuzz.py"
QBF_GENERATOR = str(Path(__file__).parent) + "/qbfuzz.py"
SMT_GENERATOR = str(Path(__file__).parent) + "/smtfuzz.py"


def terminate(process, is_timeout: List):
    """Terminate process and set timeout flag"""
    if process.poll() is None:
        try:
            process.terminate()
            is_timeout[0] = True
        except Exception as ex:
            print(f"Termination error: {ex}")


def run_subprocess(cmd: List[str], timeout: int = 15) -> str:
    """Run subprocess with timeout and return decoded output"""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    is_timeout = [False]
    timer = Timer(timeout, terminate, args=[p, is_timeout])
    timer.start()
    out = p.stdout.readlines()
    out = ' '.join([element.decode('UTF-8') for element in out])
    p.stdout.close()
    timer.cancel()
    if p.poll() is None:
        p.terminate()
    return out if not is_timeout[0] else ""


def gen_cnf_numeric_clauses() -> List[List[int]]:
    """Generate CNF formula as numeric clauses. Note: fuzzsat adds 0s that pysat doesn't like"""
    cmd = ['python3', CNF_GENERATOR,
           '-i', str(random.randint(1, 10)), '-I', str(random.randint(11, 50)),
           '-p', str(random.randint(2, 10)), '-P', str(random.randint(11, 30)),
           '-l', str(random.randint(2, 10)), '-L', str(random.randint(11, 30))]

    logging.debug(f"Generating CNF with: {cmd}")
    out = run_subprocess(cmd)
    if not out:
        return []

    result = []
    try:
        for line in out.split("\n"):
            data = line.split(" ")
            if data[0] == '' and len(data) > 1:
                result.append([int(x) for x in data[1:-1]])
        return result
    except Exception as ex:
        print(ex)
        return []


def gene_smt2string(logic="QF_BV", incremental=False) -> str:
    """Generate SMT-LIB2 string"""
    cnfratio = random.randint(2, 10)
    cntsize = random.randint(5, 20)
    strategy = random.choice(['cnf', 'ncnf', 'bool']) if incremental else 'noinc'

    cmd = ['python3', SMT_GENERATOR,
           '--strategy', strategy,
           '--cnfratio', str(cnfratio),
           '--cntsize', str(cntsize),
           '--logic', logic]

    out = run_subprocess(cmd, timeout=6)
    return out if out else ""


def generate_from_grammar_as_str(logic="QF_BV", incremental=False):
    """Generate SMT formula from grammar as string"""
    cnfratio = random.randint(2, 10)
    cntsize = random.randint(5, 20)
    strategy = random.choice(['CNFexp', 'cnf', 'ncnf', 'bool']) if incremental else 'noinc'

    cmd = ['python3', SMT_GENERATOR,
           '--strategy', strategy,
           '--cnfratio', str(cnfratio),
           '--cntsize', str(cntsize),
           '--logic', logic]

    out = run_subprocess(cmd)
    return out if out else ""


if __name__ == '__main__':
    print(gene_smt2string())
