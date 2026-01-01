#!/usr/bin/env python3

import sys
import os
import subprocess as sub
import time
import shutil
import re
import tempfile

try:
    file, solver, options = sys.argv[1].split(",")
except:
    print("Usage: run_solver.py FILE,SOLVER,OPTIONS", file=sys.stderr)
    sys.exit(2)


with tempfile.NamedTemporaryFile() as t:
    limit_time = os.environ["LIMIT_TIME"]
    limit_mem = os.environ["LIMIT_MEM"]
    solver_command = f"runlim -s {limit_mem} -r {limit_time} -o {t.name} {solver} {options} benchmarks/benchmarks/{file}"
    start = time.time()
    o = sub.run(solver_command, stdout=sub.PIPE, stderr=sub.PIPE, shell=True)
    log = open(t.name).read()
    status = re.search("status:\\s*(\S.*)", log)[1]
    terminationreason = {
        "ok": "none",
        "out of memory": "memory",
        "out of time": "cputime",
    }[status]
    end = time.time()
    exitcode = int(re.search("result:\\s*(\S.*)", log)[1])
    exitsignal = 0 if exitcode >= 0 else -exitcode
    walltime = float(re.search("real:\\s*(\S*)", log)[1])
    cputime = float(re.search("time:\\s*(\S*)", log)[1])
    memory = 2**10 * float(re.search("space:\\s*(\S*)", log)[1])

# print(f"file,exitcode,exitsignal,walltime,cputime,memory,terminationreason,options,solver")
print(
    f"{file},{exitcode},{exitsignal},{walltime},{cputime},{memory},{terminationreason},{options},{solver}"
)
