import os
import subprocess
import time

#
# try :
#     p = subprocess.run(["python3" , "smt_main.py" ,"benchmarks/Example_1.txt.smt2"] ,
#                                    timeout=3 ,
#                                    stdout=subprocess.PIPE,
#                                    text = True)
#     print(p.stdout)
#     print(p.returncode)
# except subprocess.TimeoutExpired:
#     print("out of time")

dataset_to_build = [
    "bmc-bv",
    "20170501-Heizmann-UltimateAutomizer",
    "bmc-bv-svcomp14",
    "pipe",
    "20170531-Hansen-Check",
    "brummayerbiere",
    "pspace",
    "2017-BuchwaldFried",
    "brummayerbiere2",
    "brummayerbiere3",
    "2018-Goel-hwbench",
    "rubik",
    "2018-Mann",
    "brummayerbiere4",
    "RWS",
    "20190311-bv-term-small-rw-Noetzli",
    "bruttomesso",
]
if __name__ == "__main__":
    from pathlib import Path
    import sys

    project_root = str(Path(__file__).parent)
    sys.path.append(project_root)
    Timeout_cnt = 0  # 记录超时smt
    Solved_Sat = 0  # 记录求解结果为sat的Smt Query
    Solved_UnSat = 0  # 记录求解结果为Unsat的Smt Query
    Solving_Time = 0  # 记录Solving time，包含超时任务
    log_file = open(project_root + "/log_aria_1200.txt", "a")
    for dir in dataset_to_build:
        dir_path = os.path.join(project_root, "benchmarks_mine", dir)
        files = os.listdir(dir_path)
        for file in files:
            file_path = os.path.join(dir_path, file)
            before = time.time()
            try:
                p = subprocess.run(
                    ["python3", "smt_main.py", file_path],
                    timeout=1200,
                    stdout=subprocess.PIPE,
                    text=True,
                )
                if p.stdout == "SolverResult.UNSAT\n":
                    Solved_UnSat += 1
                else:
                    Solved_Sat += 1
                log_file.write(f"{dir + '/' + file} {p.stdout}")
                log_file.flush()
            except subprocess.TimeoutExpired:
                log_file.write(f"{dir + '/' + file} OUT OF TIME\n")
                Timeout_cnt += 1
                log_file.flush()
            after = time.time()
            Solving_Time += after - before
    print("Time spent on solving is ", Solving_Time)
    print("Count(Timeout smt queries) =", Timeout_cnt)
    print("Count(Solved_Sat) =", Solved_Sat)
    print("Count(Solved_UnSat) =", Solved_UnSat)
