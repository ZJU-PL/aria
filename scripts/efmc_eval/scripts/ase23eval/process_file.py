import os
import json
import signal
import subprocess
from pathlib import Path

LIA_DIR = "./benchmarks/sygus-inv/LIA/2017.ASE_FiB"
RESULT_DIR = "./result/"
SCRIPT_PATH = "../syguslia2sygusbv_signed.py"
CVC5_BINARY = "./bin_solvers/bin/cvc5-Linux"
TIMEOUT = 5

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


def process_file(input_file, output_file):
    with open(output_file, "w") as output:
        subprocess.run(["python3", SCRIPT_PATH, input_file], stdout=output)


def run_cvc5(input_file):
    try:
        result = subprocess.run(
            [CVC5_BINARY, "--lang=sygus2", input_file],
            capture_output=True,
            timeout=TIMEOUT,
        )
        return result.stdout.decode("utf-8")
    except subprocess.TimeoutExpired:
        print(f"Timeout occurred for {input_file}")
        return None


if __name__ == "__main__":
    result_dir = "./Result/Result"
    total_time = 0

    for subdir, dirs, files in os.walk(result_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(subdir, file)
                with open(file_path, "r") as json_file:
                    data = json.load(json_file)
                    if "time" in data:
                        total_time += data["time"]

    print(f"Total time: {total_time}")
