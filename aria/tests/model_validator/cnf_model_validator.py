"""
Validate a CNF model
"""
import os
import sys
import linecache


def parse_model(file_name):
    """Parse a model file and return the status and the model."""
    status_line = linecache.getline(file_name, 1)
    parsed_status = status_line[:-1]
    model_line = linecache.getline(file_name, 2)
    parsed_model = list(map(int, model_line.split()[:-1]))
    return parsed_status, parsed_model


def parse_problem(prob_file, model):
    """Parse a problem file and return the status of the model."""
    with open(prob_file, "r", encoding='utf-8') as file:
        for line in file:
            line = line.rstrip()
            if len(line) == 0 or line[0] == 'p' or line[0] == 'c':
                continue
            values = list(map(int, line.split()))

            values = values[:-1]

            satisfied = False
            for lit in values:
                var = abs(lit) - 1
                if lit > 0:
                    if model[var] > 0:
                        satisfied = True
                        break
                else:
                    if model[var] < 0:
                        satisfied = True
                        break
            if not satisfied:
                return "UNSAT"

    return "SAT"


if __name__ == "__main__":
    problem_file = sys.argv[1]
    model_file = sys.argv[2]
    status, model = parse_model(model_file)
    calculated_status = parse_problem(problem_file, model)
    problem_name = os.path.basename(problem_file)
    if status != calculated_status:
        print(problem_name, status, calculated_status, status == calculated_status)
