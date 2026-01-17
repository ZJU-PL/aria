from aria.smt.pcdclt import solve as pcdclt_solve
from aria.smt.pcdclt import config as pcdclt_config
import sys

if __name__ == "__main__":
    input_file = sys.argv[1]

    with open(input_file, 'r', encoding='utf-8') as f:
        smt2_string = f.read()

    logic = 'ALL'
    for line in smt2_string.split('\n'):
        if line.strip().startswith('(set-logic'):
            logic = line.split()[1].rstrip(')')
            break

    result = pcdclt_solve(smt2_string, logic=logic)
    print(result.name.lower())
