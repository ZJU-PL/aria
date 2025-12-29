"""
A hack for constructing partial weighted max sat instances from wcnf files
for z3.

Authors: Anthony Lin, Matt Hague

Modified from https://github.com/matthewhague/sat-css-tool/blob/master/wcnf2z3.py
"""
import sys
from timeit import default_timer

import z3


def construct_z3_optimizer(fin):
    """
    Construct a z3 optimizer from wcnf file referenced by fin
    """
    # Ignoring comments
    header_line = None
    for line in fin:
        linestrip = line.strip()
        if len(linestrip) > 0 and linestrip[0] == 'c':
            continue
        if len(linestrip) > 0 and linestrip[0] == 'p':
            header_line = linestrip
            break
        assert False, 'Parse error: p expected after comments'

    # Extracting p-info
    if header_line is None:
        assert False, 'Empty file'  # the only possibility of error
    line = header_line
    line_array = line.split()
    assert len(line_array) == 5, 'Unexpected number of words in p line'
    assert line_array[0] == 'p'
    assert line_array[1] == 'wcnf'
    try:
        nbvar = int(line_array[2])
        assert nbvar >= 1, 'Non-positive number of variables'
        nbclauses = int(line_array[3])
        assert nbclauses >= 1, 'Non-positive number of clauses'
        top = int(line_array[4])
        assert top >= 1, 'Non-positive weight for hard constraints'
    except ValueError:
        assert False, 'Unexpected input'

    opt = z3.Optimize()
    h = None
    # Parsing clauses
    for line in fin:
        line_array = line.split()
        try:
            wt = int(line_array[0])
            clause = []
            for lit in line_array[1:-1]:  # Note last number is 0
                var = abs(int(lit))
                z3var = z3.Bool(str(var))
                if int(lit) > 0:
                    clause.append(z3var)
                elif int(lit) < 0:
                    clause.append(z3.Not(z3var))
                else:
                    assert False, 'Zero value is seen prematurely'
        except ValueError:
            assert False, 'Unexpected input'
        if wt == top:
            opt.add(z3.Or(clause))
        elif wt < top:
            h = opt.add_soft(z3.Or(clause), wt)
        else:
            assert False, 'Weight bigger than max weight declared'

    return opt, h


def main():
    """Main function for command-line execution."""
    filename = 'file.wcnf'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        print(sys.argv[1])

    with open(filename, 'r', encoding='utf-8') as fin:
        (opt, h) = construct_z3_optimizer(fin)

    opt.set('maxsat_engine', 'wmax')

    print("Checking...")
    start_t = default_timer()
    opt.check()
    end_t = default_timer()
    print("Done in", (end_t - start_t), "s")

    print(opt.model())
    if h is not None:
        print(h.value())


if __name__ == '__main__':
    main()
