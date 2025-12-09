import z3
from pysat.formula import CNF

from aria.prob import wmc_count, WMCBackend, WMCOptions


def main():
    # (x or y) and (~x or z)
    cnf = CNF(from_clauses=[[1, 2], [-1, 3]])
    # literal weights; if only one polarity is provided for a var, the other is 1-w
    weights = {1: 0.6, 2: 0.7, 3: 0.5}

    res_dnnf = wmc_count(cnf, weights, WMCOptions(backend=WMCBackend.DNNF))
    res_enum = wmc_count(cnf, weights, WMCOptions(backend=WMCBackend.ENUMERATION))

    print("WMC (DNNF):", res_dnnf)
    print("WMC (ENUM):", res_enum)


if __name__ == "__main__":
    main()
