"""Wrapper for MiniSat22 solver integration with SymPy."""
try:
    from sympy.assumptions.cnf import EncodedCNF
except ImportError:
    EncodedCNF = None

try:
    from pysat.solvers import Minisat22
except ImportError:
    Minisat22 = None


def minisat22_satisfiable(expr, all_models=False, minimal=False):
    """Check satisfiability using MiniSat22 solver."""

    if EncodedCNF is None:
        raise ImportError("sympy.assumptions.cnf.EncodedCNF is not available")

    if Minisat22 is None:
        raise ImportError("pysat.solvers.Minisat22 is not available")

    if not isinstance(expr, EncodedCNF):
        exprs = EncodedCNF()
        exprs.add_prop(expr)
        expr = exprs

    # Return UNSAT when False (encoded as 0) is present in the CNF
    if {0} in expr.data:
        if all_models:
            return (f for f in [False])
        return False

    r = Minisat22(expr.data)

    if minimal:
        r.set_phases([-(i+1) for i in range(r.nof_vars())])

    if not r.solve():
        return False

    if not all_models:
        return {expr.symbols[abs(lit) - 1]: lit > 0 for lit in r.get_model()}

    # Make solutions SymPy compatible by creating a generator
    def _gen(results):
        satisfiable = False
        while results.solve():
            sol = results.get_model()
            yield {expr.symbols[abs(lit) - 1]: lit > 0 for lit in sol}
            if minimal:
                results.add_clause([-i for i in sol if i > 0])
            else:
                results.add_clause([-i for i in sol])
            satisfiable = True
        if not satisfiable:
            yield False

    return _gen(r)
