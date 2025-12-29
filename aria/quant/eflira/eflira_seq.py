"""Sequential solver for exists-forall linear integer real arithmetic (EFLIRA)."""
import logging
from typing import List
import time

import z3

from aria.quant.efsmt_parser import EFSMTZ3Parser

logger = logging.getLogger(__name__)


def solve_with_simple_cegar(x: List[z3.ExprRef], y: List[z3.ExprRef], phi: z3.ExprRef,
                            maxloops=None) -> z3.CheckSatResult:
    """
    Solve exists-forall bit-vectors
     (The name of the engine is EFBVTactic.SIMPLE_CEGAR)
    """
    # set_param("verbose", 15)
    qf_logic = "QF_LIA"
    esolver = z3.SolverFor(qf_logic)
    fsolver = z3.SolverFor(qf_logic)
    esolver.add(z3.BoolVal(True))
    loops = 0
    while maxloops is None or loops <= maxloops:
        logger.debug("  Round: %s", loops)
        loops += 1
        eres = esolver.check()
        if eres == z3.unsat:
            return z3.unsat
        emodel = esolver.model()
        # the following lines should be done by the forall solver?
        mappings = [(var, emodel.eval(var, model_completion=True)) for var in x]
        sub_phi = z3.simplify(z3.substitute(phi, mappings))

        fsolver.push()
        fsolver.add(z3.Not(sub_phi))
        if fsolver.check() == z3.sat:
            fmodel = fsolver.model()
            # the following operations should be sequential?
            # the following line should not be dependent on z3?
            y_mappings = [(var, fmodel.eval(var, model_completion=True)) for var in y]
            sub_phi = z3.simplify(z3.substitute(phi, y_mappings))
            esolver.add(sub_phi)
            fsolver.pop()
        else:
            return z3.sat
    return z3.unknown


def solve_with_z3(y: List[z3.ExprRef], phi: z3.ExprRef) -> z3.CheckSatResult:
    """Solve exists-forall problem using Z3's built-in quantifier solver."""
    s = z3.Solver()
    s.add(z3.ForAll(y, phi))
    return s.check()


def test2():
    """Test function for solving EFLIRA problems."""
    file = "efbv.smt2"
    # ss = EFSMTParser()
    ss = EFSMTZ3Parser()
    exists_vars, forall_vars, qf_fml = ss.parse_smt2_file(file)
    print("Start solving!")
    start = time.time()
    solve_with_z3(forall_vars, qf_fml)
    print("z3 time: ", time.time() - start)
    start = time.time()
    solve_with_simple_cegar(exists_vars, forall_vars, qf_fml, maxloops=50)
    print("efsmt time: ", time.time() - start)


if __name__ == '__main__':
    test2()
