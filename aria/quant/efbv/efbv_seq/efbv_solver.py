# coding: utf-8
"""
The uniformed interface for solving Exists-ForAll problems
"""
import logging

# from enum import Enum
# from typing import List

import z3

from aria.quant.efbv.efbv_seq.efbv_bin_solvers import (
    solve_with_bin_smt,
    solve_with_bin_qbf,
)
from aria.quant.efbv.efbv_seq.efbv_cegis_solvers import simple_cegis_efsmt
from aria.quant.efbv.efbv_seq.efbv_to_bool import EFBVFormulaTranslator
from aria.quant.efbv.efbv_seq.efbv_sat_solver import solve_with_sat_solver

logger = logging.getLogger(__name__)


class EFBVSequentialSolver:
    """Solving exists forall problem"""

    def __init__(self, logic: str, **kwargs):
        self.phi = None
        self.exists_vars = None
        self.forall_vars = None

        self.logic = logic

        self.seed = kwargs.get("seed", 1)  # random seed
        self.solver = kwargs.get("solver", "z3")

        self.initialized = False
        self.pysmt_solver = kwargs.get("pysmt_solver", "z3")

    def set_tactic(self, name: str):
        """Set solver tactic (not implemented)."""
        raise NotImplementedError

    def init(self, exist_vars, forall_vars, phi: z3.ExprRef):
        """Initialize the solver with variables and formula."""
        self.exists_vars = exist_vars
        self.forall_vars = forall_vars
        self.phi = phi
        self.initialized = True

    def dump_ef_smt_file(self, smt2_file_name: str):
        """Dump the constraint from the ef engine."""
        fml_str = f"(set-logic {self.logic})\n"

        dump_strategy = 1
        if dump_strategy == 1:
            # there are duplicates in self.exists_vars???
            exits_vars_names = set()
            for v in self.exists_vars:
                name = str(v)
                if name not in exits_vars_names:
                    exits_vars_names.add(name)
                    fml_str += f"(declare-const {v.sexpr()} {v.sort().sexpr()})\n"

            quant_vars = "("
            for v in self.forall_vars:
                quant_vars += f"({v.sexpr()} {v.sort().sexpr()}) "
            quant_vars += ")\n"

            quant_fml_body = "(and \n"
            s = z3.Solver()
            s.add(self.phi)
            # self.phi is in the form of
            #  and (Init, Trans, Post)
            assert z3.is_app(self.phi)
            for fml in self.phi.children():
                quant_fml_body += f"  {fml.sexpr()}\n"
            quant_fml_body += ")"

            fml_body = f"(assert (forall {quant_vars} {quant_fml_body}))\n"
            fml_str += fml_body
            fml_str += "(check-sat)\n"
        else:
            # Another more direct strategy
            # But we cannot see the definition of the VC clearly
            sol = z3.Solver()
            sol.add(z3.ForAll(self.forall_vars, self.phi))
            fml_str += sol.to_smt2()

        with open(smt2_file_name, "w", encoding="utf-8") as tmp:
            tmp.write(fml_str)

    def dump_qbf_file(self, qdimacs_file_name: str):
        """Dump to QBF formula."""
        assert self.logic in ("BV", "UFBV")
        fml_manager = EFBVFormulaTranslator()
        qdimacs_str = fml_manager.to_qdimacs_str(
            self.phi, existential_vars=self.exists_vars, universal_vars=self.forall_vars
        )
        with open(qdimacs_file_name, "w", encoding="utf-8") as tmp:
            tmp.write(qdimacs_str)

    def dump_sat_file(self, dimacs_file_name: str):
        """Dump to SAT file (not implemented)."""
        raise NotImplementedError

    def solve(self):
        """
        Solve EFSMT(BV) formulas via different strategies.
        """
        assert self.initialized
        print(f"EFSMT solver: {self.solver}")
        # 1. Quantifier instantiation approach
        if self.solver == "z3":
            return solve_with_bin_smt(
                self.logic, self.exists_vars, self.forall_vars, self.phi, "z3"
            )
        if self.solver == "cvc5":
            return solve_with_bin_smt(
                self.logic, self.exists_vars, self.forall_vars, self.phi, "cvc5"
            )
        if self.solver == "btor":
            return solve_with_bin_smt(
                self.logic, self.exists_vars, self.forall_vars, self.phi, "boolector2"
            )
        if self.solver == "yices2":
            return solve_with_bin_smt(
                self.logic, self.exists_vars, self.forall_vars, self.phi, "yices2"
            )
        if self.solver == "mathsat":
            return solve_with_bin_smt(
                self.logic, self.exists_vars, self.forall_vars, self.phi, "mathsat"
            )
        if self.solver == "bitwuzla":
            return solve_with_bin_smt(
                self.logic, self.exists_vars, self.forall_vars, self.phi, "bitwuzla"
            )

        # 2. Bit-blasting approach
        if self.solver == "z3qbf":
            return self.solve_with_z3_qbf()
        if self.solver == "caqe":
            return self.solve_with_third_party_qbf("caqe")
        # TODO: q3b (BDD-based), z3-based QE+SAT
        if self.solver == "q3b":
            return solve_with_bin_smt(
                self.logic, self.exists_vars, self.forall_vars, self.phi, "q3b"
            )
        if self.solver == "z3sat":
            return self.solve_with_z3_sat()
        # third-party SAT solves (using pySAT)
        if self.solver in [
            "cd",
            "cd15",
            "gc3",
            "gc4",
            "g3",
            "g4",
            "lgl",
            "mcb",
            "mpl",
            "mg3",
            "mc",
            "m22",
            "mgh",
        ]:
            return self.solve_with_third_party_sat(solver_name=self.solver)

        # 3. Simple cegis-based approach
        if self.solver == "cegis":
            # TODO: other engines in pysmt
            print("solving via cegis_solver")
            return self.solve_with_simple_cegis()

        raise NotImplementedError

    def solve_with_simple_cegis(self) -> str:
        """Solve with a CEGIS-style algorithm.

        Consists of a "forall solver" and an "exists solver". This can be slow
        (perhaps not a good idea for NRA). Maybe good for LRA or BV?
        NOTE: Currently, we use pySMT for the implementation.
        """
        print("Simple, sequential, CEGIS-style EFSMT!")
        z3_res = simple_cegis_efsmt(
            self.logic,
            self.exists_vars,
            self.forall_vars,
            self.phi,
            pysmt_solver=self.pysmt_solver,
        )
        return z3_res

    def solve_with_z3_qbf(self) -> str:
        """Translate to QBF and solve."""
        assert self.logic in ("BV", "UFBV")
        fml_manager = EFBVFormulaTranslator()
        sol = z3.Solver()
        vc = fml_manager.to_z3_qbf(self.phi, self.exists_vars, self.forall_vars)
        sol.add(vc)
        res = sol.check()
        if res == z3.sat:
            return "sat"
        if res == z3.unsat:
            return "unsat"
        return "unknown"

    def solve_with_z3_sat(self) -> str:
        """Solve using quantifier elimination + SAT solving."""
        assert self.logic in ("BV", "UFBV")
        print("Quantifier elimination + SAT solving")
        fml_manager = EFBVFormulaTranslator()
        sol = z3.Solver()
        vc = fml_manager.to_z3_sat(self.phi, self.exists_vars, self.forall_vars)
        # print(vc)
        sol.add(vc)
        res = sol.check()
        if res == z3.sat:
            return "sat"
        if res == z3.unsat:
            return "unsat"
        return "unknown"

    def solve_with_third_party_qbf(self, solver_name: str) -> str:
        """Translate EFSMT(BV) to QBF and call a third-party QBF solver."""
        assert self.logic in ("BV", "UFBV")
        fml_manager = EFBVFormulaTranslator()
        qdimacs = fml_manager.to_qdimacs_str(
            self.phi, existential_vars=self.exists_vars, universal_vars=self.forall_vars
        )
        return solve_with_bin_qbf(qdimacs, solver_name)

    def solve_with_third_party_sat(self, solver_name: str) -> str:
        """
        Translate EFSMT(BV) to SAT, and call a third-party SAT solver

        cd(cadical103), cd15(cadical153),
        gc3(gluecard3), gc4(glucard4), g3(glucose3), g4(glucose4),
        lgl(lingeling), mcb(maplechrono), mcm(maplecm), mpl(maplesat)
        mg3(mergesat3), mc(minicard), m22(minisat22, mgh(minsatgh)
        sat_solvers_in_pysat = ['cd', 'cd15', 'gc3', 'gc4', 'g3',
                        'g4', 'lgl', 'mcb', 'mpl', 'mg3',
                        'mc', 'm22', 'mgh']
        """
        assert self.logic in ("BV", "UFBV")
        print("Quantifier elimination + SAT solving")
        fml_manager = EFBVFormulaTranslator()
        vc = fml_manager.to_dimacs_str(self.phi, self.exists_vars, self.forall_vars)
        res = solve_with_sat_solver(vc, solver_name=solver_name)
        return res


def demo_efsmt():
    """Demo function for EFSMT solving."""
    import time

    x, y, z = z3.BitVecs("x y z", 16)
    # x, y, z = z3.Reals("x y z")
    fmla = z3.Implies(z3.And(y > 0, y < 10), y - 2 * x < 7)

    start = time.time()
    solver = EFBVSequentialSolver(logic="BV", solver="cegis")
    solver.init(exist_vars=[x], forall_vars=[y], phi=fmla)
    # solver.dump_cegis_smt_files()
    # print(solver.solve_with_z3_sat())
    print(solver.solve_with_third_party_sat(solver_name="cd"))
    print(f"time: {time.time() - start}")


if __name__ == "__main__":
    demo_efsmt()
