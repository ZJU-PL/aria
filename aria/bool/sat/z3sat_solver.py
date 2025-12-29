# coding: utf-8
"""
Useful functions for exploring Z3's powerful SAT engine.
  - Z3SATSolver
  - Z3MaxSATSolver

Currently, we hope to use this as the Boolean solver of the parallel CDCL(T) engine.
"""
from typing import List

import z3

from aria.utils.types import SolverResult


class Z3SATSolver:
    """Z3 SAT solver wrapper."""
    def __init__(self, logic="QF_FD"):
        self.int2z3var = {}  # for initializing from int clauses
        self.solver = z3.SolverFor(logic)
        # self.solver = z3.SimpleSolver()

    def from_smt2file(self, fname: str) -> None:
        """Load formula from SMT2 file."""
        self.solver.add(z3.And(z3.parse_smt2_file(fname)))

    def from_smt2string(self, smtstring: str) -> None:
        """Load formula from SMT2 string."""
        self.solver.add(z3.And(z3.parse_smt2_string(smtstring)))

    def from_int_clauses(self, clauses: List[List[int]]) -> None:
        """
        Initialize self.solver with a list of clauses
        """
        # z3_clauses = []
        for clause in clauses:
            conds = []
            for t in clause:
                if t == 0:
                    break
                a = abs(t)
                if a in self.int2z3var:
                    b = self.int2z3var[a]
                else:
                    b = z3.Bool(f"k!{a}")
                    # b = z3.BitVec(f"k!{a}", 1)
                    self.int2z3var[a] = b
                b = z3.Not(b) if t < 0 else b
                conds.append(b)
            self.solver.add(z3.Or(*conds))
            # z3_clauses.append(z3.Or(*conds))

    def get_z3var(self, intname: int) -> z3.BoolRef:
        """
        Given an integer (labeling a Boolean var.), return its corresponding Z3 Boolean var

        NOTE: this function is only meaningful when the solver is initialized by
          from_int_clauses, from_dimacsfile, or from_dimacsstring
        """
        if intname in self.int2z3var:
            return self.int2z3var[intname]
        raise ValueError(f"{intname} not in the var list!")

    def get_consequences(
        self, prelist: List[z3.BoolRef], postlist: List[z3.BoolRef]
    ) -> List[z3.BoolRef]:
        """Get consequences using Z3's extension."""
        res, factslist = self.solver.consequences([prelist], [postlist])
        if res == z3.sat:
            return factslist
        return []

    def get_unsat_core(self, assumptions: List[z3.BoolRef]) -> List[z3.BoolRef]:
        """Get unsat core."""
        res = self.solver.check(assumptions)
        if res == z3.unsat:
            return self.solver.unsat_core()
        return []

    def check_sat_assuming(self, assumptions: List[z3.BoolRef]) -> SolverResult:
        """Check satisfiability with assumptions."""
        res = self.solver.check(assumptions)
        if res == z3.sat:
            return SolverResult.SAT
        if res == z3.unsat:
            return SolverResult.UNSAT
        return SolverResult.UNKNOWN


class Z3MaxSATSolver:
    """
    MaxSAT
    """

    def __init__(self):
        # self.fml = None
        self.int2z3var = {}  # for initializing from int clauses
        self.solver = None
        self.hard = []
        self.soft = []
        self.weight = []

    def from_wcnf_file(self, fname: str) -> None:
        self.solver = z3.Optimize()
        self.solver.from_file(fname)

    def from_int_clauses(self, hard: List[List[int]], soft: List[List[int]], weight: List[int]):
        """
        TODO: handle two different cases (each clause ends with 0 or not)
        """
        self.solver = z3.Optimize()
        # self.solver.set('maxsat_engine', 'wmax')
        self.solver.set('maxsat_engine', 'maxres')

        for clause in hard:
            conds = []
            for t in clause:
                if t == 0:
                    break
                a = abs(t)
                if a in self.int2z3var:
                    b = self.int2z3var[a]
                else:
                    b = z3.Bool(f"k!{a}")
                    # b = z3.BitVec(f"k!{a}", 1)
                    self.int2z3var[a] = b
                b = z3.Not(b) if t < 0 else b
                conds.append(b)

            cls = z3.Or(*conds)
            self.solver.add(cls)
            self.hard.append(cls)

        for i, soft_clause in enumerate(soft):
            conds = []
            for t in soft_clause:
                if t == 0:  # TODO: need this?
                    break
                a = abs(t)
                if a in self.int2z3var:
                    b = self.int2z3var[a]
                else:
                    b = z3.Bool(f"k!{a}")
                    # b = z3.BitVec(f"k!{a}", 1)
                    self.int2z3var[a] = b
                b = z3.Not(b) if t < 0 else b
                conds.append(b)

            cls = z3.Or(*conds)
            self.solver.add_soft(cls, weight=weight[i])
            self.soft.append(cls)
            self.weight.append(weight[i])

    def check(self):
        """
        Check MaxSAT and return cost.
        TODO: get rid of self.hard, self.soft, and self.cost
          use the API of Optimize() to obtain the cost...
          cost: sum of weight of falsified soft clauses
        """
        cost = 0
        # print(len(self.solver.objectives()))
        if self.solver.check() == z3.sat:
            model = self.solver.model()
            for i, soft_clause in enumerate(self.soft):
                if z3.is_false(model.eval(soft_clause)):
                    cost += self.weight[i]
            print("finish z3 MaxSAT")
            # TODO: query the weight...
        return cost
