"""
Propositional Interpolant

Perhaps integrating the following implementations
 -  https://github.com/fslivovsky/interpolatingsolver/tree/9050db1d39213e94f9cadd036754aed69a1faa5f
    (it uses C++ and some third-party libraries)
"""

from typing import List

import z3
from pysat.formula import CNF
from pysat.solvers import Solver


class BooleanInterpolant:
    """Class for computing Boolean interpolants using Z3 solvers."""

    @staticmethod
    def mk_lit(model: z3.ModelRef, expr: z3.ExprRef) -> z3.ExprRef:
        """
        Create a literal based on the model evaluation.

        :param model: The Z3 model to evaluate against
        :param expr: The expression to evaluate
        :return: The expression if true in model, otherwise its negation
        """
        if z3.is_true(model.eval(expr)):
            return expr
        return z3.Not(expr)

    @staticmethod
    def pogo(
        solver_a: z3.Solver, solver_b: z3.Solver, interpolation_vars: List[z3.ExprRef]
    ):
        """
        The pogo function takes two solvers, A and B.
        It then checks if the formula in A is satisfiable.
        If it is, it tries to prove a contradiction from the formulas in B.
        The function yields each interpolant as it goes along.

        :param solver_a: Keep track of the current state of the interpolation problem
        :param solver_b: Check the interpolant
        :param interpolation_vars: Pass the interpolation literals to the pogo function
        :return: A generator of interpolants
        """
        while z3.sat == solver_a.check():
            model = solver_a.model()
            literals = [BooleanInterpolant.mk_lit(model, x) for x in interpolation_vars]
            if z3.unsat == solver_b.check(literals):
                negated_core = z3.Not(z3.And(solver_b.unsat_core()))
                yield negated_core
                solver_a.add(negated_core)
            else:
                print("expecting unsat")
                break

    @staticmethod
    def compute_itp(
        fml_a: z3.ExprRef, fml_b: z3.ExprRef, var_list: List[z3.ExprRef]
    ) -> List[z3.ExprRef]:
        """
        Compute interpolants between two formulas.

        :param fml_a: First formula
        :param fml_b: Second formula
        :param var_list: List of variables for interpolation
        :return: List of interpolants
        """
        solver_a = z3.SolverFor("QF_FD")
        solver_a.add(fml_a)
        solver_b = z3.SolverFor("QF_FD")
        solver_b.add(fml_b)
        return list(BooleanInterpolant.pogo(solver_a, solver_b, var_list))


class PySATInterpolant:
    """Class for computing interpolants using PySAT solvers."""

    @staticmethod
    def compute_itp(fml_a: CNF, fml_b: CNF, interpolation_vars: List[int]):
        """
        Compute interpolants using PySAT.

        :param fml_a: First CNF formula
        :param fml_b: Second CNF formula
        :param interpolation_vars: List of variable indices for interpolation
        :return: Generator of interpolants
        """
        solver_a = Solver(bootstrap_with=fml_a)
        solver_b = Solver(bootstrap_with=fml_b)
        while solver_a.solve():
            # TODO: check the value of a var in the model, and build assumption
            # using interpolation_vars
            _ = interpolation_vars  # Unused until TODO is implemented
            cube = []
            if solver_b.solve(assumptions=cube):
                core = solver_b.get_core()
                not_core = [-v for v in core]
                yield not_core
                # Add the negated core as a clause to solver_a
                solver_a.add_clause(not_core)
            else:
                print("expecting unsat")
                break


def demo_itp():
    x = z3.Bool("x")
    y = z3.Bool("y")
    # A: x and not y
    fml_a = z3.And(x, z3.Not(y))
    # B: not x and y
    fml_b = z3.And(z3.Not(x), y)
    itp = BooleanInterpolant.compute_itp(fml_a, fml_b, [x, y])
    print(list(itp))


if __name__ == "__main__":
    demo_itp()
