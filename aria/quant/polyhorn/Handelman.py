from typing import List, Tuple

from aria.quant.polyhorn.Constraint import PolynomialConstraint
from aria.quant.polyhorn.Solver import Solver
from aria.quant.polyhorn.Polynomial import Polynomial
from aria.quant.polyhorn.Constraint import CoefficientConstraint


class Handelman:
    """ A class that performs handelman algorithm.

        Attributes:
                variables ([UnknownVariable]): The list of variables of the polynomials.
                LHS ([PolynomialConstraint]): The list of left hand side constraints in farkas algorithm
                RHS (PolynomialConstraint) : The right hand side constraint in algorithm
                max_d_for_sat (int) : maximum degree of monoids when finding sat constraints
                max_d_for_unsat (int) : maximum degree of monoids when finding unsat constraints
    """

    def __init__(self, variables, LHS: List[PolynomialConstraint], RHS: PolynomialConstraint,
                 max_d_for_sat: int = 0, max_d_for_unsat: int = 0):
        self.variables = variables
        self.RHS = RHS
        self.LHS = LHS
        self.max_d_for_sat = max_d_for_sat
        self.max_d_for_unsat = max_d_for_unsat

    def get_lists_with_fixed_len(self, lst, n, max_d):
        if len(lst) == n:
            return [lst]
        ans = []
        for i in range(max_d - sum(lst)+1):
            ans = ans + self.get_lists_with_fixed_len(lst + [i], n, max_d)
        return ans
    def get_monoids(self, max_d: int) -> Tuple[List[Polynomial], List[bool]]:
        """ this function creates monoids of given the degree of all left hand side polynomials

        :param max_d: maximum degree of each monoid
        :return: list of monoids and a list of boolean which the ith element is true if the ith monoid is strict.
        """
        monoids = []
        is_strict = []
        all_degrees = self.get_lists_with_fixed_len([], len(self.LHS), max_d)
        for degree_of_each_lhs in all_degrees:
            poly = Solver.get_constant_polynomial(self.variables, '1')
            is_the_new_monoid_strict = True

            for i in range(len(self.LHS)):
                if (not self.LHS[i].is_strict()) and (degree_of_each_lhs[i] > 0):
                    is_the_new_monoid_strict = False
                for d in range(degree_of_each_lhs[i]):
                    poly = poly * self.LHS[i].polynomial

            is_strict.append(is_the_new_monoid_strict)
            monoids.append(poly)

        return monoids, is_strict

    def get_poly_sum(self, max_d: int, need_strict: bool = False) -> Tuple[Polynomial, CoefficientConstraint]:
        """ This function returns a polynomial y_0 + y_1*f_1 + y_2*f_2 ...+ y_n*f_n where f_i are monoids of left hand side
        polynomials and y_i are newly created variable in handelman theorem

        :param max_d: maximum degree of each monoid
        :param need_strict: is it generated for the 0 > 0 case or not.
        :return: polynomial of sum of all monoids of left hand side with new variables and corresponding constraints.
        """
        polynomial_of_sum = Polynomial(self.variables, [])

        all_monoid, is_strict = self.get_monoids(max_d)
        constraints = []
        sum_of_strict = Solver.get_constant_polynomial(self.variables, '0')

        if self.RHS.is_strict():
            new_var = Solver.get_variable_polynomial(self.variables, 'y0', 'generated_for_handelman_in_strict_case')
            polynomial_of_sum = polynomial_of_sum + new_var
            constraints.append(CoefficientConstraint(new_var.monomials[0].coefficient, '>='))
            sum_of_strict = sum_of_strict + new_var

        for i, monoid in enumerate(all_monoid):
            new_var_poly = Solver.get_variable_polynomial(self.variables, f'y{i + 1}',
                                                          'generated_for_Handelman')
            polynomial_of_sum = polynomial_of_sum + (new_var_poly * monoid)
            constraints.append(CoefficientConstraint(new_var_poly.monomials[0].coefficient, '>='))

            if is_strict[i]:
                sum_of_strict = sum_of_strict + new_var_poly

        if need_strict or self.RHS.is_strict():
            constraints.append(CoefficientConstraint(sum_of_strict.monomials[0].coefficient, '>'))
        return polynomial_of_sum, constraints

    def get_SAT_constraint(self) -> List[CoefficientConstraint]:
        """ a function to find the constraints when the LHS => RHS is satisfiable

        :return: list of coefficient constraints when it is satisfiable
        """
        polynomial_of_sum, constraints = self.get_poly_sum(self.max_d_for_sat)
        return Solver.find_equality_constraint(polynomial_of_sum, self.RHS.polynomial) + constraints

    def get_UNSAT_constraint(self, need_strict: bool = False) -> List[CoefficientConstraint]:
        """ a function to find the constraints when it is not satisfiable.\n
        two set of constraint can be generated:\n
         1)LHS => -1 >= 0\n
         2)LHS => 0 > 0\n

        :param need_strict: determine which set of constraint to generate.
        :return: list of coefficient constraints when it is not satisfiable
        """
        if need_strict:
            polynomial_of_sum, constraints = self.get_poly_sum(self.max_d_for_unsat, need_strict)
            return Solver.find_equality_constraint(polynomial_of_sum,
                                                   Polynomial(self.variables, [])) + constraints

        polynomial_of_sum, constraints = self.get_poly_sum(self.max_d_for_unsat)
        return Solver.find_equality_constraint(polynomial_of_sum,
                                               Solver.get_constant_polynomial(self.variables, '-1')) + constraints
