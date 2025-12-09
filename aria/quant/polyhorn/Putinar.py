from typing import List, Tuple

import numpy as np

from .Constraint import CoefficientConstraint, PolynomialConstraint
from .Polynomial import Polynomial
from .Solver import Solver
from .UnknownVariable import UnknownVariable


class Putinar:
    """ A class that performs putinar algorithm.

        Attributes:
                variables ([UnknownVariable]): The list of variables of the polynomials.
                LHS ([PolynomialConstraint]): The list of left hand side constraints in farkas algorithm
                RHS (PolynomialConstraint) : The right hand side constraint in algorithm
                max_d_for_sat (int) : maximum degree of monoids when finding sat constraints
                max_d_for_unsat (int) : maximum degree of monoids when finding unsat constraints
                max_d_for_unsat_strict (int) : maximum degree of monoids when finding unsat constraints in strict case
                degree_for_new_var (int) : degree of new variable that is generated for strict case in right hand side of equality
    """

    def __init__(self, variables, LHS: List[PolynomialConstraint], RHS: PolynomialConstraint,
                 max_d_for_sat=0, max_d_for_unsat=0, max_d_for_unsat_strict=0, degree_for_new_var=0):
        self.variables = variables
        self.RHS = RHS
        self.LHS = LHS
        self.max_d_for_sat = max_d_for_sat
        self.max_d_for_unsat = max_d_for_unsat
        self.max_d_for_unsat_strict = max_d_for_unsat_strict
        self.degree_for_new_var = degree_for_new_var

    def get_lower_triangular_matrix(self, dim: int) -> Tuple[List[List[Polynomial]], List[CoefficientConstraint]]:
        """this function generates a lower triangular matrix L with positive elements in diagonal and returns L * L^T

        :param dim: The dimension of the generated matrix
        :return: a positive semideinite matrix with corresponding constraints.
        """
        matrix = np.array(
            [[Polynomial(self.variables, []) for __ in range(dim)]
             for _ in range(dim)]
        )
        constraints = []
        for i in range(dim):
            for j in range(i + 1):
                matrix[i][j] = Solver.get_variable_polynomial(self.variables, f'l_{i}{j}',
                                                              'generated_for_semidefinite_matrix')
            constraints.append(CoefficientConstraint(
                matrix[i][i].monomials[0].coefficient, '>='))

        return np.matmul(matrix, matrix.T), constraints

    @staticmethod
    def get_monoids(variables, max_d: int) -> List[Polynomial]:
        """ This function gives the monoid of a maximum degree on set of variables.

        :param variables: the set of variable
        :param max_d: maximum degree of generated monoids
        :return: list of monoids defined on the set of variable
        """
        all_monoid = []
        for mask in range((max_d + 1) ** len(variables)):
            mask_copy = mask
            degrees = []
            for i in range(len(variables)):
                degrees.append(mask_copy % (max_d + 1))
                mask_copy //= (max_d + 1)
            if sum(degrees) > max_d:
                continue
            all_monoid.append(Solver.get_degree_polynomial(variables, degrees))

        return all_monoid

    def get_sum_of_square(self, max_d: int) -> Tuple[Polynomial, List[CoefficientConstraint]]:
        """ the function generate a new sum-of-square template on set of class variable and given maximum degree

        :param max_d: maximum degree of the generated template
        :return: polynomial represents the sum-of-square template and corresponding constraints.
        """
        monoid = np.array(Putinar.get_monoids(
            self.variables, (max_d // 2))).reshape(1, -1)
        semi_definite_matrix, constraint = self.get_lower_triangular_matrix(
            monoid.shape[1])
        poly_with_more_degree = np.matmul(
            np.matmul(monoid, semi_definite_matrix), monoid.T)[0][0]
        monomials = []
        for i, monomial in enumerate(poly_with_more_degree.monomials):
            new_poly = Solver.get_variable_polynomial(self.variables, f't_{i}', 'eta_generated_for_sum_of_square') \
                * Solver.get_degree_polynomial(self.variables, monomial.degrees)
            constraint.append(
                CoefficientConstraint(
                    new_poly.monomials[0].coefficient - monomial.coefficient, '=')
            )
            monomials.append(new_poly.monomials[0])
        return Polynomial(poly_with_more_degree.variables, monomials), constraint

    def get_poly_sum(self, max_d: int) -> Tuple[Polynomial, List[CoefficientConstraint]]:
        """ This function returns a polynomial y_0(rhs is struct) + h_0 + h_1*g_1 ... + h_n*g_n where h_i are
        sum-of-square polynomial and g_i are left hand side polynomials

        :param max_d: maximum degree of each sum-of-square
        :return: polynomial of sum of all left hand side with new variables and corresponding constraints.
        """

        polynomial_of_sum, all_constraints = self.get_sum_of_square(max_d)
        strict_poly = Solver.get_constant_polynomial(self.variables, 0)
        if self.RHS.is_strict():
            new_var = Solver.get_variable_polynomial(
                self.variables, 'y0', 'generated_for_putinar_in_strict_case')
            strict_poly = strict_poly + new_var
            polynomial_of_sum = polynomial_of_sum + new_var
            all_constraints.append(CoefficientConstraint(
                new_var.monomials[0].coefficient, '>='))
        for i, left_constraint in enumerate(self.LHS):
            left_poly = left_constraint.polynomial
            new_sum_of_square, constraint = self.get_sum_of_square(max_d)
            if self.RHS.is_strict() and left_constraint.is_strict():
                new_var = Solver.get_variable_polynomial(
                    self.variables, 'y0', 'generated_for_putinar_in_strict_case')
                strict_poly = strict_poly + new_var
                new_sum_of_square = new_sum_of_square + new_var

                all_constraints.append(CoefficientConstraint(
                    new_var.monomials[0].coefficient, '>='))
            all_constraints = all_constraints + constraint
            polynomial_of_sum = polynomial_of_sum + \
                (new_sum_of_square * left_poly)
        if self.RHS.is_strict():
            all_constraints.append(CoefficientConstraint(
                strict_poly.monomials[0].coefficient, '>'))
        return polynomial_of_sum, all_constraints

    def get_SAT_constraint(self):
        """ a function to find the constraints when the LHS => RHS is satisfiable

        :return: list of coefficient constraints when it is satisfiable
        """
        polynomial_of_sum, constraints = self.get_poly_sum(self.max_d_for_sat)
        return constraints + Solver.find_equality_constraint(polynomial_of_sum, self.RHS.polynomial)

    def get_UNSAT_constraint(self, need_strict=False):
        """ a function to find the constraints when it is not satisfiable.\n
        two set of constraint can be generated:\n
         1) LHS => -1 >= 0\n
         2) w_j^2*d = h_1*(g_1 - w_1^2) + h_2*(g_2 - w_2^2) ... h_n*(g_n - w_n^2) for some j where g_j is strict  \n

        :param need_strict: determine which set of constraint to generate.
        :return: list of coefficient constraints when it is not satisfiable
        """
        if need_strict:
            return self.handel_strict(self.max_d_for_unsat_strict, self.degree_for_new_var)
        polynomial_of_sum, constraints = self.get_poly_sum(
            self.max_d_for_unsat)
        return constraints + Solver.find_equality_constraint(polynomial_of_sum,
                                                             Solver.get_constant_polynomial(
                                                                 self.RHS.polynomial.variables, '-1'))

    def handel_strict(self, max_d: int, power_of_new_var: int) -> List[List[CoefficientConstraint]]:
        """ it handles the second form of unsatisfiablity

        :param max_d: maximum degree for template polynomials
        :param power_of_new_var: power of newly generated variables
        :return: list of constraint for each equality
        """
        new_variables = [UnknownVariable(name=f'w_{i + 1}', type_of_var='generated_for_strict_unsat') for i in
                         range(len(self.LHS))]
        new_set_of_cons = []
        for left_constraint in self.LHS:
            new_set_of_cons.append(PolynomialConstraint(left_constraint.polynomial.add_variables(new_variables), left_constraint.sign
                                                        )
                                   )
        all_variable = self.variables + new_variables
        all_monoids = Putinar.get_monoids(all_variable, max_d)
        all_constraints = []
        for j, left_constraint in enumerate(new_set_of_cons):
            if left_constraint.is_strict():
                poly_sum = Polynomial(all_variable, [])
                for i, g in enumerate(new_set_of_cons):
                    degrees = [0] * (len(all_variable))
                    degrees[len(self.variables) + i] = 1
                    new_var_poly = Solver.get_degree_polynomial(
                        all_variable, degrees)
                    poly_sum = poly_sum + \
                        (Putinar.get_general_template(all_variable, all_monoids) *
                         (g.polynomial - (new_var_poly * new_var_poly)))

                degrees = [0] * (len(all_variable))
                degrees[len(self.variables) + j] = 1
                new_var_poly = Solver.get_degree_polynomial(
                    all_variable, degrees)
                poly_of_var_to_power = Solver.get_constant_polynomial(
                    all_variable, '1')
                for _ in range(2 * power_of_new_var):
                    poly_of_var_to_power = poly_of_var_to_power * new_var_poly
                all_constraints.append(Solver.find_equality_constraint(
                    poly_of_var_to_power, poly_sum))
        return all_constraints

    @staticmethod
    def get_general_template(all_variables: List[UnknownVariable], all_monoids: List[Polynomial]) -> List[Polynomial]:
        """ generate a new variable for every monoid and multiply it and return the sum of all.

        :param all_variables: variable of the polynomials
        :param all_monoids: monoids which should be multiply by new variable
        :return: A polynomial of the sum of all monoid multiplied by new variable
        """
        new_unknown_variables = [
            Solver.get_variable_polynomial(
                all_variables, name=f'eta{i + 1}', type_of_var='generated_for_template_poly_strict')
            for i in range(len(all_monoids))]
        return np.matmul(np.array(new_unknown_variables).reshape(1, -1), np.array(all_monoids).reshape(-1, 1))[0][0]
