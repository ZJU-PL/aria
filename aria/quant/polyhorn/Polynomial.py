from typing import List

import numpy as np

from .Coefficient import Coefficient
from .UnknownVariable import UnknownVariable


class Monomial:
    """ A class that represent the Monomial\n
        It consists a set of variables with corresponding degree in an array and a Coefficient.
        For example (2*a*b + 1/2*c*c*c + 3.5)*x^2*y^3 is a Monomial.

        Attributes:
            variables ([UnknownVariable]): The sorted list of variables of the Monomial.
            degrees ([int]): Degree of each variable.
            coefficient (Coefficient): the Coefficient of the Monomial

    """

    def __init__(self, variables: List[UnknownVariable], degrees: List[int], coefficient: Coefficient):
        variables.sort()
        self.variables = variables
        self.degrees = degrees
        self.coefficient = coefficient

    def __str__(self):
        """ convert Monomial to string.

            :return: string format of the class.
        """
        return '*'.join(['(' + str(self.coefficient) + ')'] +
                        [str(self.variables[i]) + '^' + str(self.degrees[i]) for i in range(len(self.degrees))])

    def __mul__(self, other):
        """ multiply two Monomial\n the result is a Monomial which the degree of each variable is sum of degrees in
        multiplicands and the coefficient is the multiply of the two coefficient in multiplicands.

                :param other (Monomial): the other Monomial that should be multiplied
                :return: new Monomial that is the result of multiplication of two Coefficient
        """
        return Monomial(self.variables, np.array(self.degrees) + np.array(other.degrees),
                        self.coefficient * other.coefficient)

    def __eq__(self, other) -> bool:
        """ compare two Monomials\n the comparison is based on their length of set of variable and if that is equal
        and all variables and degree are equal they are equal.

             :param other(Monomial): the other Monomial that should be compared to.
             :return: boolean that determine are they equal or not.
         """
        if len(self.variables) != len(other.variables):
            return False
        for i in range(len(self.variables)):
            if self.variables[i] != other.variables[i] or self.degrees[i] != other.degrees[i]:
                return False
        return True

    def __lt__(self, other):
        """ compare two Monomials\n the comparison is based on their length of set of variable and if that is equal
        based on degree in lexicographical order.

             :param other(Monomial): the other Monomial that should be compared to.
             :return: boolean that determine which one is less than the other.
         """
        if len(self.variables) != len(other.variables):
            return len(self.variables) < len(other.variables)
        for i in range(len(self.variables)):
            if self.degrees[i] != other.degrees[i]:
                return self.degrees[i] < other.degrees[i]
        return True

    def __neg__(self):
        """ negate a Monomial\n
                For negating a Monomial it is sufficient to just negate the Coefficient.

                :return: a Monomial that is the negated form of the Monomial.
                """
        return Monomial(self.variables, self.degrees, -self.coefficient)

    def is_mono(self):
        number_of_nonzero = 0
        for deg in self.degrees:
            if deg != 0 and deg != 1:
                return False
            number_of_nonzero += deg
        return (number_of_nonzero <= 1)

    def convert_to_preorder(self) -> str:
        """ convert Monomial to preorder format.

        :return: string in preorder format of the class.
        """
        preorder = '( * 1 '
        preorder += self.coefficient.convert_to_preorder()

        for i, var in enumerate(self.variables):
            for _ in range(self.degrees[i]):
                preorder += str(var)
        preorder += ' )'
        return preorder


class Polynomial:
    """ A class that represent the Polynomial\n
        It consists a set of Monomials which should be added together to form the Polynomial.
        For example (2*a*b + 1/2*c*c*c + 3.5)*x^2*y^3 + (3 + 2*a*b)*x^0*y^1 is a Polynomial.

        Attributes:
            variables ([UnknownVariable]): The sorted list of variables of the Polynomial
            monomials ([Monomial]): The sorted list of Monomials that should be added together
            dict_from_degrees_to_monomials (dictionary): a dictionary that maps each degree to corresponding Monomial

    """

    def __init__(self, variables: List[UnknownVariable], monomials: List[Monomial]):
        variables.sort()
        monomials.sort()
        self.variables = variables
        self.monomials = monomials
        self.dict_from_degrees_to_monomials = {}
        for monomial in self.monomials:
            self.dict_from_degrees_to_monomials[tuple(
                monomial.degrees)] = monomial

    def get_monomial_by_degree(self, degree: List[int]) -> Monomial:
        if degree in self.dict_from_degrees_to_monomials.keys():
            return self.dict_from_degrees_to_monomials[degree]
        return Monomial(self.variables, [0] * len(self.variables), Coefficient([]))

    def __str__(self) -> str:
        """ convert Polynomial to string.

            :return: string format of the class.
        """
        if len(self.monomials) == 0:
            return '0'
        return '+'.join([str(monomial) for monomial in self.monomials])

    def __add__(self, other):
        """ sum of two Polynomial\n
            Sum of two Polynomial is a union of their Monomials.

            :param other: the other Polynomial that should be added.
            :return: new Polynomial that is sum of two Coefficient
        """
        return Polynomial(self.variables, self.monomials + other.monomials).revise()

    def __neg__(self):
        """ negate a Polynomial\n
            For negating a Polynomial it is sufficient to negate all its Monomial.

            :return: a Polynomial that is the negated form of the Polynomial.
        """
        monomials = [-monomial for monomial in self.monomials]
        return Polynomial(self.variables, monomials)

    def __sub__(self, other):
        """ subtract two Polynomial\n
            subtract of two Polynomial is adding one with the negated of the other.

            :param other (Polynomial): the other Polynomial that should be subtracted.
            :return: new Polynomial that is subtracted of two Polynomial
        """
        return self + (-other)

    def __mul__(self, other):
        """ multiply two Polynomial\n the result is a Polynomial which consist Monomials that are equal to the
            multiply of two Monomial from each Polynomial. numpy is used for that.

            :param other: the other Polynomial that should be multiplied
            :return: new Polynomial that is multiplied of two Polynomial
        """
        return Polynomial(self.variables,
                          np.array(
                              np.matmul(
                                  np.array(self.monomials).reshape(1, -1).T,
                                  np.array(other.monomials).reshape(1, -1)
                              )
                          ).reshape(1, -1)[0]
                          ).revise()

    def revise(self):
        """ revise the Polynomial\n
        It means that the Monomials that are the same added together.\n
        For example (3*a)*x^1 + (2*a*b + 1*a)*x^1  is a Polynomial and after revise it returns (3*a + 2*a*b + 1*a)x^1.

        :return: the revised format of the Polynomial.
        """
        self.monomials.sort()
        new_list = []
        i = 0
        while i < len(self.monomials):
            coefficient = Coefficient([])
            j = i
            while j < len(self.monomials):
                if self.monomials[i] == self.monomials[j]:
                    coefficient += self.monomials[j].coefficient
                else:
                    break
                j += 1
            new_list.append(
                Monomial(self.monomials[i].variables, self.monomials[i].degrees, coefficient))
            i = j

        return Polynomial(self.variables, new_list)

    def add_variables(self, new_variables: List[UnknownVariable]):
        """ This function add a set of new variable to each Monomial of a Polynomial.

        :param new_variables :  the list of the new variables that should add to the Polynomial.
        :return: new Polynomial with the new list of the variables.
        """
        monomials = []
        for monomial in self.monomials:
            monomials.append(Monomial(monomial.variables + new_variables,
                                      monomial.degrees +
                                      [0] * len(new_variables),
                                      monomial.coefficient)
                             )
        return Polynomial(self.variables + new_variables, monomials)

    def is_linear(self):
        for mono in self.monomials:
            if not mono.is_mono():
                return False
        return True

    def convert_to_preorder(self) -> str:
        """ convert Polynomial to preorder format.

        :return: string in preorder format of the class.
        """
        if len(self.monomials) == 0:
            return '0'
        if len(self.monomials) == 1:
            return self.monomials[0].convert_to_preorder()
        preorder = '( + 0 '
        for i in range(0, len(self.monomials)):
            preorder += self.monomials[i].convert_to_preorder() + ' '
        preorder += ' )'
        return preorder
