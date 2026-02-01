"""Coefficient elements and linear algebra for polynomial coefficients."""

from fractions import Fraction
from typing import List

import numpy as np

from aria.quant.polyhorn.UnknownVariable import UnknownVariable


class Element:
    """A class that represent the elements\n
    It consists a constant and a set of variables which should be multiplied together to form the element.
    For example 2*a*b, 1/2*c*c*c are elements

    Attributes:
        constant (Fraction): It is the constant to multiply with other variables
        variables ([UnknownVariable]): The sorted list of variables that should be multiplied together

    """

    def __init__(self, constant, variables: List[UnknownVariable] = []):
        self.constant = Fraction(constant)
        variables.sort()
        self.variables = variables

    def __str__(self) -> str:
        """convert Element to string.

        :return: string format of the class.
        """
        return "*".join([str(self.constant)] + [str(var) for var in self.variables])

    def __mul__(self, other):
        """multiply two elements\n
        the constant of each element are multiplied for the new element. The set of variable for the new element is the uion of the two other elements.

        :param other: the other element that should be multiplied
        :return: new element that is multiplied of two element
        """
        return Element(
            (self.constant * other.constant), (self.variables + other.variables)
        )

    def __add__(self, other):
        """sum of two elements\n
        Sum of two or more elements is a Coefficient.

        :param other: the other element that should be added.
        :return: new coefficient that is sum of two element
        """
        return Coefficient([self, other])

    def __lt__(self, other) -> bool:
        """compare two elements\n
        the comparison is based on their length of set of variable and if that is equal based on the set of variable lexicographically and if that is equal too based on their constant.

        :param other: the other element that should be compared to.
        :return: boolean that determine which one is less than the other.
        """
        if len(self.variables) == len(other.variables):
            for i in range(len(self.variables)):
                if not (self.variables[i] == other.variables[i]):
                    return self.variables[i] < other.variables[i]
            return self.constant < other.constant
        else:
            return len(self.variables) < len(other.variables)

    def __eq__(self, other) -> bool:
        """compare two elements\n
        the comparison is based on their length of set of variable and if that is equal based on the set of variable lexicographically and if that is equal too based on their constant.

        :param other: the other element that should be compared to.
        :return: boolean that determine are they equal or not.
        """

        if len(self.variables) == len(other.variables):
            for i in range(len(self.variables)):
                if not (self.variables[i] == other.variables[i]):
                    return False
            return True
        else:
            return False

    def __neg__(self):
        """negate an element\n
        For negating an element it is sufficient to just negate the constant.

        :return: an element that is the negated form of the element.
        """
        return Element(-self.constant, self.variables)

    def convert_to_preorder(self) -> str:
        """convert Element to preorder format.

        :return: string in preorder format of the class.
        """
        if self.constant == 0:
            return "0"
        if self.constant.denominator == 1:
            preorder = str(self.constant.numerator)
            if self.constant < 0:
                preorder = f"(- {-self.constant.numerator})"
        else:
            preorder = f"(/ {self.constant.numerator} {self.constant.denominator})"
            if self.constant < 0:
                preorder = (
                    f"(- (/ {-self.constant.numerator} {self.constant.denominator}))"
                )
        if len(self.variables) == 0:
            return preorder
        preorder = "(* 1 " + preorder + " "
        for var in self.variables:
            preorder += str(var) + " "
        preorder += " )"
        return preorder


class Coefficient:
    """A class that represent the Coefficient\n
    It consists a set of elements which should be added together to form the coefficient.
    For example 2*a*b + 1/2*c*c*c + 3.5 is a coefficient.

    Attributes:
        elements ([Element]): The sorted list of elements that should be added together

    """

    def __init__(self, elements: List[Element] = []):
        elements.sort()
        self.elements = elements

    def __str__(self) -> str:
        """convert Coefficient to string.

        :return: string format of the class.
        """
        if len(self.elements) == 0:
            return "0"
        return "+".join([str(element) for element in self.elements])

    def __mul__(self, other):
        """multiply two Coefficient\n the result is a coefficient which consist elements that are equal to the
        multiply of two element from each coefficient. numpy is used for that.

            :param other: the other Coefficient that should be multiplied
            :return: new Coefficient that is multiplied of two Coefficient
        """
        return Coefficient(
            np.array(
                np.matmul(
                    np.array(self.elements).reshape(1, -1).T,
                    np.array(other.elements).reshape(1, -1),
                )
            ).reshape(1, -1)[0]
        ).revise()

    def __add__(self, other):
        """sum of two Coefficient\n
        Sum of two Coefficient is a union of their elements.

        :param other: the other Coefficient that should be added.
        :return: new coefficient that is sum of two Coefficient
        """
        if type(other) is Coefficient:
            return Coefficient(self.elements + other.elements).revise()
        if type(other) is Element:
            return Coefficient(self.elements + [other]).revise()

    def revise(self):
        """revise the Coefficient\n
        It means that the elements with the same set of variables are added together.\n
        For example 3*a + 2*a*b + 1*a is a Coefficient and after revise it returns 4*a + 2*a*b.

        :return: the revised format of the Coefficient.
        """
        self.elements.sort()
        new_list = []
        i = 0
        while i < len(self.elements):
            constant = Fraction("0")
            j = i
            while j < len(self.elements):
                if self.elements[i] == self.elements[j]:
                    constant += self.elements[j].constant
                else:
                    break
                j += 1
            new_list.append(Element(constant, self.elements[i].variables))
            i = j

        return Coefficient(new_list)

    def __neg__(self):
        """negate a Coefficient\n
        For negating a Coefficient it is sufficient to negate all its elements.

        :return: a Coefficient that is the negated form of the Coefficient.
        """
        elements = [-element for element in self.elements]
        return Coefficient(elements)

    def __sub__(self, other):
        """subtract two Coefficient\n
        subtract of two Coefficient is adding one with the negated of the other.

        :param other: the other Coefficient that should be subtracted.
        :return: new coefficient that is subtracted of two Coefficient
        """
        return self + (-other)

    def convert_to_preorder(self) -> str:
        """convert Coefficient to preorder format.

        :return: string in preorder format of the class.
        """
        if len(self.elements) == 0:
            return "0"
        if len(self.elements) == 1:
            return self.elements[0].convert_to_preorder()
        preorder = "(+ 0 "
        not_zero = False
        for i in range(0, len(self.elements)):
            if self.elements[i].constant == 0:
                continue
            else:
                not_zero = True
            preorder += self.elements[i].convert_to_preorder() + " "
        preorder += ")"
        if not_zero:
            return preorder
        else:
            return "0"
