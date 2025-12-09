from aria.quant.polyhorn.Coefficient import Coefficient
from aria.quant.polyhorn.Polynomial import Polynomial


class PolynomialConstraint:
    """ A class that represent the constraint with a Polynomial on one side and zero on the other side\n

        Attributes:
            polynomial (Polynomial): The Polynomial represented in constraint
            sign (str): The between Polynomial and zero

    """

    def __init__(self, polynomial: Polynomial, sign: str):
        self.polynomial = polynomial
        self.sign = sign
        if sign == '<':
            self.sign = '>'
            self.polynomial = -self.polynomial
        if sign == '<=':
            self.sign = '>='
            self.polynomial = -self.polynomial

    def __str__(self) -> str:
        """ convert Constraint to string.

        :return: string format of the class.
        """
        return str(self.polynomial) + self.sign + "0"

    def is_strict(self) -> bool:
        """ determine if the constraint is strict.

        :return: a boolean that is true if the constraint is strict.
        """
        return self.sign == '>'

    def __neg__(self):
        """ negate a constraint by negating the polynomial and reversing the sign

        :return: new constraint that represent the negation of the previous constraint.
        """
        if self.sign == '>':
            return PolynomialConstraint(-self.polynomial, '>=')
        elif self.sign == '>=':
            return PolynomialConstraint(-self.polynomial, '>')
        elif self.sign == '=':
            return PolynomialConstraint(self.polynomial, '!=')
        elif self.sign == '!=':
            return PolynomialConstraint(self.polynomial, '=')

    def convert_to_preorder(self) -> str:
        """ convert constraint to preorder format.

        :return: string in preorder format of the class.
        """
        return '(' + self.sign + ' ' + self.polynomial.convert_to_preorder() + ' ' + '0)'


class CoefficientConstraint:
    """ A class that represent the constraint with a Coefficient on one side and zero on the other side\n

            Attributes:
                coefficient (Coefficient): The Coefficient represented in constraint
                sign (str): The between Coefficient and zero
    """

    def __init__(self, coefficient: Coefficient, sign: str):
        self.coefficient = coefficient
        self.sign = sign

    def __str__(self) -> str:
        """ convert Constraint to string.

        :return: string format of the class.
        """
        return str(self.coefficient) + self.sign + "0"

    def is_strict(self) -> bool:
        """ determine if the constraint is strict.

        :return: a boolean that is true if the constraint is strict.
        """
        return self.sign == '>'

    def convert_to_preorder(self) -> str:
        """ convert constraint to preorder format.

        :return: string in preorder format of the class.
        """
        return '(' + self.sign + ' ' + self.coefficient.convert_to_preorder() + ' ' + '0)'

    def is_equality(self) -> bool:
        """ determine if the constraint is in form of a variable equal to constant.

        :return: a boolean that is true if the constraint is equality.
        """
        if self.sign != '=' or len(self.coefficient.elements) > 2:
            return False
        if len(self.coefficient.elements) == 2:
            if len(self.coefficient.elements[0].variables) + len(self.coefficient.elements[1].variables) == 1:
                return True
            else:
                return False
        if len(self.coefficient.elements) == 1:
            if len(self.coefficient.elements[0].variables) == 1:
                return True
            else:
                return False

    def __neg__(self):
        """ negate a constraint by negating the coefficient and reversing the sign

        :return: new constraint that represent the negation of the previous constraint.
        """
        if self.sign == '>':
            return CoefficientConstraint(-self.coefficient, '>=')
        elif self.sign == '>=':
            return CoefficientConstraint(-self.coefficient, '>')
        elif self.sign == '=':
            return CoefficientConstraint(self.coefficient, '!=')
        elif self.sign == '!=':
            return CoefficientConstraint(self.coefficient, '=')
