from typing import List

from aria.quant.polyhorn.Coefficient import Element
from aria.quant.polyhorn.Constraint import CoefficientConstraint
from aria.quant.polyhorn.DNF import DNF
from aria.quant.polyhorn.Polynomial import Coefficient, Monomial, Polynomial, UnknownVariable


class Solver:
    """ This class consist of some static method which are used in other classes

    """

    @staticmethod
    def find_equality_constraint(LHS: Polynomial, RHS: Polynomial) -> List[CoefficientConstraint]:
        """ given two polynomial that should be equal together, this function finds constraint for the equality.
        Moreover, all the Coefficient of each Monomial in each side should be equal.


        :param LHS: left hand side of the equality
        :param RHS: right hand side of the equality
        :return: list of the constraints that needs to be true for equality
        """
        all_degree = set(
            LHS.dict_from_degrees_to_monomials.keys()
        ).union(
            set(RHS.dict_from_degrees_to_monomials.keys())
        )

        all_constraint = []
        for degree in all_degree:
            mono1 = LHS.get_monomial_by_degree(degree)
            mono2 = RHS.get_monomial_by_degree(degree)
            constraint = CoefficientConstraint(
                mono1.coefficient - mono2.coefficient, '=')
            all_constraint.append(constraint)

        return all_constraint

    @staticmethod
    def get_constant_polynomial(variables: List[UnknownVariable], constant) -> Polynomial:
        """generate new polynomial with one monomial and a constant as its coefficient

                :param variables: polynomial variables
                :param constant: the constant of the new polynomial
                :return: polynomial with constant as its coefficient
                """
        return Polynomial(variables,
                          [Monomial(
                              variables, [0] * len(variables),
                              Coefficient(
                                  [Element(constant, [])]
                              )
                          )]
                          )

    @staticmethod
    def get_variable_polynomial(variables: List[UnknownVariable], name: str, type_of_var: str = None) -> Polynomial:
        """generate new polynomial with one monomial and a new generated variable as its coefficient

        :param variables: polynomial variables
        :param name: name of the new variable
        :param type_of_var: type of the new variable
        :return: polynomial with new variable as its coefficient
        """
        new_variable = UnknownVariable(name=name, type_of_var=type_of_var)
        return Polynomial(variables,
                          [Monomial(
                              variables, [0] * len(variables),
                              Coefficient(
                                  [Element('1', [new_variable])]
                              )
                          )]
                          )

    @staticmethod
    def get_degree_polynomial(variables: List[UnknownVariable], degrees: List[int]) -> Polynomial:
        """ generate new polynomial with one monomial and given degrees

        :param variables: polynomial variables
        :param degrees: degree of the monomial
        :return: polynomial with given degree set
        """
        return Polynomial(variables,
                          [Monomial(
                              variables, degrees,
                              Coefficient(
                                  [Element('1', [])]
                              )
                          )]
                          )

    @staticmethod
    def convert_constraints_to_smt_format(all_constraint: List[DNF], precondition, names: List[str] = None) -> str:
        """ generate string for declaring constraint in smt format

        :param all_constraint: constraint that should be converted to smt format
        :param names: list of name for each constraint
        :return: smt string format of constraints
        """
        smt_string = ''
        for i, constraint in enumerate(all_constraint):
            if names is None:
                smt_string = smt_string + \
                    f'(assert  {constraint.convert_to_preorder()} )\n'
            else:
                smt_string = smt_string + \
                    f'(assert ( ! {constraint.convert_to_preorder()} :named {names[i]}))\n'

        for constraint in precondition:
            if len(constraint) == 1:
                smt_string = smt_string + \
                    f'(assert  {constraint[0].convert_to_preorder()} )\n'
            elif len(constraint) == 2:
                smt_string = smt_string + \
                    f'(assert (=> {constraint[0].convert_to_preorder()} {constraint[1].convert_to_preorder()}))\n'
        return smt_string

    @staticmethod
    def smt_declare_variable_phase(all_constraint: List[DNF], real: bool = True,
                                   pre_variables: List[UnknownVariable] = []) -> str:
        """ generate string format for declaring the variables in smt format

        :param all_constraint: constraint that their variable should be generated
        :param real: variables should be declared as integer or real valued
        :param pre_variables: list of variables that should be defined but might not be in constraints
        :return: smt string format of declaration phase
        """
        all_variables = Solver.get_all_variable(all_constraint, pre_variables)

        smt_string = ''

        for var in all_variables:
            if real:
                smt_string = smt_string + f'(declare-const {var} Real)\n'
            else:
                smt_string = smt_string + f'(declare-const {var} Int)\n'

        return smt_string

    @staticmethod
    def get_all_variable(all_constraints: List[DNF], pre_variables: List[UnknownVariable] = []) -> List[UnknownVariable]:
        """ find all the variables in a list of constraint

        :param all_constraints: list of constraints
        :param pre_variables: list of variables that should be defined but might not be in constraints
        :return: list of variables used in constraints
        """
        all_variables = set(pre_variables)
        for dnf in all_constraints:
            for literal in dnf.literals:
                for constraint in literal:
                    for element in constraint.coefficient.elements:
                        all_variables = all_variables.union(
                            set([var for var in element.variables]))

        return all_variables
