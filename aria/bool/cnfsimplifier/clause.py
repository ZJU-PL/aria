"""
Clause
"""
import uuid
from typing import List, Set, Optional, Union, Any

from .variable import Variable

def create_id() -> str:
    return str(uuid.uuid4())


class Clause:
    """Representation of a Boolean clause"""
    variable_list: List[Variable]  # 变量列表
    literals_set: Set[int]        # 文字集合
    id: str                      # 唯一标识符
    __size: int                  # 变量列表大小
    __tautology: bool           # 是否为重言式

    def __init__(self, variable_list: List[Union[Variable, int, float]]) -> None:
        """
        Initialize a clause with a list of variables.

        Args:
            variable_list: list of variables or integers
        """
        # Convert integers to Variable objects if needed
        if variable_list and isinstance(variable_list[0], (int, float)):
            variable_list = [Variable(int(var)) for var in variable_list]
        elif isinstance(variable_list, set):
            variable_list = [Variable(int(var)) for var in variable_list]

        self.variable_list = variable_list  # type: List[Variable]
        self.__size = len(self.variable_list)
        self.id = create_id()

        self.literals_set = set()  # type: Set[int]
        for var in self.variable_list:
            self.literals_set.add(var.variable_value)

        self.__tautology = self.__setup_tautology()

    def get_size(self) -> int:
        """
        :return: variable list size (might contain duplicated variables)
        """
        return len(self.variable_list)

    def get_set_size(self) -> int:
        """
        :return: variable set size
        """
        return len(self.literals_set)

    def __setup_tautology(self) -> bool:
        """
        Check if the clause is a tautology
        :complexity: O(n), where n is the number of literals on clause
        :return: boolean
        """
        for var in self.variable_list:
            if -var.variable_value in self.literals_set:
                return True
        return False

    def __update_tautology(self, variable_added: Variable) -> bool:
        """
        Check if the clause is a tautology
        :complexity: O(1)
        :return: boolean
        """
        return -variable_added.variable_value in self.literals_set

    def add_literal(self, lit: Union[Variable, int]) -> None:
        """
        Add a new literal to the clause
        :complexity: O(n)
        :param lit: the literal object of the Variable class or an integer
        :return: None
        """
        if isinstance(lit, int):
            lit = Variable(lit)
        self.variable_list.append(lit)
        self.size = len(self.variable_list)
        self.literals_set.add(lit.variable_value)
        self.__tautology = self.__update_tautology(lit)

    def get_diff(self, other_clause: 'Clause') -> List[Variable]:
        """
        Get the difference between two literal sets, this literal set and the other literal set
        :complexity: O(n) where n is the size of the sets
        :param other_clause: the other clause
        :return: a list of Variables that is the difference between the two literals set.
        """
        return [Variable(var) for var in self.literals_set.difference(other_clause.literals_set)]

    def get_resolvent(self, other_clause: 'Clause', lit: Variable) -> 'Clause':
        """
        Get the resolvent of two clauses
        :complexity: O(n) where n is the size of the clauses
        :param other_clause: the other clause
        :param lit: literal to get the resolvent based on it
        :return: return a new clause, the resolvent clause
        """
        if other_clause == self \
                or not other_clause.is_literal_value_present(-lit.variable_value) \
                or not self.is_literal_value_present(lit.variable_value):
            raise Exception("")

        self_set = self.literals_set.copy()
        other_set = other_clause.literals_set.copy()

        self_set.remove(lit.variable_value)
        other_set.remove(-lit.variable_value)

        new_set = self_set.union(other_set)

        resolvent_clause = Clause(list(new_set))

        return resolvent_clause

    def copy_with_new_id(self):
        """
        Copy this clause to a new one with new id and new literals
        :complexity: O(n) where n in the number of literals inside the clause
        :return: Clause
        """
        new_variable_list = [lit.copy() for lit in self.variable_list]
        new_clause = Clause(new_variable_list)
        return new_clause

    def copy_with_same_id(self):
        """
        Copy this clause to a new one with the same id and new literals
        :complexity: O(n) where n in the number of literals inside the clause
        :return: Clause
        """
        new_clause = self.copy_with_new_id()
        new_clause.id = self.id
        return new_clause

    def is_tautology(self):
        """
        checks whether the clause is a tautology
        :complexity: O(1)
        :return: boolean
        """
        return self.__tautology

    def is_sub_clause_of(self, other_clause):
        """
        checks whether a clause is a subclause of the other one
        :complexity: O(n) where n is the number of literal present on this class
        :param other_clause: the other clause
        :return: boolean
        """
        for lit in self.get_literals():
            if not other_clause.has_literal(lit):
                return False
        return True

    def has_literal(self, lit: Variable):
        """
        Checks whether a literal object is present on a clause
        :complexity: O(1)
        :param lit: literal object of class Variable
        :return: boolean
        """
        return self.is_literal_value_present(lit.variable_value)

    def is_literal_value_present(self, literal_value: int):
        """
        checks whether a literal value int is present on literal set
        :complexity: O(1) -> set check uses a hashtable, worse case could be O(n)
        :param literal_value: literal value to check
        :return: boolean
        """
        return literal_value in self.literals_set

    def get_literals(self):
        """
        :complexity: O(1)
        :return: the variable list
        """
        return self.variable_list

    def get_literals_sub_sets(self):
        """
        Get all literals sub sets excluding the empty set
        :complexity: this is done in O(n2^n), where n is the number of literals in the clause
        :return: all sub sets
        """

        def rec(i, curr, literals, ans):
            if i == len(literals):
                if curr:
                    ans.append(curr)
            else:
                copied = curr.copy()
                copied.append(literals[i])
                rec(i + 1, curr, literals, ans)
                rec(i + 1, copied, literals, ans)

        variables = self.variable_list
        sub_sets = []
        rec(0, [], variables, sub_sets)

        return sub_sets

    def get_blocking_clause(self, cnf):
        """
        Checks whether a CNF's clause is a bocked clause or not
        :complexity: O(c*l^2), where c is the number of clauses in cnf,
        and l is the number of literals per clause
        :param cnf: the cnf that the clause belogs
        :return: boolean
        """
        for lit in self.get_literals():
            for other_clause in cnf.get_clauses():
                if other_clause != self:
                    if other_clause.is_literal_value_present(-lit.variable_value):
                        resolvent_clause = self.get_resolvent(other_clause, lit)
                        if resolvent_clause.is_tautology():
                            return other_clause
        return None

    def is_subsumed(self, cnf):
        """
        :complexity: O(c*l), where l is the number of literals per clause,
         and c is the number of clause in the cnf
        :param cnf: The cnf that the clause belong
        :return: boolean
        """
        for other_clause in cnf.get_clauses():
            if other_clause != self:
                if other_clause.is_sub_clause_of(self):
                    return True
        return False

    def hla(self, f):
        """
        Hidden Literal Addition HLA(F,C)
        :complexity: O(c*l^2), where c is the number of clauses on CNF,
        and l is the number of literals on one clause
        :param f: CNF
        :return: HLA(F,C)
        """
        cls = self

        c_hla = cls.copy_with_same_id()

        for lit in cls.get_literals():
            for clause in f.get_clauses():
                if clause != cls and clause.get_size() == 2:
                    lit_clause = Clause([lit])
                    if lit_clause.is_sub_clause_of(clause):

                        dif = clause.get_diff(lit_clause)

                        if dif:
                            lit = dif[0].copy()
                        else:
                            lit = lit

                        c_hla.add_literal(lit)

        return c_hla

    def ala(self, f):
        """
        Asymmetric Literal Addition ALA(F,C)
        :complexity: O(c*(l^2)*(2^l))
        :param f: CNF
        :return: ALA(F,C)
        """
        c = self

        c_ala = c.copy_with_same_id()

        sub_sets = c.get_literals_sub_sets()

        for literal_sub_set in sub_sets:
            for clause in f.get_clauses():
                if clause != c:
                    sub_clause = Clause(literal_sub_set)
                    if clause.get_set_size() == sub_clause.get_set_size() + 1:
                        if sub_clause.is_sub_clause_of(clause):
                            dif = clause.get_diff(sub_clause)
                            lit = dif[0].copy()

                            c_ala.add_literal(lit)

        return c_ala

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id
