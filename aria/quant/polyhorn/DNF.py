from typing import List


class DNF:
    """A class that represents disjunctive normal form.

        Attributes:
            literals ([]): list of elements that the or of these elements form the DNF.
    """

    def __init__(self, literals: List):
        self.literals = literals

    def __or__(self, other):
        """ Or two DNF.\n
        Or of two DNF is the union of literals of each.

        :param other: the other DNF that should be or with this class.
        :return: new DNF that is or of two DNF.
        """
        return DNF(self.literals + other.literals)

    def __and__(self, other):
        """ And two DNF.\n
        The result is a DNF where each of the literal is the union of two literal in each DNF.

        :param other: the other DNF that should be and with this class.
        :return: new DNF that is and of two DNF.
        """
        if len(self.literals) == 0:
            return DNF(other.literals)
        if len(other.literals) == 0:
            return DNF(self.literals)

        literal_list = []
        for first_literal in self.literals:
            for second_literal in other.literals:
                literal_list.append(first_literal + second_literal)
        return DNF(literal_list)

    def __neg__(self):
        """ negate a DNF\n For negating a DNF it is sufficient to negate all its literal and And them together.
        negate a literal makes a DNF.

            :return: a DNF that is the negated form of the DNF.
        """
        result_DNF = DNF([])
        for literal in self.literals:
            new_arr = []
            for item in literal:
                new_arr.append([-item])
            result_DNF = result_DNF & DNF(new_arr)
        return result_DNF

    def __str__(self) -> str:
        """ convert DNF to string.

            :return: string format of the class.
        """
        res = ''
        for literal in self.literals:
            res += '\n  AND '.join(["\t" + str(item) for item in literal])
            res += '\n OR \n'
        return '(\n' + res + ')\n'

    def convert_to_preorder(self) -> str:
        """ convert DNF to preorder format.

        :return: string in preorder format of the class.
        """
        res = '( or '
        for literal in self.literals:
            res += '( and '
            for item in literal:
                res += item.convert_to_preorder()
                res += ' '
            res += ') '
        res += ' )'
        return res
