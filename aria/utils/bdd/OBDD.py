"""Module for Ordered Binary Decision Diagrams (OBDD)."""

import ast

from .BDD import BDDNode
from .BDD import apply as BDDapply
from .ordering import Ordering, ListOrdering


def parse_args(args_node):
    """Parse arguments node to create an Ordering."""
    return Ordering([id(arg) for arg in args_node.args])


def parse_name(ordering, node):
    """Parse a name node into an OBDD."""
    if node.id == "True":
        return OBDD(BDDNode(True), ordering)
    if node.id == "False":
        return OBDD(BDDNode(False), ordering)

    return OBDD(BDDNode(node.id, BDDNode(False), BDDNode(True)), ordering)


def parse_binary_unary_op(ordering, node):
    """Parse a unary operation node."""
    if isinstance(node.op, (ast.Not, ast.Invert)):
        return ~parse_binary_expr(ordering, node.operand)
    return None


def parse_binary_op(ordering, node):
    """Parse a binary operation node."""
    if isinstance(node.op, ast.BitAnd):
        return parse_binary_expr(ordering, node.left) & parse_binary_expr(
            ordering, node.right
        )

    if isinstance(node.op, ast.BitOr):
        return parse_binary_expr(ordering, node.left) | parse_binary_expr(
            ordering, node.right
        )

    raise SyntaxError(f"expected a binary operator, got a {node.op.__class__}")


def parse_binary_binary_op(ordering, node):
    """Parse a binary boolean operation node."""
    if isinstance(node.op, ast.And):
        result = OBDD(BDDNode(True), ordering)
        for arg in node.values:
            result = result & parse_binary_expr(ordering, arg)

        return result

    if isinstance(node.op, ast.Or):
        result = OBDD(BDDNode(False), ordering)
        for arg in node.values:
            result = result | parse_binary_expr(ordering, arg)

        return result

    raise SyntaxError(f"expected a binary operator, got a {node.op.__class__}")


def parse_binary_expr(ordering, node):
    """Parse a binary expression node."""
    if isinstance(node, ast.BinOp):
        return parse_binary_op(ordering, node)

    if isinstance(node, ast.BoolOp):
        return parse_binary_binary_op(ordering, node)

    if isinstance(node, ast.UnaryOp):
        return parse_binary_unary_op(ordering, node)

    if isinstance(node, ast.Name):
        return parse_name(ordering, node)

    if isinstance(node, ast.Constant):
        if node.value in [0, 1]:
            return OBDD(BDDNode(node.value), ordering)

        raise SyntaxError(f"expected a binary expression, got number {node.value}")

    raise SyntaxError(f"expected a binary expression, got a {node.__class__}")


class BinaryParser:
    r"""
    A class to represent parsers of binary functions.

    The objects of this class can parse strings representing binary functions
    are return the corresponding OBDD.

    """

    def __init__(self, ordering):
        r"""Initialize a binary parser.

        :param ordering: an ordering for the variables to be parsed
        :type ordering: Ordering
        """
        if not isinstance(ordering, Ordering):
            try:
                ordering = Ordering(ordering)
            except TypeError as exc:
                raise TypeError(
                    f"expected an Ordering of variables, got {ordering}"
                ) from exc

        self.ordering = ordering

    @staticmethod
    def parse_function(binary_funct):
        r"""Parse a binary function.

        :param ordering: an ordering for the variables to be parsed
        :type ordering: Ordering
        """
        st = ast.parse(binary_funct)

        try:
            args_node = st.body[0].value.args

            body_node = st.body[0].value.body

        except Exception as exc:
            raise SyntaxError(
                f"expected a binary lambda function, got {binary_funct}"
            ) from exc

        ordering = parse_args(args_node)
        return parse_binary_expr(ordering, body_node)

    def parse(self, binary_expr):
        st = ast.parse(binary_expr)
        try:
            binary_expr_node = st.body[0].value
        except Exception as exc:
            raise SyntaxError(
                f"expected a binary expression, got {binary_expr}"
            ) from exc

        return parse_binary_expr(self.ordering, binary_expr_node)


class OBDD:
    r"""
    A class to represent Ordered Binary Decision Diagrams (OBDDs).
    """

    def __init__(self, bfunct, ordering=None, check_ordering=True):
        r"""Initialize an OBDD.

        :param bfunct: either a BDDNode,  a string that represents a binary
                expression,  or a string that represent a binary function in
                lambda notation
        :type bfunct: str or function
        :param ordering: an ordering for the variables to be parsed
        :type ordering: Ordering
        :param check_ordering: a binary flag: if bfunct is a BDDNode and this
                               flag is set to True,  then this method check
                               whether the BDDNode respects the ordering
        :type check_ordering: bool
        """
        if ordering is None:
            obdd = BinaryParser.parse_function(bfunct)
            self.ordering = obdd.ordering
            self.root = obdd.root

            return

        if not isinstance(ordering, Ordering):
            try:
                ordering = Ordering(ordering)
            except TypeError:
                raise TypeError(
                    "expected an Ordering of variables, got " + "{}".format(ordering)
                )

        self.ordering = ordering

        if isinstance(bfunct, BDDNode):
            if check_ordering and not bfunct.respect_ordering(ordering):
                raise ValueError(
                    f"the BDDNode {bfunct} "
                    f"does not respect {ordering}"
                )

            self.root = bfunct
        else:
            if isinstance(bfunct, str):
                bparser = BinaryParser(ordering)
                self.root = (bparser.parse(bfunct)).root
            else:
                raise TypeError(
                    f"expected a BDDNode or a str, got {bfunct}"
                )

    def restrict(self, var, value):
        r"""Partially evaluate the binary function encoded by an OBDD.

        :param var: name of the variable to be set
        :type var: str
        :param value: value to be set
        :type value: bool or int
        :returns: the OBDD representing the partial evaluation of :math:`f`
                  with :param var: set to :param value: where :math:`f` is
                  the function encoded by current object
        :rtype: OBDD
        """
        return OBDD(self.root.restrict(var, value), self.ordering)

    def variables(self):
        r"""Return the variables in an OBDD.

        :returns: the set of the variables represented in the OBDD
        :rtype: set
        """
        return self.root.variables()

    def __eq__(self, other):
        r"""Check whether two OBDD are the same.

        :param other: either a BDDNode,  an OBDD,  a binary,  or a Boolean value
        :type other: BDDNode,  OBDD,  binary,  or Boolean value
        :returns: True if the two OBDD are the same and respect the same
                  ordering,  False otherwise
        :rtype: bool
        """
        if isinstance(other, BDDNode):
            return self == OBDD(other, self.ordering)

        if isinstance(other, OBDD):
            return self.root is other.root and self.ordering == other.ordering

        if other in set([0, 1, True, False]):
            return self == OBDD(BDDNode(other), self.ordering)

        raise TypeError(f"expected an OBDD, got {other}")

    def __req__(self, other):
        r"""Check whether two OBDD are the same.

        :param other: either a BDDNode,  an OBDD,  a binary,  or a Boolean value
        :type other: BDDNode,  OBDD,  binary,  or Boolean value
        :returns: True if the two OBDD are the same and respect the same
                  ordering,  False otherwise
        :rtype: bool
        """
        return self == other

    def apply(self, operator, other):
        r"""Apply a binary binary operator to two OBDD.

        :param operator: a binary boolean operator
        :type operator: function
        :param other: an OBDD
        :type other: OBDD
        :returns: The OBDD obtained by applying the operator to the two OBDD
        :rtype: OBDD
        """

        if not isinstance(other, OBDD):
            raise TypeError(f"expected an OBDD, got {other}")

        if self.ordering != other.ordering:
            raise RuntimeError(
                f"Unsupported operation: {self} "
                f"and {other} "
                f"have different variable ordering"
            )

        result_cache = {}

        bdd = BDDapply(operator, self.root, other.root, self.ordering, result_cache)
        return OBDD(bdd, self.ordering, check_ordering=False)

    def __and__(self, other):
        r"""Build the conjunction of two OBDD.

        :param other: an OBDD
        :type other: OBDD
        :returns: the OBDD that represents the logical conjunction of
                  the two OBDD
        :rtype: OBDD
        """
        return self.apply((lambda a, b: a and b), other)

    def __or__(self, other):
        r"""Build the non-exclusive disjunction of two OBDD.

        :param other: an OBDD
        :type other: OBDD
        :returns: the OBDD that represents the logical non-exclusive
                  disjunction of the two OBDD
        :rtype: OBDD
        """
        return self.apply((lambda a, b: a or b), other)

    def __xor__(self, other):
        r"""Build the exclusive disjunction of two OBDD.

        :param other: an OBDD
        :type other: OBDD
        :returns: the OBDD that represents the logical exclusive disjunction
                  of the two OBDD
        :rtype: OBDD
        """
        return self.apply((lambda a, b: a ^ b), other)

    def __invert__(self):
        r"""Build the negation of an OBDD.

        :param A: an OBDD
        :type A: OBDD
        :returns: the OBDD that represents the logical negation of the OBDD
        :rtype: OBDD
        """
        return OBDD(~self.root, self.ordering)

    def __str__(self):
        r"""Return a string that represents an OBDD

        :returns: a string that represents the OBDD
        :rtype: str
        """
        if isinstance(self.ordering, ListOrdering):
            result = "lambda"
            sep = " "
            for var in self.ordering.get_list():
                result += f"{sep}{var}"
                sep = ","

            return f"{result}: {self.root}"

        return f"{self.root}"

    def __repr__(self):
        r"""Return a string that represents an OBDD

        :returns: a string that represents the OBDD
        :rtype: str
        """
        return f"{str(self)}"
