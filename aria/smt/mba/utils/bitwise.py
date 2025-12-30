#!/usr/bin/python3
"""Bitwise expression AST utilities.

This module provides classes for representing and manipulating bitwise
expressions as abstract syntax trees.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Sequence, Optional


class BitwiseType(Enum):
    """The type of a node representing a bitwise (sub-)expression."""

    TRUE = 0
    VARIABLE = 1
    CONJUNCTION = 2
    EXCL_DISJUNCTION = 3
    INCL_DISJUNCTION = 4

    # Function for comparing types.
    def __lt__(self, other: "BitwiseType") -> bool:  # type: ignore[override]
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


# An abstract syntax tree structure representing a bitwise (sub-)expression.
# Each node has a type (constant, variable or binary operation) and is possibly
# negated. The bitwise negation is not realized via a separate node, but via a
# flag which each node is equipped with.
class Bitwise:
    """AST for a bitwise (sub-)expression.

    Nodes can be constants, variables, or n-ary boolean operators, possibly
    negated.
    """

    def __init__(self, b_type: BitwiseType, negated: bool = False, vidx: int = -1) -> None:
        """Initialize a bitwise AST node.

        Args:
            b_type: The type of the node (constant, variable, or operation).
            negated: Whether the node is negated.
            vidx: Variable index (required if b_type is VARIABLE).
        """
        assert (vidx >= 0) == (b_type == BitwiseType.VARIABLE)

        self.__type: BitwiseType = b_type
        self.__vidx: int = vidx
        self.__negated: bool = negated
        self.__children: List[Bitwise] = []

    def add_child(self, child: "Bitwise") -> None:
        """Add the given node as a child node."""
        self.__children.append(child)

    def add_variable(self, vidx: int, negated: bool = False) -> None:
        """Add a child node for a variable with given index."""
        self.add_child(Bitwise(BitwiseType.VARIABLE, negated, vidx))

    def child_count(self) -> int:
        """Return the number of children."""
        return len(self.__children)

    def first_child(self) -> "Bitwise":
        """Return this node's first child."""
        return self.__children[0]

    def __op_to_string(self) -> str:
        """Return a string representation of this node's operation.

        Requires that this node represents a binary operation.
        """
        assert self.__type > BitwiseType.VARIABLE

        if self.__type == BitwiseType.CONJUNCTION:
            return "&"
        if self.__type == BitwiseType.EXCL_DISJUNCTION:
            return "^"
        return "|"

    def to_string(self, variables: Sequence[str] = None, with_parentheses: bool = False) -> str:
        """Return a string representation of this node, including all its children.

        Args:
            variables: Optional sequence of variable names.
            with_parentheses: Whether to add parentheses around the expression.
        """
        if variables is None:
            variables = []
        if self.__type == BitwiseType.TRUE:
            return "0" if self.__negated else "1"

        if self.__type == BitwiseType.VARIABLE:
            if len(variables) == 0:
                return ("~x" if self.__negated else "x") + str(self.__vidx)
            return ("~" if self.__negated else "") + variables[self.__vidx]

        with_parentheses = with_parentheses or self.__negated
        assert self.child_count() > 1

        s = "~" if self.__negated else ""
        s += "(" if with_parentheses else ""
        s += self.__children[0].to_string(variables, True)
        for child in self.__children[1:]:
            s += self.__op_to_string() + child.to_string(variables, True)
        s += ")" if with_parentheses else ""

        return s

    def __are_all_children_contained(self, other: "Bitwise") -> bool:
        """Return true iff this node's children are all contained in the given one's children."""
        assert other.__type == self.__type

        o_indices = list(range(len(other.__children)))
        for child in self.__children:
            found = False
            for i in o_indices:
                if child.equals(other.__children[i]):
                    o_indices.remove(i)
                    found = True

            if not found:
                return False

        return True

    def equals(self, other: "Bitwise", negated: bool = False) -> bool:
        """Return true iff this node equals the given one."""
        if self.__type != other.__type:
            return False
        if self.__vidx != other.__vidx:
            return False
        if (self.__negated == other.__negated) == negated:
            return False
        if len(self.__children) != len(other.__children):
            return False

        return self.__are_all_children_contained(other)

    def __pull_up_child(self) -> None:
        """Copy the only child's content to this node."""
        assert len(self.__children) == 1
        child = self.__children[0]

        self.__type = child.__type
        self.__vidx = child.__vidx
        self.__negated ^= child.__negated
        self.__children = child.__children

    def __copy(self, node: "Bitwise") -> None:
        """Copy the given node's content to this node."""
        self.__type = node.__type
        self.__vidx = node.__vidx
        self.__negated = node.__negated
        self.__children = node.__children

    def __get_copy(self) -> "Bitwise":
        """Copy this node's content to a new node."""
        n: Bitwise = Bitwise(self.__type, self.__negated, self.__vidx)
        n.__children = []

        for child in self.__children:
            n.__children.append(child.__get_copy())

        return n

    def refine(self) -> None:
        """Refine the structure of the tree.

        Try to write the bitwise expression more simply, possibly introducing
        exclusive disjunctions, flipping negations or extracting common nodes
        of children.
        """
        max_it = 10

        for _ in range(max_it):
            if not self.__refine_step():
                return

    def __refine_step(self) -> bool:
        """Perform one step of the refinement, see refine()."""
        changed = False

        for child in self.__children:
            if child.__refine_step():
                changed = True

        if self.__check_insert_xor():
            changed = True
        if self.__check_flip_negation():
            changed = True
        if self.__check_extract():
            changed = True

        return changed

    def __check_insert_xor(self) -> bool:
        """Try to replace any subexpressions with an exclusive disjunction.

        See __try_insert_xor().
        """
        if self.__type not in [BitwiseType.CONJUNCTION, BitwiseType.INCL_DISJUNCTION]:
            return False

        changed = False
        for i in range(len(self.__children) - 1):
            # The range above is not updated when a child is deleted.
            if i >= len(self.__children) - 1:
                break

            for j in range(1, len(self.__children)):
                if self.__try_insert_xor(i, j):
                    changed = True
                    break

        if changed and len(self.__children) == 1:
            self.__pull_up_child()

        return changed

    def __try_insert_xor(self, i: int, j: int) -> bool:
        """Try to insert an exclusive disjunction.

        Applies the rules "(x|y) & (~x|~y) -> (x^y)" or
        "(x&y) | (~x&~y) -> (x^~y)".
        """
        child1 = self.__children[i]
        child2 = self.__children[j]

        t = self.__type
        ot = (BitwiseType.INCL_DISJUNCTION if t == BitwiseType.CONJUNCTION
              else BitwiseType.CONJUNCTION)

        if child1.__type != ot or child2.__type != ot:
            return False

        # TODO: Group children of children together if there are more.
        if len(child1.__children) != 2 or len(child2.__children) != 2:
            return False

        for perm in [[0, 1], [1, 0]]:
            if child1.__children[0].equals(child2.__children[perm[0]], True):
                if not child1.__children[1].equals(child2.__children[perm[1]], True):
                    return False

                child1.__type = BitwiseType.EXCL_DISJUNCTION

                # (x|y) & (~x|~y) -> (x^y)
                if t == BitwiseType.CONJUNCTION:
                    if child1.__children[0].__negated:
                        child1.__children[0].__negated = False
                        child1.__children[1].__negated = not child1.__children[1].__negated

                # (x&y) | (~x&~y) -> (x^~y)
                else:
                    if child1.__children[0].__negated:
                        child1.__children[0].__negated = False
                    else:
                        child1.__children[1].__negated = not child1.__children[1].__negated

                del self.__children[j]
                return True

        return False

    def __check_flip_negation(self) -> bool:
        """Check if flipping negation makes the subexpression simpler.

        Check whether the subexpression corresponding to this node becomes
        simpler when flipping the negation of this node together with that
        of its children.
        """
        if len(self.__children) == 0:
            return False

        changed = False

        for child in self.__children:
            if child.__check_flip_negation():
                changed = True

        cnt = len(self.__children)
        neg_cnt = sum(1 if c.__negated else 0 for c in self.__children)
        if 2 * neg_cnt < cnt:
            return changed
        if (2 * neg_cnt == cnt and
                (not self.__negated or self.__type == BitwiseType.EXCL_DISJUNCTION)):
            return changed

        if self.__type != BitwiseType.EXCL_DISJUNCTION:
            self.__negated = not self.__negated
            if self.__type == BitwiseType.INCL_DISJUNCTION:
                self.__type = BitwiseType.CONJUNCTION
            else:
                self.__type = BitwiseType.INCL_DISJUNCTION

        for child in self.__children:
            child.__negated = not child.__negated

        return True

    def __do_all_children_have_type(self, t: BitwiseType) -> bool:
        """Return true iff all this node's children have the given type."""
        for child in self.__children:
            if child.__type != t:
                return False
            if child.__negated:
                return False
        return True

    def __check_extract(self) -> bool:
        """Check whether a common node can be factored out of this node's children."""
        t = self.__type
        if t not in (BitwiseType.CONJUNCTION, BitwiseType.INCL_DISJUNCTION):
            return False

        ot = (BitwiseType.INCL_DISJUNCTION if t == BitwiseType.CONJUNCTION
              else BitwiseType.CONJUNCTION)
        if not self.__do_all_children_have_type(ot):
            return False

        commons = []
        while True:
            common = self.__try_extract()
            if common is None:
                break
            commons.append(common)

        if len(commons) == 0:
            return False

        for child in self.__children:
            assert len(child.__children) > 0
            if len(child.__children) == 1:
                child.__pull_up_child()

        node: Bitwise = Bitwise(ot, self.__negated)
        self.__negated = False
        node.__children = commons + [self.__get_copy()]
        self.__copy(node)

        return True

    def __try_extract(self) -> Optional["Bitwise"]:
        """Try to factor a common node out of this node's children."""
        assert self.__type in [BitwiseType.CONJUNCTION, BitwiseType.INCL_DISJUNCTION]

        common = self.__get_common_child()
        if common is None:
            return None

        for child in self.__children:
            child.__remove_child(common)

        return common

    def __get_common_child(self) -> Optional["Bitwise"]:
        """Return a node that appears in all children and can be factored out."""
        assert self.__type in [BitwiseType.CONJUNCTION, BitwiseType.INCL_DISJUNCTION]

        # It is enough to consider the first child and check for all of its
        # children whether they appear in the other children too.

        first = self.__children[0]
        for child in first.__children:
            if self.__has_child_in_remaining_children(child):
                return child.__get_copy()
        return None

    def __has_child_in_remaining_children(
            self, node: "Bitwise") -> bool:
        """Return true iff the given node can be factored out.

        Return true iff the given node can be factored out from all children
        but the first one.
        """
        assert self.__type in [BitwiseType.CONJUNCTION, BitwiseType.INCL_DISJUNCTION]

        for child in self.__children[1:]:
            if not child.__has_child(node):
                return False
        return True

    def __has_child(self, node: "Bitwise") -> bool:
        """Return true iff this node has a child equal to the given node."""
        for child in self.__children:
            if child.equals(node):
                return True
        return False

    def __remove_child(self, node: "Bitwise") -> None:
        """Remove the given node from this node's children."""
        for i, child in enumerate(self.__children):
            if child.equals(node):
                del self.__children[i]
                return

        assert False
