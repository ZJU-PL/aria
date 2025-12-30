"""Implicant utilities for DNF construction.

This module provides classes for representing implicants (conjunctions
of possibly negated variables) used in Disjunctive Normal Form.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from aria.smt.mba.utils.bitwise import Bitwise, BitwiseType


class Implicant:
    """Represents a conjunction of possibly-negated variables.

    The implicant is encoded as a vector where each entry denotes how a
    variable occurs in the conjunction:
    - 1 for unnegated occurrence
    - 0 for negated occurrence
    - None if the variable does not influence the conjunction
    """

    def __init__(self, vnumber: int, value: int) -> None:
        """Initialize an implicant.

        - vnumber: number of variables
        - value: integer whose binary representation encodes the conjunction; if
          -1, create an empty implicant that can be populated later.
        """
        self.vec: List[Optional[int]] = []
        self.minterms: List[int] = [value] if value != -1 else []
        self.obsolete: bool = False

        if value != -1:
            self.__init_vec(vnumber, value)

    def __init_vec(self, vnumber: int, value: int) -> None:
        """Initialize the implicant's vector.

        Initialize with 1s for variables that appear unnegatedly and 0s
        for those which appear negatedly.
        """
        for _ in range(vnumber):
            self.vec.append(value & 1)
            value >>= 1

        assert len(self.vec) == vnumber

    def __str__(self) -> str:
        """Return a string representation of this implicant."""
        return str(self.vec)

    def __get_copy(self) -> "Implicant":
        """Return a copy of this implicant."""
        cpy: Implicant = Implicant(len(self.vec), -1)
        cpy.vec = list(self.vec)
        cpy.minterms = list(self.minterms)
        return cpy

    def count_ones(self) -> int:
        """Return the number of ones in the implicant's vector."""
        return self.vec.count(1)

    def try_merge(self, other: "Implicant") -> Optional["Implicant"]:
        """Try to merge this implicant with the given one.

        Returns a merged implicant if this is possible and None otherwise.
        """
        assert len(self.vec) == len(other.vec)

        diff_idx = -1
        for i, (self_val, other_val) in enumerate(zip(self.vec, other.vec)):
            if self_val == other_val:
                continue

            # Already found a difference, no other difference allowed.
            if diff_idx != -1:
                return None

            diff_idx = i

        new_impl: Implicant = self.__get_copy()
        new_impl.minterms += other.minterms
        if diff_idx != -1:
            new_impl.vec[diff_idx] = None

        return new_impl

    def get_indifferent_hash(self) -> int:
        """Get a number that uniquely identifies the indifferent positions.

        The indifferent positions are those for which either 0 or 1 would fit.
        """
        h = 0
        n = 1

        for val in self.vec:
            # The position is indifferent.
            if not val:
                h += n
            n <<= 1

        return h

    def to_bitwise(self) -> Bitwise:
        """Create an abstract syntax tree structure corresponding to this implicant."""
        root: Bitwise = Bitwise(BitwiseType.CONJUNCTION)
        for i, val in enumerate(self.vec):
            # The variable has no influence.
            if val is None:
                continue

            root.add_variable(i, val == 0)

        cnt = root.child_count()
        if cnt == 0:
            return Bitwise(BitwiseType.TRUE)
        if cnt == 1:
            return root.first_child()
        return root

    def get(self, variables: Sequence[str]) -> str:
        """Return a more detailed string representation."""
        assert len(variables) == len(self.vec)

        s = ""
        for i, val in enumerate(self.vec):
            # The variable has no influence.
            if val is None:
                continue

            if len(s) > 0:
                s += "&"
            if val == 0:
                s += "~"
            s += variables[i]

        return s if len(s) > 0 else "-1"
