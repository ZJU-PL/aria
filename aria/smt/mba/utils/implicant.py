
from __future__ import annotations

from typing import List, Optional, Sequence

from aria.smt.mba.utils.bitwise import Bitwise, BitwiseType


# A structure representing a conjunction of possibly negated variables. It is
# mainly represented as a vector whose entry with index 0 is 1 if the i-th
# variable occurs unnegatedly in the conjunction, 0 if it appears negatedly and
# None if it has no influence.
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

    # Initialize the implicant's vector with 1s for variables that appear
    # unnegatedly and 0s for those which appear negatedly.
    def __init_vec(self, vnumber: int, value: int) -> None:
        for i in range(vnumber):
            self.vec.append(value & 1)
            value >>= 1

        assert (len(self.vec) == vnumber)

    # Returns a string representation of this implicant.
    def __str__(self) -> str:
        return str(self.vec)

    # Returns a copy of this implicant.
    def __get_copy(self) -> "Implicant":
        cpy: Implicant = Implicant(len(self.vec), -1)
        cpy.vec = list(self.vec)
        cpy.minterms = list(self.minterms)
        return cpy

    # Returns the number of ones in the implicant's vector.
    def count_ones(self) -> int:
        return self.vec.count(1)

    # Try to merge this implicant with the given one. Returns a merged
    # implicant if this is possible and None otherwise.
    def try_merge(self, other: "Implicant") -> Optional["Implicant"]:
        assert (len(self.vec) == len(other.vec))

        diffIdx = -1
        for i in range(len(self.vec)):
            if self.vec[i] == other.vec[i]:
                continue

            # Already found a difference, no other difference allowed.
            if diffIdx != -1:
                return None

            diffIdx = i

        newImpl: Implicant = self.__get_copy()
        newImpl.minterms += other.minterms
        if diffIdx != -1:
            newImpl.vec[diffIdx] = None

        return newImpl

    # Get a number that uniquely identifies the indifferent positions, i.e.,
    # the positions for which either 0 or 1 would fit.
    def get_indifferent_hash(self) -> int:
        h = 0
        n = 1

        for i in range(len(self.vec)):
            # The position is indifferent.
            if not self.vec[i]: h += n
            n << 1

        return h

    # Create an abstract syntax tree structure corresponding to this implicant.
    def to_bitwise(self) -> Bitwise:
        root: Bitwise = Bitwise(BitwiseType.CONJUNCTION)
        for i in range(len(self.vec)):
            # The variable has no influence.
            if self.vec[i] == None:
                continue

            root.add_variable(i, self.vec[i] == 0)

        cnt = root.child_count()
        if cnt == 0: return Bitwise(BitwiseType.TRUE)
        if cnt == 1: return root.first_child()
        return root

    # Returns a more detailed string representation.
    def get(self, variables: Sequence[str]) -> str:
        assert (len(variables) == len(self.vec))

        s = ""
        for i in range(len(self.vec)):
            # The variable has no influence.
            if self.vec[i] == None:
                continue

            if len(s) > 0: s += "&"
            if self.vec[i] == 0: s += "~"
            s += variables[i]

        return s if len(s) > 0 else "-1"
