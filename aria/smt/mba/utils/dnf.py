"""Disjunctive Normal Form (DNF) utilities.

This module provides classes for representing and constructing DNFs
from truth vectors using the Quine-McCluskey algorithm.
"""

from __future__ import annotations

import re
from typing import List, Sequence

from aria.smt.mba.utils.implicant import Implicant
from aria.smt.mba.utils.bitwise import Bitwise, BitwiseType

using_gmpy: bool = True
try:
    import gmpy2
except ModuleNotFoundError:
    using_gmpy = False


def popcount(x: int) -> int:
    """Return the number of ones in the given number's binary representation."""
    if using_gmpy:
        return gmpy2.popcount(x)
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count("1")


# A structure representing a disjunctive normal form, i.e., a disjunction of
# conjunctions of possibly negated variables.
class Dnf:
    """A disjunctive normal form (DNF): disjunction of conjunctions.

    Internally constructed via Quine–McCluskey from a truth vector.
    """

    def __init__(self, vnumber: int, vec: Sequence[int]) -> None:
        """Build a DNF for a given number of variables and truth vector."""
        self.__groups: List[dict[int, List[Implicant]]] = []
        self.primes: List[Implicant] = []

        self.__init_groups(vnumber, vec)
        self.__merge()
        self.__drop_unrequired_implicants(vec)

    def __init_groups(self, vnumber: int, vec: Sequence[int]) -> None:
        """Initialize groups of implicants.

        Create a vector representation of conjunction for each 1 in the given
        vector, equivalent to the corresponding evaluation of variables according
        to its position, and classify the conjunctions according to their numbers
        of ones.
        """
        assert len(vec) == 2**vnumber

        self.__groups = [{} for _ in range(vnumber + 1)]
        for i, bit in enumerate(vec):
            if bit == 0:
                continue
            assert bit == 1

            impl = Implicant(vnumber, i)
            ones_cnt = popcount(i)
            group = self.__groups[ones_cnt]

            if "0" in group:
                group["0"].append(impl)
            else:
                group["0"] = [impl]

    def __merge_step(self) -> bool:
        """Try to merge implicants whose vectors differ in just one position.

        Note that, e.g., the disjunction of "x&y&z" and "x&y&~z" can be
        simplified to "x&y" since the "z" has no influence on its values.
        """
        changed = False
        new_groups: List[dict[int, List[Implicant]]] = [{} for _ in self.__groups]

        for ones_cnt, group in enumerate(self.__groups):
            if ones_cnt < len(self.__groups) - 1:
                next_group = self.__groups[ones_cnt + 1]

                # Iterate over hashes of indifferent positions.
                for h in group:
                    # The next group has no implicants with coincident
                    # indifferent positions.
                    if h not in next_group:
                        continue

                    for impl1 in group[h]:
                        for impl2 in next_group[h]:
                            new_impl = impl1.try_merge(impl2)
                            # Could not merge the implicants.
                            if new_impl is None:
                                continue

                            changed = True
                            impl1.obsolete = True
                            impl2.obsolete = True

                            new_group: dict[int, List[Implicant]] = new_groups[
                                new_impl.count_ones()
                            ]
                            new_h: int = new_impl.get_indifferent_hash()

                            if new_h in new_group:
                                new_group[new_h].append(new_impl)
                            else:
                                new_group[new_h] = [new_impl]

            for h in group:
                for impl in group[h]:
                    if not impl.obsolete:
                        self.primes.append(impl)

        self.__groups = new_groups
        # The only group which may vanish is the last one, since it was not
        # empty before and its elements can only be merged into the second-last
        # group.
        if len(self.__groups[-1]) == 0:
            del self.__groups[-1]
        return changed

    def __merge(self) -> None:
        """Try to merge implicants iteratively until nothing can be merged any more."""
        while True:
            changed = self.__merge_step()

            if not changed:
                return

    def __drop_unrequired_implicants(self, vec: Sequence[int]) -> None:
        """Remove implicants which are already represented by others."""
        requ = {i for i in range(len(vec)) if vec[i] == 1}

        i = 0
        while i < len(self.primes):
            impl = self.primes[i]

            mt_set = set(impl.minterms)
            # The implicant has still required terms.
            if mt_set & requ:
                requ -= mt_set
                i += 1
                continue

            del self.primes[i]

    def __str__(self) -> str:
        """Return a string representation of this DNF."""
        s = "implicants:\n"

        for impl in self.primes:
            s += "    " + str(impl) + "\n"

        return s

    def to_bitwise(self) -> Bitwise:
        """Create an abstract syntax tree structure corresponding to this DNF."""
        cnt = len(self.primes)
        if cnt == 0:
            return Bitwise(BitwiseType.TRUE, True)
        if cnt == 1:
            return self.primes[0].to_bitwise()

        root = Bitwise(BitwiseType.INCL_DISJUNCTION)
        for p in self.primes:
            root.add_child(p.to_bitwise())

        return root

    def get(self, variables: Sequence[str]) -> str:
        """Return a more detailed string representation."""
        if len(self.primes) == 0:
            return "0"

        s = ""
        for p in self.primes:
            if len(s) > 0:
                s += "|"

            ps = p.get(variables)
            with_par = len(self.primes) > 1 and bool(re.search("([&])", ps))
            s += "(" + ps + ")" if with_par else ps

        return s
