#!/usr/bin/python3
"""Module for creating bitwise expressions from truth value vectors."""

from __future__ import annotations

import os
import re
import sys
from typing import List, Optional, Sequence

from aria.smt.mba.utils.dnf import Dnf


# Class for creating bitwise expressions for a given number of variables.
# pylint: disable=too-few-public-methods
class BitwiseFactory:
    """Factory to build minimal bitwise expressions for a truth vector."""

    def __init__(self, vnumber: int,  # pylint: disable=redefined-outer-name
                 variables: Optional[Sequence[str]] = None,  # pylint: disable=redefined-outer-name
                 no_table: bool = False) -> None:  # pylint: disable=redefined-outer-name
        assert variables is None or len(variables) >= vnumber

        self.__vnumber: int = vnumber
        self.__variables: Sequence[str] | None = variables
        self.__uses_default_vars: bool = variables is None
        self.__table: Optional[Sequence[str]] = None
        self.__no_table: bool = no_table

        if self.__uses_default_vars:
            self.__variables = [self.__get_alt_vname(i) for i in range(vnumber)]

    # Get the alternative name of the variable with given index used if no
    # variable names are specified.
    def __get_alt_vname(self, i: int) -> str:
        return "X[" + str(i) + "]"

    # Initializes the lookup table containing 2^2^t base expressions for the
    # used number t of variables. Requires that the number of variables is not
    # larger than 3.
    def __init_table(self) -> None:
        if self.__vnumber == 1:
            self.__init_table_1var()
        elif self.__vnumber == 2:
            self.__init_table_2vars()
        elif self.__vnumber == 3:
            self.__init_table_3vars()
        else:
            assert False

    # Initializes the lookup table for 1 variable.
    def __init_table_1var(self) -> None:
        self.__table = [
            "0",  # [0 0]
            "X[0]"  # [0 1]
        ]

    # Initializes the lookup table for 2 variables.
    def __init_table_2vars(self) -> None:
        self.__table = [
            "0",  # [0 0 0 0]
            "(X[0]&~X[1])",  # [0 1 0 0]
            "(~X[0]&X[1])",  # [0 0 1 0]
            "(X[0]^X[1])",  # [0 1 1 0]
            "(X[0]&X[1])",  # [0 0 0 1]
            "X[0]",  # [0 1 0 1]
            "X[1]",  # [0 0 1 1]
            "(X[0]|X[1])"  # [0 1 1 1]
        ]

    # Initializes the lookup table for 3 variables.
    def __init_table_3vars(self) -> None:
        """Initialize lookup table for 3 variables from file."""
        utils_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "utils"
        )
        truthfile = os.path.join(
            utils_dir, "bitwise_list_" + str(self.__vnumber) + "vars.txt"
        )
        bitwise_expr_list = []

        with open(truthfile, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                b = re.split("#", line)[0].rstrip()
                bitwise_expr_list.append(b)

        self.__table = bitwise_expr_list

    # Create a bitwise expression with given truth value vector.
    def __create_bitwise(self, vector: Sequence[int]) -> str:
        """Create a bitwise expression from a truth value vector."""
        d = Dnf(self.__vnumber, vector)
        b = d.to_bitwise()
        b.refine()
        s = b.to_string(self.__variables)

        # Add parentheses if necessary.
        if bool(re.search("([&|^])", s)):
            s = "(" + s + ")"
        return s

    # For the given vector of truth values, returns the truth value vector for
    # the corresponding expression after subtracting the given offset. That
    # is, returns a vector which has zeros exactly in the positions where the
    # given vector contains the given offset.
    def __get_bitwise_vector(self, vector: Sequence[int],
                              offset_val: int) -> List[int]:
        """Get bitwise vector after subtracting offset."""
        return [(0 if v == offset_val else 1) for v in vector]

    # For the given vector of truth values, returns the index of the
    # corresponding expression in the lookup table after subtracting the given
    # offset. That is, returns the index of the truth table entry for a truth
    # value vector which has zeros exactly in the positions where the given
    # vector contains the given offset.
    def __get_bitwise_index_for_vector(self, vector: Sequence[int],
                                        offset_val: int) -> int:
        """Get index in lookup table for vector after subtracting offset."""
        index = 0
        add = 1
        for pos in range(len(vector) - 1):
            if vector[pos + 1] != offset_val:
                index += add
            add <<= 1

        return index

    # For the given vector of truth values, returns the corresponding bitwise
    # expression from the lookup table after subtracting the given offset, if
    # given. Initializes the table if not yet initialized.
    def __get_bitwise_from_table(self, vector: Sequence[int],
                                  offset_val: int) -> str:
        """Get bitwise expression from lookup table."""
        if self.__table is None:
            self.__init_table()

        index = self.__get_bitwise_index_for_vector(vector, offset_val)
        bitwise = self.__table[index]

        if not self.__uses_default_vars:
            # Exchange variables.
            for var_idx in range(self.__vnumber):
                bitwise = bitwise.replace(
                    self.__get_alt_vname(var_idx), self.__variables[var_idx]
                )

        return bitwise

    # For the given vector of truth values, creates the corresponding bitwise
    # expression after subtracting the given offset. Uses the Quine-McCluskey
    # algorithm and some addiitonal refinement.
    def __create_bitwise_with_offset(self, vector: Sequence[int],
                                      offset_val: int) -> str:
        """Create bitwise expression with offset using Quine-McCluskey."""
        vector = self.__get_bitwise_vector(vector, offset_val)
        return self.__create_bitwise(vector)

    # For the given vector of truth values, returns the corresponding bitwise
    # expression after subtracting the given offset, if given.
    def __create_bitwise_unnegated(self, vector: Sequence[int],
                                    offset_val: int = 0) -> str:
        """Create unnegated bitwise expression."""
        if not self.__no_table and self.__vnumber <= 3:
            return self.__get_bitwise_from_table(vector, offset_val)
        return self.__create_bitwise_with_offset(vector, offset_val)

    # For the given vector of truth values, returns the corresponding bitwise
    # expression after subtracting the given offset, if given. That is, returns
    # the truth table entry for a truth value vector which has zeros exactly in
    # the positions where the given vector contains the given offset. If
    # negated is True, the bitwise expression is negated.
    def create_bitwise(self, vector: Sequence[int], negated: bool = False,
                       offset_val: int = 0) -> str:
        """Create bitwise expression from truth vector."""
        # If the vector's first entry is nonzero after subtracting the offset,
        # negate the truth values and negate the bitwise thereafter.
        if (not self.__no_table and self.__vnumber <= 3 and
                vector[0] != offset_val):
            assert vector[0] == offset_val + 1
            for pos in range(len(vector)):  # pylint: disable=consider-using-enumerate
                vector[pos] = offset_val + (vector[pos] - offset_val + 1) % 2
            negated = True

        e = self.__create_bitwise_unnegated(vector, offset_val)
        if negated:
            return e[1:] if e[0] == "~" else "~" + e
        return e


# Returns a bitwise expression in string representation corresponding to the
# given truth value vector If an offset is given, the truth value vector is
# derived via subtracting the offset from the given vector. If no variables are
# passed, the variables haven names "X[i]".
def create_bitwise(vnumber: int,  # pylint: disable=redefined-outer-name
                   vec: Sequence[int],  # pylint: disable=redefined-outer-name
                   offset_val: int = 0,
                   variables: Optional[Sequence[str]] = None,  # pylint: disable=redefined-outer-name
                   no_table: bool = False) -> str:  # pylint: disable=redefined-outer-name
    """Create bitwise expression from truth vector."""
    factory = BitwiseFactory(vnumber, variables, no_table)
    return factory.create_bitwise(vec, False, offset_val)


# Print usage.
def print_usage() -> None:
    """Print usage information."""
    print("Usage: python3 create_bitwise.py <vnumber> <truth values>")
    print("The truth value list starts with \"[\", ends with \"]\" and "
          "contains values separated by spaces.")
    print("The variables are expected to start with letters and consist of "
          "letters, underscores and digits.")
    print("Command line options:")
    print("    -h:    print usage")
    print("    -v:    specify the variables (in same notation as the "
          "truth values)")
    print("    -o:    specify some offset for the truth value vector")
    print("    -n:    disable usage of lookup tables")


if __name__ == "__main__":
    argc = len(sys.argv)

    if argc == 2 and sys.argv[1] == "-h":
        print_usage()
        sys.exit(0)

    if argc < 3:
        sys.exit("Requires vnumber and truth values as arguments!")

    vnumber = int(sys.argv[1])
    vec = list(map(int, sys.argv[2].strip('[]').split(' ')))

    if len(vec) != 2 ** vnumber:
        sys.exit("Incorrect number of truth values! Requires exactly " +
                 str(2 ** vnumber) + " values.")

    offset = 0
    variables = None
    no_table = False

    arg_idx = 2
    while arg_idx < argc - 1:
        arg_idx = arg_idx + 1

        if sys.argv[arg_idx] == "-h":
            print_usage()
            sys.exit(0)

        elif sys.argv[arg_idx] == "-o":
            arg_idx += 1
            if arg_idx == argc:
                print_usage()
                sys.exit("Error: No offset list!")

            offset = int(sys.argv[arg_idx])

        elif sys.argv[arg_idx] == "-v":
            arg_idx += 1
            if arg_idx == argc:
                print_usage()
                sys.exit("Error: No variable list!")

            variables = sys.argv[arg_idx].strip('[]').split(' ')

            if len(variables) < vnumber:
                sys.exit("Incorrect number of variables! Requires at least " +
                         str(vnumber) + " values.")

        elif sys.argv[arg_idx] == "-n":
            no_table = True

        else:
            sys.exit("Unknown option " + sys.argv[arg_idx] + "!")

    if vec.count(offset) + vec.count(offset + 1) != len(vec):
        sys.exit("Error: Only offset and offset+1 allowed in truth vector!")

    print("*** Truth values " + str(vec))
    bw = create_bitwise(vnumber, vec, 0, variables, no_table)
    print("*** ... yields " + bw)
