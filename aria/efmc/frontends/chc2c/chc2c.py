#!/usr/bin/python3

#   Copyright 2025 Budapest University of Technology and Economics
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import argparse
import os

from .LinearCHC2C import LinearCHC2C, RecursiveException
from .NonLinearCHC2C import NonLinearCHC2C


def main():
    parser = argparse.ArgumentParser(
        prog="CHC2C",
        description="Converts CHC files in the SMT-LIB format to C code for verification.",
        epilog="See https://chc-comp.github.io/format.html for the documentation of the format.",
    )

    parser.add_argument("filename")

    parser.add_argument("-o", "--out", default="chc.c")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--no-retry", action="store_true")

    args = parser.parse_args()
    with open(args.filename, "r") as f_in:
        text = f_in.read()
        program = None
        if not args.recursive:
            try:
                program = LinearCHC2C().chc_to_c_program(text, args.out)
            except RecursiveException:
                if args.no_retry:
                    print(
                        "Unable to map without recursion, but --no-retry is enabled. Exiting."
                    )
                    return
                print("Retrying with recursive mapping...")
        if not program:
            program = NonLinearCHC2C().chc_to_c_program(text, args.out)
        with open(args.out, "w") as f_out:
            f_out.write(program)

    print(f"Mapping done. Output: {os.path.abspath(args.filename)}")


if __name__ == "__main__":
    main()
