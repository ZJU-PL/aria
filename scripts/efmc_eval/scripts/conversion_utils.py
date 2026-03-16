"""
Common utilities for format conversion scripts
"""

import os
from typing import List
from pathlib import Path
import z3
from efmc.frontends.mini_sygus_parser import parse_sexpression


class BitvectorConverter:
    """Base class for bitvector conversions with common functionality"""

    def __init__(self, bitvector_width=32, signedness="signed"):
        self.bitvector_width = bitvector_width
        self.signedness = signedness

    def rep_operand(self, op: str) -> str:
        """Replace operand with corresponding BV operator"""
        if self.signedness == "signed":
            rep_rules = {
                "+": "bvadd",
                "-": "bvsub",
                "*": "bvmul",
                "%": "bvsdiv",
                "div": "bvudiv",
                ">=": "bvsge",
                "<=": "bvsle",
                ">": "bvsgt",
                "<": "bvslt",
            }
        else:
            rep_rules = {
                "+": "bvadd",
                "-": "bvsub",
                "*": "bvmul",
                "%": "bvurem",
                "div": "bvudiv",
                ">=": "bvuge",
                "<=": "bvule",
                ">": "bvugt",
                "<": "bvult",
            }
        return rep_rules.get(op, op)

    def to_bv_sexpr_misc(self, line: List) -> List[str]:
        """Convert LIRA expressions to BV s-expressions"""
        res = ["("]
        if not isinstance(line[0], list):
            if line[0] == "-" and len(line) == 2 and not isinstance(line[1], list):
                # Handle negative numbers using two's complement: -x = bvnot(x) + 1
                if isinstance(line[1], int):
                    var = f"(_ bv{line[1]} {self.bitvector_width})"
                else:
                    var = line[1]
                new_line = ["bvadd", f"(_ bv1 {self.bitvector_width})", ["bvnot", var]]
            else:
                new_line = [self.rep_operand(line[0])] + line[1:]
        else:
            new_line = line

        for element in new_line:
            if isinstance(element, list):
                if not isinstance(element[0], list):
                    if (
                        element[0] == "-"
                        and len(element) == 2
                        and not isinstance(element[1], list)
                    ):
                        if isinstance(element[1], int):
                            var = f"(_ bv{element[1]} {self.bitvector_width})"
                        else:
                            var = element[1]
                        new_element = [
                            "bvadd",
                            f"(_ bv1 {self.bitvector_width})",
                            ["bvnot", var],
                        ]
                    else:
                        new_element = [self.rep_operand(element[0])] + element[1:]
                else:
                    new_element = element
                res.extend(self.to_bv_sexpr_misc(new_element))
            else:
                if isinstance(element, int):
                    res.append(f"(_ bv{element} {self.bitvector_width})")
                else:
                    res.append(str(element))
        res.append(")")
        return res

    def ira2bv(self, tt: str) -> str:
        """Convert integer arithmetic expression to bitvector"""
        return " ".join(self.to_bv_sexpr_misc(parse_sexpression(tt)))


class FileProcessor:
    """Common file processing utilities"""

    @staticmethod
    def get_files_with_extensions(path: str, extensions: List[str]) -> List[str]:
        """Get all files with specified extensions from path (file or directory)"""
        files = []
        if os.path.isfile(path):
            if any(path.endswith(ext) for ext in extensions):
                files.append(path)
        else:
            for root, dirs, file_list in os.walk(path):
                for fname in file_list:
                    if any(fname.endswith(ext) for ext in extensions):
                        files.append(os.path.join(root, fname))
        return files

    @staticmethod
    def safe_process_file(filename: str, processor_func, *args, **kwargs):
        """Safely process a file with error handling"""
        print(f"Processing {filename}")
        try:
            with open(filename, "r") as f:
                content = f.read()
                return processor_func(content, *args, **kwargs)
        except Exception as ex:
            print(f"Error processing {filename}: {ex}")
            return None

    @staticmethod
    def write_output_file(content: str, output_path: str, create_dirs=True):
        """Write content to output file, optionally creating directories"""
        if create_dirs:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            f.write(content)
        print(f"Written to {output_path}")


def build_bv_variable_signatures(vars_list, bitvector_width: int) -> List[str]:
    """Build bitvector variable signatures for SMT-LIB"""
    return [
        f"({var} {z3.BitVec(str(var), bitvector_width).sort().sexpr()})"
        for var in vars_list
    ]


def build_inv_signature(vars_list, bitvector_width: int, format_type="sygus") -> str:
    """Build invariant signature for different formats"""
    if format_type == "sygus":
        sig = "(synth-inv inv_fun ("
        for var in vars_list:
            sig += f"({var} {z3.BitVec(str(var), bitvector_width).sort().sexpr()})"
        sig += "))\n\n"
    else:  # CHC format
        sig = "(declare-fun inv ("
        for _ in vars_list:
            sig += f"(_ BitVec {bitvector_width}) "
        sig += ") Bool)\n"
    return sig
