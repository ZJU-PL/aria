"""
SyGuS(LIA) to SyGuS(BV) converter
"""

import os
import sys
from pathlib import Path
import argparse
import z3

sys.path.append(str(Path(__file__).parent.parent))
from efmc.frontends.mini_sygus_parser import SyGusInVParser, parse_sexpression


class SyGusLIAToSyGuSBVConverter:
    def __init__(self, bitvector_width=32, signedness="signed"):
        self.bitvector_width = bitvector_width
        self.signedness = signedness

    def rep_operand(self, op: str) -> str:
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
                "%": "bvsdiv",
                "div": "bvsdiv",
                ">=": "bvuge",
                "<=": "bvule",
                ">": "bvugt",
                "<": "bvult",
            }
        return rep_rules.get(op, op)

    def to_bv_sexpr_misc(self, line):
        """Convert LIRA expressions to BV"""
        res = ["("]
        if not isinstance(line[0], list):
            if line[0] == "-" and len(line) == 2 and not isinstance(line[1], list):
                # Handle negative numbers using two's complement
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
        return " ".join(self.to_bv_sexpr_misc(parse_sexpression(tt)))

    def convert(self, content: str) -> str:
        ss = SyGusInVParser(content, to_real=False)
        all_vars, init, trans, post = ss.get_transition_system()

        init_vars = all_vars[: len(all_vars) // 2]
        trans_vars = all_vars[len(all_vars) // 2 :]

        # Build invariant signature
        bv_inv_sig = "(synth-inv inv_fun ("
        for init_var in init_vars:
            bv_inv_sig += f"({init_var} {z3.BitVec(str(init_var), self.bitvector_width).sort().sexpr()})"
        bv_inv_sig += "))\n\n"

        # Build formula string
        bv_fml_str = "(set-logic BV)\n\n" + bv_inv_sig

        # Build variable signatures
        bv_init_vars_sig = [
            f"({var} {z3.BitVec(str(var), self.bitvector_width).sort().sexpr()})"
            for var in init_vars
        ]
        bv_all_vars_sig = [
            f"({var} {z3.BitVec(str(var), self.bitvector_width).sort().sexpr()})"
            for var in all_vars
        ]

        # Add function definitions
        bv_fml_str += f"(define-fun pre_fun ({' '.join(bv_init_vars_sig)}) Bool\n       {self.ira2bv(init.sexpr())})\n"
        bv_fml_str += f"(define-fun trans_fun ({' '.join(bv_all_vars_sig)}) Bool\n       {self.ira2bv(trans.sexpr())})\n"
        bv_fml_str += f"(define-fun post_fun ({' '.join(bv_init_vars_sig)}) Bool\n       {self.ira2bv(post.sexpr())})\n\n"
        bv_fml_str += (
            "(inv-constraint inv_fun pre_fun trans_fun post_fun)\n\n(check-synth)\n\n"
        )

        return bv_fml_str


def process_file(filename: str, target_dir: str, converter: SyGusLIAToSyGuSBVConverter):
    print(f"Processing {filename}")
    try:
        with open(filename, "r") as f:
            content = f.read()
            fml_str = converter.convert(content)

            filename_base = os.path.basename(filename)
            new_file_name = os.path.join(
                target_dir,
                f"{filename_base}_{converter.bitvector_width}bits_{converter.signedness}.sl",
            )

            os.makedirs(target_dir, exist_ok=True)
            with open(new_file_name, "w") as new_f:
                print(f"Writing to {new_file_name}")
                new_f.write(fml_str)
    except Exception as ex:
        print(f"Error processing {filename}: {ex}")


def process_folder(path: str, target_dir: str, converter: SyGusLIAToSyGuSBVConverter):
    for root, dirs, files in os.walk(path):
        for fname in files:
            if os.path.splitext(fname)[1] in [".smt2", ".sl"]:
                process_file(os.path.join(root, fname), target_dir, converter)


def main():
    parser = argparse.ArgumentParser(description="Convert SyGuS LIA to SyGuS BV")
    parser.add_argument("input_path", help="Input file or directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument(
        "--width", type=int, default=32, help="Bitvector width (default: 32)"
    )
    parser.add_argument(
        "--signedness",
        choices=["signed", "unsigned"],
        default="signed",
        help="Bitvector signedness (default: signed)",
    )

    args = parser.parse_args()

    converter = SyGusLIAToSyGuSBVConverter(args.width, args.signedness)

    if os.path.isfile(args.input_path):
        process_file(args.input_path, args.output_dir, converter)
    else:
        process_folder(args.input_path, args.output_dir, converter)


if __name__ == "__main__":
    main()
