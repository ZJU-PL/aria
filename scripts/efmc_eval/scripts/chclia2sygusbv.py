"""
CHC(LIA) to SyGuS(BV) converter
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from efmc.frontends.chc_parser import CHCParser
from conversion_utils import (
    BitvectorConverter,
    FileProcessor,
    build_bv_variable_signatures,
    build_inv_signature,
)


class CHCLIAToSyGuSBVConverter(BitvectorConverter):
    def convert(self, content: str) -> str:
        ss = CHCParser(content, to_real=False)
        all_vars, init, trans, post = ss.get_transition_system()

        init_vars = all_vars[: len(all_vars) // 2]

        # Build formula string
        bv_fml_str = "(set-logic BV)\n\n"
        bv_fml_str += build_inv_signature(init_vars, self.bitvector_width, "sygus")

        # Build variable signatures
        bv_init_vars_sig = build_bv_variable_signatures(init_vars, self.bitvector_width)
        bv_all_vars_sig = build_bv_variable_signatures(all_vars, self.bitvector_width)

        # Add function definitions
        bv_fml_str += f"(define-fun pre_fun ({' '.join(bv_init_vars_sig)}) Bool\n       {self.ira2bv(init.sexpr())})\n"
        bv_fml_str += f"(define-fun trans_fun ({' '.join(bv_all_vars_sig)}) Bool\n       {self.ira2bv(trans.sexpr())})\n"
        bv_fml_str += f"(define-fun post_fun ({' '.join(bv_init_vars_sig)}) Bool\n       {self.ira2bv(post.sexpr())})\n\n"
        bv_fml_str += (
            "(inv-constraint inv_fun pre_fun trans_fun post_fun)\n\n(check-synth)\n\n"
        )

        return bv_fml_str


def main():
    parser = argparse.ArgumentParser(description="Convert CHC LIA to SyGuS BV")
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
    converter = CHCLIAToSyGuSBVConverter(args.width, args.signedness)

    files = FileProcessor.get_files_with_extensions(args.input_path, [".smt2", ".sl"])

    for filename in files:
        result = FileProcessor.safe_process_file(filename, converter.convert)
        if result:
            filename_base = os.path.basename(filename)
            output_path = os.path.join(
                args.output_dir,
                f"{filename_base}_{args.width}bits_{args.signedness}.sl",
            )
            FileProcessor.write_output_file(result, output_path)


if __name__ == "__main__":
    main()
