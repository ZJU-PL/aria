"""
CHC(LIA) to CHC(BV) converter
"""

import os
import sys
import argparse
from pathlib import Path
import z3

sys.path.append(str(Path(__file__).parent.parent))
from efmc.frontends.chc_parser import CHCParser, ground_quantifier
from efmc.utils.z3_expr_utils import get_variables
from conversion_utils import (
    BitvectorConverter,
    FileProcessor,
    build_bv_variable_signatures,
    build_inv_signature,
)


class CHCLIAToCHCBVConverter(BitvectorConverter):
    def convert(self, content: str) -> str:
        ss = CHCParser(content, to_real=False)
        all_vars, init, trans, post = ss.get_transition_system()

        init_vars = all_vars[: len(all_vars) // 2]

        # Build formula string
        bv_fml_str = "(set-logic HORN)\n"
        bv_fml_str += build_inv_signature(init_vars, self.bitvector_width, "chc")

        # Build variable signatures
        bv_init_vars_sig = build_bv_variable_signatures(init_vars, self.bitvector_width)
        bv_all_vars_sig = build_bv_variable_signatures(all_vars, self.bitvector_width)

        init_var_names = [str(var) for var in init_vars]
        trans_var_names = [str(var) for var in all_vars[len(all_vars) // 2 :]]

        # Add assertions
        bv_fml_str += f"(assert (forall ({' '.join(bv_init_vars_sig)}) \n"
        bv_fml_str += f"      (=> {self.ira2bv(init.sexpr())} (inv {' '.join(init_var_names)}))))\n"

        bv_fml_str += f"(assert (forall ({' '.join(bv_all_vars_sig)}) \n"
        bv_fml_str += f"      (=> (and (inv {' '.join(init_var_names)}) {self.ira2bv(trans.sexpr())}) "
        bv_fml_str += f"(inv {' '.join(trans_var_names)}))))\n"

        bv_fml_str += f"(assert (forall ({' '.join(bv_init_vars_sig)}) \n"
        bv_fml_str += f"      (=> (inv {' '.join(init_var_names)}) {self.ira2bv(post.sexpr())})))\n"

        bv_fml_str += "(check-sat)\n"

        # Handle variable renaming for multi-phase benchmarks
        fmls = z3.parse_smt2_string(bv_fml_str)
        if len(fmls) >= 3:
            trans_fml = fmls[1]
            trans_body, trans_vars_list = ground_quantifier(trans_fml)

            # Create variable mappings for renaming
            mappings = []
            for var in trans_vars_list:
                var_name = str(var)
                if var_name.endswith("1"):
                    new_var_name = var_name[:-1] + "!"
                elif var_name.endswith("0"):
                    new_var_name = var_name[:-1]
                else:
                    continue
                new_var = z3.BitVec(new_var_name, var.sort())
                mappings.append((var, new_var))

            if mappings:
                # Apply variable renaming
                new_init_qf = z3.substitute(ground_quantifier(fmls[0])[0], mappings)
                new_trans_qf = z3.substitute(trans_body, mappings)
                new_post_qf = z3.substitute(ground_quantifier(fmls[2])[0], mappings)

                # Rebuild with renamed variables
                sol = z3.Solver()
                new_init_post_vars_list = get_variables(new_post_qf)
                new_trans_vars_list = get_variables(new_trans_qf)

                sol.add(z3.ForAll(new_init_post_vars_list, new_init_qf))
                sol.add(z3.ForAll(new_trans_vars_list, new_trans_qf))
                sol.add(z3.ForAll(new_init_post_vars_list, new_post_qf))

                bv_fml_str = "(set-logic HORN)\n" + sol.sexpr() + "(check-sat)\n"

        return bv_fml_str


def main():
    parser = argparse.ArgumentParser(description="Convert CHC LIA to CHC BV")
    parser.add_argument("input_path", help="Input file or directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument(
        "--width", type=int, default=64, help="Bitvector width (default: 64)"
    )
    parser.add_argument(
        "--signedness",
        choices=["signed", "unsigned"],
        default="signed",
        help="Bitvector signedness (default: signed)",
    )

    args = parser.parse_args()
    converter = CHCLIAToCHCBVConverter(args.width, args.signedness)

    files = FileProcessor.get_files_with_extensions(args.input_path, [".smt2", ".sl"])

    for filename in files:
        result = FileProcessor.safe_process_file(filename, converter.convert)
        if result:
            filename_base = os.path.basename(filename)
            output_path = os.path.join(
                args.output_dir,
                f"{filename_base}_{args.width}bits_{args.signedness}.smt2",
            )
            FileProcessor.write_output_file(result, output_path)


if __name__ == "__main__":
    main()
