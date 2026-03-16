#!/usr/bin/env python3
"""
Unified format converter for various SMT-LIB and SyGuS format conversions
Supports: CHC<->SyGuS, LIA<->BV, and various combinations
"""
import os
import sys
import argparse
from pathlib import Path
from enum import Enum

sys.path.append(str(Path(__file__).parent.parent))
from efmc.frontends.chc_parser import CHCParser
from efmc.frontends.mini_sygus_parser import SyGusInVParser
from conversion_utils import (
    BitvectorConverter,
    FileProcessor,
    build_bv_variable_signatures,
    build_inv_signature,
)


class ConversionType(Enum):
    SYGUS_LIA_TO_SYGUS_BV = "sygus_lia_to_sygus_bv"
    CHC_LIA_TO_SYGUS_BV = "chc_lia_to_sygus_bv"
    CHC_LIA_TO_CHC_BV = "chc_lia_to_chc_bv"
    SYGUS_BV_TO_CHC_BV = "sygus_bv_to_chc_bv"
    CHC_BV_TO_SYGUS_BV = "chc_bv_to_sygus_bv"
    SYGUS_LIA_TO_CHC_LIA = "sygus_lia_to_chc_lia"


class UnifiedConverter:
    def __init__(
        self, conversion_type: ConversionType, bitvector_width=32, signedness="signed"
    ):
        self.conversion_type = conversion_type
        self.bv_converter = (
            BitvectorConverter(bitvector_width, signedness)
            if self._needs_bv_conversion()
            else None
        )
        self.bitvector_width = bitvector_width
        self.signedness = signedness

    def _needs_bv_conversion(self) -> bool:
        """Check if the conversion involves bitvector operations"""
        return "bv" in self.conversion_type.value.lower()

    def convert(self, content: str) -> str:
        """Main conversion dispatcher"""
        if self.conversion_type == ConversionType.SYGUS_LIA_TO_SYGUS_BV:
            return self._sygus_lia_to_sygus_bv(content)
        elif self.conversion_type == ConversionType.CHC_LIA_TO_SYGUS_BV:
            return self._chc_lia_to_sygus_bv(content)
        elif self.conversion_type == ConversionType.CHC_LIA_TO_CHC_BV:
            return self._chc_lia_to_chc_bv(content)
        elif self.conversion_type == ConversionType.SYGUS_BV_TO_CHC_BV:
            return self._sygus_bv_to_chc_bv(content)
        elif self.conversion_type == ConversionType.CHC_BV_TO_SYGUS_BV:
            return self._chc_bv_to_sygus_bv(content)
        elif self.conversion_type == ConversionType.SYGUS_LIA_TO_CHC_LIA:
            return self._sygus_lia_to_chc_lia(content)
        else:
            raise ValueError(f"Unsupported conversion type: {self.conversion_type}")

    def _sygus_lia_to_sygus_bv(self, content: str) -> str:
        """Convert SyGuS LIA to SyGuS BV"""
        ss = SyGusInVParser(content, to_real=False)
        all_vars, init, trans, post = ss.get_transition_system()

        init_vars = all_vars[: len(all_vars) // 2]

        bv_fml_str = "(set-logic BV)\n\n"
        bv_fml_str += build_inv_signature(init_vars, self.bitvector_width, "sygus")

        bv_init_vars_sig = build_bv_variable_signatures(init_vars, self.bitvector_width)
        bv_all_vars_sig = build_bv_variable_signatures(all_vars, self.bitvector_width)

        bv_fml_str += f"(define-fun pre_fun ({' '.join(bv_init_vars_sig)}) Bool\n       {self.bv_converter.ira2bv(init.sexpr())})\n"
        bv_fml_str += f"(define-fun trans_fun ({' '.join(bv_all_vars_sig)}) Bool\n       {self.bv_converter.ira2bv(trans.sexpr())})\n"
        bv_fml_str += f"(define-fun post_fun ({' '.join(bv_init_vars_sig)}) Bool\n       {self.bv_converter.ira2bv(post.sexpr())})\n\n"
        bv_fml_str += (
            "(inv-constraint inv_fun pre_fun trans_fun post_fun)\n\n(check-synth)\n\n"
        )

        return bv_fml_str

    def _chc_lia_to_sygus_bv(self, content: str) -> str:
        """Convert CHC LIA to SyGuS BV"""
        ss = CHCParser(content, to_real=False)
        all_vars, init, trans, post = ss.get_transition_system()

        init_vars = all_vars[: len(all_vars) // 2]

        bv_fml_str = "(set-logic BV)\n\n"
        bv_fml_str += build_inv_signature(init_vars, self.bitvector_width, "sygus")

        bv_init_vars_sig = build_bv_variable_signatures(init_vars, self.bitvector_width)
        bv_all_vars_sig = build_bv_variable_signatures(all_vars, self.bitvector_width)

        bv_fml_str += f"(define-fun pre_fun ({' '.join(bv_init_vars_sig)}) Bool\n       {self.bv_converter.ira2bv(init.sexpr())})\n"
        bv_fml_str += f"(define-fun trans_fun ({' '.join(bv_all_vars_sig)}) Bool\n       {self.bv_converter.ira2bv(trans.sexpr())})\n"
        bv_fml_str += f"(define-fun post_fun ({' '.join(bv_init_vars_sig)}) Bool\n       {self.bv_converter.ira2bv(post.sexpr())})\n\n"
        bv_fml_str += (
            "(inv-constraint inv_fun pre_fun trans_fun post_fun)\n\n(check-synth)\n\n"
        )

        return bv_fml_str

    def _chc_lia_to_chc_bv(self, content: str) -> str:
        """Convert CHC LIA to CHC BV (with variable renaming support)"""
        from efmc.frontends.chc_parser import ground_quantifier
        from efmc.utils.z3_expr_utils import get_variables
        import z3

        ss = CHCParser(content, to_real=False)
        all_vars, init, trans, post = ss.get_transition_system()

        init_vars = all_vars[: len(all_vars) // 2]

        bv_fml_str = "(set-logic HORN)\n"
        bv_fml_str += build_inv_signature(init_vars, self.bitvector_width, "chc")

        bv_init_vars_sig = build_bv_variable_signatures(init_vars, self.bitvector_width)
        bv_all_vars_sig = build_bv_variable_signatures(all_vars, self.bitvector_width)

        init_var_names = [str(var) for var in init_vars]
        trans_var_names = [str(var) for var in all_vars[len(all_vars) // 2 :]]

        bv_fml_str += f"(assert (forall ({' '.join(bv_init_vars_sig)}) \n"
        bv_fml_str += f"      (=> {self.bv_converter.ira2bv(init.sexpr())} (inv {' '.join(init_var_names)}))))\n"

        bv_fml_str += f"(assert (forall ({' '.join(bv_all_vars_sig)}) \n"
        bv_fml_str += f"      (=> (and (inv {' '.join(init_var_names)}) {self.bv_converter.ira2bv(trans.sexpr())}) "
        bv_fml_str += f"(inv {' '.join(trans_var_names)}))))\n"

        bv_fml_str += f"(assert (forall ({' '.join(bv_init_vars_sig)}) \n"
        bv_fml_str += f"      (=> (inv {' '.join(init_var_names)}) {self.bv_converter.ira2bv(post.sexpr())})))\n"

        bv_fml_str += "(check-sat)\n"

        # Handle variable renaming for multi-phase benchmarks
        try:
            fmls = z3.parse_smt2_string(bv_fml_str)
            if len(fmls) >= 3:
                trans_fml = fmls[1]
                trans_body, trans_vars_list = ground_quantifier(trans_fml)

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
                    new_init_qf = z3.substitute(ground_quantifier(fmls[0])[0], mappings)
                    new_trans_qf = z3.substitute(trans_body, mappings)
                    new_post_qf = z3.substitute(ground_quantifier(fmls[2])[0], mappings)

                    sol = z3.Solver()
                    new_init_post_vars_list = get_variables(new_post_qf)
                    new_trans_vars_list = get_variables(new_trans_qf)

                    sol.add(z3.ForAll(new_init_post_vars_list, new_init_qf))
                    sol.add(z3.ForAll(new_trans_vars_list, new_trans_qf))
                    sol.add(z3.ForAll(new_init_post_vars_list, new_post_qf))

                    bv_fml_str = "(set-logic HORN)\n" + sol.sexpr() + "(check-sat)\n"
        except Exception:
            # If variable renaming fails, return the original formula
            pass

        return bv_fml_str

    def _sygus_bv_to_chc_bv(self, content: str) -> str:
        """Convert SyGuS BV to CHC BV"""
        ss = SyGusInVParser(content, to_real=False)
        all_vars, init, trans, post = ss.get_transition_system()

        init_vars = all_vars[: len(all_vars) // 2]
        trans_vars = all_vars[len(all_vars) // 2 :]

        bv_fml_str = "(set-logic HORN)\n"
        bv_fml_str += build_inv_signature(init_vars, self.bitvector_width, "chc")

        init_vars_sig = [f"({var} Int)" for var in init_vars]
        all_vars_sig = [f"({var} Int)" for var in all_vars]

        init_var_names = [str(var) for var in init_vars]
        trans_var_names = [str(var) for var in trans_vars]

        bv_fml_str += f"(assert (forall ({' '.join(init_vars_sig)}) \n"
        bv_fml_str += f"      (=> {init.sexpr()} (inv {' '.join(init_var_names)}))))\n"

        bv_fml_str += f"(assert (forall ({' '.join(all_vars_sig)}) \n"
        bv_fml_str += (
            f"      (=> (and (inv {' '.join(init_var_names)}) {trans.sexpr()}) "
        )
        bv_fml_str += f"(inv {' '.join(trans_var_names)}))))\n"

        bv_fml_str += f"(assert (forall ({' '.join(init_vars_sig)}) \n"
        bv_fml_str += f"      (=> (inv {' '.join(init_var_names)}) {post.sexpr()})))\n"

        bv_fml_str += "(check-sat)\n"
        return bv_fml_str

    def _chc_bv_to_sygus_bv(self, content: str) -> str:
        """Convert CHC BV to SyGuS BV"""
        ss = CHCParser(content, to_real=False)
        all_vars, init, trans, post = ss.get_transition_system()

        init_vars = all_vars[len(all_vars) // 2 :]  # Note: swapped for CHC BV

        bv_fml_str = "(set-logic BV)\n\n"
        bv_fml_str += build_inv_signature(init_vars, self.bitvector_width, "sygus")

        bv_init_vars_sig = build_bv_variable_signatures(init_vars, self.bitvector_width)
        bv_all_vars_sig = build_bv_variable_signatures(all_vars, self.bitvector_width)

        bv_fml_str += f"(define-fun pre_fun ({' '.join(bv_init_vars_sig)}) Bool\n       {init.sexpr()})\n"
        bv_fml_str += f"(define-fun trans_fun ({' '.join(bv_all_vars_sig)}) Bool\n       {trans.sexpr()})\n"
        bv_fml_str += f"(define-fun post_fun ({' '.join(bv_init_vars_sig)}) Bool\n       {post.sexpr()})\n\n"
        bv_fml_str += (
            "(inv-constraint inv_fun pre_fun trans_fun post_fun)\n\n(check-synth)\n\n"
        )

        return bv_fml_str

    def _sygus_lia_to_chc_lia(self, content: str) -> str:
        """Convert SyGuS LIA to CHC LIA"""
        ss = SyGusInVParser(content, to_real=False)
        all_vars, init, trans, post = ss.get_transition_system()

        init_vars = all_vars[: len(all_vars) // 2]
        trans_vars = all_vars[len(all_vars) // 2 :]

        fml_str = "(set-logic HORN)\n"

        inv_sig = "(declare-fun inv ("
        for _ in init_vars:
            inv_sig += "Int "
        inv_sig += ") Bool)\n"
        fml_str += inv_sig

        init_vars_sig = [f"({var} Int)" for var in init_vars]
        all_vars_sig = [f"({var} Int)" for var in all_vars]

        init_var_names = [str(var) for var in init_vars]
        trans_var_names = [str(var) for var in trans_vars]

        fml_str += f"(assert (forall ({' '.join(init_vars_sig)}) \n"
        fml_str += f"      (=> {init.sexpr()} (inv {' '.join(init_var_names)}))))\n"

        fml_str += f"(assert (forall ({' '.join(all_vars_sig)}) \n"
        fml_str += f"      (=> (and (inv {' '.join(init_var_names)}) {trans.sexpr()}) "
        fml_str += f"(inv {' '.join(trans_var_names)}))))\n"

        fml_str += f"(assert (forall ({' '.join(init_vars_sig)}) \n"
        fml_str += f"      (=> (inv {' '.join(init_var_names)}) {post.sexpr()})))\n"

        fml_str += "(check-sat)\n"
        return fml_str

    def get_output_extension(self) -> str:
        """Get the appropriate file extension for the output format"""
        if "sygus" in self.conversion_type.value:
            return ".sl"
        else:
            return ".smt2"


def main():
    parser = argparse.ArgumentParser(
        description="Unified format converter for SMT-LIB and SyGuS"
    )
    parser.add_argument(
        "conversion_type",
        type=str,
        choices=[ct.value for ct in ConversionType],
        help="Type of conversion to perform",
    )
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

    conversion_type = ConversionType(args.conversion_type)
    converter = UnifiedConverter(conversion_type, args.width, args.signedness)

    files = FileProcessor.get_files_with_extensions(args.input_path, [".smt2", ".sl"])

    for filename in files:
        result = FileProcessor.safe_process_file(filename, converter.convert)
        if result:
            filename_base = os.path.basename(filename)
            output_path = os.path.join(
                args.output_dir,
                f"{filename_base}_{args.width}bits_{args.signedness}{converter.get_output_extension()}",
            )
            FileProcessor.write_output_file(result, output_path)


if __name__ == "__main__":
    main()
