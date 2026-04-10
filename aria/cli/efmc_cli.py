"""Command-line interface for EFMC.
This module provides the main CLI interface for running different verification engines on transition systems specified in various formats.
"""

import argparse
import logging
import os
import signal
import sys

# from pathlib import Path
# import logging
# import subprocess
# from threading import Timer
from typing import List
import psutil

from aria.efmc.frontends import parse_sygus, parse_chc, parse_boogie, parse_c
from aria.efmc.sts import TransitionSystem
from aria.efmc.efmc_config import g_verifier_args

from aria.efmc.engines.ef import EFProver
from aria.efmc.engines.kinduction import KInductionProver
from aria.efmc.engines.pdr import PDRProver
from aria.efmc.engines.llm4inv import LLM4InvProver
from aria.efmc.utils.verification_utils import VerificationResult

# for the less-used prover, I move them to the corresponding functions.
# the followings only serve as "docs"
""" 
from aria.efmc.engines.qi import QuantifierInstantiationProver
from aria.efmc.engines.houdini import HoudiniProver
from aria.efmc.engines.abduction import AbductionProver
from aria.efmc.engines.qe import QuantifierEliminationProver
from aria.efmc.engines.predabs import PredicateAbstractionProver
from aria.efmc.engines.symabs import SymbolicAbstractionProver
from aria.efmc.engines.bdd.bdd_prover import BDDProver
from aria.efmc.engines.symabs.symabs_prover import SymbolicAbstractionProver
from aria.efmc.engines.predabs.predabs_prover import PredicateAbstractionProver
"""

# Available templates
TEMPLATES = {
    "int_real": [
        "interval",
        "power_interval",
        "zone",
        "octagon",
        "affine",
        "power_affine",
        "poly",
        "power_poly",
    ],
    "bitvector": [
        "bv_interval",
        "power_bv_interval",
        "bv_zone",
        "power_bv_zone",
        "bv_octagon",
        "power_bv_octagon",
        "bv_poly",
        "power_bv_poly",
        "knownbits",
        "bitpredabs",
        "bv_pattern",
        "power_bv_pattern",
        "bv_xor_parity",
        "power_bv_xor_parity",
        "power_bv_rotation",
        "bv_rotation",
    ],
    "floating_point": ["fp_interval", "fp_poly"],
}


class EFMCRunner:
    """Main runner class for EFMC verification tasks"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        """Configure signal handlers for graceful termination"""

        def handler(sig, frame):
            self.logger.info("Handling termination signal")
            parent = psutil.Process(os.getpid())
            for child in parent.children(recursive=True):
                child.kill()
            sys.exit(0)

        for sig in [signal.SIGINT, signal.SIGQUIT, signal.SIGABRT, signal.SIGTERM]:
            signal.signal(sig, handler)

    def print_verification_result(self, result: VerificationResult) -> None:
        """Print verification result in a consistent format"""
        if result.is_safe:
            print("safe")
            if result.invariant is not None:
                print(f"Invariant: {result.invariant}")
        elif result.is_unknown:
            print("unknown")
        else:
            if result.counterexample is not None:
                print("unsafe")
                print(f"Counterexample: {result.counterexample}")
            else:
                print("unknown")

    def run_ef_prover(self, sts: TransitionSystem) -> None:
        """Run template-based invariant generation using EF prover"""
        # Determine template and available templates based on variable types
        if sts.has_bv:
            available_templates = TEMPLATES["bitvector"]
            default_template = "bv_interval"
        elif sts.has_fp:
            available_templates = TEMPLATES["floating_point"]
            default_template = "fp_interval"
        else:
            available_templates = TEMPLATES["int_real"]
            default_template = "interval"

        # Handle auto template selection
        if g_verifier_args.template == "auto":
            g_verifier_args.template = default_template
        elif g_verifier_args.template not in available_templates:
            self.logger.error(f"Unsupported template: {g_verifier_args.template}")
            self.logger.info(f"Available templates: {available_templates}")
            sys.exit(1)

        # Create EF prover with appropriate configuration
        ef_prover = EFProver(
            sts,
            prop_strengthen=g_verifier_args.prop_strengthen,
            abs_refine=g_verifier_args.abs_refine,
            validate_invariant=g_verifier_args.validate_invariant,
            no_overflow=(
                g_verifier_args.prevent_over_under_flows > 0 if sts.has_bv else False
            ),
            no_underflow=(
                g_verifier_args.prevent_over_under_flows > 0 if sts.has_bv else False
            ),
            pysmt_solver=g_verifier_args.pysmt_solver,
            strengthen_templates=g_verifier_args.strengthen_templates,
            num_disjunctions=g_verifier_args.num_disjunctions,
            cegis_solver_timeout=getattr(g_verifier_args, "cegis_solver_timeout", 10),
            cegis_dump_dir=getattr(g_verifier_args, "cegis_dump_dir", None),
            cegis_dump_threshold=getattr(g_verifier_args, "cegis_dump_threshold", 5),
        )

        # Set template and solver
        ef_prover.set_template(
            g_verifier_args.template, num_disjunctions=g_verifier_args.num_disjunctions
        )
        ef_prover.set_solver(g_verifier_args.efsmt_solver)

        if g_verifier_args.dump_ef_smt2 or g_verifier_args.dump_qbf:
            ef_prover.dump_constraint(g_verifier_args)
        else:
            result = ef_prover.solve()
            self.print_verification_result(result)

    def run_pdr(self, sts: TransitionSystem) -> None:
        """Run PDR/IC3 verification"""
        pdr_prover = PDRProver(sts)
        result = pdr_prover.solve()
        self.print_verification_result(result)

    def run_qi(self, sts: TransitionSystem) -> None:
        """Run the Quantifier Instantiation (QI) based verification"""
        from aria.efmc.engines.qi.qi_prover import QuantifierInstantiationProver

        qi_prover = QuantifierInstantiationProver(sts)

        # Set the QI strategy based on the command-line argument
        if hasattr(g_verifier_args, "qi_strategy"):
            qi_prover.set_strategy(g_verifier_args.qi_strategy)
            self.logger.info(f"Using QI strategy: {g_verifier_args.qi_strategy}")

        result = qi_prover.solve()
        self.print_verification_result(result)

    def run_k_induction(self, sts: TransitionSystem) -> None:
        """Run k-induction verification"""
        self.logger.info("Starting k-induction...")

        # Choose between incremental and non-incremental k-induction
        if g_verifier_args.kind_incremental:
            from aria.efmc.engines.kinduction.kind_prover_inc import KInductionProverInc

            kind_prover = KInductionProverInc(sts)
        else:
            kind_prover = KInductionProver(sts)

        if g_verifier_args.kind_aux_inv:
            kind_prover.use_aux_invariant = True
        result = kind_prover.solve(int(g_verifier_args.kind_k))
        self.print_verification_result(result)

    def run_qe(self, sts: TransitionSystem) -> None:
        """Run quantifier elimination based verification"""
        from aria.efmc.engines.qe.qe_prover import QuantifierEliminationProver

        qe_prover = QuantifierEliminationProver(sts)
        result = qe_prover.solve()
        self.print_verification_result(result)

    def verify_chc(self, filename: str) -> None:
        """Verify CHC format file"""
        all_vars, init, trans, post = parse_chc(filename, to_real_type=False)
        self.logger.info("CHC file parsing completed")

        sts = TransitionSystem()
        sts.from_z3_cnts([all_vars, init, trans, post])

        if sts.has_bv and g_verifier_args.signedness in ["signed", "unsigned"]:
            sts.set_signedness(g_verifier_args.signedness)

        self.run_verifier(sts)

    def verify_sygus(self, filename: str) -> None:
        """Verify SyGuS format file"""

        all_vars, init, trans, post = parse_sygus(filename, to_real_type=False)
        self.logger.info("SyGuS file parsing completed")

        sts = TransitionSystem()
        sts.from_z3_cnts([all_vars, init, trans, post])

        if sts.has_bv and g_verifier_args.signedness in ["signed", "unsigned"]:
            sts.set_signedness(g_verifier_args.signedness)

        self.run_verifier(sts)

    def verify_boogie(self, filename: str) -> None:
        """Verify Boogie format file"""
        self.logger.info(f"Parsing Boogie file: {filename}")

        # Parse Boogie file and get transition system directly
        sts = parse_boogie(filename, to_real_type=False)
        self.logger.info("Boogie file parsing completed")

        if sts.has_bv and g_verifier_args.signedness in ["signed", "unsigned"]:
            sts.set_signedness(g_verifier_args.signedness)

        self.run_verifier(sts)

    def verify_c(self, filename: str) -> None:
        """Verify C file"""
        self.logger.info(f"Parsing C file: {filename}")
        sts = parse_c(filename, to_real_type=False)
        self.logger.info("C file parsing completed")

        if sts.has_bv and g_verifier_args.signedness in ["signed", "unsigned"]:
            sts.set_signedness(g_verifier_args.signedness)

        self.run_verifier(sts)

    def run_verifier(self, sts: TransitionSystem) -> None:
        """Run the selected verification engine"""
        engine_map = {
            "ef": self.run_ef_prover,
            "pdr": self.run_pdr,
            "kind": self.run_k_induction,
            "qe": self.run_qe,
            "qi": self.run_qi,
            "houdini": self.run_houdini,
            "abduction": self.run_abduction,
            "bdd": self.run_bdd,
            "predabs": self.run_predabs,
            "symabs": self.run_symabs,
            "llm4inv": self.run_llm4inv,
        }

        if g_verifier_args.engine not in engine_map:
            self.logger.error(f"Unsupported engine: {g_verifier_args.engine}")
            self.logger.info(f"Available engines: {list(engine_map.keys())}")
            sys.exit(1)

        engine_map[g_verifier_args.engine](sts)

    def run_houdini(self, sts: TransitionSystem) -> None:
        """Run Houdini-based invariant inference"""
        from aria.efmc.engines.houdini.houdini_prover import HoudiniProver

        houdini_prover = HoudiniProver(sts)
        result = houdini_prover.solve()
        self.print_verification_result(result)

    def run_abduction(self, sts: TransitionSystem) -> None:
        """Run abduction-based invariant inference"""
        self.logger.info("Starting abduction-based verification...")
        from aria.efmc.engines.abduction.abduction_prover import AbductionProver

        abduction_prover = AbductionProver(sts)
        if hasattr(g_verifier_args, "abduction_max_iterations"):
            abduction_prover.max_iterations = g_verifier_args.abduction_max_iterations
        if hasattr(g_verifier_args, "abduction_timeout"):
            abduction_prover.timeout = g_verifier_args.abduction_timeout
        result = abduction_prover.solve()
        self.print_verification_result(result)

    def run_bdd(self, sts: TransitionSystem) -> None:
        """Run BDD-based verification"""
        from aria.efmc.engines.bdd.bdd_prover import BDDProver

        self.logger.info("Starting BDD-based verification...")
        use_forward = True
        if hasattr(g_verifier_args, "bdd_backward") and g_verifier_args.bdd_backward:
            use_forward = False
        max_iterations = 1000
        if hasattr(g_verifier_args, "bdd_max_iterations"):
            max_iterations = g_verifier_args.bdd_max_iterations

        bdd_prover = BDDProver(
            sts, use_forward=use_forward, max_iterations=max_iterations
        )
        result = bdd_prover.solve()
        self.print_verification_result(result)

    def run_predabs(self, sts: TransitionSystem) -> None:
        """Run predicate abstraction-based verification"""
        self.logger.info("Starting predicate abstraction-based verification...")
        from aria.efmc.engines.predabs.predabs_prover import PredicateAbstractionProver

        predabs_prover = PredicateAbstractionProver(sts)

        # Set predicates if provided
        if (
            hasattr(g_verifier_args, "predabs_predicates")
            and g_verifier_args.predabs_predicates
        ):
            try:
                # Parse predicates from strings if they are provided as strings
                import z3

                predicates = []
                for pred_str in g_verifier_args.predabs_predicates:
                    # Try to parse as SMT-LIB2 string
                    try:
                        pred = z3.parse_smt2_string(f"(assert {pred_str})")[0]
                        predicates.append(pred)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to parse predicate '{pred_str}': {e}"
                        )

                if predicates:
                    predabs_prover.set_predicates(predicates)
                else:
                    self.logger.warning(
                        "No valid predicates provided, using default predicates"
                    )
            except Exception as e:
                self.logger.error(f"Error setting predicates: {e}")

        result = predabs_prover.solve()
        self.print_verification_result(result)

    def run_symabs(self, sts: TransitionSystem) -> None:
        """Run symbolic abstraction-based verification"""
        self.logger.info("Starting symbolic abstraction-based verification...")
        from aria.efmc.engines.symabs.symabs_prover import SymbolicAbstractionProver

        symabs_prover = SymbolicAbstractionProver(sts)

        # Set domain if provided
        if hasattr(g_verifier_args, "symabs_domain") and g_verifier_args.symabs_domain:
            symabs_prover.domain = g_verifier_args.symabs_domain

        result = symabs_prover.solve()
        self.print_verification_result(result)

    def run_llm4inv(self, sts: TransitionSystem) -> None:
        """Run the simplified LLM4Inv guess-and-check engine"""
        self.logger.info("Starting LLM4Inv (guess-and-check) verification...")
        prover = LLM4InvProver(sts)
        result = prover.solve()
        self.print_verification_result(result)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="EFMC - A Software Model Checker",
        usage="efmc --file FILE [options]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input and general options
    input_group = parser.add_argument_group("Input options")
    input_group.add_argument(
        "--file", type=str, required=True, help="Input file to verify"
    )
    input_group.add_argument(
        "--lang",
        type=str,
        choices=["chc", "sygus", "boogie", "auto"],
        default="auto",
        help="Input language format",
    )
    input_group.add_argument(
        "--engine",
        type=str,
        choices=[
            "ef",
            "pdr",
            "kind",
            "qe",
            "qi",
            "houdini",
            "abduction",
            "bdd",
            "predabs",
            "symabs",
            "llm4inv",
        ],
        default="ef",
        help="Verification engine to use",
    )
    input_group.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    # Template-based verification options
    template_group = parser.add_argument_group("Template-based verification options")
    template_group.add_argument(
        "--template",
        type=str,
        default="auto",
        help="Template for invariant generation. For integer/real: interval, power_interval, zone, octagon, affine, power_affine, poly, power_poly. For bitvector: bv_interval, power_bv_interval, bv_zone, power_bv_zone, bv_octagon, power_bv_octagon, bv_poly, power_bv_poly, bv_pattern, bv_xor_parity, bv_rotation. For floating-point: fp_interval, fp_poly",
    )
    template_group.add_argument(
        "--num-disjunctions",
        type=int,
        default=1,
        help="Number of disjunctions in template",
    )
    template_group.add_argument(
        "--prop-strengthen", action="store_true", help="Enable property strengthening"
    )
    template_group.add_argument(
        "--abs-refine", action="store_true", help="Enable abstraction refinement"
    )
    template_group.add_argument(
        "--validate-invariant",
        action="store_true",
        help="Validate generated invariants",
    )
    template_group.add_argument(
        "--strengthen-templates",
        type=str,
        default="",
        help="Add some templates to strengthen the present template",
    )
    # Solver options
    # currently, only affect the ef engine?
    solver_group = parser.add_argument_group("Solver options")

    # "z3api" means we use the z3py API to solve the constraints "in memory"
    solver_group.add_argument(
        "--efsmt-solver",
        dest="efsmt_solver",
        default="z3api",
        type=str,
        help="Solver for the exists-forall SMT problem. Options include:\n"
        "1. Quantifier instantiation approach: [z3, cvc5, btor, yices2, mathsat, bitwuzla, z3api]\n"
        "2. Bit-blasting approach: [z3qbf, caqe, q3b, z3sat]\n"
        "3. CEGIS approach: [cegis] (implemented via pysmt, need to set pysmt_solver)",
    )

    # We implement the CEGIS solver using pysmt's API, so we need to set the PySMT solver. However, pysmt may have limited support for some theories such as QF_FP and QF_SLIA.
    solver_group.add_argument(
        "--pysmt-solver",
        dest="pysmt_solver",
        default="z3",
        type=str,
        help="Set the PySMT solver for the CEGIS solver's backend. Options include: [z3, cvc5, btor, yices2, mathsat, bitwuzla]",
    )

    # Currently, the timeout is not used for the engines inside efmc
    # It is better to control the timeout in some "upper-level" scripts.
    solver_group.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for external solvers and binary verifiers",
    )

    # Bitvector options (for controling the bit-vector expressions in the template)
    bv_group = parser.add_argument_group("Bitvector options")
    bv_group.add_argument(
        "--prevent-over-under-flows",
        type=int,
        default=0,
        help="Prevent over/underflows in bitvector operations",
    )
    bv_group.add_argument(
        "--signedness",
        type=str,
        default="signed",
        choices=["signed", "unsigned"],
        help="Signedness for bitvector operations",
    )

    # Output options
    output_group = parser.add_argument_group("EFSMT constraint dumping options")
    output_group.add_argument(
        "--dump-ef-smt2",
        action="store_true",
        default=False,
        help="Dump EF constraints in SMT2 format",
    )
    output_group.add_argument(
        "--dump-qbf",
        action="store_true",
        default=False,
        help="Dump constraints in QBF format",
    )
    output_group.add_argument(
        "--dump-cnt-dir",
        dest="dump_cnt_dir",
        default="/tmp",
        type=str,
        help="The dir for storing the dumped constraints",
    )

    # CEGIS-specific options (for FP logic)
    cegis_group = parser.add_argument_group("CEGIS solver options (for FP logic)")
    cegis_group.add_argument(
        "--cegis-solver-timeout",
        type=int,
        default=10,
        help="Timeout per solver call in seconds for CEGIS (default: 10)",
    )
    cegis_group.add_argument(
        "--cegis-dump-dir",
        type=str,
        default=None,
        help="Directory to dump challenging QF_FP queries (default: None, no dumping)",
    )
    cegis_group.add_argument(
        "--cegis-dump-threshold",
        type=int,
        default=5,
        help="Dump queries that timeout or exceed this many iterations (default: 5)",
    )

    # K-induction options
    kind_group = parser.add_argument_group("K-induction options")
    kind_group.add_argument(
        "--kind-k", type=int, default=30, help="K value for k-induction"
    )
    kind_group.add_argument(
        "--kind-incremental",
        dest="kind_incremental",
        default=False,
        action="store_true",
        help="Use incremental k-induction (default: False)",
    )
    kind_group.add_argument(
        "--kind-aux-inv",
        action="store_true",
        help="Use auxiliary invariants in k-induction",
    )
    kind_group.add_argument(
        "--kind-aux-inv-alg",
        dest="kind_aux_inv_alg",
        default="default",
        type=str,
        help="Select the approach for generating auxiliary invariants. Options include: [top, int, houdini]",
    )

    # LLM4Inv options
    llm4inv_group = parser.add_argument_group("LLM4Inv options")
    llm4inv_group.add_argument(
        "--llm4inv-max-candidates-per-iter",
        type=int,
        default=5,
        help="Maximum number of candidates for LLM4Inv (default: 5)",
    )
    llm4inv_group.add_argument(
        "--llm4inv-provider",
        type=str,
        default="local",
        help="Provider for LLM4Inv: local, remote (default: local)",
    )
    llm4inv_group.add_argument(
        "--llm4inv-local-provider",
        type=str,
        default="lm-studio",
        help="Local provider for LLM4Inv: lm-studio, vllm, sglang",
    )
    llm4inv_group.add_argument(
        "--llm4inv-local-model",
        type=str,
        default="qwen/qwen3-coder-30b",
        help="Local model for LLM4Inv (default: qwen/qwen3-coder-30b)",
    )
    llm4inv_group.add_argument(
        "--llm4inv-remote-model",
        type=str,
        default="deepseek-v3",
        help="Remote model for LLM4Inv: deepseek-v3, gpt-4o, etc. (default: deepseek-v3)",
    )
    llm4inv_group.add_argument(
        "--llm4inv-temperature",
        type=float,
        default=0.1,
        help="Temperature for LLM4Inv (default: 0.1)",
    )
    llm4inv_group.add_argument(
        "--llm4inv-max-output-length",
        type=int,
        default=4096,
        help="Maximum output length for LLM4Inv (default: 4096)",
    )
    llm4inv_group.add_argument(
        "--llm4inv-measure-cost",
        action="store_true",
        default=False,
        help="Measure cost for LLM4Inv (default: False)",
    )
    llm4inv_group.add_argument(
        "--llm4inv-max-iterations",
        type=int,
        default=10,
        help="Maximum number of iterations for LLM4Inv (default: 10)",
    )

    # Quantifier instantiation-based verification options
    qi_group = parser.add_argument_group(
        "Quantifier instantiation-based verification options"
    )
    qi_group.add_argument(
        "--qi-strategy",
        type=str,
        default="mbqi",
        help="Quantifier instantiation strategy (mbqi, ematching, combined, auto)",
    )

    # Abduction-based verification options
    abduction_group = parser.add_argument_group("Abduction-based verification options")
    abduction_group.add_argument(
        "--abduction-timeout",
        type=int,
        default=10000,
        help="Timeout in milliseconds for abduction solver queries",
    )
    abduction_group.add_argument(
        "--abduction-max-iterations",
        type=int,
        default=300,
        help="Maximum number of strengthening iterations for abduction",
    )

    # BDD-based verification options
    bdd_group = parser.add_argument_group("BDD-based verification options")
    bdd_group.add_argument(
        "--bdd-backward",
        action="store_true",
        help="Use backward reachability analysis instead of forward",
    )
    bdd_group.add_argument(
        "--bdd-max-iterations",
        type=int,
        default=1000,
        help="Maximum number of iterations for fixed-point computation",
    )

    # Predicate abstraction options
    predabs_group = parser.add_argument_group("Predicate abstraction options")
    predabs_group.add_argument(
        "--predabs-predicates",
        type=str,
        nargs="+",
        help="List of predicates to use for predicate abstraction",
    )

    # Symbolic abstraction options
    symabs_group = parser.add_argument_group("Symbolic abstraction options")
    symabs_group.add_argument(
        "--symabs-domain",
        type=str,
        default="interval",
        choices=["interval", "bits", "known_bits"],
        help="Abstract domain to use for symbolic abstraction",
    )

    # PDR engine options: Allow for parsing options to Z3's PDR engine (named Spacer)
    # FIXME: refer to the Z3 documentation for the options
    # pdr_group = parser.add_argument_group('PDR engine options')
    return parser.parse_args()


def main():
    """Main entry point for the CLI"""
    global g_verifier_args
    g_verifier_args = parse_arguments()

    # Configure logging based on the specified log level
    logging.basicConfig(
        level=getattr(logging, g_verifier_args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    runner = EFMCRunner()

    if g_verifier_args.lang == "chc":
        runner.verify_chc(g_verifier_args.file)
    elif g_verifier_args.lang == "sygus":
        runner.verify_sygus(g_verifier_args.file)
    elif g_verifier_args.lang == "boogie":
        runner.verify_boogie(g_verifier_args.file)
    elif g_verifier_args.lang == "c":
        runner.verify_c(g_verifier_args.file)
    else:  # the default value is "auto"
        file_extension = os.path.splitext(g_verifier_args.file)[1].lower()
        if file_extension == ".smt2":
            runner.verify_chc(g_verifier_args.file)
        elif file_extension == ".sy" or file_extension == ".sl":
            runner.verify_sygus(g_verifier_args.file)
        elif file_extension == ".bpl":
            runner.verify_boogie(g_verifier_args.file)
        elif file_extension == ".c":
            runner.verify_c(g_verifier_args.file)
        else:
            logging.error(f"Unsupported file extension: {file_extension}")
            logging.info(
                "Supported extensions: .smt2 (CHC), .sy/sl (SyGuS), .bpl (Boogie)"
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
