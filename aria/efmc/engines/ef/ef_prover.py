# coding: utf-8
"""
Constraint/template based invariant inference using EFSMT solving
NOTE:
 - We assume there is only one invariant to be synthesized
 - There is only one pair of <pre, post> from TransitionSystem
 - This file relies heavily on the templates in efmc.templates
"""

import logging
import time
from typing import Optional
from aria.efmc.utils import big_and
from aria.efmc.utils import big_or

import z3

from aria.efmc.sts import TransitionSystem
from aria.efmc.engines.ef.templates import *
from aria.efmc.engines.ef.efsmt.efsmt_solver import EFSMTSolver
from aria.efmc.utils import extract_all
from aria.efmc.utils.verification_utils import VerificationResult

logger = logging.getLogger(__name__)


class EFProver:
    """Template-based invariant inference using exists-forall SMT solving."""

    def __init__(self, sts: TransitionSystem, **kwargs):
        self.sts = sts
        self.ct = None
        self.logic = "ALL"
        self.inductive_invaraint = None

        # Configuration options
        self.ignore_post_cond = False
        self.print_vc = False
        self.seed = kwargs.get("seed", 1)
        self.prop_strengthening = kwargs.get("prop_strengthen", False)
        self.prop_strengthening_templates = kwargs.get("strengthen_templates", "")
        self.abs_refine = kwargs.get("abs_refine", False)
        self.solver = kwargs.get("solver", "z3api")
        self.validate_invariant = kwargs.get("validate_invariant", False)
        self.no_overflow = kwargs.get("no_overflow", False)
        self.no_underflow = kwargs.get("no_underflow", False)
        self.pysmt_solver = kwargs.get("pysmt_solver", "z3")
        self.num_disjunctions = kwargs.get("num_disjunctions", 2)
        # CEGIS-specific parameters
        self.cegis_solver_timeout = kwargs.get("cegis_solver_timeout", 10)
        self.cegis_dump_dir = kwargs.get("cegis_dump_dir", None)
        self.cegis_dump_threshold = kwargs.get("cegis_dump_threshold", 5)
        # split the input into list if not empty str
        if self.prop_strengthening_templates != "":
            if self.prop_strengthening:
                self.prop_strengthening_templates = (
                    self.prop_strengthening_templates.split(",")
                )
                print(
                    "Use templates",
                    self.prop_strengthening_templates,
                    "for property strengthening",
                )
            else:
                print("ERROR: Enable --prop-strengthen first")
                exit(-1)

        print(f"Prevent over/under flow: {self.no_overflow}, {self.no_underflow}")

    def set_solver(self, solver_name: str):
        self.solver = solver_name

    def set_template(self, template_name: str, num_disjunctions=2, **kwargs):
        """Set self.ct (the template to use)"""
        self.ct = self._create_template(template_name, num_disjunctions)
        self.setup_logic()

    def _create_template(self, template_name: str, num_disjunctions: int):
        """Factory method to create templates based on name and system type"""
        template_map = self._get_template_map(num_disjunctions)

        if template_name not in template_map:
            raise NotImplementedError(f"Template '{template_name}' not implemented")

        return template_map[template_name]()

    def _get_template_map(self, num_disjunctions: int):
        """Get template mapping based on system variable types"""
        common_kwargs = {
            "no_overflow": self.no_overflow,
            "no_underflow": self.no_underflow,
            "num_disjunctions": num_disjunctions,
        }

        if self.sts.has_bv:
            return self._get_bv_template_map(common_kwargs)
        elif self.sts.has_fp:
            return self._get_fp_template_map(common_kwargs)
        else:
            return self._get_int_template_map(common_kwargs)

    def _get_bv_template_map(self, kwargs):
        """Bit-vector template mapping"""
        return {
            "bv_interval": lambda: BitVecIntervalTemplate(self.sts),
            "power_bv_interval": lambda: DisjunctiveBitVecIntervalTemplate(
                self.sts, **kwargs
            ),
            "bv_zone": lambda: self._get_zone_or_interval_template(
                BitVecZoneTemplate, BitVecIntervalTemplate, kwargs
            ),
            "power_bv_zone": lambda: DisjunctiveBitVecZoneTemplate(self.sts, **kwargs),
            "bv_octagon": lambda: self._get_zone_or_interval_template(
                BitVecOctagonTemplate, BitVecIntervalTemplate, kwargs
            ),
            "power_bv_octagon": lambda: DisjunctiveBitVecOctagonTemplate(
                self.sts, **kwargs
            ),
            "bv_affine": lambda: BitVecAffineTemplate(self.sts),
            "power_bv_affine": lambda: DisjunctiveBitVecAffineTemplate(
                self.sts, **kwargs
            ),
            "bv_poly": lambda: BitVecPolyhedronTemplate(self.sts),
            "power_bv_poly": lambda: DisjunctiveBitVecPolyhedronTemplate(
                self.sts, **kwargs
            ),
            "knownbits": lambda: KnownBitsTemplate(self.sts),
            "bitpredabs": lambda: BitPredAbsTemplate(self.sts),
            "bv_pattern": lambda: BitVecPatternTemplate(self.sts),
            "bv_rotation": lambda: BitVecRotationTemplate(self.sts),
            "bv_xor_parity": lambda: BitVecXorParityTemplate(self.sts),
            "power_bv_pattern": lambda: DisjunctiveBitVecPatternTemplate(
                self.sts, **kwargs
            ),
            "power_bv_rotation": lambda: DisjunctiveBitVecRotationTemplate(
                self.sts, **kwargs
            ),
            "power_bv_xor_parity": lambda: DisjunctiveBitVecXorParityTemplate(
                self.sts, **kwargs
            ),
        }

    def _get_int_template_map(self, kwargs):
        """Integer/real template mapping"""
        return {
            "interval": lambda: IntervalTemplate(self.sts),
            "power_interval": lambda: DisjunctiveIntervalTemplate(self.sts, **kwargs),
            "zone": lambda: self._get_zone_or_interval_template(
                ZoneTemplate, IntervalTemplate, kwargs
            ),
            "octagon": lambda: self._get_zone_or_interval_template(
                OctagonTemplate, IntervalTemplate, kwargs
            ),
            "affine": lambda: AffineTemplate(self.sts),
            "power_affine": lambda: DisjunctiveAffineTemplate(self.sts, **kwargs),
            "poly": lambda: PolyTemplate(self.sts),
            "power_poly": lambda: DisjunctivePolyTemplate(self.sts, **kwargs),
        }

    def _get_fp_template_map(self, kwargs):
        """Floating-point template mapping"""
        return {
            "fp_interval": lambda: FPIntervalTemplate(self.sts),
            "fp_poly": lambda: FPPolyhedronTemplate(
                self.sts, num_constraints=kwargs.get("num_disjunctions", 2)
            ),
        }

    def _get_zone_or_interval_template(self, zone_class, interval_class, kwargs):
        """Return zone template if multiple variables, otherwise interval template"""
        if len(self.sts.variables) < 2:
            return interval_class(self.sts)
        return zone_class(
            self.sts,
            **{k: v for k, v in kwargs.items() if k in ["no_overflow", "no_underflow"]},
        )

    def setup_logic(self):
        """Setup self.logic based on template and transition system types"""
        if self.sts.has_bv:
            self.logic = "BV"
        elif self.sts.has_fp:
            self.logic = "FP"
        elif self.sts.has_real:
            self.logic = "UFLRA" if self._is_linear_template() else "UFNRA"
        elif self.sts.has_int:
            self.logic = "UFLIA" if self._is_linear_template() else "UFNIA"
        elif self.sts.has_bool:
            self.logic = "UF"
        else:
            raise NotImplementedError

    def _is_linear_template(self):
        """Check if template is linear (zone, interval, or disjunctive interval)"""
        linear_types = {
            TemplateType.ZONE,
            TemplateType.INTERVAL,
            TemplateType.DISJUNCTIVE_INTERVAL,
        }
        return self.ct.template_type in linear_types

    def dump_constraint(self, g_verifier_args) -> bool:
        """Dumping the verification condition"""
        assert g_verifier_args.dump_ef_smt2 or g_verifier_args.dump_qbf
        assert g_verifier_args.dump_ef_smt2 != g_verifier_args.dump_qbf

        qf_vc = self.generate_quantifier_free_vc()
        ef_solver = EFSMTSolver(logic=self.logic, solver=self.solver)

        exists_vars = extract_all(self.ct.template_vars)
        ef_solver.init(
            exist_vars=exists_vars, forall_vars=self.sts.all_variables, phi=qf_vc
        )

        # Generate filename
        import os

        file_name = (
            f"{g_verifier_args.dump_cnt_dir}/{os.path.basename(g_verifier_args.file)}"
        )
        file_name += f"+{g_verifier_args.template}+d{g_verifier_args.num_disjunctions}"
        file_name += f"+strength_{g_verifier_args.prop_strengthen}"
        file_name += f"+ouflow_{g_verifier_args.prevent_over_under_flows}"

        if g_verifier_args.dump_ef_smt2:
            file_name += ".smt2"
            print(f"Dumping SMT constraint to {file_name}")
            ef_solver.dump_ef_smt_file(smt2_file_name=file_name)
        elif g_verifier_args.dump_qbf:
            file_name += ".qdimacs"
            print(f"Dumping QBF constraint to {file_name}")
            ef_solver.dump_qbf_file(qdimacs_file_name=file_name)

    def solve(self, timeout: Optional[int] = None) -> VerificationResult:
        """The interface for calling different engines"""
        print("Start solving:")
        print(f"Used template: {self.ct.template_type}")
        print(f"Used logic: {self.logic}")

        if self.solver == "z3api":
            return self.solve_with_z3()
        else:
            return self._solve_with_external_solver()

    def _solve_with_external_solver(self) -> VerificationResult:
        """Solve using external EFSMT solver"""
        qf_vc = self.generate_quantifier_free_vc()
        print("EFSMT starting!!!")

        ef_solver = EFSMTSolver(
            logic=self.logic,
            solver=self.solver,
            pysmt_solver=self.pysmt_solver,
            solver_timeout=self.cegis_solver_timeout,
            dump_dir=self.cegis_dump_dir,
            dump_threshold=self.cegis_dump_threshold,
        )
        exists_vars = extract_all(self.ct.template_vars)
        ef_solver.init(
            exist_vars=exists_vars, forall_vars=self.sts.all_variables, phi=qf_vc
        )

        res = ef_solver.solve()
        if res == "sat":
            model = ef_solver.get_model()
            invariant = self.ct.build_invariant(model) if model else None
            return VerificationResult(True, invariant)
        else:
            return VerificationResult(False, None, is_unknown=True)

    def generate_quantifier_free_vc(self) -> z3.ExprRef:
        """Generate the quantifier-free part of the verification condition"""
        constraints = []

        if self.prop_strengthening:
            constraints.extend(self._generate_strengthened_constraints())
        else:
            constraints.extend(self._generate_standard_constraints())

        # Add template variable restrictions for specific template types
        if self._needs_additional_template_constraints():
            constraints.append(self.ct.get_additional_cnts_for_template_vars())

        return z3.And(constraints)

    def _generate_strengthened_constraints(self):
        """Generate constraints for property strengthening mode"""
        var_map = list(zip(self.sts.variables, self.sts.prime_variables))
        post_in_prime = z3.substitute(self.sts.post, var_map)
        self.induction = [self.ct]

        if self.prop_strengthening_templates != "":
            for template_name in self.prop_strengthening_templates:
                self.induction.append(
                    self._create_template(template_name, self.num_disjunctions)
                )

        constraints = [
            z3.Implies(
                self.sts.init,
                z3.And(
                    self.sts.post,
                    big_and([j.template_cnt_init_and_post for j in self.induction]),
                ),
            ),
            z3.Implies(
                z3.And(
                    self.sts.post,
                    big_and([j.template_cnt_init_and_post for j in self.induction]),
                    self.sts.trans,
                ),
                z3.And(
                    post_in_prime,
                    big_and([j.template_cnt_trans for j in self.induction]),
                ),
            ),
        ]

        if not self.ignore_post_cond:
            constraints.append(
                z3.Implies(
                    z3.And(
                        self.sts.post,
                        big_and([j.template_cnt_init_and_post for j in self.induction]),
                    ),
                    self.sts.post,
                )
            )

        constraint_types = {
            TemplateType.INTERVAL,
            TemplateType.DISJUNCTIVE_INTERVAL,
            TemplateType.ZONE,
            TemplateType.OCTAGON,
            TemplateType.FLOAT,
        }

        for i in range(len(self.induction) - 1):
            if self.induction[i + 1].template_type in constraint_types:
                constraints.append(
                    self.induction[i + 1].get_additional_cnts_for_template_vars()
                )
        return constraints

    def _generate_standard_constraints(self):
        """Generate standard verification constraints"""
        self.induction = [self.ct]
        constraints = [
            z3.Implies(self.sts.init, self.ct.template_cnt_init_and_post),
            z3.Implies(
                z3.And(self.ct.template_cnt_init_and_post, self.sts.trans),
                self.ct.template_cnt_trans,
            ),
        ]

        if not self.ignore_post_cond:
            constraints.append(
                z3.Implies(self.ct.template_cnt_init_and_post, self.sts.post)
            )

        return constraints

    def _needs_additional_template_constraints(self):
        """Check if template needs additional constraints"""
        constraint_types = {
            TemplateType.INTERVAL,
            TemplateType.DISJUNCTIVE_INTERVAL,
            TemplateType.ZONE,
            TemplateType.OCTAGON,
            TemplateType.FLOAT,
        }
        return self.ct.template_type in constraint_types

    def solve_with_z3(self) -> VerificationResult:
        """Solve using Z3 Python API"""
        s = z3.SolverFor(self.logic)
        qf_vc = self.generate_quantifier_free_vc()
        vc = z3.ForAll(self.sts.all_variables, qf_vc)

        # print(f"VC: {vc}")

        if self.print_vc:
            print(vc)

        s.add(vc)
        # print(s)
        print("EFSMT starting (via z3py API)!!!")

        start = time.time()
        check_res = s.check()
        print(f"EFSMT time: {time.time() - start}")

        if check_res == z3.sat:
            print("sat")
            model = s.model()
            invariant = big_and([t.build_invariant(model) for t in self.induction])
            # invariant = self.ct.build_invariant(model)
            return VerificationResult(True, invariant)
        elif check_res == z3.unsat:
            print("unsat")
            # unsat means no invariant exists with the current template
            # This could mean the property is unsafe or the template is too weak
            return VerificationResult(False, None, is_unknown=True)
        else:  # z3.unknown
            print("unknown")
            return VerificationResult(False, None, is_unknown=True)
