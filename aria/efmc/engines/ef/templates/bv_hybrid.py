"""
Hybrid bit-vector templates combining multiple abstract domains via conjunction.

This module provides conjunctive combinations of bit-vector templates to achieve
higher precision by leveraging the complementary strengths of different domains:
- Interval: Numerical bounds tracking
- KnownBits: Bit-level precision
- Zone: Relational constraints between variables
- ...?

TODO: by LLM, to be reviewed
"""

import z3
from typing import List, Dict, Any, Optional
from aria.efmc.engines.ef.templates.abstract_template import Template, TemplateType
from aria.efmc.engines.ef.templates.bv_interval import BitVecIntervalTemplate
from aria.efmc.engines.ef.templates.bv_zone import BitVecZoneTemplate
from aria.efmc.engines.ef.templates.bv_bitwise import KnownBitsTemplate
from aria.efmc.sts import TransitionSystem
from aria.utils.z3.expr import big_and


class ConjunctiveBitVecTemplate(Template):
    """
    Conjunctive combination of multiple bit-vector templates.

    Combines multiple base templates using conjunction (AND) to create
    stronger invariants by leveraging complementary strengths:
    - Better precision than individual templates
    - Can capture complex relationships missed by single domains
    """

    def __init__(self, sts: TransitionSystem, **kwargs):
        self.template_type = TemplateType.BV_ENHANCED_PATTERN
        self.sts = sts

        # Configuration: which base templates to combine
        self.combine_interval = kwargs.get("combine_interval", True)
        self.combine_zone = kwargs.get("combine_zone", True)
        self.combine_knownbits = kwargs.get("combine_knownbits", True)

        # Base template instances
        self.base_templates = []
        self.template_vars = []

        # Initialize base templates
        if self.combine_interval:
            interval_template = BitVecIntervalTemplate(sts, **kwargs)
            self.base_templates.append(("interval", interval_template))

        if self.combine_zone and len(sts.variables) >= 2:
            zone_template = BitVecZoneTemplate(sts, **kwargs)
            self.base_templates.append(("zone", zone_template))

        if self.combine_knownbits:
            knownbits_template = KnownBitsTemplate(sts)
            self.base_templates.append(("knownbits", knownbits_template))

        # Collect all template variables
        self.add_template_vars()

        # Initialize constraints
        self.template_cnt_init_and_post = None
        self.template_cnt_trans = None
        self.add_template_cnts()

    def add_template_vars(self):
        """Collect template variables from all base templates."""
        for name, template in self.base_templates:
            if hasattr(template, "template_vars"):
                # Handle nested template variables (list of lists)
                if isinstance(template.template_vars, list) and template.template_vars:
                    if isinstance(template.template_vars[0], list):
                        # Flatten nested structure
                        for var_group in template.template_vars:
                            self.template_vars.extend(var_group)
                    else:
                        self.template_vars.extend(template.template_vars)

            # Special handling for KnownBitsTemplate
            if name == "knownbits" and hasattr(template, "bit_vars"):
                for var_idx in template.bit_vars:
                    for must0, must1 in template.bit_vars[var_idx]:
                        self.template_vars.extend([must0, must1])

    def add_template_cnts(self):
        """Combine constraints from all base templates using conjunction."""
        init_post_constraints = []
        trans_constraints = []

        for name, template in self.base_templates:
            # Add base template constraints
            if hasattr(template, "template_cnt_init_and_post"):
                init_post_constraints.append(template.template_cnt_init_and_post)
            if hasattr(template, "template_cnt_trans"):
                trans_constraints.append(template.template_cnt_trans)

            # Add additional constraints if available
            if hasattr(template, "get_additional_cnts_for_template_vars"):
                additional = template.get_additional_cnts_for_template_vars()
                if not z3.is_true(additional):
                    init_post_constraints.append(additional)

        # Combine all constraints using conjunction
        self.template_cnt_init_and_post = big_and(init_post_constraints)
        self.template_cnt_trans = big_and(trans_constraints)

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables: bool = False
    ) -> z3.ExprRef:
        """Build invariant by combining invariants from all base templates."""
        invariant_parts = []

        for name, template in self.base_templates:
            if hasattr(template, "build_invariant_expr"):
                part = template.build_invariant_expr(model, use_prime_variables)
                invariant_parts.append(part)

        return big_and(invariant_parts) if invariant_parts else z3.BoolVal(True)

    """
    Reduced product combination maintaining consistency between template views.
    
    Maintains multiple abstract domains simultaneously and ensures consistency
    through cross-domain constraints. Each operation uses the most precise
    abstract transformer available.
    """

    def __init__(self, sts: TransitionSystem, **kwargs):
        self.template_type = TemplateType.BV_ENHANCED_PATTERN
        self.sts = sts

        # Component domains
        self.domains = {}
        self.consistency_constraints = []

        if kwargs.get("use_interval", True):
            self.domains["interval"] = BitVecIntervalTemplate(sts, **kwargs)

        if kwargs.get("use_zone", True) and len(sts.variables) >= 2:
            self.domains["zone"] = BitVecZoneTemplate(sts, **kwargs)

        if kwargs.get("use_knownbits", True):
            self.domains["knownbits"] = KnownBitsTemplate(sts)

        # Collect template variables and build consistency constraints
        self.template_vars = []
        self.add_template_vars()

        self.template_cnt_init_and_post = None
        self.template_cnt_trans = None
        self.add_template_cnts()

    def add_template_vars(self):
        """Collect variables from all domains and add consistency constraints."""
        for domain_name, domain in self.domains.items():
            if hasattr(domain, "template_vars"):
                if isinstance(domain.template_vars, list):
                    if domain.template_vars and isinstance(
                        domain.template_vars[0], list
                    ):
                        for var_group in domain.template_vars:
                            self.template_vars.extend(var_group)
                    else:
                        self.template_vars.extend(domain.template_vars)

        # Add consistency constraints between domains
        self._add_consistency_constraints()

    def _add_consistency_constraints(self):
        """Add constraints to maintain consistency between different domain views."""
        # Example: If interval says x ∈ [a,b] and known bits says bit i must be 1,
        # ensure these are consistent

        if "interval" in self.domains and "knownbits" in self.domains:
            interval_domain = self.domains["interval"]
            knownbits_domain = self.domains["knownbits"]

            # For each variable, ensure interval bounds are consistent with known bits
            for i, var in enumerate(self.sts.variables):
                if i < len(interval_domain.template_vars) and z3.is_bv(var):
                    lower_bound = interval_domain.template_vars[i][0]
                    upper_bound = interval_domain.template_vars[i][1]

                    # If certain bits are known, they constrain the possible range
                    if i in knownbits_domain.bit_vars:
                        for bit_pos, (must0, must1) in enumerate(
                            knownbits_domain.bit_vars[i]
                        ):
                            # If bit must be 1, add minimum value constraint
                            min_val_if_bit_set = z3.BitVecVal(1 << bit_pos, var.size())
                            self.consistency_constraints.append(
                                z3.Implies(
                                    must1, z3.UGE(upper_bound, min_val_if_bit_set)
                                )
                            )

    def add_template_cnts(self):
        """Combine constraints from all domains plus consistency constraints."""
        all_constraints = []

        # Add constraints from each domain
        for domain in self.domains.values():
            if hasattr(domain, "template_cnt_init_and_post"):
                all_constraints.append(domain.template_cnt_init_and_post)

        # Add consistency constraints
        all_constraints.extend(self.consistency_constraints)

        self.template_cnt_init_and_post = big_and(all_constraints)

        # Similar for transition constraints
        trans_constraints = []
        for domain in self.domains.values():
            if hasattr(domain, "template_cnt_trans"):
                trans_constraints.append(domain.template_cnt_trans)

        self.template_cnt_trans = big_and(trans_constraints)

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables: bool = False
    ) -> z3.ExprRef:
        """Build invariant combining all domain views."""
        invariant_parts = []

        for domain in self.domains.values():
            if hasattr(domain, "build_invariant_expr"):
                part = domain.build_invariant_expr(model, use_prime_variables)
                invariant_parts.append(part)

        return big_and(invariant_parts) if invariant_parts else z3.BoolVal(True)
