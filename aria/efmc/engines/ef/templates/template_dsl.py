"""
DSL for defining templates (allow the user to define templates in a more declarative way)

This module provides a domain-specific language (DSL) for defining abstract domain templates
in a declarative manner, reducing code duplication and making template creation more intuitive.

Example Usage:
    @template_domain("interval")
    class MyIntervalTemplate:
        constraints = [
            LinearConstraint("lower", [0, 1], ">="),
            LinearConstraint("upper", [0, -1], ">=")
        ]
"""

from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional, Callable
from enum import Enum
import z3

from aria.efmc.engines.ef.templates.abstract_template import Template, TemplateType
from aria.efmc.sts import TransitionSystem
from aria.efmc.utils import big_and, big_or
from aria.utils.z3.bv import Signedness


class DomainType(Enum):
    """Types of abstract domains supported by the DSL."""

    INTERVAL = "interval"
    ZONE = "zone"
    AFFINE = "affine"
    POLYHEDRON = "polyhedron"
    BV_INTERVAL = "bv_interval"
    BV_AFFINE = "bv_affine"


@dataclass
class LinearConstraint:
    """Represents a linear constraint: sum(coeffs[i] * vars[i]) relation rhs"""

    name: str
    coeffs: List[Union[int, str]]  # Constants or template variable names
    relation: str  # "==", ">=", "<=", ">", "<"
    rhs: Union[int, str] = 0


@dataclass
class LogicalConstraint:
    """Base class for logical constraints."""

    pass


@dataclass
class Or(LogicalConstraint):
    constraints: List[Union["LogicalConstraint", "Equal"]]


@dataclass
class And(LogicalConstraint):
    constraints: List[Union["LogicalConstraint", "Equal"]]


@dataclass
class Equal(LogicalConstraint):
    var_name: str
    value: Union[int, str]


@dataclass
class TemplateSpec:
    """Specification for a template domain."""

    domain_type: DomainType
    constraints: List[LinearConstraint] = field(default_factory=list)
    additional_constraints: List[LogicalConstraint] = field(default_factory=list)
    num_templates: int = 1


class TemplateBuilder:
    """Builds template instances from declarative specifications."""

    def __init__(self, sts: TransitionSystem, spec: TemplateSpec):
        self.sts = sts
        self.spec = spec
        self.arity = len(sts.variables)

        # Determine variable type
        if spec.domain_type.value.startswith("bv_"):
            self.is_bv = True
            self.bv_size = sts.variables[0].sort().size() if sts.variables else 32
            self.signedness = (
                Signedness.SIGNED if sts.signedness == "signed" else Signedness.UNSIGNED
            )
        else:
            self.is_bv = False
            self.use_real = sts.has_real

    def create_variable(self, name: str) -> z3.ExprRef:
        """Create a Z3 variable of the appropriate type."""
        if self.is_bv:
            return z3.BitVec(name, self.bv_size)
        elif self.use_real:
            return z3.Real(name)
        else:
            return z3.Int(name)

    def generate_template_variables(self) -> List[List[z3.ExprRef]]:
        """Generate template variables based on domain type."""
        domain_generators = {
            DomainType.INTERVAL: self._gen_interval_vars,
            DomainType.BV_INTERVAL: self._gen_bv_interval_vars,
            DomainType.ZONE: self._gen_zone_vars,
            DomainType.AFFINE: self._gen_linear_vars,
            DomainType.POLYHEDRON: self._gen_linear_vars,
            DomainType.BV_AFFINE: self._gen_linear_vars,
        }
        return domain_generators.get(self.spec.domain_type, self._gen_linear_vars)()

    def _gen_interval_vars(self) -> List[List[z3.ExprRef]]:
        """Generate variables for interval domain: [lower_const, lower_coeff, upper_const, upper_coeff]"""
        return [
            [self.create_variable(f"i{v}_{i}") for i in range(4)]
            for v in self.sts.variables
        ]

    def _gen_bv_interval_vars(self) -> List[List[z3.ExprRef]]:
        """Generate variables for BV interval: [lower_bound, upper_bound]"""
        return [
            [self.create_variable(f"l_{v}"), self.create_variable(f"u_{v}")]
            for v in self.sts.variables
        ]

    def _gen_zone_vars(self) -> List[List[z3.ExprRef]]:
        """Generate variables for zone domain (x-y differences)."""
        import itertools

        template_vars = []
        for x, y in itertools.combinations(self.sts.variables, 2):
            name = f"{x}{y}"
            template_vars.append(
                [self.create_variable(f"i{name}_{i}") for i in range(4)]
            )
        return template_vars

    def _gen_linear_vars(self) -> List[List[z3.ExprRef]]:
        """Generate variables for linear templates: [const, coeff1, coeff2, ...]"""
        template_vars = []
        for i in range(self.spec.num_templates):
            tvars = [self.create_variable(f"p{i+1}_0")]  # constant
            tvars.extend(
                [self.create_variable(f"p{i+1}_{j}") for j in range(1, self.arity + 1)]
            )  # coefficients
            template_vars.append(tvars)
        return template_vars


class DSLTemplate(Template):
    """Template implementation that uses the DSL specification."""

    def __init__(self, sts: TransitionSystem, spec: TemplateSpec, **kwargs):
        self.sts = sts
        self.spec = spec

        # Map domain type to template type
        type_mapping = {
            DomainType.INTERVAL: TemplateType.INTERVAL,
            DomainType.ZONE: TemplateType.ZONE,
            DomainType.AFFINE: TemplateType.AFFINE,
            DomainType.POLYHEDRON: TemplateType.POLYHEDRON,
            DomainType.BV_INTERVAL: TemplateType.BV_INTERVAL,
            DomainType.BV_AFFINE: TemplateType.BV_AFFINE,
        }
        self.template_type = type_mapping.get(spec.domain_type, TemplateType.POLYHEDRON)

        # Build template using the DSL
        self.builder = TemplateBuilder(sts, spec)
        self.template_vars = self.builder.generate_template_variables()

        # Pre-compute constraints
        self.add_template_cnts()

    def add_template_vars(self) -> None:
        """Template variables are already generated by the builder."""
        pass

    def add_template_cnts(self) -> None:
        """Add constraints based on the DSL specification."""
        init_post_cnts = self._build_constraints(use_prime=False)
        trans_cnts = self._build_constraints(use_prime=True)

        self.template_cnt_init_and_post = (
            big_and(init_post_cnts) if init_post_cnts else z3.BoolVal(True)
        )
        self.template_cnt_trans = (
            big_and(trans_cnts) if trans_cnts else z3.BoolVal(True)
        )

    def _build_constraints(self, use_prime: bool) -> List[z3.ExprRef]:
        """Build constraints for the template."""
        variables = self.sts.prime_variables if use_prime else self.sts.variables
        constraints = []

        # Domain-specific constraint building
        if self.spec.domain_type == DomainType.INTERVAL:
            constraints.extend(self._build_interval_constraints(variables))
        elif self.spec.domain_type == DomainType.BV_INTERVAL:
            constraints.extend(self._build_bv_interval_constraints(variables))
        elif self.spec.domain_type == DomainType.ZONE:
            constraints.extend(self._build_zone_constraints(variables, use_prime))
        elif self.spec.domain_type in [
            DomainType.AFFINE,
            DomainType.POLYHEDRON,
            DomainType.BV_AFFINE,
        ]:
            constraints.extend(self._build_linear_constraints(variables))

        return constraints

    def _build_interval_constraints(
        self, variables: List[z3.ExprRef]
    ) -> List[z3.ExprRef]:
        """Build interval constraints: lower <= var <= upper"""
        constraints = []
        for i, var in enumerate(variables):
            i0, i1, i2, i3 = self.template_vars[i]
            constraints.extend(
                [i0 + var * i1 >= 0, i2 + var * i3 >= 0]  # lower bound  # upper bound
            )
        return constraints

    def _build_bv_interval_constraints(
        self, variables: List[z3.ExprRef]
    ) -> List[z3.ExprRef]:
        """Build bitvector interval constraints."""
        constraints = []
        for i, var in enumerate(variables):
            l_bound, u_bound = self.template_vars[i]
            if self.builder.signedness == Signedness.UNSIGNED:
                constraints.append(z3.And(z3.UGE(var, l_bound), z3.ULE(var, u_bound)))
            else:
                constraints.append(z3.And(var >= l_bound, var <= u_bound))
        return constraints

    def _build_zone_constraints(
        self, variables: List[z3.ExprRef], use_prime: bool
    ) -> List[z3.ExprRef]:
        """Build zone constraints for variable differences."""
        import itertools

        constraints = []
        zone_vars = list(itertools.combinations(variables, 2))

        for i, (x, y) in enumerate(zone_vars):
            diff = x - y
            i0, i1, i2, i3 = self.template_vars[i]
            constraints.extend([i0 + diff * i1 >= 0, i2 + diff * i3 >= 0])
        return constraints

    def _build_linear_constraints(
        self, variables: List[z3.ExprRef]
    ) -> List[z3.ExprRef]:
        """Build linear constraints: p0 + p1*x1 + ... + pn*xn relation 0"""
        constraints = []
        for tvars in self.template_vars:
            term = tvars[0]  # constant term
            for i, var in enumerate(variables):
                if i + 1 < len(tvars):
                    term = term + var * tvars[i + 1]

            # Default relation based on domain type
            if self.spec.domain_type == DomainType.AFFINE:
                constraints.append(term == 0)
            else:  # POLYHEDRON, BV_AFFINE
                constraints.append(term >= 0)

        return constraints

    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables: bool
    ) -> z3.ExprRef:
        """Build invariant expression from model."""
        variables = (
            self.sts.prime_variables if use_prime_variables else self.sts.variables
        )
        constraints = []

        if self.spec.domain_type == DomainType.BV_INTERVAL:
            # Simple BV interval invariant
            for i, var in enumerate(variables):
                l_val, u_val = (
                    model[self.template_vars[i][0]],
                    model[self.template_vars[i][1]],
                )
                if self.builder.signedness == Signedness.UNSIGNED:
                    constraints.append(z3.And(z3.UGE(var, l_val), z3.ULE(var, u_val)))
                else:
                    constraints.append(z3.And(var >= l_val, var <= u_val))
        else:
            # Generic linear invariant building
            for tvars in self.template_vars:
                term = model[tvars[0]]
                for i, var in enumerate(variables):
                    if i + 1 < len(tvars):
                        term = term + var * model[tvars[i + 1]]

                if self.spec.domain_type == DomainType.AFFINE:
                    constraints.append(term == 0)
                else:
                    constraints.append(term >= 0)

        return big_and(constraints) if constraints else z3.BoolVal(True)


def template_domain(domain_type: str, **kwargs):
    """Decorator for creating template domains using the DSL."""

    def decorator(cls):
        constraints = getattr(cls, "constraints", [])
        additional_constraints = getattr(cls, "additional_constraints", [])

        spec = TemplateSpec(
            domain_type=DomainType(domain_type),
            constraints=constraints,
            additional_constraints=additional_constraints,
            **kwargs,
        )

        class DSLTemplateClass(DSLTemplate):
            def __init__(self, sts: TransitionSystem, **init_kwargs):
                super().__init__(sts, spec, **init_kwargs)

        DSLTemplateClass.__name__ = cls.__name__
        return DSLTemplateClass

    return decorator


# Example templates using the DSL
@template_domain("interval")
class IntervalDSLTemplate:
    """Interval template: lower <= var <= upper"""

    pass


@template_domain("affine", num_templates=1)
class AffineDSLTemplate:
    """Affine template: p0 + p1*x1 + ... + pn*xn == 0"""

    pass


@template_domain("bv_interval")
class BitVecIntervalDSLTemplate:
    """Bitvector interval template"""

    pass


# Utility function
def create_template_from_spec(
    sts: TransitionSystem, spec: TemplateSpec, **kwargs
) -> Template:
    """Create a template instance from a specification."""
    return DSLTemplate(sts, spec, **kwargs)
