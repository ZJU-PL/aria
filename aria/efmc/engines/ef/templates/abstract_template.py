"""
This module contains the abstract base class for template-based verification.

The Template class is an abstract base class that defines the interface for
template-based program verification. It provides methods for managing template
variables, constraints, and building invariants and ranking functions.
"""

from enum import Enum, auto
from typing import Optional, List
from abc import ABC, abstractmethod
import z3


class TemplateType(Enum):
    """Enumeration of supported template types for program verification.

    Categories:
    - Basic: Simple numeric domains (INTERVAL, ZONE, etc.)
    - Disjunctive: Unions of basic domains
    - Bit-Vector: Templates for bit-vector arithmetic
    - Special: Arrays, strings, floating-point
    """

    # Basic numeric domains
    INTERVAL = auto()
    ZONE = auto()
    OCTAGON = auto()
    AFFINE = auto()
    POLYHEDRON = auto()
    POLYNOMIAL = auto()

    # Disjunctive domains
    DISJUNCTIVE_INTERVAL = auto()
    DISJUNCTIVE_ZONE = auto()
    DISJUNCTIVE_OCTAGON = auto()
    DISJUNCTIVE_AFFINE = auto()
    DISJUNCTIVE_POLYHEDRON = auto()
    DISJUNCTIVE_POLYNOMIAL = auto()
    # Special domains
    ARRAY = auto()
    STRING = auto()
    FLOAT = auto()

    # Floating-point specific domains
    FP_INTERVAL = auto()
    FP_POLYHEDRON = auto()

    # Bit-vector domains
    BV_INTERVAL = auto()
    BV_ZONE = auto()
    BV_OCTAGON = auto()
    BV_AFFINE = auto()
    BV_POLYHEDRON = auto()
    BV_DISJUNCTIVE_INTERVAL = auto()
    BV_DISJUNCTIVE_ZONE = auto()
    BV_DISJUNCTIVE_OCTAGON = auto()
    BV_DISJUNCTIVE_AFFINE = auto()
    BV_DISJUNCTIVE_POLYHEDRON = auto()
    BV_KNOWNBITS = auto()
    BV_BITS_PREDICATE_ABSTRACTION = auto()
    BV_ENHANCED_PATTERN = auto()

    # Bit-level domains
    BV_PATTERN = auto()
    BV_ROTATION = auto()
    BV_XOR_PARITY = auto()
    BV_DISJUNCTIVE_XOR_PARITY = auto()
    BV_DISJUNCTIVE_PATTERN = auto()
    BV_DISJUNCTIVE_ROTATION = auto()

    @classmethod
    def is_disjunctive(cls, template_type: "TemplateType") -> bool:
        """Check if the template type is disjunctive."""
        return "DISJUNCTIVE" in template_type.name

    @classmethod
    def is_bitvector(cls, template_type: "TemplateType") -> bool:
        """Check if the template type uses bit-vector arithmetic."""
        return template_type.name.startswith("BV_")

    @classmethod
    def is_interval(cls, template_type: "TemplateType") -> bool:
        """Check if the template type is an interval."""
        return template_type.name.startswith("INTERVAL")

    @classmethod
    def is_floating_point(cls, template_type: "TemplateType") -> bool:
        """Check if the template type is floating-point specific."""
        return template_type.name.startswith("FP_")


class Template(ABC):
    """Abstract base class for template-based verification.

    This class defines the interface for template-based program verification,
    including methods for managing template variables, constraints, and
    building invariants and ranking functions.
    """

    @abstractmethod
    def add_template_vars(self) -> None:
        """Initialize the template variables.

        This method should set up all necessary template variables
        required for the verification process.
        """
        pass

    @abstractmethod
    def add_template_cnts(self) -> None:
        """Add constraints for the template variables.

        Adds constraints according to the specification of inductive loop invariant.
        These constraints define the shape and properties of the invariant.
        """
        pass

    @abstractmethod
    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables: bool
    ) -> z3.ExprRef:
        """Build an invariant expression from a model.

        Args:
            model: Z3 model containing variable assignments
            use_prime_variables: If True, use primed variables (x', y'), otherwise use (x, y)

        Returns:
            Z3 expression representing the invariant
            If use_prime_variables is True, the expression should be in terms of the primed variables.
            Else, the expression should be in terms of the original variables.
        """
        pass

    def build_invariant(self, model: z3.ModelRef) -> z3.ExprRef:
        """Build an invariant from a model using current state variables.

        This is a convenience method that calls build_invariant_expr with
        use_prime_variables=False.

        Args:
            model: Z3 model containing variable assignments

        Returns:
            Z3 expression representing the invariant in terms of current state variables
        """
        return self.build_invariant_expr(model, use_prime_variables=False)

    @staticmethod
    def _init_signedness_from_sts(sts) -> Optional["Signedness"]:
        """Helper method to initialize signedness from a transition system.

        Args:
            sts: Transition system with signedness attribute

        Returns:
            Signedness enum value or None if not applicable
        """
        try:
            from aria.efmc.utils.bv_utils import Signedness

            if sts.signedness == "signed":
                return Signedness.SIGNED
            elif sts.signedness == "unsigned":
                return Signedness.UNSIGNED
        except (ImportError, AttributeError):
            pass
        return None

    def _build_linear_term(
        self,
        template_vars: List[z3.ExprRef],
        variables: List[z3.ExprRef],
        start_idx: int = 0,
    ) -> z3.ExprRef:
        """Helper method to build a linear term from template variables.

        Builds a term of the form: template_vars[start_idx] +
                                   variables[0] * template_vars[start_idx+1] +
                                   variables[1] * template_vars[start_idx+2] + ...

        Args:
            template_vars: List of template variable expressions
            variables: List of program variables
            start_idx: Starting index in template_vars (default: 0)

        Returns:
            Z3 expression representing the linear term
        """
        term = template_vars[start_idx]
        for j in range(len(variables)):
            term = term + variables[j] * template_vars[start_idx + j + 1]
        return term

    def _get_variable(self, index: int, use_prime_variables: bool) -> z3.ExprRef:
        """Helper method to get a variable (prime or non-prime) by index.

        Args:
            index: Index of the variable
            use_prime_variables: If True, return prime variable, else return regular variable

        Returns:
            Z3 variable expression
        """
        if use_prime_variables:
            return self.sts.prime_variables[index]
        else:
            return self.sts.variables[index]
