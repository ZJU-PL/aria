"""
Coordinate system for managing term-to-coordinate mappings.

This module provides functionality for creating and managing coordinate systems
that map symbolic terms to coordinate identifiers, supporting polynomial
representations and type analysis for symbolic computation.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Union, Optional, Any, Callable, Hashable
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import functools
import logging

# Import from other SRK modules
from aria.srk.syntax import Context, Symbol, Expression, Type, mk_const, mk_real, mk_add, mk_mul, mk_div, mk_mod, mk_floor, mk_app, typ_symbol, Var
from aria.srk.linear import QQVector
from aria.srk.polynomial import Polynomial, Monomial
from aria.srk.qQ import QQ

# Set up logging
logger = logging.getLogger(__name__)


class TermType(Enum):
    """Types of terms in coordinate systems."""
    TY_INT = "int"
    TY_REAL = "real"


class CSTermType(Enum):
    """Coordinate system term types - corresponds to OCaml cs_term variants."""
    MUL = "Mul"
    INV = "Inv"
    MOD = "Mod"
    FLOOR = "Floor"
    APP = "App"
    ONE = "One"
    ZERO = "Zero"


@dataclass(frozen=True)
class CSTerm(Hashable):
    """Coordinate system term - represents how coordinates are defined.

    This corresponds to the OCaml cs_term type with variants:
    - Mul of V.t * V.t
    - Inv of V.t
    - Mod of V.t * V.t
    - Floor of V.t
    - App of symbol * (V.t list)
    """
    term_type: CSTermType
    vectors: Tuple[QQVector, ...] = field(default_factory=tuple, hash=False)
    func: Optional[Symbol] = field(default=None, hash=False)
    args: Tuple[QQVector, ...] = field(default_factory=tuple, hash=False)

    def __post_init__(self):
        # Validate the term structure based on type
        if self.term_type == CSTermType.MUL:
            if len(self.vectors) != 2:
                raise ValueError("Mul requires exactly 2 vectors")
        elif self.term_type == CSTermType.INV:
            if len(self.vectors) != 1:
                raise ValueError("Inv requires exactly 1 vector")
        elif self.term_type == CSTermType.MOD:
            if len(self.vectors) != 2:
                raise ValueError("Mod requires exactly 2 vectors")
        elif self.term_type == CSTermType.FLOOR:
            if len(self.vectors) != 1:
                raise ValueError("Floor requires exactly 1 vector")
        elif self.term_type == CSTermType.APP:
            if self.func is None:
                raise ValueError("App requires a function symbol")
            if not self.args:  # For backwards compatibility, also check args
                object.__setattr__(self, 'args', self.vectors)
        elif self.term_type in [CSTermType.ONE, CSTermType.ZERO]:
            if len(self.vectors) != 0:
                raise ValueError("One/Zero should not have vectors")
        else:
            raise ValueError(f"Invalid term type: {self.term_type}")

    def __hash__(self):
        # Custom hash that matches OCaml's hash function
        if self.term_type == CSTermType.MUL:
            return hash((0, tuple(self.vectors[0].entries.items()), tuple(self.vectors[1].entries.items())))
        elif self.term_type == CSTermType.INV:
            return hash((1, tuple(self.vectors[0].entries.items())))
        elif self.term_type == CSTermType.MOD:
            return hash((2, tuple(self.vectors[0].entries.items()), tuple(self.vectors[1].entries.items())))
        elif self.term_type == CSTermType.FLOOR:
            return hash((3, tuple(self.vectors[0].entries.items())))
        elif self.term_type == CSTermType.APP:
            return hash((4, self.func, tuple(tuple(v.entries.items()) for v in self.args)))
        elif self.term_type == CSTermType.ONE:
            return hash(5)
        elif self.term_type == CSTermType.ZERO:
            return hash(6)
        else:
            raise ValueError(f"Unknown term type: {self.term_type}")

    @classmethod
    def mul(cls, x: QQVector, y: QQVector) -> 'CSTerm':
        """Create a multiplication term: x * y"""
        return cls(CSTermType.MUL, (x, y))

    @classmethod
    def inv(cls, x: QQVector) -> 'CSTerm':
        """Create an inverse term: 1/x"""
        return cls(CSTermType.INV, (x,))

    @classmethod
    def mod(cls, x: QQVector, y: QQVector) -> 'CSTerm':
        """Create a modulo term: x mod y"""
        return cls(CSTermType.MOD, (x, y))

    @classmethod
    def floor(cls, x: QQVector) -> 'CSTerm':
        """Create a floor term: floor(x)"""
        return cls(CSTermType.FLOOR, (x,))

    @classmethod
    def app(cls, func: Symbol, args: Tuple[QQVector, ...]) -> 'CSTerm':
        """Create a function application term: func(args...)"""
        return cls(CSTermType.APP, args, func)

    @classmethod
    def one(cls) -> 'CSTerm':
        """Create a one (constant 1) term."""
        return cls(CSTermType.ONE)

    @classmethod
    def zero(cls) -> 'CSTerm':
        """Create a zero (constant 0) term."""
        return cls(CSTermType.ZERO)


class CoordinateSystem:
    """Coordinate system for mapping terms to coordinates.

    This class manages the mapping between symbolic terms and coordinate identifiers,
    supporting polynomial representations and type analysis for symbolic computation.
    """

    # Constants (matching OCaml)
    CONST_ID = -1

    def __init__(self, context: Context):
        """Initialize a new coordinate system.

        Args:
            context: The SRK context for symbolic operations
        """
        self.context = context
        self.term_id: Dict[CSTerm, int] = {}
        self.id_def: List[Tuple[CSTerm, int, TermType]] = []
        self.next_id = 0

    @property
    def dim(self) -> int:
        """Number of dimensions in the coordinate system."""
        return len(self.id_def)

    @property
    def int_dim(self) -> int:
        """Number of integer dimensions."""
        return sum(1 for _, _, typ in self.id_def if typ == TermType.TY_INT)

    @property
    def real_dim(self) -> int:
        """Number of real dimensions."""
        return sum(1 for _, _, typ in self.id_def if typ == TermType.TY_REAL)

    def copy(self) -> 'CoordinateSystem':
        """Create a copy of the coordinate system."""
        new_cs = CoordinateSystem(self.context)
        new_cs.term_id = self.term_id.copy()
        new_cs.id_def = self.id_def.copy()
        new_cs.next_id = self.next_id
        return new_cs

    def admit_cs_term(self, term: CSTerm) -> int:
        """Admit a coordinate system term, returning its ID."""
        if term in self.term_id:
            return self.term_id[term]

        # Determine type and level
        typ, level = self._analyze_term_type_and_level(term)

        # Add to system
        self.id_def.append((term, level, typ))
        coord_id = self.next_id
        self.term_id[term] = coord_id
        self.next_id += 1

        return coord_id

    def _analyze_term_type_and_level(self, term: CSTerm) -> Tuple[TermType, int]:
        """Analyze the type and level of a coordinate system term."""
        if term.term_type == CSTermType.MUL:
            x_typ = self._get_vector_type(term.vectors[0])
            y_typ = self._get_vector_type(term.vectors[1])
            typ = self._join_types(x_typ, y_typ)
            x_level = self._get_vector_level(term.vectors[0])
            y_level = self._get_vector_level(term.vectors[1])
            level = max(x_level, y_level)

        elif term.term_type == CSTermType.INV:
            typ = TermType.TY_REAL
            level = self._get_vector_level(term.vectors[0])

        elif term.term_type == CSTermType.MOD:
            x_typ = self._get_vector_type(term.vectors[0])
            y_typ = self._get_vector_type(term.vectors[1])
            typ = self._join_types(x_typ, y_typ)
            x_level = self._get_vector_level(term.vectors[0])
            y_level = self._get_vector_level(term.vectors[1])
            level = max(x_level, y_level)

        elif term.term_type == CSTermType.FLOOR:
            typ = TermType.TY_INT
            level = self._get_vector_level(term.vectors[0])

        elif term.term_type == CSTermType.APP:
            # Determine function return type
            func_typ = self._get_function_type(term.func)
            typ = func_typ
            # Level is max of argument levels
            level = max((self._get_vector_level(arg) for arg in term.args), default=0)

        elif term.term_type in [CSTermType.ONE, CSTermType.ZERO]:
            # Constants have level 0
            typ = TermType.TY_REAL if term.term_type == CSTermType.ONE else TermType.TY_INT
            level = 0

        else:
            raise ValueError(f"Unknown term type: {term.term_type}")

        return typ, level

    def _get_vector_type(self, vec: QQVector) -> TermType:
        """Get the type of a vector."""
        # Check if vector represents an integer value
        is_integral = True
        for coord_id, coeff in vec.entries.items():
            if coord_id == -1:  # constant coefficient
                if not coeff.is_integer():
                    is_integral = False
            else:
                coord_typ = self._get_coordinate_type(coord_id)
                if coord_typ != TermType.TY_INT:
                    is_integral = False

        return TermType.TY_INT if is_integral else TermType.TY_REAL

    def _get_vector_level(self, vec: QQVector) -> int:
        """Get the level of a vector (max coordinate level)."""
        max_level = -1
        for coord_id, _ in vec.entries.items():
            if coord_id != -1:  # not constant
                coord_level = self._get_coordinate_level(coord_id)
                max_level = max(max_level, coord_level)
        return max_level

    def _get_coordinate_type(self, coord_id: int) -> TermType:
        """Get the type of a coordinate."""
        if 0 <= coord_id < len(self.id_def):
            return self.id_def[coord_id][2]
        return TermType.TY_REAL  # default

    def _get_coordinate_level(self, coord_id: int) -> int:
        """Get the level of a coordinate."""
        if 0 <= coord_id < len(self.id_def):
            return self.id_def[coord_id][1]
        return 0  # default

    def _get_function_type(self, func: Optional[Symbol]) -> TermType:
        """Get the return type of a function symbol."""
        if func is None:
            return TermType.TY_REAL

        # Get the symbol's type from the context
        try:
            symbol_type = typ_symbol(self.context, func)
            if symbol_type == Type.INT:
                return TermType.TY_INT
            elif symbol_type == Type.REAL:
                return TermType.TY_REAL
            elif symbol_type == Type.BOOL:
                return TermType.TY_INT  # Booleans are treated as integers (0/1)
            else:
                return TermType.TY_REAL  # Default for unknown types
        except Exception:
            # Fallback if symbol type cannot be determined
            return TermType.TY_REAL

    def _join_types(self, typ1: TermType, typ2: TermType) -> TermType:
        """Join two types (Int + Real = Real)."""
        if typ1 == TermType.TY_INT and typ2 == TermType.TY_INT:
            return TermType.TY_INT
        return TermType.TY_REAL

    def destruct_coordinate(self, coord_id: int) -> CSTerm:
        """Get the coordinate system term for a coordinate ID."""
        if 0 <= coord_id < len(self.id_def):
            return self.id_def[coord_id][0]
        raise ValueError(f"Invalid coordinate ID: {coord_id}")

    def term_of_coordinate(self, coord_id: int) -> Expression:
        """Convert a coordinate ID to its corresponding arithmetic term.

        This corresponds to the OCaml term_of_coordinate function.
        """
        term = self.destruct_coordinate(coord_id)

        if term.term_type == CSTermType.MUL:
            x_term = self._vector_to_term(term.vectors[0])
            y_term = self._vector_to_term(term.vectors[1])
            return self.context.mk_mul([x_term, y_term])

        elif term.term_type == CSTermType.INV:
            # 1 / x
            x_term = self._vector_to_term(term.vectors[0])
            one_const = mk_real(self.context, QQ.one())
            return self.context.mk_div(one_const, x_term)

        elif term.term_type == CSTermType.MOD:
            # x mod y
            x_term = self._vector_to_term(term.vectors[0])
            y_term = self._vector_to_term(term.vectors[1])
            return self.context.mk_mod(x_term, y_term)

        elif term.term_type == CSTermType.FLOOR:
            # floor(x)
            x_term = self._vector_to_term(term.vectors[0])
            return self.context.mk_floor(x_term)

        elif term.term_type == CSTermType.APP:
            arg_terms = [self._vector_to_term(arg) for arg in term.args]
            return self.context.mk_app(term.func, arg_terms)

        elif term.term_type == CSTermType.ONE:
            return mk_real(self.context, QQ.one())

        elif term.term_type == CSTermType.ZERO:
            return mk_real(self.context, QQ.zero())

        else:
            raise ValueError(f"Unknown term type: {term.term_type}")

    def vec_of_term(self, term: Expression, admit: bool = False) -> QQVector:
        """Convert an arithmetic term to a vector representation.

        This corresponds to the OCaml vec_of_term function.

        Args:
            term: The arithmetic expression to convert
            admit: If True, admit new terms into the coordinate system

        Returns:
            A vector representing the term

        Raises:
            KeyError: If the term cannot be represented and admit=False
        """
        # Use the recursive algorithm to convert the term
        return self._vec_of_term_recursive(term, admit)

    def _vec_of_term_recursive(self, term: Expression, admit: bool) -> QQVector:
        """Recursive implementation of vec_of_term algorithm."""
        from .syntax import Const, Var, App, Add, Mul

        # Handle constants
        const_val = self._extract_constant(term)
        if const_val is not None:
            return QQVector({self.CONST_ID: const_val})

        # Handle variables (Var expressions)
        if isinstance(term, Var):
            # Variables are represented as function applications with no arguments
            # Create a symbol for this variable - we need to create a unique symbol
            # For now, use the var_id as the symbol id and var_type as the type
            symbol = Symbol(term.var_id, None, term.var_type)
            cs_term = CSTerm.app(symbol, ())
            coord_id = self._cs_term_id(cs_term, admit)
            return QQVector({coord_id: QQ.one()})

        # Handle function applications (App expressions)
        if isinstance(term, App):
            # Convert arguments first
            arg_vectors = []
            for arg in term.args:
                # Check if argument is an arithmetic term (not array or formula)
                arg_vec = self._vec_of_term_recursive(arg, admit)
                arg_vectors.append(arg_vec)

            # Create coordinate system term for the application
            cs_term = CSTerm.app(term.symbol, tuple(arg_vectors))
            coord_id = self._cs_term_id(cs_term, admit)
            return QQVector({coord_id: QQ.one()})

        # Handle addition (Add expressions)
        if isinstance(term, Add):
            if not term.args:
                return QQVector({self.CONST_ID: QQ.zero()})
            elif len(term.args) == 1:
                return self._vec_of_term_recursive(term.args[0], admit)
            else:
                # Sum all arguments: arg1 + arg2 + ... + argn
                result_vec = QQVector()
                for arg in term.args:
                    arg_vec = self._vec_of_term_recursive(arg, admit)
                    result_vec = result_vec.add(arg_vec)
                return result_vec

        # Handle multiplication (Mul expressions)
        if isinstance(term, Mul):
            if not term.args:
                return QQVector({self.CONST_ID: QQ.one()})
            elif len(term.args) == 1:
                return self._vec_of_term_recursive(term.args[0], admit)
            else:
                # For multiple multiplication: arg1 * arg2 * ... * argn
                # We need to chain the multiplications properly
                arg_vectors = [self._vec_of_term_recursive(arg, admit) for arg in term.args]

                # Start with the first argument
                result_vec = arg_vectors[0]

                # Multiply with each subsequent argument
                for arg_vec in arg_vectors[1:]:
                    cs_term = CSTerm.mul(result_vec, arg_vec)
                    coord_id = self._cs_term_id(cs_term, admit)
                    result_vec = QQVector({coord_id: QQ.one()})

                return result_vec

        # For other expression types, create a generic coordinate
        if admit:
            # Create a coordinate for expression types we don't fully support yet
            # This includes Div, Mod, Floor, and other complex operations
            # For now, we represent them as a constant (1) but this should be improved
            # to properly handle the specific operation type
            cs_term = CSTerm.one()
            coord_id = self.admit_cs_term(cs_term)
            return QQVector({coord_id: QQ.one()})

        raise KeyError(f"Cannot convert term to vector: {term} (unsupported expression type: {type(term)})")

    def _cs_term_id(self, cs_term: CSTerm, admit: bool) -> int:
        """Get or admit a coordinate system term ID."""
        if cs_term in self.term_id:
            return self.term_id[cs_term]

        if admit:
            return self.admit_cs_term(cs_term)
        else:
            raise KeyError(f"Coordinate system term not found: {cs_term}")

    def admits(self, term: Expression) -> bool:
        """Check if the coordinate system admits a given term.

        This corresponds to the OCaml admits function.
        """
        try:
            self.vec_of_term(term, admit=False)
            return True
        except (KeyError, ValueError):
            return False

    def polynomial_of_coordinate(self, coord_id: int) -> Polynomial:
        """Get polynomial representation of a coordinate.

        This corresponds to the OCaml polynomial_of_coordinate function.
        """
        term = self.destruct_coordinate(coord_id)

        if term.term_type == CSTermType.MUL:
            x_poly = self.polynomial_of_vec(term.vectors[0])
            y_poly = self.polynomial_of_vec(term.vectors[1])
            return x_poly * y_poly
        else:
            # For non-multiplicative terms, create a polynomial with the coordinate as a variable
            return Polynomial.of_dim(coord_id)

    def polynomial_of_vec(self, vec: QQVector) -> Polynomial:
        """Convert a vector to a polynomial.

        This corresponds to the OCaml polynomial_of_vec function.
        """
        const_coeff, rest_vec = vec.entries.get(self.CONST_ID, QQ.zero()), QQVector(
            {k: v for k, v in vec.entries.items() if k != self.CONST_ID}
        )

        poly = Polynomial.scalar(const_coeff)

        for coord_id, coeff in rest_vec.entries.items():
            coord_poly = self.polynomial_of_coordinate(coord_id)
            term_poly = coord_poly * coeff
            poly = poly + term_poly

        return poly

    def _extract_constant(self, term: Expression) -> Optional[QQ]:
        """Extract a constant value from a term if possible."""
        from .syntax import Const
        from fractions import Fraction

        # Check if this is a constant expression
        if isinstance(term, Const):
            # Try to extract the numeric value from the constant symbol
            symbol = term.symbol
            if symbol.name and symbol.name.startswith("real_"):
                try:
                    # Extract the numeric value from "real_X" format
                    value_str = symbol.name[5:]  # Remove "real_" prefix
                    return Fraction(float(value_str))
                except (ValueError, IndexError):
                    pass

            # Check if it's an integer constant (name like "int_X")
            if symbol.name and symbol.name.startswith("int_"):
                try:
                    value_str = symbol.name[4:]  # Remove "int_" prefix
                    return Fraction(int(value_str))
                except (ValueError, IndexError):
                    pass

        # For other expression types, return None
        return None

    def _vector_to_term(self, vec: QQVector) -> Expression:
        """Convert a vector to an arithmetic term.

        This corresponds to the OCaml term_of_vec function.
        """
        terms = []

        for coord_id, coeff in vec.entries.items():
            if coord_id == self.CONST_ID:  # constant
                terms.append(mk_real(self.context, coeff))
            elif QQ.equal(coeff, QQ.one()):
                terms.append(self.term_of_coordinate(coord_id))
            else:
                coeff_term = mk_real(self.context, coeff)
                coord_term = self.term_of_coordinate(coord_id)
                terms.append(self.context.mk_mul([coeff_term, coord_term]))

        if not terms:
            return mk_real(self.context, QQ.zero())
        elif len(terms) == 1:
            return terms[0]
        else:
            return self.context.mk_add(terms)

    def _type_string(self, typ: TermType) -> str:
        """Convert TermType to string for logging."""
        return typ.value

    def type_of_id(self, coord_id: int) -> TermType:
        """Get the type of a coordinate."""
        if 0 <= coord_id < len(self.id_def):
            return self.id_def[coord_id][2]
        raise ValueError(f"Invalid coordinate ID: {coord_id}")

    def type_of_vec(self, vec: QQVector) -> TermType:
        """Get the type of a vector."""
        return self._get_vector_type(vec)

    def type_of_monomial(self, monomial: Monomial) -> TermType:
        """Get the type of a monomial."""
        is_integral = True
        for coord_id, _ in monomial.entries.items():
            if self.type_of_id(coord_id) != TermType.TY_INT:
                is_integral = False
                break
        return TermType.TY_INT if is_integral else TermType.TY_REAL

    def type_of_polynomial(self, poly: Polynomial) -> TermType:
        """Get the type of a polynomial."""
        is_integral = True
        for coeff, monomial in poly.entries.items():
            if not coeff.is_integer() or self.type_of_monomial(monomial) != TermType.TY_INT:
                is_integral = False
                break
        return TermType.TY_INT if is_integral else TermType.TY_REAL

    def level_of_id(self, coord_id: int) -> int:
        """Get the level of a coordinate."""
        if 0 <= coord_id < len(self.id_def):
            return self.id_def[coord_id][1]
        return 0  # default

    def level_of_vec(self, vec: QQVector) -> int:
        """Get the level of a vector (max coordinate level)."""
        return self._get_vector_level(vec)

    def cs_term_id(self, cs_term: CSTerm, admit: bool = False) -> int:
        """Get the coordinate ID for a coordinate system term.

        This is equivalent to the OCaml cs_term_id function.
        """
        if cs_term in self.term_id:
            return self.term_id[cs_term]

        if admit:
            return self.admit_cs_term(cs_term)
        else:
            raise KeyError(f"Coordinate system term not found: {cs_term}")

    def admit_term(self, term: Expression) -> None:
        """Admit a term into the coordinate system."""
        self.vec_of_term(term, admit=True)

    def subcoordinates(self, coord_id: int) -> Set[int]:
        """Find all subcoordinates of a given coordinate."""
        result = set()

        def add_subcoordinates(term: CSTerm, coord_set: Set[int]):
            if term.term_type == CSTermType.MUL:
                for vec in term.vectors:
                    for sub_coord, _ in vec.entries.items():
                        if sub_coord != self.CONST_ID:
                            coord_set.add(sub_coord)
                            # Recursively add subcoordinates of the sub-coordinate
                            sub_term = self.destruct_coordinate(sub_coord)
                            add_subcoordinates(sub_term, coord_set)
            elif term.term_type in [CSTermType.INV, CSTermType.FLOOR]:
                for sub_coord, _ in term.vectors[0].entries.items():
                    if sub_coord != self.CONST_ID:
                        coord_set.add(sub_coord)
                        sub_term = self.destruct_coordinate(sub_coord)
                        add_subcoordinates(sub_term, coord_set)
            elif term.term_type == CSTermType.MOD:
                for vec in term.vectors:
                    for sub_coord, _ in vec.entries.items():
                        if sub_coord != self.CONST_ID:
                            coord_set.add(sub_coord)
                            sub_term = self.destruct_coordinate(sub_coord)
                            add_subcoordinates(sub_term, coord_set)
            elif term.term_type == CSTermType.APP:
                for vec in term.args:
                    for sub_coord, _ in vec.entries.items():
                        if sub_coord != self.CONST_ID:
                            coord_set.add(sub_coord)
                            sub_term = self.destruct_coordinate(sub_coord)
                            add_subcoordinates(sub_term, coord_set)

        term = self.destruct_coordinate(coord_id)
        add_subcoordinates(term, result)
        result.discard(self.CONST_ID)  # Remove constant ID
        return result

    def direct_subcoordinates(self, coord_id: int) -> Set[int]:
        """Find direct subcoordinates of a given coordinate."""
        result = set()
        term = self.destruct_coordinate(coord_id)

        if term.term_type == CSTermType.MUL:
            for vec in term.vectors:
                for sub_coord, _ in vec.entries.items():
                    if sub_coord != self.CONST_ID:
                        result.add(sub_coord)
        elif term.term_type in [CSTermType.INV, CSTermType.FLOOR]:
            for sub_coord, _ in term.vectors[0].entries.items():
                if sub_coord != self.CONST_ID:
                    result.add(sub_coord)
        elif term.term_type == CSTermType.MOD:
            for vec in term.vectors:
                for sub_coord, _ in vec.entries.items():
                    if sub_coord != self.CONST_ID:
                        result.add(sub_coord)
        elif term.term_type == CSTermType.APP:
            for vec in term.args:
                for sub_coord, _ in vec.entries.items():
                    if sub_coord != self.CONST_ID:
                        result.add(sub_coord)

        result.discard(self.CONST_ID)  # Remove constant ID
        return result

    def __repr__(self) -> str:
        """String representation of the coordinate system."""
        lines = []
        for i, (term, level, typ) in enumerate(self.id_def):
            lines.append(f"{i} -> {term} (level={level}, type={typ.value})")
        return f"CoordinateSystem([\n  {chr(10).join(lines)}\n])"

    def __str__(self) -> str:
        """String representation for pretty printing."""
        return self.__repr__()


# Pretty printing functions (matching OCaml)
def pp_cs_term(cs: CoordinateSystem, term: CSTerm) -> str:
    """Pretty print a coordinate system term."""
    if term.term_type == CSTermType.MUL:
        x_str = pp_vector(cs, term.vectors[0])
        y_str = pp_vector(cs, term.vectors[1])
        return f"({x_str}) * ({y_str})"
    elif term.term_type == CSTermType.INV:
        x_str = pp_vector(cs, term.vectors[0])
        return f"1/({x_str})"
    elif term.term_type == CSTermType.MOD:
        x_str = pp_vector(cs, term.vectors[0])
        y_str = pp_vector(cs, term.vectors[1])
        return f"({x_str}) mod ({y_str})"
    elif term.term_type == CSTermType.FLOOR:
        x_str = pp_vector(cs, term.vectors[0])
        return f"floor({x_str})"
    elif term.term_type == CSTermType.APP:
        if not term.args:
            return str(term.func)
        else:
            arg_strs = [pp_vector(cs, arg) for arg in term.args]
            return f"{term.func}({', '.join(arg_strs)})"
    elif term.term_type == CSTermType.ONE:
        return "1"
    elif term.term_type == CSTermType.ZERO:
        return "0"
    else:
        return str(term)


def pp_vector(cs: CoordinateSystem, vec: QQVector) -> str:
    """Pretty print a vector."""
    if not vec.entries:
        return "0"

    terms = []
    for coord_id, coeff in vec.entries.items():
        if coord_id == cs.CONST_ID:
            terms.append(str(coeff))
        elif QQ.equal(coeff, QQ.one()):
            coord_term = cs.term_of_coordinate(coord_id)
            terms.append(str(coord_term))
        else:
            coord_term = cs.term_of_coordinate(coord_id)
            terms.append(f"{coeff} * {coord_term}")

    if not terms:
        return "0"
    elif len(terms) == 1:
        return terms[0]
    else:
        return " + ".join(terms)


# Convenience functions
def mk_empty(context: Context) -> CoordinateSystem:
    """Create an empty coordinate system."""
    return CoordinateSystem(context)


def get_context(cs: CoordinateSystem) -> Context:
    """Get the context from a coordinate system."""
    return cs.context
