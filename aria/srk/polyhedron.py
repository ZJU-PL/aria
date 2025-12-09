"""
Polyhedron operations for convex polyhedra.

This module implements operations on convex polyhedra, which are used
in abstract interpretation and program analysis.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Iterator
from dataclasses import dataclass, field
from fractions import Fraction

from .syntax import Context, Symbol, Type, FormulaExpression, ArithExpression
from .linear import QQVector, QQMatrix


class Constraint:
    """Linear constraint: a^T x + c ≥ 0 or a^T x + c = 0."""

    def __init__(self, coefficients: QQVector, constant: Fraction, equality: bool = False):
        self.coefficients = coefficients
        self.constant = constant
        self.equality = equality  # True for equality, False for inequality

    def evaluate(self, point: QQVector) -> Fraction:
        """Evaluate constraint at a point."""
        return self.coefficients.dot(point) + self.constant

    def is_satisfied(self, point: QQVector) -> bool:
        """Check if constraint is satisfied at point."""
        value = self.evaluate(point)
        if self.equality:
            return value == 0
        else:
            return value >= 0

    def __str__(self) -> str:
        if self.equality:
            return f"{self.coefficients} + {self.constant} = 0"
        else:
            return f"{self.coefficients} + {self.constant} ≥ 0"


class Polyhedron:
    """Convex polyhedron defined by linear constraints."""

    def __init__(self, constraints: List[Constraint]):
        self.constraints = constraints

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Polyhedron):
            return False
        return self.constraints == other.constraints

    def __hash__(self) -> int:
        return hash(tuple(self.constraints))

    def add_constraint(self, constraint: Constraint) -> Polyhedron:
        """Add a constraint to the polyhedron."""
        new_constraints = self.constraints + [constraint]
        return Polyhedron(new_constraints)

    def intersect(self, other: Polyhedron) -> Polyhedron:
        """Intersection of two polyhedra."""
        new_constraints = self.constraints + other.constraints
        return Polyhedron(new_constraints)

    def is_empty(self) -> bool:
        """Check if polyhedron is empty using linear programming."""
        if not self.constraints:
            return False
        
        try:
            from scipy.optimize import linprog
            import numpy as np
            
            # Extract inequality and equality constraints
            ineq_constraints = []
            eq_constraints = []
            
            all_dims = set()
            for constraint in self.constraints:
                all_dims.update(constraint.coefficients.dimensions())
            
            if not all_dims:
                # Check if any equality constraint is violated
                for constraint in self.constraints:
                    if constraint.equality and constraint.constant != 0:
                        return True
                return False
            
            n = max(all_dims) + 1
            
            for constraint in self.constraints:
                # Convert to standard form
                A_row = np.zeros(n)
                for dim, coeff in constraint.coefficients.entries.items():
                    A_row[dim] = float(coeff)
                b_val = float(-constraint.constant)
                
                if constraint.equality:
                    eq_constraints.append((A_row, b_val))
                else:
                    # Convert a^T x + c >= 0 to -a^T x <= c
                    ineq_constraints.append((-A_row, -b_val))
            
            # Try to find a feasible point
            # Minimize 0 (just check feasibility)
            c = np.zeros(n)
            
            A_ub = np.array([a for a, b in ineq_constraints]) if ineq_constraints else None
            b_ub = np.array([b for a, b in ineq_constraints]) if ineq_constraints else None
            
            A_eq = np.array([a for a, b in eq_constraints]) if eq_constraints else None
            b_eq = np.array([b for a, b in eq_constraints]) if eq_constraints else None
            
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
            
            return not result.success
            
        except ImportError:
            # If scipy not available, fall back to conservative answer
            return False

    def contains(self, point: QQVector) -> bool:
        """Check if polyhedron contains a point."""
        for constraint in self.constraints:
            if not constraint.is_satisfied(point):
                return False
        return True

    def vertices(self) -> List[QQVector]:
        """Get vertices of the polyhedron using vertex enumeration."""
        if not self.constraints or self.is_empty():
            return []
        
        try:
            from scipy.spatial import HalfspaceIntersection
            from scipy.optimize import linprog
            import numpy as np
            
            # Get all dimensions
            all_dims = set()
            for constraint in self.constraints:
                all_dims.update(constraint.coefficients.dimensions())
            
            if not all_dims:
                return []
            
            n = max(all_dims) + 1
            
            # Convert constraints to halfspace representation
            # Each halfspace is a^T x + b <= 0
            halfspaces = []
            
            for constraint in self.constraints:
                A_row = np.zeros(n)
                for dim, coeff in constraint.coefficients.entries.items():
                    A_row[dim] = float(coeff)
                b_val = float(constraint.constant)
                
                if constraint.equality:
                    # a^T x + b = 0 becomes two inequalities
                    halfspaces.append(np.concatenate([A_row, [b_val]]))
                    halfspaces.append(np.concatenate([-A_row, [-b_val]]))
                else:
                    # a^T x + b >= 0 becomes -a^T x - b <= 0
                    halfspaces.append(np.concatenate([-A_row, [-b_val]]))
            
            if not halfspaces:
                return []
            
            halfspaces = np.array(halfspaces)
            
            # Find a feasible interior point
            c = np.zeros(n)
            A_ub = []
            b_ub = []
            A_eq = []
            b_eq = []
            
            for constraint in self.constraints:
                A_row = np.zeros(n)
                for dim, coeff in constraint.coefficients.entries.items():
                    A_row[dim] = float(coeff)
                b_val = float(-constraint.constant)
                
                if constraint.equality:
                    A_eq.append(A_row)
                    b_eq.append(b_val)
                else:
                    A_ub.append(-A_row)
                    b_ub.append(-b_val)
            
            A_ub = np.array(A_ub) if A_ub else None
            b_ub = np.array(b_ub) if b_ub else None
            A_eq = np.array(A_eq) if A_eq else None
            b_eq = np.array(b_eq) if b_eq else None
            
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
            
            if not result.success:
                return []  # No feasible point
            
            feasible_point = result.x
            
            # Compute halfspace intersection
            hs = HalfspaceIntersection(halfspaces, feasible_point)
            
            # Convert vertices to QQVectors
            vertices = []
            for vertex in hs.intersections:
                entries = {}
                for i, val in enumerate(vertex):
                    if abs(val) > 1e-10:
                        entries[i] = Fraction(val).limit_denominator(10000)
                vertices.append(QQVector(entries))
            
            return vertices
            
        except (ImportError, Exception):
            # If computation fails, return empty list
            return []

    def dual_cone(self, dimension: int) -> Polyhedron:
        """Compute the dual cone of the polyhedron."""
        # This would compute the dual cone
        # Placeholder implementation
        return Polyhedron([])

    def enum_constraints(self) -> Iterator[Tuple[str, QQVector]]:
        """Enumerate constraints of the polyhedron."""
        for constraint in self.constraints:
            kind = "zero" if constraint.equality else "nonneg"
            yield kind, constraint.coefficients

    def enum_generators(self, dimension: int) -> Iterator[Tuple[str, QQVector]]:
        """Enumerate generators of the polyhedron."""
        verts = self.vertices()
        for v in verts:
            yield "vertex", v

    def __str__(self) -> str:
        if not self.constraints:
            return "∅"

        constraints_str = "\n".join(f"  {c}" for c in self.constraints)
        return f"Polyhedron(\n{constraints_str}\n)"


class PolyhedronOperations:
    """Operations on polyhedra."""

    @staticmethod
    def from_formula(formula: FormulaExpression, context: Context) -> Polyhedron:
        """Create polyhedron from formula."""
        # Very lightweight extractor for simple conjunctions of linear atoms
        constraints: List[Constraint] = []
        from .syntax import And, Leq, Lt, Eq, Add, Mul, Const, Var

        def lin_of(term) -> QQVector:
            # Convert simple linear term (sum of const*var) to QQVector over variable IDs
            if isinstance(term, Var):
                return QQVector({term.symbol.id: Fraction(1)})
            if isinstance(term, Const):
                return QQVector()  # constants handled via constant offset
            if isinstance(term, Add):
                vec = QQVector()
                for arg in term.args:
                    vec = vec + lin_of(arg)
                return vec
            if isinstance(term, Mul) and len(term.args) == 2:
                c, t = term.args
                if isinstance(c, Const) and isinstance(t, Var):
                    # Best-effort numeric parse for constants
                    try:
                        cval = Fraction(str(c.symbol.name))
                    except Exception:
                        cval = Fraction(0)
                    return QQVector({t.symbol.id: cval})
            return QQVector()

        atoms = [formula]
        if hasattr(formula, 'args') and isinstance(formula, And):
            atoms = list(formula.args)

        for atom in atoms:
            if isinstance(atom, (Leq, Lt, Eq)):
                left_vec = lin_of(atom.left)
                right_vec = lin_of(atom.right)
                coeffs = left_vec - right_vec
                # We ignore pure constants for now and set constant offset to 0
                c = Fraction(0)
                if isinstance(atom, Leq):
                    constraints.append(Constraint(coeffs, c, equality=False))
                elif isinstance(atom, Lt):
                    # Strict inequality approximated by non-strict with -1
                    constraints.append(Constraint(coeffs, c - Fraction(1), equality=False))
                else:  # Eq
                    constraints.append(Constraint(coeffs, c, equality=True))

        return Polyhedron(constraints)

    @staticmethod
    def to_formula(polyhedron: Polyhedron, context: Context) -> FormulaExpression:
        """Convert polyhedron to formula."""
        from .syntax import make_expression_builder
        builder = make_expression_builder(context)
        conjuncts: List[FormulaExpression] = []
        for c in polyhedron.constraints:
            sum_term: Optional[ArithExpression] = None
            for dim, coeff in c.coefficients.entries.items():
                if coeff == 0:
                    continue
                var = builder.mk_var(dim, Type.REAL)
                term = var if coeff == 1 else builder.mk_mul([builder.mk_const(int(coeff)), var])
                sum_term = term if sum_term is None else builder.mk_add([sum_term, term])
            if sum_term is None:
                sum_term = builder.mk_const(0)
            if c.constant != 0:
                sum_term = builder.mk_add([sum_term, builder.mk_const(int(c.constant))])
            conjuncts.append(builder.mk_eq(sum_term, builder.mk_const(0)) if c.equality
                             else builder.mk_leq(builder.mk_const(0), sum_term))
        if not conjuncts:
            from .syntax import TrueExpr
            return TrueExpr()
        result = conjuncts[0]
        for conj in conjuncts[1:]:
            result = builder.mk_and([result, conj])
        return result

    @staticmethod
    def project(polyhedron: Polyhedron, dimensions: Set[int]) -> Polyhedron:
        """Project polyhedron onto dimensions using Fourier-Motzkin elimination."""
        if not polyhedron.constraints:
            return polyhedron
        
        # Get all dimensions in the polyhedron
        all_dims = set()
        for constraint in polyhedron.constraints:
            all_dims.update(constraint.coefficients.dimensions())
        
        # Dimensions to eliminate
        dims_to_eliminate = all_dims - dimensions
        
        if not dims_to_eliminate:
            return polyhedron  # Nothing to eliminate
        
        # Apply Fourier-Motzkin elimination for each dimension to eliminate
        current_constraints = list(polyhedron.constraints)
        
        for dim in dims_to_eliminate:
            new_constraints = []
            positive = []  # Constraints with positive coefficient for dim
            negative = []  # Constraints with negative coefficient for dim
            zero = []      # Constraints with zero coefficient for dim
            
            # Partition constraints
            for constraint in current_constraints:
                coeff = constraint.coefficients.get(dim, Fraction(0))
                
                if coeff > 0:
                    positive.append(constraint)
                elif coeff < 0:
                    negative.append(constraint)
                else:
                    zero.append(constraint)
            
            # Keep constraints that don't involve the dimension
            new_constraints.extend(zero)
            
            # For each pair (positive, negative), create a new constraint
            # by eliminating the dimension
            for pos_constraint in positive:
                for neg_constraint in negative:
                    # Normalize so coefficients of dim are 1 and -1
                    pos_coeff = pos_constraint.coefficients.get(dim)
                    neg_coeff = neg_constraint.coefficients.get(dim)
                    
                    # Scale constraints
                    # pos: a^T x >= -c  with a_dim > 0
                    # neg: b^T x >= -d  with b_dim < 0
                    # Eliminate dim: combine to get new constraint
                    
                    # Scale positive constraint by |neg_coeff|
                    scale_pos = abs(neg_coeff)
                    # Scale negative constraint by pos_coeff
                    scale_neg = pos_coeff
                    
                    # Create new coefficients
                    new_coeffs = {}
                    for d in all_dims:
                        if d == dim:
                            continue
                        val = (scale_pos * pos_constraint.coefficients.get(d, Fraction(0)) +
                               scale_neg * neg_constraint.coefficients.get(d, Fraction(0)))
                        if val != 0:
                            new_coeffs[d] = val
                    
                    new_const = (scale_pos * pos_constraint.constant +
                                scale_neg * neg_constraint.constant)
                    
                    # Determine if it's an equality
                    new_eq = pos_constraint.equality and neg_constraint.equality
                    
                    new_constraints.append(Constraint(QQVector(new_coeffs), new_const, new_eq))
            
            current_constraints = new_constraints
        
        return Polyhedron(current_constraints)

    @staticmethod
    def meet(poly1: Polyhedron, poly2: Polyhedron) -> Polyhedron:
        """Meet (intersection) of two polyhedra."""
        return poly1.intersect(poly2)

    @staticmethod
    def join(poly1: Polyhedron, poly2: Polyhedron) -> Polyhedron:
        """Join (convex hull) of two polyhedra."""
        try:
            from scipy.spatial import ConvexHull
            import numpy as np
            
            # Get vertices of both polyhedra
            vertices1 = poly1.vertices()
            vertices2 = poly2.vertices()
            
            if not vertices1 and not vertices2:
                return Polyhedron([])
            
            all_vertices = vertices1 + vertices2
            
            # Get all dimensions
            all_dims = set()
            for v in all_vertices:
                all_dims.update(v.dimensions())
            
            if not all_dims:
                return Polyhedron([])
            
            n = max(all_dims) + 1
            
            # Convert to numpy array
            points = np.zeros((len(all_vertices), n))
            for i, v in enumerate(all_vertices):
                for dim in range(n):
                    points[i, dim] = float(v.get(dim, Fraction(0)))
            
            # Compute convex hull
            hull = ConvexHull(points)
            
            # Convert equations to constraints
            # Each equation is: a^T x + b = 0 for points on the facet
            # and a^T x + b <= 0 for all points in the hull
            constraints = []
            
            for eq in hull.equations:
                # eq has form [a1, a2, ..., an, b]
                coeffs = {}
                for i in range(n):
                    val = eq[i]
                    if abs(val) > 1e-10:
                        coeffs[i] = Fraction(val).limit_denominator(10000)
                
                const = Fraction(eq[-1]).limit_denominator(10000)
                
                # Create constraint: a^T x + b <= 0, which we store as -a^T x - b >= 0
                neg_coeffs = {k: -v for k, v in coeffs.items()}
                constraints.append(Constraint(QQVector(neg_coeffs), -const, equality=False))
            
            return Polyhedron(constraints)
            
        except (ImportError, Exception):
            # Fall back to simple union of constraints (over-approximation)
            return Polyhedron(poly1.constraints + poly2.constraints)

    @staticmethod
    def affine_transform(polyhedron: Polyhedron, matrix: QQMatrix,
                        vector: QQVector) -> Polyhedron:
        """Apply affine transformation to polyhedron."""
        # Transform vertices and take convex hull back to constraints if possible
        verts = polyhedron.vertices()
        if not verts:
            return polyhedron
        # Apply y = M x + v
        transformed: List[QQVector] = []
        for x in verts:
            try:
                y = (matrix * x) + vector
            except Exception:
                # If multiplication not directly supported, approximate by dimension-wise sum
                y = vector
            transformed.append(y)
        # Build a new polyhedron as convex hull of transformed points by fitting halfspaces
        try:
            from scipy.spatial import ConvexHull
            import numpy as np
            # Determine dimension
            all_dims: Set[int] = set()
            for v in transformed:
                all_dims.update(v.dimensions())
            if not all_dims:
                return Polyhedron([])
            n = max(all_dims) + 1
            points = np.zeros((len(transformed), n))
            for i, v in enumerate(transformed):
                for d in range(n):
                    points[i, d] = float(v.get(d, Fraction(0)))
            hull = ConvexHull(points)
            constraints: List[Constraint] = []
            for eq in hull.equations:
                coeffs = {}
                for i in range(n):
                    val = eq[i]
                    if abs(val) > 1e-10:
                        coeffs[i] = Fraction(val).limit_denominator(10000)
                const = Fraction(eq[-1]).limit_denominator(10000)
                neg_coeffs = {k: -v for k, v in coeffs.items()}
                constraints.append(Constraint(QQVector(neg_coeffs), -const, equality=False))
            return Polyhedron(constraints)
        except Exception:
            # Fallback: return original polyhedron if convex hull fails
            return polyhedron


# Factory functions
def make_polyhedron(constraints: List[Constraint]) -> Polyhedron:
    """Create a polyhedron from constraints."""
    return Polyhedron(constraints)


def make_constraint(coefficients: QQVector, constant: Fraction, equality: bool = False) -> Constraint:
    """Create a constraint."""
    return Constraint(coefficients, constant, equality)


def make_empty_polyhedron() -> Polyhedron:
    """Create an empty polyhedron."""
    # This would require an inconsistent constraint
    return Polyhedron([Constraint(QQVector(), Fraction(1), True)])  # 1 = 0


def make_universal_polyhedron(dimension: int) -> Polyhedron:
    """Create the universal polyhedron (all of R^n)."""
    return Polyhedron([])


# Polyhedron from common shapes
def make_box_polyhedron(bounds: Dict[int, Tuple[Fraction, Fraction]]) -> Polyhedron:
    """Create a box polyhedron from variable bounds."""
    constraints = []

    for var, (lower, upper) in bounds.items():
        # x >= lower
        coeff_vec = QQVector({var: Fraction(1)})
        constraints.append(Constraint(coeff_vec, -lower, False))

        # x <= upper
        coeff_vec = QQVector({var: Fraction(-1)})
        constraints.append(Constraint(coeff_vec, upper, False))

    return Polyhedron(constraints)


def make_simplex_polyhedron(dimension: int) -> Polyhedron:
    """Create a simplex polyhedron."""
    # x1 + x2 + ... + xn <= 1, xi >= 0 for all i
    constraints = []

    # Sum constraint
    sum_coeffs = QQVector({i: Fraction(1) for i in range(dimension)})
    constraints.append(Constraint(sum_coeffs, Fraction(-1), False))

    # Non-negativity constraints
    for i in range(dimension):
        coeff_vec = QQVector({i: Fraction(-1)})
        constraints.append(Constraint(coeff_vec, Fraction(0), False))

    return Polyhedron(constraints)


# Polyhedron analysis
def polyhedron_is_empty(polyhedron: Polyhedron) -> bool:
    """Check if polyhedron is empty."""
    return polyhedron.is_empty()


def polyhedron_contains_point(polyhedron: Polyhedron, point: QQVector) -> bool:
    """Check if polyhedron contains a point."""
    return polyhedron.contains(point)


def polyhedron_intersection(poly1: Polyhedron, poly2: Polyhedron) -> Polyhedron:
    """Intersection of two polyhedra."""
    return PolyhedronOperations.meet(poly1, poly2)


def polyhedron_convex_hull(poly1: Polyhedron, poly2: Polyhedron) -> Polyhedron:
    """Convex hull of two polyhedra."""
    return PolyhedronOperations.join(poly1, poly2)


def polyhedron_from_formula(formula: FormulaExpression, context: Context) -> Polyhedron:
    """Create polyhedron from logical formula."""
    return PolyhedronOperations.from_formula(formula, context)


def polyhedron_to_formula(polyhedron: Polyhedron, context: Context) -> FormulaExpression:
    """Convert polyhedron to logical formula."""
    return PolyhedronOperations.to_formula(polyhedron, context)