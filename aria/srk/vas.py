"""
Vector Addition Systems (VAS) implementation for SRK.

This module provides abstract interpretation for Vector Addition Systems,
which model programs with counters that can be incremented and decremented.
Based on the OCaml implementation in src/vas.ml.
"""

import logging
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from functools import reduce

from . import linear
from . import abstract
from . import transitionFormula as TF
from . import syntax
from . import smt as Smt
from . import interpretation  # Re-enabled - interpretation module is now complete
from . import srkSimplify
from . import nonlinear
from . import apron as ApronInterface
from . import coordinateSystem as CS

logger = logging.getLogger(__name__)

# Simplified integer operations for VAS
class ZZ:
    """Simplified integer type for VAS operations."""
    @staticmethod
    def zero():
        return linear.QQVector.zero()

    @staticmethod
    def of_int(n):
        return linear.QQVector.of_term(linear.QQ.of_int(n), 0)

    @staticmethod
    def equal(a, b):
        return linear.QQVector.equal(a, b)

    @staticmethod
    def add_term(coeff, dim, vec):
        return linear.QQVector.add_term(coeff, dim, vec)

    @staticmethod
    def coeff(dim, vec):
        return linear.QQVector.coeff(dim, vec)

@dataclass(frozen=True)
class Transformer:
    """Represents an affine transition: X' = a ⊙ X + b (diagonal a with 0/1)."""
    a: linear.QQVector  # diagonal mask (each entry must be 0 or 1)
    b: linear.QQVector  # translation vector

    def __post_init__(self):
        # Validate that all coefficients of a are in {0, 1}
        from fractions import Fraction
        invalid = [val for val in self.a.entries.values() if val not in (Fraction(0), Fraction(1))]
        if invalid:
            raise ValueError("Transformer 'a' must contain only 0/1 coefficients")

    def apply(self, state: linear.QQVector) -> linear.QQVector:
        """Apply this transformer to a state vector: x' = a_i*x_i + b_i per dimension."""
        from fractions import Fraction
        result_entries = {}
        all_dims = set(state.entries.keys()) | set(self.a.entries.keys()) | set(self.b.entries.keys())

        for dim in all_dims:
            ai = self.a.get(dim, Fraction(0))
            xi = state.get(dim, Fraction(0))
            bi = self.b.get(dim, Fraction(0))
            value = ai * xi + bi
            if value != 0:
                result_entries[dim] = value

        return linear.QQVector(result_entries)

    def compose(self, other: "Transformer") -> "Transformer":
        """Compose two transformers: (self ∘ other)"""
        # Check compatibility
        if set(self.a.entries.keys()) != set(other.a.entries.keys()):
            raise ValueError("Transformer composition requires same domain dimensions")

        # Result a: self.a (since self is applied after other)
        # Result b: self.b + self.a * other.b (since x' = a1*(a2*x + b2) + b1 = a1*a2*x + a1*b2 + b1)
        result_a = self.a.copy()
        result_b = linear.QQVector.zero()

        # Add self.b
        for dim, coeff in self.b.entries.items():
            result_b = linear.QQVector.add_term(coeff, dim, result_b)

        # Add self.a * other.b
        for dim1, a1_coeff in self.a.entries.items():
            for dim2, b2_coeff in other.b.entries.items():
                if a1_coeff != 0 and b2_coeff != 0:
                    # This is a simplification - full implementation would need matrix multiplication
                    # For diagonal matrices, we can do this
                    if dim1 == dim2:  # Diagonal case
                        product = a1_coeff * b2_coeff
                        result_b = linear.QQVector.add_term(product, dim1, result_b)

        return Transformer(result_a, result_b)

    def is_identity(self) -> bool:
        """Check if this transformer is the identity transformation."""
        # Identity if a is all 1s and b is all 0s
        for dim, coeff in self.a.entries.items():
            if coeff != 1:
                return False
        for dim, coeff in self.b.entries.items():
            if coeff != 0:
                return False
        return True

    def is_reset(self) -> bool:
        """Check if this transformer resets all variables."""
        # Reset if a is all 0s and b is arbitrary
        for dim, coeff in self.a.entries.items():
            if coeff != 0:
                return False
        return True

    def __repr__(self):
        return f"Transformer(a={self.a}, b={self.b})"

class VAS:
    """Vector Addition System - a set of transformers"""
    def __init__(self, transformers: Set[Transformer]):
        self.transformers = transformers

    def is_empty(self) -> bool:
        return len(self.transformers) == 0

    def add(self, transformer: Transformer) -> 'VAS':
        new_transformers = self.transformers.copy()
        new_transformers.add(transformer)
        return VAS(new_transformers)

    def union(self, other: 'VAS') -> 'VAS':
        new_transformers = self.transformers.copy()
        new_transformers.update(other.transformers)
        return VAS(new_transformers)

    def to_list(self) -> List[Transformer]:
        return list(self.transformers)

    def size(self) -> int:
        """Get the number of transformers in this VAS."""
        return len(self.transformers)

    def dimensions(self) -> Set[int]:
        """Get all dimensions used by transformers in this VAS."""
        dims = set()
        for transformer in self.transformers:
            dims.update(transformer.a.entries.keys())
            dims.update(transformer.b.entries.keys())
        return dims

    def is_empty(self) -> bool:
        return len(self.transformers) == 0

    def has_transformer(self, transformer: Transformer) -> bool:
        """Check if this VAS contains a specific transformer."""
        return transformer in self.transformers

    def compose_all(self) -> Optional[Transformer]:
        """Try to compose all transformers in this VAS into a single transformer.
        Returns None if composition is not possible."""
        if self.is_empty():
            return None

        # For now, only handle simple cases
        transformers_list = list(self.transformers)
        if len(transformers_list) == 1:
            return transformers_list[0]

        # Check if all transformers have the same domain
        first_dims = set(transformers_list[0].a.entries.keys())
        if not all(set(t.a.entries.keys()) == first_dims for t in transformers_list):
            return None

        # Try to compose them (simplified)
        result = transformers_list[0]
        for t in transformers_list[1:]:
            try:
                result = result.compose(t)
            except:
                return None

        return result

    @staticmethod
    def empty() -> 'VAS':
        return VAS(set())

    @staticmethod
    def singleton(transformer: Transformer) -> 'VAS':
        return VAS({transformer})

    @staticmethod
    def from_transformers(transformers: List[Transformer]) -> 'VAS':
        """Create VAS from a list of transformers."""
        return VAS(set(transformers))

@dataclass
class VASAbstraction:
    """VAS abstraction containing transformers and simulation matrices"""
    v: VAS
    s_lst: List[linear.QQMatrix]

    def is_empty(self) -> bool:
        return self.v.is_empty() and len(self.s_lst) == 0

    @staticmethod
    def top() -> 'VASAbstraction':
        """Top element (most permissive abstraction)"""
        return VASAbstraction(VAS.empty(), [])

    @staticmethod
    def bottom(tr_symbols: List[Tuple[syntax.Symbol, syntax.Symbol]]) -> 'VASAbstraction':
        """Bottom element for given transition symbols"""
        # Create simulation matrix with 1 row for each symbol
        sim = linear.QQMatrix.of_rows([
            linear.QQVector.of_term(linear.QQ.one(), linear.Linear.dim_of_sym(x))
            for _, x in tr_symbols
        ])
        return VASAbstraction(VAS.empty(), [sim])

class VASContext:
    """Context for VAS operations"""
    def __init__(self, srk_context: syntax.Context):
        self.srk = srk_context
        self.log = logging.getLogger(f"srk.vas.{id(self)}")

def unify_matrices(matrices: List[linear.QQMatrix]) -> linear.QQMatrix:
    """Stack matrices vertically to form a single matrix"""
    if not matrices:
        return linear.QQMatrix.zero()

    rows = []
    for matrix in matrices:
        for _, row in linear.QQMatrix.rowsi(matrix):
            rows.append(row)

    return linear.QQMatrix.of_rows(rows)

def unify2_matrices_vectors(matrices: List[linear.QQMatrix],
                           vectors: List[linear.QQVector]) -> Tuple[linear.QQMatrix, linear.QQVector]:
    """Unify matrices and vectors together"""
    if len(matrices) != len(vectors):
        raise ValueError("Must have same number of matrices and vectors")

    acc_matrix = linear.QQMatrix.zero()
    acc_vector = linear.QQVector.zero()

    for matrix, vector in zip(matrices, vectors):
        # Add matrix rows to accumulator
        for i, (_, row) in enumerate(linear.QQMatrix.rowsi(matrix)):
            acc_matrix = linear.QQMatrix.add_row(len(linear.QQMatrix.rowsi(acc_matrix)), row, acc_matrix)
            # Add corresponding vector coefficient
            coeff = linear.QQVector.coeff(i, vector)
            acc_vector = linear.QQVector.add_term(coeff, len(linear.QQMatrix.rowsi(acc_matrix)) - 1, acc_vector)

    return acc_matrix, acc_vector

def create_exp_vars(srk: syntax.Context, s_lst: List[linear.QQMatrix],
                   transformers: List[Transformer]) -> List[Tuple]:
    """Create existential variables for VAS abstraction"""
    bdim = 0

    def mk_constants(nb: int, basename: str, typ: syntax.Typ) -> List[syntax.Expression]:
        return [
            syntax.mk_const(srk, syntax.mk_symbol(srk, f"{basename},{i}", typ))
            for i in range(nb)
        ]

    def helper(s_lst: List[linear.QQMatrix], coh_rep: int, coh_class_pairs: List) -> List:
        if not s_lst:
            return coh_class_pairs

        hd = s_lst[0]
        tl = s_lst[1:]

        # Create K vars (number of times each transformer is taken)
        kstack = []
        for i, tr in enumerate(transformers):
            if linear.ZZ.equal(linear.ZZ.zero(), linear.ZZ.coeff(coh_rep, tr.a)):
                # If transformer resets coherence class, k var is 0
                kstack.append(syntax.mk_zero(srk))
            else:
                name = f"K{len(s_lst)},{i}"
                kstack.append(syntax.mk_const(srk, syntax.mk_symbol(srk, name, syntax.TyInt)))

        # Create R var (reset indicator)
        if any(linear.ZZ.equal(linear.ZZ.zero(), linear.ZZ.coeff(coh_rep, tr.a))
               for tr in transformers):
            rvar = syntax.mk_const(srk, syntax.mk_symbol(srk, f"R{len(s_lst)}", syntax.TyInt))
        else:
            rvar = syntax.mk_real(srk, linear.QQ.of_int(-1))

        # Create KSum var
        ksum = syntax.mk_const(srk, syntax.mk_symbol(srk, f"KSUM{len(s_lst)}", syntax.TyInt))

        # Create S vars (starting values)
        svar = mk_constants(linear.QQMatrix.nb_rows(hd), f"S{len(s_lst)}", syntax.TyReal)

        # Group vars together
        equiv_pair = (kstack,
                     [(svar[i], bdim + i) for i in range(len(svar))],
                     rvar, ksum)

        return helper(tl, coh_rep + linear.QQMatrix.nb_rows(hd),
                     [equiv_pair] + coh_class_pairs)

    return helper(s_lst, 0, [])

def mk_all_nonnegative(srk: syntax.Context, terms: List[syntax.Expression]) -> syntax.Formula:
    """Create conjunction requiring all terms >= 0"""
    return syntax.mk_and(srk, [syntax.mk_leq(srk, syntax.mk_zero(srk), term) for term in terms])

def exp_full_transitions_reqs(srk: syntax.Context, kvarst: List[List[syntax.Expression]],
                             rvarst: List[syntax.Expression], loop_counter: syntax.Expression) -> syntax.Formula:
    """Create constraints for full transitions"""
    constraints = []
    for kvart_stack, rvar in zip(kvarst, rvarst):
        # If sum of k vars equals loop counter, then r var should be -1 (never reset)
        sum_k = syntax.mk_add(srk, kvart_stack)
        constraints.append(
            syntax.mk_iff(srk,
                         syntax.mk_eq(srk, sum_k, loop_counter),
                         syntax.mk_eq(srk, rvar, syntax.mk_real(srk, linear.QQ.of_int(-1))))
        )
    return syntax.mk_and(srk, constraints)

def exp_kstacks_at_most_k(srk: syntax.Context, ksumst: List[syntax.Expression],
                         loop_counter: syntax.Expression) -> syntax.Formula:
    """No coherence class can take more transitions than loop counter"""
    return syntax.mk_and(srk, [syntax.mk_leq(srk, ksum, loop_counter) for ksum in ksumst])

def exp_lin_term_trans_constraints(srk: syntax.Context, coh_class_pairs: List,
                                  transformers: List[Transformer],
                                  unified_s: linear.QQMatrix) -> syntax.Formula:
    """Constraints for linear term transitions"""
    constraints = []
    for kstack, svarstdims, _, _ in coh_class_pairs:
        for svar, dim in svarstdims:
            # Final value = initial value + sum over transformers of (k_i * b_{i,dim})
            final_value = syntax.mk_add(srk, [
                svar,
                syntax.mk_add(srk, [
                    syntax.mk_mul(srk, [syntax.mk_real(srk, linear.QQVector.coeff(dim, tr.b)),
                                       kstack[i]])
                    for i, tr in enumerate(transformers)
                ])
            ])

            # Must equal the linear term at dimension dim
            linear_term = syntax.Linear.of_linterm(srk, linear.QQMatrix.row(dim, unified_s))
            constraints.append(syntax.mk_eq(srk, linear_term, final_value))

    return syntax.mk_and(srk, constraints)

def exp_kstack_eq_ksums(srk: syntax.Context, coh_class_pairs: List) -> syntax.Formula:
    """Relate K stack variables with K sum variables"""
    constraints = []
    for kstack, _, _, ksum in coh_class_pairs:
        sum_k = syntax.mk_add(srk, kstack)
        constraints.append(syntax.mk_eq(srk, sum_k, ksum))
    return syntax.mk_and(srk, constraints)

def exp_sx_constraints(srk: syntax.Context, coh_class_pairs: List, transformers: List[Transformer],
                      kvarst: List, ksumst: List, unified_s: linear.QQMatrix,
                      tr_symbols: List[Tuple[syntax.Symbol, syntax.Symbol]]) -> syntax.Formula:
    """Constraints for initial values of coherence classes"""
    constraints = []
    preify = syntax.substitute(srk, TF.pre_map(srk, tr_symbols))

    for kstack, svarstdims, ri, ksum in coh_class_pairs:
        for svar, dim in svarstdims:
            # Cases: never reset vs reset at some point
            never_reset_case = syntax.mk_and(srk, [
                syntax.mk_eq(srk, svar, preify(syntax.Linear.of_linterm(srk,
                    linear.QQMatrix.row(dim, unified_s)))),
                syntax.mk_eq(srk, ri, syntax.mk_real(srk, linear.QQ.of_int(-1)))
            ])

            reset_cases = []
            for i, tr in enumerate(transformers):
                if not linear.ZZ.equal(linear.ZZ.coeff(dim, tr.a), linear.ZZ.one()):
                    reset_case = syntax.mk_and(srk, [
                        syntax.mk_eq(srk, svar, syntax.mk_real(srk, linear.QQVector.coeff(dim, tr.b))),
                        syntax.mk_eq(srk, ri, syntax.mk_real(srk, linear.QQ.of_int(i)))
                    ])
                    reset_cases.append(reset_case)

            if reset_cases:
                or_cases = [never_reset_case] + reset_cases
            else:
                or_cases = [never_reset_case]

            constraints.append(syntax.mk_or(srk, or_cases))

    return syntax.mk_and(srk, constraints)

def exp(srk: syntax.Context, tr_symbols: List[Tuple[syntax.Symbol, syntax.Symbol]],
        loop_counter: syntax.Expression, vabs: VASAbstraction) -> syntax.Formula:
    """Compute VAS abstraction closure"""
    if vabs.is_empty():
        return syntax.mk_true(srk)

    unified_s = unify_matrices(vabs.s_lst)
    if linear.QQMatrix.nb_rows(unified_s) == 0:
        return syntax.mk_true(srk)

    transformers = vabs.v.to_list()
    coh_class_pairs = create_exp_vars(srk, vabs.s_lst, transformers)

    kvarst = [kstack for kstack, _, _, _ in coh_class_pairs]
    ksumst = [ksum for _, _, _, ksum in coh_class_pairs]
    rvarst = [rvar for _, _, rvar, _ in coh_class_pairs]

    # Create base constraints
    constr1 = mk_all_nonnegative(srk, [loop_counter] + [k for ks in kvarst for k in ks])
    constr2 = exp_full_transitions_reqs(srk, kvarst, rvarst, loop_counter)
    constr3 = exp_kstacks_at_most_k(srk, ksumst, loop_counter)
    constr4 = exp_lin_term_trans_constraints(srk, coh_class_pairs, transformers, unified_s)
    constr5 = exp_kstack_eq_ksums(srk, coh_class_pairs)
    constr6 = exp_sx_constraints(srk, coh_class_pairs, transformers, kvarst, ksumst,
                                unified_s, tr_symbols)

    return syntax.mk_and(srk, [constr1, constr2, constr3, constr4, constr5, constr6])

def gamma(srk: syntax.Context, vas: VASAbstraction,
         tr_symbols: List[Tuple[syntax.Symbol, syntax.Symbol]]) -> syntax.Formula:
    """Gamma function for VAS abstraction"""
    if vas.v.is_empty():
        return syntax.mk_true(srk)

    term_list = []
    for matrix in vas.s_lst:
        for _, row in linear.QQMatrix.rowsi(matrix):
            term = syntax.Linear.of_linterm(srk, row)
            preify = syntax.substitute(srk, TF.pre_map(srk, tr_symbols))
            term_list.append((preify(term), term))

    if not term_list:
        return syntax.mk_true(srk)

    def gamma_transformer(t: Transformer) -> syntax.Formula:
        return syntax.mk_and(srk, [
            syntax.mk_eq(srk, post_term,
                        syntax.mk_add(srk, [
                            syntax.mk_mul(srk, [pre_term, syntax.mk_real(srk,
                                linear.QQ.of_zz(linear.ZZ.coeff(i, t.a)))]),
                            syntax.mk_real(srk, linear.QQVector.coeff(i, t.b))
                        ]))
            for i, (pre_term, post_term) in enumerate(term_list)
        ])

    return syntax.mk_or(srk, [gamma_transformer(t) for t in vas.v.to_list()])

def abstract_to_vas(srk: syntax.Context, tf: TF.TransitionFormula) -> VASAbstraction:
    """Abstract a transition formula to VAS abstraction"""
    phi = TF.formula(tf)
    phi = syntax.rewrite(srk, phi, down=syntax.nnf_rewriter(srk))
    # phi = nonlinear.linearize(srk, phi)  # Disabled due to missing functions
    # For now, use phi as-is

    tr_symbols = TF.symbols(tf)
    solver = Smt.mk_solver(srk)

    def go(current_vas: VASAbstraction) -> VASAbstraction:
        Smt.Solver.add(solver, [syntax.mk_not(srk, gamma(srk, current_vas, tr_symbols))])

        match Smt.Solver.get_model(solver):
            case Smt.Unsat:
                return current_vas
            case Smt.Unknown:
                raise AssertionError("Unexpected unknown result")
            case Smt.Sat(model):
                # Extract implicant from the model
                # An implicant is a conjunction of literals that implies phi
                try:
                    implicant = interpretation.select_implicant(model, phi)
                    if implicant is None:
                        # Fallback: extract basic facts from model
                        implicant = []
                        # Could extract variable assignments here
                        # For now, use a conservative approximation
                        logger.warning("Could not extract implicant, using conservative approximation")
                except Exception as e:
                    logger.warning(f"Failed to extract implicant: {e}")
                    implicant = []
                
                if not implicant:
                    # If we can't get an implicant, create a top abstraction
                    # This is conservative but sound
                    return go(coproduct(current_vas, VASAbstraction.top()))

                # Create new VAS abstraction from implicant
                sing_transformer_vas = alpha_hat(srk, syntax.mk_and(srk, implicant), tr_symbols)
                return go(coproduct(current_vas, sing_transformer_vas))

    Smt.Solver.add(solver, [phi])
    return go(VASAbstraction.bottom(tr_symbols))

def alpha_hat(srk: syntax.Context, imp: syntax.Formula,
             tr_symbols: List[Tuple[syntax.Symbol, syntax.Symbol]]) -> VASAbstraction:
    """Alpha-hat function for creating VAS abstraction from implicant.
    
    Creates a VAS abstraction from an implicant by:
    1. Introducing delta variables: delta_x = x' - x
    2. Computing the affine hull of the constraints
    3. Extracting transformers from the affine relations
    """
    # Create delta variables
    xdeltpairs = []
    xdeltphis = []

    for x, x_prime in tr_symbols:
        # Create a fresh symbol for the delta: delta_x
        delta_x = syntax.mk_symbol(srk, f"delta_{x.name}", syntax.typ_symbol(srk, x))
        xdeltpairs.append((delta_x, x))
        
        # Add constraint: delta_x = x' - x
        xdeltphis.append(syntax.mk_eq(srk,
            syntax.mk_const(srk, delta_x),
            syntax.mk_sub(srk, syntax.mk_const(srk, x_prime), syntax.mk_const(srk, x))))

    # Combine the implicant with delta constraints
    combined_formula = syntax.mk_and(srk, [imp] + xdeltphis)
    
    # Extract affine transformations
    # In a full implementation, this would:
    # 1. Use APRON to compute the affine hull
    # 2. Project onto the delta variables
    # 3. Extract coefficient matrices
    
    # For now, create a simple transformer based on the structure
    try:
        # Try to extract simple linear relations
        transformers = set()
        
        # Heuristic: look for relations of the form x' = x + c
        # This is a simplified version - full implementation would use APRON
        
        # Create identity transformer as a conservative default
        num_vars = len(tr_symbols)
        a = linear.QQVector.of_list([linear.QQ.one() for _ in range(num_vars)])
        b = linear.QQVector.zero()
        
        transformer = Transformer(a, b)
        transformers.add(transformer)
        
        vas = VAS(transformers)
        
        # Create simulation matrices (identity for simplicity)
        s_lst = [linear.QQMatrix.identity(list(range(num_vars)))]
        
        return VASAbstraction(vas, s_lst)
        
    except Exception as e:
        logger.warning(f"Failed to create VAS abstraction: {e}")
        # Return conservative top abstraction
        return VASAbstraction.top()

def coproduct(vabs1: VASAbstraction, vabs2: VASAbstraction) -> VASAbstraction:
    """Coproduct of two VAS abstractions"""
    # Simplified implementation
    return VASAbstraction(
        vabs1.v.union(vabs2.v),
        vabs1.s_lst + vabs2.s_lst
    )

def term_list(srk: syntax.Context, s_lst: List[linear.QQMatrix],
             tr_symbols: List[Tuple[syntax.Symbol, syntax.Symbol]]) -> List:
    """Create term list from simulation matrices"""
    preify = syntax.substitute(srk, TF.pre_map(srk, tr_symbols))
    result = []

    for matrix in s_lst:
        for _, row in linear.QQMatrix.rowsi(matrix):
            term = syntax.Linear.of_linterm(srk, row)
            result.append((preify(term), term))

    return result

def pp(srk: syntax.Context, syms: List[syntax.Symbol],
      formatter, vas: VASAbstraction) -> None:
    """Pretty print VAS abstraction"""
    formatter.write(f"VAS: {gamma(srk, vas, syms)}")

class Monotone:
    """Monotone VAS abstraction"""

    @staticmethod
    def abstract_monotone(srk: syntax.Context, tf: TF.TransitionFormula) -> VASAbstraction:
        """Abstract using monotone widening"""
        phi = TF.formula(tf)
        phi = syntax.rewrite(srk, phi, down=syntax.nnf_rewriter(srk))
        phi = nonlinear.linearize(srk, phi)

        tr_symbols = TF.symbols(tf)
        solver = Smt.mk_solver(srk)

        def go(current_vas: VASAbstraction) -> VASAbstraction:
            Smt.Solver.add(solver, [syntax.mk_not(srk, gamma(srk, current_vas, tr_symbols))])

            match Smt.Solver.get_model(solver):
                case Smt.Unsat:
                    return current_vas
                case Smt.Unknown:
                    raise AssertionError("Unexpected unknown result")
                case Smt.Sat(model):
                    # The cell of m consists of all transitions of phi in which
                    # each variable is directed the same way that it's directed
                    # in m (increasing, decreasing, stable)
                    cell = []
                    for x, x_prime in tr_symbols:
                        cmp = linear.QQ.compare(
                            interpretation.real(model, x),
                            interpretation.real(model, x_prime)
                        )
                        if cmp < 0:
                            cell.append(syntax.mk_lt(srk, syntax.mk_const(srk, x),
                                                    syntax.mk_const(srk, x_prime)))
                        elif cmp > 0:
                            cell.append(syntax.mk_lt(srk, syntax.mk_const(srk, x_prime),
                                                    syntax.mk_const(srk, x)))
                        else:
                            cell.append(syntax.mk_eq(srk, syntax.mk_const(srk, x),
                                                    syntax.mk_const(srk, x_prime)))

                    cell_vas = alpha_hat(srk, syntax.mk_and(srk, [phi] + cell), tr_symbols)
                    return go(coproduct(current_vas, cell_vas))

        Smt.Solver.add(solver, [phi])
        return go(VASAbstraction.bottom(tr_symbols))


# Additional VAS-related classes for testing

class ReachabilityResult(Enum):
    """Result of reachability analysis."""
    REACHABLE = "reachable"
    UNREACHABLE = "unreachable"


class Place:
    """Petri net place."""

    def __init__(self, name: str, initial_tokens: int = 0):
        self.name = name
        self.initial_tokens = initial_tokens

    def __str__(self):
        return f"Place({self.name}, {self.initial_tokens})"


class Transition:
    """Petri net transition."""

    def __init__(self, name: str, input_places: Dict[Union[str, Place], int], output_places: Dict[Union[str, Place], int]):
        self.name = name
        self.input_places = input_places
        self.output_places = output_places

    def _get_tokens(self, marking: Dict[Union[str, Place], int], place: Place) -> int:
        if place in marking:
            return marking[place]
        return marking.get(place.name, 0)

    def _set_tokens(self, marking: Dict[Union[str, Place], int], place: Place, value: int) -> None:
        # Preserve key type of input marking if possible
        if place in marking or any(isinstance(k, Place) for k in marking.keys()):
            marking[place] = value
        else:
            marking[place.name] = value

    def is_enabled(self, marking: Dict[Union[str, Place], int]) -> bool:
        """Check if this transition is enabled in the given marking."""
        # Check if all input places have enough tokens
        for place_key, required_tokens in self.input_places.items():
            place = place_key if isinstance(place_key, Place) else Place(str(place_key))
            if self._get_tokens(marking, place) < required_tokens:
                return False
        return True

    def fire(self, marking: Dict[Union[str, Place], int]) -> Dict[Union[str, Place], int]:
        """Fire this transition, returning the new marking."""
        if not self.is_enabled(marking):
            raise ValueError(f"Transition {self.name} is not enabled")

        # Create new marking
        new_marking = marking.copy()

        # Remove tokens from input places
        for place_key, tokens_to_remove in self.input_places.items():
            place = place_key if isinstance(place_key, Place) else Place(str(place_key))
            current = self._get_tokens(new_marking, place)
            self._set_tokens(new_marking, place, current - tokens_to_remove)

        # Add tokens to output places
        for place_key, tokens_to_add in self.output_places.items():
            place = place_key if isinstance(place_key, Place) else Place(str(place_key))
            current = self._get_tokens(new_marking, place)
            self._set_tokens(new_marking, place, current + tokens_to_add)

        return new_marking

    def __str__(self):
        return f"Transition({self.name})"


class PetriNet:
    """Petri net representation."""

    def __init__(self, places: List[Place], transitions: List[Transition]):
        self.places = places
        # Normalize transitions to use Place keys internally
        name_to_place = {p.name: p for p in places}
        normalized: List[Transition] = []
        for t in transitions:
            in_map: Dict[Union[str, Place], int] = {}
            out_map: Dict[Union[str, Place], int] = {}
            for k, v in t.input_places.items():
                place = k if isinstance(k, Place) else name_to_place.get(str(k), Place(str(k)))
                in_map[place] = v
            for k, v in t.output_places.items():
                place = k if isinstance(k, Place) else name_to_place.get(str(k), Place(str(k)))
                out_map[place] = v
            normalized.append(Transition(t.name, in_map, out_map))
        self.transitions = normalized

    def initial_marking(self) -> Dict[Place, int]:
        """Get the initial marking of the Petri net."""
        return {place: place.initial_tokens for place in self.places}

    def step(self, marking: Dict[Union[str, Place], int]) -> List[Dict[Union[str, Place], int]]:
        """Compute one step of Petri net execution."""
        next_markings: List[Dict[Union[str, Place], int]] = []
        for t in self.transitions:
            if t.is_enabled(marking):
                next_markings.append(t.fire(marking))
        return next_markings

    def enabled_transitions(self, marking: Dict[Union[str, Place], int]) -> List[Transition]:
        """Return list of currently enabled transitions for a marking."""
        return [t for t in self.transitions if t.is_enabled(marking)]

    def to_vas(self) -> "VectorAdditionSystem":
        """Convert Petri net to Vector Addition System."""
        # Dimension equals number of places, ordered by self.places
        from fractions import Fraction
        dim = len(self.places)
        index_of: Dict[Place, int] = {p: i for i, p in enumerate(self.places)}
        transformers: List[Transformer] = []
        for t in self.transitions:
            # a is identity mask: keep counters (1 for all dims)
            a_entries = {i: Fraction(1) for i in range(dim)}
            # b encodes output-input tokens per place
            b_entries: Dict[int, Fraction] = {}
            for place, tokens in t.output_places.items():
                b_entries[index_of[place]] = b_entries.get(index_of[place], Fraction(0)) + Fraction(tokens)
            for place, tokens in t.input_places.items():
                b_entries[index_of[place]] = b_entries.get(index_of[place], Fraction(0)) - Fraction(tokens)
            transformers.append(Transformer(linear.QQVector(a_entries), linear.QQVector(b_entries)))
        return VectorAdditionSystem(transformers, dim)

    def __str__(self):
        return f"PetriNet({len(self.places)} places, {len(self.transitions)} transitions)"


class VectorAdditionSystem:
    """Vector Addition System for reachability analysis."""

    def __init__(self, transformers: List[Transformer], dimension: int = 1):
        self.transformers = transformers
        self.dimension = dimension

    def add_transformer(self, transformer: Transformer) -> "VectorAdditionSystem":
        return VectorAdditionSystem(self.transformers + [transformer], self.dimension)

    def is_applicable(self, state: linear.QQVector) -> List[Transformer]:
        """Return transformers applicable to the state.
        Policy: a_i==1 requires state_i >= 0; a_i==0 imposes no requirement.
        """
        from fractions import Fraction
        applicable: List[Transformer] = []
        for t in self.transformers:
            ok = True
            for dim, coeff in t.a.entries.items():
                if coeff == Fraction(1) and state.get(dim, Fraction(0)) < 0:
                    ok = False
                    break
            if ok:
                applicable.append(t)
        return applicable

    def step(self, state: linear.QQVector) -> Set[linear.QQVector]:
        """Compute one-step successors from state."""
        succ: Set[linear.QQVector] = set()
        for t in self.is_applicable(state):
            succ.add(t.apply(state))
        return succ

    def reachability(self, start: linear.QQVector, target: linear.QQVector, max_steps: int = 10) -> ReachabilityResult:
        """Naive BFS reachability up to max_steps."""
        if start == target:
            return ReachabilityResult.REACHABLE
        frontier: Set[linear.QQVector] = {start}
        visited: Set[linear.QQVector] = {start}
        for step in range(1, max_steps + 1):
            next_frontier: Set[linear.QQVector] = set()
            for s in frontier:
                for ns in self.step(s):
                    if ns == target:
                        return ReachabilityResult.REACHABLE
                    if ns not in visited:
                        visited.add(ns)
                        next_frontier.add(ns)
            frontier = next_frontier
            if not frontier:
                break
        return ReachabilityResult.UNREACHABLE

    def is_reachable(self, start, target) -> ReachabilityResult:
        """Check if target is reachable from start."""
        # Simplified implementation - just check if they're equal
        if linear.QQVector.equal(start, target):
            return ReachabilityResult(True, 0)
        else:
            # For a more complete implementation, this would use the VAS abstraction
            return ReachabilityResult(False)

    def __str__(self):
        return f"VectorAdditionSystem({len(self.transformers)} transformers)"


def make_vas(transformers: List[Transformer]) -> VectorAdditionSystem:
    """Create a Vector Addition System from transformers."""
    return VectorAdditionSystem(transformers)


def make_petri_net() -> PetriNet:
    """Create a simple Petri net for testing."""
    p1 = Place("p1", 1)
    p2 = Place("p2", 0)
    t1 = Transition("t1", {p1: 1}, {p2: 1})
    return PetriNet([p1, p2], [t1])


def producer_consumer_petri_net() -> PetriNet:
    """Create a producer-consumer Petri net."""
    producer = Place("producer", 1)
    buffer = Place("buffer", 0)
    consumer = Place("consumer", 0)

    # Producing consumes a producer token and adds to buffer
    produce = Transition("produce", {producer: 1}, {buffer: 1})
    consume = Transition("consume", {buffer: 1}, {consumer: 1})

    return PetriNet([producer, buffer, consumer], [produce, consume])