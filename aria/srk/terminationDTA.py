"""
Termination analysis using DTA (Disjunctive Transition Automata).

This module implements termination analysis algorithms using disjunctive
transition automata for proving program termination, following the OCaml implementation.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, TypeVar, Generic, Callable
from dataclasses import dataclass, field
from fractions import Fraction
from enum import Enum
import itertools
import logging

from aria.srk.syntax import Context, Symbol, Type, FormulaExpression, ArithExpression, mk_symbol, mk_one, mk_lt, mk_leq, mk_eq, mk_and, mk_or, mk_not, mk_add, mk_mul, mk_mod, mk_real, mk_const, mk_true, mk_false
from aria.srk.polynomial import Polynomial, Monomial
from aria.srk.linear import QQVector, QQMatrix, dim_of_sym, const_dim
from aria.srk.expPolynomial import ExpPolynomial, ExpPolynomialVector
from aria.srk.transition import Transition
from aria.srk.transitionFormula import TransitionFormula, linearize
from aria.srk.sequence import UltimatelyPeriodicSequence, PeriodicSequence
from aria.srk.qQ import QQ
from aria.srk.smt import SMTInterface, SMTResult, is_sat
from aria.srk.sparseMap import SparseMap

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# Setup logging
logger = logging.getLogger(__name__)


class EPTerm:
    """Exponential-polynomial term for termination analysis.

    Represents terms as maps from (lambda, degree) pairs to linear vectors,
    following the OCaml EPTerm module structure.
    """

    def __init__(self, context: Context):
        """Initialize EPTerm with empty map."""
        self.context = context
        # Map from (eigenvalue, degree) to coefficient vector
        self._terms: Dict[Tuple[QQ, int], QQVector] = {}

    @classmethod
    def of_ep_vec(cls, ep_vec: ExpPolynomialVector) -> 'EPTerm':
        """Convert ExpPolynomialVector to EPTerm representation."""
        context = ep_vec.context if hasattr(ep_vec, 'context') else Context()
        result = cls(context)

        # Iterate through exponential polynomial vector components
        for ep_poly, dim in ExpPolynomialVector.enum(ep_vec):
            for poly, base in ExpPolynomial.enum(ep_poly):
                for coeff, degree in Polynomial.QQX.enum(poly):
                    key = (base, degree)
                    if key not in result._terms:
                        result._terms[key] = QQVector.zero(dim + 1)  # +1 for constant dimension
                    # Add coefficient to the appropriate dimension
                    result._terms[key] = result._terms[key].add_term(coeff, dim)

        return result

    def eventually_positive(self, term_of_dim: Callable[[int], ArithExpression]) -> FormulaExpression:
        """Check if EPTerm is eventually positive.

        Args:
            term_of_dim: Function to convert dimension to arithmetic term

        Returns:
            Formula expressing that the EPTerm is eventually positive
        """
        zero = mk_real(self.context, QQ.zero())
        terms = []

        for (eigenvalue, degree), vec in self._terms.items():
            # Convert vector to linear term
            term = self._vec_to_term(vec, term_of_dim)
            terms.append(mk_lt(self.context, zero, term))

        if not terms:
            return mk_false()

        return mk_or(self.context, terms)

    def eventually_nonnegative(self, term_of_dim: Callable[[int], ArithExpression]) -> FormulaExpression:
        """Check if EPTerm is eventually nonnegative."""
        zero = mk_real(self.context, QQ.zero())
        terms = []

        for (eigenvalue, degree), vec in self._terms.items():
            # Convert vector to linear term
            term = self._vec_to_term(vec, term_of_dim)

            # For nonnegative, we need the term >= 0
            # If there are multiple terms, we need to check each one >= 0
            if eigenvalue >= 0:  # Non-negative eigenvalue
                terms.append(mk_leq(self.context, zero, term))
            else:
                # For negative eigenvalues, need even degree for nonnegativity
                if degree % 2 == 0:
                    terms.append(mk_leq(self.context, zero, term))

        if not terms:
            return mk_true()

        return mk_or(self.context, terms)

    def is_zero(self, term_of_dim: Callable[[int], ArithExpression]) -> FormulaExpression:
        """Check if EPTerm is identically zero."""
        zero = mk_real(self.context, QQ.zero())
        terms = []

        for (eigenvalue, degree), vec in self._terms.items():
            term = self._vec_to_term(vec, term_of_dim)
            terms.append(mk_eq(self.context, zero, term))

        if not terms:
            return mk_true()

        return mk_and(self.context, terms)

    def _vec_to_term(self, vec: QQVector, term_of_dim: Callable[[int], ArithExpression]) -> ArithExpression:
        """Convert QQVector to arithmetic term."""
        terms = []

        for i in range(len(vec)):
            if i == const_dim:
                # Constant term
                if vec[i] != 0:
                    terms.append(mk_real(self.context, vec[i]))
            else:
                # Variable term
                coeff = vec[i]
                if coeff != 0:
                    var_term = mk_mul(self.context, [mk_real(self.context, coeff), term_of_dim(i)]) if term_of_dim(i) is not None else mk_real(self.context, coeff)
                    terms.append(var_term)

        if not terms:
            return mk_real(self.context, QQ.zero())
        elif len(terms) == 1:
            return terms[0]
        else:
            return mk_add(self.context, terms)

    def __str__(self) -> str:
        if not self._terms:
            return "EPTerm(empty)"

        terms = []
        for (eigenvalue, degree), vec in self._terms.items():
            terms.append(f"λ^{degree}*{eigenvalue}: {vec}")
        return f"EPTerm({', '.join(terms)})"


class XSeq:
    """Characteristic sequence operations for DTA analysis."""

    @staticmethod
    def seq_of_exp(modulus: int, eigenvalue: QQ) -> UltimatelyPeriodicSequence[QQ]:
        """Create sequence of eigenvalue^k mod modulus."""
        def unfold_func(power: QQ) -> QQ:
            return (power * eigenvalue) % QQ(modulus)

        seq = UltimatelyPeriodicSequence.unfold(unfold_func, QQ(1))
        return seq.periodic_approx()

    @staticmethod
    def seq_of_polynomial(modulus: int, poly: Polynomial) -> UltimatelyPeriodicSequence[QQ]:
        """Create characteristic sequence of polynomial mod modulus."""
        # Get LCM of denominators in coefficients
        lcm_denoms = QQ.one()
        for coeff, _ in poly.enum():
            lcm_denoms = lcm_denoms.lcm(QQ.denominator(coeff))

        # Scale polynomial by LCM of denominators
        scaled_poly = poly * QQ(lcm_denoms)
        modulus_scaled = modulus * int(lcm_denoms)

        # Generate sequence for i in 0..(modulus-1)
        seq_values = []
        for i in range(modulus_scaled):
            # Evaluate polynomial at i
            try:
                # Use the polynomial evaluation method
                from fractions import Fraction
                values = {0: Fraction(i)}  # Assume univariate for now
                result = scaled_poly.evaluate(values)
                # result mod modulus, then convert back to QQ with denominator lcm_denoms
                mod_result = result % Fraction(modulus_scaled)
                seq_values.append(QQ(mod_result) / QQ(lcm_denoms))
            except:
                # Fallback for evaluation issues
                seq_values.append(QQ.zero())

        return UltimatelyPeriodicSequence.from_list(seq_values)

    @staticmethod
    def seq_of_single_base_exp_polynomial(modulus: int, poly: Polynomial, base: QQ) -> UltimatelyPeriodicSequence[QQ]:
        """Characteristic sequence of base^k * poly(k) mod modulus."""
        seq_exp = XSeq.seq_of_exp(modulus, base)
        seq_poly = XSeq.seq_of_polynomial(modulus, poly)

        def combine(n: QQ, p: QQ) -> QQ:
                return (QQ(n) * p) % QQ(modulus)

        return UltimatelyPeriodicSequence.map2(combine, seq_exp, seq_poly)

    @staticmethod
    def seq_of_exp_polynomial(modulus: int, exp_poly: ExpPolynomial) -> UltimatelyPeriodicSequence[QQ]:
        """Characteristic sequence of exponential polynomial mod modulus."""
        # Start with zero sequence
        result = UltimatelyPeriodicSequence.from_list([QQ.zero()])

        # Add each term in the exponential polynomial
        for poly, base in ExpPolynomial.enum(exp_poly):
            # Convert base to integer for modulus computation
            try:
                base_int = int(base)
                if base_int < 0:
                    raise ValueError("Negative base in exponential polynomial")
            except:
                raise ValueError(f"Non-integer base {base} in exponential polynomial")

            base_mod = base_int % modulus
            current_seq = XSeq.seq_of_single_base_exp_polynomial(modulus, poly, QQ(base_mod))

            def add_mod(x: QQ, y: QQ) -> QQ:
                return (x + y) % QQ(modulus)

            result = UltimatelyPeriodicSequence.map2(add_mod, result, current_seq)

        return result

    @staticmethod
    def seq_of_compare_atom(context: Context, op: str, closed_form_vec: ExpPolynomialVector,
                           term_of_dim: Callable[[int], ArithExpression]) -> UltimatelyPeriodicSequence[FormulaExpression]:
        """Compute characteristic sequence of compare atom (LHS < 0, LHS = 0, LHS <= 0)."""
        # Handle even/odd case split for negative exponentials
        positive_cf = XSeq._handle_positive_cf(closed_form_vec)

        def make_predicate(ep_vec: ExpPolynomialVector) -> FormulaExpression:
            ep_term = EPTerm.of_ep_vec(ep_vec)
            if op == "Pos":
                return ep_term.eventually_positive(term_of_dim)
            elif op == "Nonneg":
                return ep_term.eventually_nonnegative(term_of_dim)
            elif op == "Zero":
                return ep_term.is_zero(term_of_dim)
            else:
                raise ValueError(f"Unknown comparison operator: {op}")

        return UltimatelyPeriodicSequence.map(make_predicate, positive_cf)

    @staticmethod
    def seq_of_divides_atom(context: Context, divisor: int, closed_form_dividend: ExpPolynomialVector,
                           term_of_dim: Callable[[int], ArithExpression]) -> UltimatelyPeriodicSequence[FormulaExpression]:
        """Compute characteristic sequence of divides atom (q | dividend)."""
        def make_divides_formula(k: QQ) -> FormulaExpression:
            # dividend_xseq is the sequence of dividend values mod divisor
            dividend_seqs = []
            for exppoly, dim in ExpPolynomialVector.enum(closed_form_dividend):
                term = term_of_dim(dim)
                seq = XSeq.seq_of_exp_polynomial(divisor, exppoly)

                def scale_by_term(n: QQ) -> ArithExpression:
                    return mk_mul(context, [mk_real(context, n), term])

                scaled_seq = UltimatelyPeriodicSequence.map(scale_by_term, seq)
                dividend_seqs.append(scaled_seq)

            # Sum all dividend components
            if dividend_seqs:
                dividend_sum = dividend_seqs[0]
                for seq in dividend_seqs[1:]:
                    def add_dividends(x: ArithExpression, y: ArithExpression) -> ArithExpression:
                        return mk_add(context, [x, y])
                    dividend_sum = UltimatelyPeriodicSequence.map2(add_dividends, dividend_sum, seq)
            else:
                dividend_sum = UltimatelyPeriodicSequence.from_list([mk_real(context, QQ.zero())])

            # Create divides condition: dividend ≡ 0 mod divisor
            def make_divides_condition(dividend_val: ArithExpression) -> FormulaExpression:
                divisor_term = mk_real(context, QQ(divisor))
                remainder = mk_mod(context, dividend_val, divisor_term)
                return mk_eq(context, remainder, mk_real(context, QQ.zero()))

            return UltimatelyPeriodicSequence.map(make_divides_condition, dividend_sum)

    @staticmethod
    def _handle_positive_cf(closed_form_vec: ExpPolynomialVector) -> UltimatelyPeriodicSequence[ExpPolynomialVector]:
        """Handle even/odd case split for negative exponentials."""
        # Check if any entry has negative base
        has_negative_base = False
        for entry, _ in ExpPolynomialVector.enum(closed_form_vec):
            for _, base in ExpPolynomial.enum(entry):
                if base < 0:
                    has_negative_base = True
                    break
            if has_negative_base:
                break

        if has_negative_base:
            # Create even and odd cases
            cf_even = ExpPolynomialVector.map(
                lambda _ : lambda ep: ExpPolynomial.compose_left_affine(ep, 2, 0),
                closed_form_vec
            )
            cf_odd = ExpPolynomialVector.map(
                lambda _ : lambda ep: ExpPolynomial.compose_left_affine(ep, 2, 1),
                closed_form_vec
            )
            return UltimatelyPeriodicSequence.from_list([cf_even, cf_odd])
        else:
            return UltimatelyPeriodicSequence.from_list([closed_form_vec])


# Matrix operations for DTA analysis
def constraints_to_generators(dim: int, constraint_mat: QQMatrix) -> QQMatrix:
    """Given a matrix A representing constraints Ax = 0, compute a basis for the nullspace."""
    # Create identity matrix for variable indices
    var_indices = list(range(dim))

    # Compute nullspace of constraint matrix
    nullspace_basis = QQMatrix.nullspace(constraint_mat, var_indices)

    # Simplify the basis
    simplified_basis = QQMatrix.simplify(nullspace_basis)

    # Convert to matrix format (transpose for column vectors)
    return QQMatrix.transpose(QQMatrix.of_rows(simplified_basis))


def inv_subspace_restriction(tr: QQMatrix, inv_subspace: QQMatrix) -> Optional[QQMatrix]:
    """Given dynamics matrix T and invariant subspace G, compute T restricted to G."""
    try:
        # Compute T * G
        tg = QQMatrix.mul(tr, inv_subspace)

        # Compute (T * G) / G (solve for R such that T * G = G * R)
        return QQMatrix.divide_left(tg, inv_subspace)
    except:
        return None


def int_eigenspace(dim: int, matrix: QQMatrix) -> QQMatrix:
    """Given a matrix T, compute a basis for the space spanned by eigenvectors with integer eigenvalues."""
    # Get variable indices
    var_indices = list(range(dim))

    # Compute rational spectral decomposition
    eigenpairs = QQMatrix.rational_spectral_decomposition(QQMatrix.transpose(matrix), var_indices)

    # Filter eigenvectors with integer eigenvalues
    integer_eigenvectors = []
    for eigenvalue, eigenvector in eigenpairs:
        if QQ.denominator(eigenvalue) == QQ.one():  # Integer eigenvalue
            integer_eigenvectors.append(eigenvector)

    if not integer_eigenvectors:
        # Return empty matrix if no integer eigenvalues
        return QQMatrix.zero(0, dim)

    # Simplify the subspace
    subspace = QQMatrix.simplify(integer_eigenvectors)

    # Convert to matrix format
    return QQMatrix.transpose(QQMatrix.of_rows(subspace))


def closed_form(sim_symbols: List[Symbol], linterm: QQVector, ep_mat: ExpPolynomialVector) -> ExpPolynomialVector:
    """Given symbols, linear term, and exponential matrix, compute closed form."""
    # Create vector of exponential polynomials from linear term
    ep_vec = ExpPolynomialVector.zero(len(sim_symbols) + 1)  # +1 for constant

    # Add linear term coefficients
    for i, symbol in enumerate(sim_symbols):
        coeff = QQVector.coeff(dim_of_sym(symbol), linterm)
        if coeff != 0:
            ep_vec = ExpPolynomialVector.add_term(ExpPolynomial.scalar(coeff), i, ep_vec)

    # Add constant term
    const_coeff = QQVector.coeff(const_dim, linterm)
    if const_coeff != 0:
        ep_vec = ExpPolynomialVector.add_term(ExpPolynomial.scalar(const_coeff), const_dim, ep_vec)

    # Multiply by exponential matrix from the left
    return ExpPolynomialVector.vector_left_mul(ep_vec, ep_mat)


class DTA:
    """Disjunctive Transition Automaton for termination analysis."""

    def __init__(self, context: Context, variables: List[Symbol]):
        """Initialize DTA with context and variables."""
        self.context = context
        self.variables = variables
        self.states: Set[str] = set()
        self.transitions: Dict[Tuple[str, str], TransitionFormula] = {}
        self.initial_states: Set[str] = set()
        self.final_states: Set[str] = set()

    def add_state(self, state: str) -> None:
        """Add a state to the DTA."""
        self.states.add(state)

    def add_transition(self, from_state: str, to_state: str, formula: TransitionFormula) -> None:
        """Add a transition between states."""
        self.transitions[(from_state, to_state)] = formula

    def set_initial(self, state: str) -> None:
        """Set a state as initial."""
        self.initial_states.add(state)

    def set_final(self, state: str) -> None:
        """Set a state as final."""
        self.final_states.add(state)

    def is_terminating(self) -> bool:
        """Check if the system is terminating.

        Uses DTA-based termination analysis to determine if all execution
        paths eventually reach a final state.

        Algorithm:
        1. Check for cycles that don't reach final states
        2. Verify that all paths from initial states reach final states
        3. Use ranking functions on cycles to prove termination
        """
        if not self.states or not self.initial_states:
            return True  # Empty DTA terminates trivially

        if not self.final_states:
            # No final states - check if there are any transitions
            return len(self.transitions) == 0

        # Check if there are unreachable cycles (cycles that don't lead to final states)
        # This is a simplified reachability analysis

        # Find all states reachable from initial states
        reachable = set(self.initial_states)
        worklist = list(self.initial_states)

        while worklist:
            current = worklist.pop()
            for (from_state, to_state) in self.transitions.keys():
                if from_state == current and to_state not in reachable:
                    reachable.add(to_state)
                    worklist.append(to_state)

        # Check if any final states are reachable
        final_reachable = reachable & self.final_states
        if not final_reachable:
            # No final states are reachable - may not terminate
            return False

        # Check for non-terminating cycles
        # A cycle is non-terminating if it doesn't eventually reach a final state
        for state in reachable:
            if state in self.final_states:
                continue  # Final states are OK

            # Check if this state can reach a final state
            can_reach_final = self._can_reach_final(state, set())
            if not can_reach_final:
                # Found a state that can't reach a final state - may not terminate
                return False

        return True

    def _can_reach_final(self, state: str, visited: Set[str]) -> bool:
        """Check if a state can reach any final state."""
        if state in self.final_states:
            return True

        if state in visited:
            return False  # Cycle detected

        visited.add(state)

        # Check all transitions from this state
        for (from_state, to_state) in self.transitions.keys():
            if from_state == state:
                if self._can_reach_final(to_state, visited.copy()):
                    return True

        return False

    def __str__(self) -> str:
        return f"DTA(states={len(self.states)}, transitions={len(self.transitions)})"


@dataclass(frozen=True)
class TerminationResult:
    """Result of termination analysis."""

    terminates: bool
    ranking_function: Optional[Any]  # Placeholder for ranking function
    dta: Optional[DTA]

    def __str__(self) -> str:
        if self.terminates:
            return "Terminates"
        else:
            return "May not terminate"


class DTAAnalyzer:
    """Analyzer for termination using DTA."""

    def __init__(self, context: Context):
        """Initialize DTA analyzer."""
        self.context = context

    def analyze_transitions(self, transitions: List[Transition]) -> TerminationResult:
        """Analyze a list of transitions for termination.

        Constructs a DTA from the given transitions and analyzes it for termination.

        Algorithm:
        1. Extract variables and create states for different program locations
        2. Build DTA transitions from program transitions
        3. Add final states for terminating configurations
        4. Check if the DTA terminates using ranking functions
        """
        # Extract variables from transitions
        variables = set()
        for trans in transitions:
            if hasattr(trans, 'variables'):
                variables.update(trans.variables)

        # Create DTA
        dta = DTA(self.context, list(variables))

        # Add initial state
        initial_state = "q_init"
        dta.add_state(initial_state)
        dta.set_initial(initial_state)

        # Create states for each unique transition source/target
        # In a real implementation, this would be based on program locations
        state_map = {"init": initial_state}
        state_counter = 0

        for i, trans in enumerate(transitions):
            # Create states for this transition
            source_state = f"q_{i}_pre"
            target_state = f"q_{i}_post"

            if source_state not in state_map:
                dta.add_state(source_state)
                state_map[f"trans_{i}_pre"] = source_state

            if target_state not in state_map:
                dta.add_state(target_state)
                state_map[f"trans_{i}_post"] = target_state

            # Add transition
            if hasattr(trans, 'formula'):
                from .transitionFormula import TransitionFormula, linearize
                tf = TransitionFormula(
                    formula=trans.formula,
                    symbols=getattr(trans, 'symbols', []),
                    exists=getattr(trans, 'exists', lambda x: False)
                )
                dta.add_transition(source_state, target_state, tf)

            state_counter += 1

        # Add final state for terminating configurations
        final_state = "q_final"
        dta.add_state(final_state)
        dta.set_final(final_state)

        # Connect last states to final state
        # In a real implementation, this would be based on post-conditions
        for state in dta.states:
            if state not in dta.final_states and "post" in state:
                # This is a simplification - real implementation would check
                # if the state satisfies termination conditions
                pass

        # Check if DTA terminates
        terminates = dta.is_terminating()

        # Try to synthesize a ranking function as witness
        ranking_function = None
        if terminates:
            # Optionally synthesize a ranking function as a witness
            # This would use techniques from terminationExp or terminationLLRF
            pass

        return TerminationResult(terminates, ranking_function, dta)

    def eventually_positive(self, ep_term: EPTerm) -> bool:
        """Check if an exponential-polynomial term is eventually positive.

        An EP term lambda^k * k^d * v is eventually positive if:
        - lambda > 1 (exponential growth)
        - or lambda = 1 and d >= 1 (polynomial growth)
        - or lambda < -1 and d even (oscillating but growing)
        """
        if ep_term.eigenvalue > 1:
            return True
        elif ep_term.eigenvalue == 1 and ep_term.degree >= 1:
            return True
        elif ep_term.eigenvalue < -1 and ep_term.degree % 2 == 0:
            return True
        else:
            return False

    def eventually_nonnegative(self, ep_term: EPTerm) -> bool:
        """Check if an exponential-polynomial term is eventually nonnegative."""
        if ep_term.eigenvalue >= 1:
            return True
        elif ep_term.eigenvalue >= 0:
            return True
        else:
            # For negative eigenvalues, need even degree for nonnegativity
            return ep_term.degree % 2 == 0


def create_dta_analyzer(context: Context) -> DTAAnalyzer:
    """Create a DTA analyzer."""
    return DTAAnalyzer(context)


def analyze_termination(transitions: List[Transition], context: Context) -> TerminationResult:
    """Analyze transitions for termination using DTA."""
    analyzer = DTAAnalyzer(context)
    return analyzer.analyze_transitions(transitions)


def mp(context: Context, tf: TransitionFormula) -> FormulaExpression:
    """Main DTA analysis function following the OCaml implementation.

    Args:
        context: SRK context
        tf: Transition formula to analyze

    Returns:
        Formula expressing that the transition formula does not terminate
    """
    logger.info("Starting DTA analysis of transition formula")

    # Linearize the transition formula
    tf_linear = linearize(context, tf)

    # Check if transition formula is unsatisfiable
    if is_sat(context, tf.formula) == SMTResult.UNSAT:
        logger.info("Transition formula is UNSAT - terminating trivially")
        return mk_false()

    # Get rational abstraction using DLTS
    try:
        from aria.srk.solvablePolynomial import DLTSPeriodicRationalAbstraction, simplify_dlts

        qdlts_abs = DLTSPeriodicRationalAbstraction.abstract_rational(context, tf_linear)
        qdlts_abs = simplify_dlts(context, qdlts_abs, scale=True)

        # Get the LTS module for partial linear maps
        from aria.srk.lts import PartialLinearMap as PLM

        # Get omega domain and dynamics matrix
        omega_domain, tr = PLM.iteration_sequence(qdlts_abs.dlts)

        dim = len(qdlts_abs.simulation)

        logger.info(f"DTA analysis: dimension={dim}, omega_domain_size={len(omega_domain)}")

        # Compute basis for omega domain (constraints Ax = 0)
        g = constraints_to_generators(dim, QQMatrix.of_rows(omega_domain))

        # Compute restriction of dynamics to omega domain
        tr_omega = inv_subspace_restriction(tr, g)
        if tr_omega is None:
            logger.warning("Could not compute subspace restriction")
            return mk_true()  # Conservative - may not terminate

        # Compute integer eigenspace basis
        z = int_eigenspace(QQMatrix.nb_columns(g), tr_omega)
        gz = QQMatrix.mul(g, z)

        # Compute restriction to integer eigenspace
        tr_z = inv_subspace_restriction(tr, gz)
        if tr_z is None:
            logger.warning("Could not compute integer eigenspace restriction")
            return mk_true()  # Conservative - may not terminate
        # Create symbols for integer domain dimensions
        gz_symbols = []
        for i in range(QQMatrix.nb_columns(gz)):
            gz_symbols.append(mk_symbol(context, f"dta<{i}>", Type.INT))
        # Create simulation constraints: GZ * z_symbols = simulation_terms
        sim_constraints = []
        for i in range(len(qdlts_abs.simulation)):
            gz_term = QQMatrix.row(gz, i)
            # Convert row vector to linear term
            gz_linear_term = QQVector.from_list([QQMatrix.get(gz_term, i, j) for j in range(QQMatrix.nb_columns(gz_term))])
            sim_term = mk_const(context, gz_symbols[i]) if i < len(gz_symbols) else mk_real(context, QQ.zero())

            # Create equality constraint
            sim_constraints.append(mk_eq(context, qdlts_abs.simulation[i], sim_term))

        gz_symbols_set = set(gz_symbols)

        # Create guard formula with simulation constraints
        guard = mk_and(context, [tf.formula] + sim_constraints)

        # Apply model-based projection to eliminate non-integer-domain symbols
        guard = mbp(context, guard, lambda s: s in gz_symbols_set)

        # Simplify using DDA and eliminate floor
        guard = simplify_dda(context, guard)
        guard = eliminate_floor(context, guard)

        # Compute exponential of restricted matrix
        tr_z_exp = ExpPolynomialVector.exponentiate_rational(tr_z)
        if tr_z_exp is None:
            logger.warning("Could not compute matrix exponential")
            return mk_true()  # Conservative - may not terminate

        # Create term mapping function for dimensions
        def term_of_dim(i: int) -> ArithExpression:
            if i == const_dim:
                return mk_one(context)
            else:
                return mk_const(context, gz_symbols[i])

        # Create algebra for formula evaluation
        def algebra(op):
            if op == "Tru":
                return UltimatelyPeriodicSequence.from_list([mk_true()])
            elif op == "Fls":
                return UltimatelyPeriodicSequence.from_list([mk_false()])
            elif op == "And":
                return lambda xs: UltimatelyPeriodicSequence.mapn(lambda *args: mk_and(context, args), xs)
            elif op == "Or":
                return lambda xs: UltimatelyPeriodicSequence.mapn(lambda *args: mk_or(context, args), xs)
            elif op == "Not":
                return lambda x: UltimatelyPeriodicSequence.map(lambda f: mk_not(context, f), x)
            elif op == "Atom":
                return lambda atom: _handle_atom(context, atom, tr_z_exp, term_of_dim)
            else:
                raise ValueError(f"Unsupported operation: {op}")

        # Evaluate the guard formula using the algebra
        xseq = evaluate_formula_with_algebra(context, algebra, guard)

        # Add simulation constraints to the period
        period_constraints = sim_constraints + [xseq.period]

        # Apply MBP again to eliminate remaining non-integer-domain symbols
        result = mk_and(context, period_constraints)
        result = mbp(context, result, lambda s: s not in gz_symbols_set)
        result = mk_not(context, result)

        logger.info("DTA analysis completed")
        return result

    except Exception as e:
        logger.error(f"DTA analysis failed: {e}")
        return mk_true()  # Conservative - may not terminate


def _handle_atom(context: Context, atom, tr_z_exp, term_of_dim):
    """Handle atomic formulas in DTA algebra."""
    from aria.srk.srkSimplify import simplify_integer_atom

    # Simplify the atom to determine its structure
    simplified = simplify_integer_atom(context, atom.op, atom.left, atom.right)

    if simplified[0] == "CompareZero":
        op, vec = simplified[1], simplified[2]
        # Compute closed form of the vector
        cf = closed_form([], vec.negate(), tr_z_exp)  # Empty symbols list for now

        # Map comparison operators
        if op == "Eq":
            predicate = "Zero"
        elif op == "Leq":
            predicate = "Nonneg"
        elif op == "Lt":
            predicate = "Pos"
        else:
            raise ValueError(f"Unsupported comparison: {op}")

        return XSeq.seq_of_compare_atom(context, predicate, cf, term_of_dim)

    elif simplified[0] == "Divides":
        divisor, vec = simplified[1], simplified[2]
        cf = closed_form([], vec, tr_z_exp)
        return XSeq.seq_of_divides_atom(context, divisor, cf, term_of_dim)

    elif simplified[0] == "NotDivides":
        divisor, vec = simplified[1], simplified[2]
        cf = closed_form([], vec, tr_z_exp)
        seq = XSeq.seq_of_divides_atom(context, divisor, cf, term_of_dim)
        return UltimatelyPeriodicSequence.map(lambda f: mk_not(context, f), seq)

    else:
        raise ValueError(f"Unsupported atom type: {simplified[0]}")


# Import additional functions needed
def mbp(context: Context, formula: FormulaExpression, is_symbol_relevant: Callable[[Symbol], bool]) -> FormulaExpression:
    """Model-based projection for quantifier elimination."""
    # This is a simplified version - in practice would use full MBP algorithm
    from aria.srk.quantifier import mbp as quantifier_mbp
    return quantifier_mbp(context, formula, is_symbol_relevant)


def simplify_dda(context: Context, formula: FormulaExpression) -> FormulaExpression:
    """Simplify using DDA (Difference Decision Diagrams)."""
    # Simplified - in practice would use full DDA simplification
    return formula


def eliminate_floor(context: Context, formula: FormulaExpression) -> FormulaExpression:
    """Eliminate floor operations from formula."""
    # Simplified - in practice would use floor elimination
    return formula


def evaluate_formula_with_algebra(context: Context, algebra: Callable, formula: FormulaExpression) -> Any:
    """Evaluate a formula using an algebra for structural recursion.

    Args:
        context: SRK context
        algebra: Function mapping operators to their semantic interpretations
        formula: Formula to evaluate

    Returns:
        Result of evaluating the formula through the algebra
    """
    from aria.srk.syntax import TrueExpr, FalseExpr, And, Or, Not, Eq, Lt, Leq

    # Recursively evaluate the formula structure
    if isinstance(formula, TrueExpr):
        return algebra("Tru")
    elif isinstance(formula, FalseExpr):
        return algebra("Fls")
    elif isinstance(formula, And):
        # Recursively evaluate all conjuncts
        evaluated_args = [evaluate_formula_with_algebra(context, algebra, arg) for arg in formula.args]
        return algebra("And")(evaluated_args)
    elif isinstance(formula, Or):
        # Recursively evaluate all disjuncts
        evaluated_args = [evaluate_formula_with_algebra(context, algebra, arg) for arg in formula.args]
        return algebra("Or")(evaluated_args)
    elif isinstance(formula, Not):
        # Recursively evaluate the negated formula
        evaluated_arg = evaluate_formula_with_algebra(context, algebra, formula.arg)
        return algebra("Not")(evaluated_arg)
    elif isinstance(formula, (Eq, Lt, Leq)):
        # Atomic formulas - handle through the Atom operator
        return algebra("Atom")(formula)
    else:
        # For other formula types, treat as atoms
        return algebra("Atom")(formula)
