"""
Transition formulas for representing binary relations on program states.

This module implements transition formulas as used in program verification
and abstract interpretation. A transition formula represents a binary relation
over pre-state and post-state variables, with support for Skolem constants
and symbolic constants.

This follows the OCaml transitionFormula.ml module.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Callable, TypeVar, Any
from dataclasses import dataclass, field

T = TypeVar('T')


@dataclass(frozen=True)
class TransitionFormula:
    """
    A transition formula representing a binary relation on states.
    
    A transition formula consists of:
    - formula: A logical formula over pre and post-state variables
    - symbols: A list of (pre_symbol, post_symbol) pairs
    - exists: A predicate identifying existentially quantified (Skolem) symbols
    """

    # The underlying logical formula
    formula: Any  # FormulaExpression type
    
    # Pre-state to post-state symbol mapping
    symbols: List[Tuple[Any, Any]]  # List[(Symbol, Symbol)]
    
    # Predicate identifying Skolem constants (existentially quantified)
    exists: Callable[[Any], bool] = field(default=lambda s: True)

    def pre_symbols(self) -> Set[Any]:
        """Get the set of pre-state symbols."""
        return {pre for pre, _ in self.symbols}

    def post_symbols(self) -> Set[Any]:
        """Get the set of post-state symbols."""
        return {post for _, post in self.symbols}

    def symbolic_constants(self, srk: Any) -> Set[Any]:
        """
        Get the set of symbolic constants in the formula.
        
        Symbolic constants are symbols that:
        1. Appear in the formula
        2. Are not pre-state or post-state variables
        3. Satisfy the exists predicate (are existentially quantified)
        """
        from .syntax import symbols as get_symbols
        
        all_symbols = get_symbols(self.formula)
        pre_symbols = self.pre_symbols()
        post_symbols = self.post_symbols()
        
        # Filter for existentially quantified non-state symbols
        return {s for s in all_symbols 
                if s not in pre_symbols and s not in post_symbols and self.exists(s)}

    def is_symbolic_constant(self, symbol: Any) -> bool:
        """Check if a symbol is a symbolic constant."""
        pre_symbols = self.pre_symbols()
        post_symbols = self.post_symbols()
        return (self.exists(symbol) and 
                symbol not in pre_symbols and 
                symbol not in post_symbols)

    def __str__(self) -> str:
        return f"TransitionFormula({self.formula}, {len(self.symbols)} var pairs)"


def make(formula: Any, symbols: List[Tuple[Any, Any]], 
         exists: Optional[Callable[[Any], bool]] = None) -> TransitionFormula:
    """
    Construct a transition formula.
    
    Args:
        formula: The logical formula
        symbols: List of (pre_symbol, post_symbol) pairs
        exists: Predicate for existential quantification (default: all True)
    """
    if exists is None:
        exists = lambda s: True
    
    return TransitionFormula(formula=formula, symbols=symbols, exists=exists)


def identity(srk: Any, symbols: List[Tuple[Any, Any]]) -> TransitionFormula:
    """
    Create an identity transition formula (pre-state = post-state for all variables).
    
    Args:
        srk: The context/builder
        symbols: List of (pre_symbol, post_symbol) pairs
    """
    from .syntax import mk_eq, mk_const, mk_and
    
    # Create equality constraints for all symbol pairs
    equalities = [
        mk_eq(srk, mk_const(srk, pre), mk_const(srk, post))
        for pre, post in symbols
    ]
    
    # Combine with conjunction
    if not equalities:
        from .syntax import mk_true
        formula = mk_true(srk)
    else:
        formula = mk_and(srk, equalities)
    
    exists = lambda s: True
    return TransitionFormula(formula=formula, symbols=symbols, exists=exists)


def zero(srk: Any, symbols: List[Tuple[Any, Any]]) -> TransitionFormula:
    """
    Create a zero transition formula (always false).
    
    Args:
        srk: The context/builder
        symbols: List of (pre_symbol, post_symbol) pairs
    """
    from .syntax import mk_false
    
    formula = mk_false(srk)
    exists = lambda s: True
    return TransitionFormula(formula=formula, symbols=symbols, exists=exists)


def pre_symbols(tr_symbols: List[Tuple[Any, Any]]) -> Set[Any]:
    """Extract pre-state symbols from symbol pairs."""
    return {pre for pre, _ in tr_symbols}


def post_symbols(tr_symbols: List[Tuple[Any, Any]]) -> Set[Any]:
    """Extract post-state symbols from symbol pairs."""
    return {post for _, post in tr_symbols}


def post_map(srk: Any, tr_symbols: List[Tuple[Any, Any]]) -> Dict[Any, Any]:
    """Create a mapping from pre-state symbols to their post-state counterparts."""
    from .syntax import mk_const
    return {sym: mk_const(srk, sym_prime) for sym, sym_prime in tr_symbols}


def pre_map(srk: Any, tr_symbols: List[Tuple[Any, Any]]) -> Dict[Any, Any]:
    """Create a mapping from post-state symbols to their pre-state counterparts."""
    from .syntax import mk_const
    return {sym_prime: mk_const(srk, sym) for sym, sym_prime in tr_symbols}


def mul(srk: Any, tf1: TransitionFormula, tf2: TransitionFormula) -> TransitionFormula:
    """
    Compose two transition formulas sequentially (multiplicatively).
    
    This implements the sequential composition tf1 ; tf2.
    The composition requires introducing intermediate "mid" variables
    to connect the post-state of tf1 with the pre-state of tf2.
    
    Args:
        srk: The context/builder
        tf1: First transition formula
        tf2: Second transition formula
        
    Returns:
        Composed transition formula
    """
    from .syntax import mk_and, mk_const, mk_symbol, substitute, substitute_const, symbols as get_symbols, typ_symbol, show_symbol
    from .memo import memo
    
    if tf1.symbols != tf2.symbols:
        raise ValueError(f"TransitionFormula.mul: incompatible transition formulas - "
                        f"tf1 has {len(tf1.symbols)} symbols, tf2 has {len(tf2.symbols)} symbols")
    
    fresh_symbols: Set[Any] = set()
    
    # Create substitution maps for the composition
    map1 = {}  # Maps post-state vars of tf1 to mid vars
    map2 = {}  # Maps pre-state vars of tf2 to mid vars
    
    for sym, sym_prime in tf1.symbols:
        # Create a fresh "mid" symbol for the intermediate state
        mid_name = f"mid_{show_symbol(srk, sym)}"
        mid_symbol = mk_symbol(srk, name=mid_name, typ=typ_symbol(srk, sym))
        fresh_symbols.add(mid_symbol)
        
        mid = mk_const(srk, mid_symbol)
        map1[sym_prime] = mid  # Post-state of tf1 -> mid
        map2[sym] = mid        # Pre-state of tf2 -> mid
    
    # Substitute in tf1: replace post-state vars with mid vars
    subst1 = substitute(srk, map1, tf1.formula)
    
    # For tf2, we need to:
    # 1. Replace pre-state vars with mid vars (map2)
    # 2. Rename Skolem constants to avoid conflicts
    
    # Create a renaming function for Skolem constants in tf2
    renamed_skolems: Dict[Any, Any] = {}
    
    def rename_skolem(x: Any) -> Any:
        """Rename a symbol (used for Skolem constants in tf2)."""
        if x in map2:
            # It's a pre-state var that should map to mid
            return map2[x]
        elif tf2.exists(x):
            # It's a Skolem constant in tf2 - keep it
            return mk_const(srk, x)
        else:
            # It's a symbolic constant - rename it
            if x not in renamed_skolems:
                fresh = mk_symbol(srk, name=show_symbol(srk, x), typ=typ_symbol(srk, x))
                fresh_symbols.add(fresh)
                renamed_skolems[x] = fresh
                return mk_const(srk, fresh)
            else:
                return mk_const(srk, renamed_skolems[x])
    
    subst2 = substitute_const(srk, rename_skolem, tf2.formula)
    
    # Combine the formulas
    combined_formula = mk_and(srk, [subst1, subst2])
    
    # The exists predicate for the result: true for both original existentials
    # but false for the freshly introduced mid variables
    def combined_exists(x: Any) -> bool:
        return (tf1.exists(x) or tf2.exists(x)) and x not in fresh_symbols
    
    return TransitionFormula(
        formula=combined_formula,
        symbols=tf1.symbols,
        exists=combined_exists
    )


def add(srk: Any, tf1: TransitionFormula, tf2: TransitionFormula) -> TransitionFormula:
    """
    Union two transition formulas (non-deterministic choice).
    
    This implements tf1 + tf2 (disjunction).
    
    Args:
        srk: The context/builder
        tf1: First transition formula
        tf2: Second transition formula
        
    Returns:
        Union of the two transition formulas
    """
    from .syntax import mk_or
    
    if tf1.symbols != tf2.symbols:
        raise ValueError(f"TransitionFormula.add: incompatible transition formulas - "
                        f"tf1 has {len(tf1.symbols)} symbols, tf2 has {len(tf2.symbols)} symbols")
    
    # Disjoin the formulas
    union_formula = mk_or(srk, [tf1.formula, tf2.formula])
    
    return TransitionFormula(
        formula=union_formula,
        symbols=tf1.symbols,
        exists=tf1.exists
    )


def linearize(srk: Any, tf: TransitionFormula) -> TransitionFormula:
    """
    Linearize a transition formula to linear arithmetic.
    
    Args:
        srk: The context/builder
        tf: Transition formula to linearize
        
    Returns:
        Linearized transition formula
    """
    try:
        from .nonlinear import linearize as linearize_formula
        linearized_formula = linearize_formula(srk, tf.formula)
        return TransitionFormula(
            formula=linearized_formula,
            symbols=tf.symbols,
            exists=tf.exists
        )
    except ImportError:
        # If nonlinear module not available, return as-is
        return tf


def map_formula(f: Callable[[Any], Any], tf: TransitionFormula) -> TransitionFormula:
    """Apply a transformation to the formula."""
    return TransitionFormula(
        formula=f(tf.formula),
        symbols=tf.symbols,
        exists=tf.exists
    )


def wedge_hull(srk: Any, tf: TransitionFormula) -> Any:
    """
    Compute the wedge hull of a transition formula.
    
    This abstracts the transition formula using the wedge domain,
    eliminating post-state variables.
    
    Args:
        srk: The context/builder
        tf: Transition formula
        
    Returns:
        Formula representing the wedge hull
    """
    try:
        from .wedge import abstract as wedge_abstract
        
        post_syms = post_symbols(tf.symbols)
        
        # Define subterm predicate: symbols that are not post-state variables
        def subterm(x: Any) -> bool:
            return x not in post_syms
        
        return wedge_abstract(srk, tf.formula, exists=tf.exists, subterm=subterm)
    except ImportError:
        # If wedge module not available, return formula as-is
        return tf.formula


def preimage(srk: Any, tf: TransitionFormula, state: Any) -> Any:
    """
    Compute the preimage of a state formula under a transition formula.
    
    Given a transition formula TF and a state formula φ(post-vars),
    compute the formula ψ(pre-vars) such that:
      ψ(x) ⟺ ∃x'. TF(x, x') ∧ φ(x')
    
    This is a critical operation for symbolic reachability analysis.
    
    Args:
        srk: The context/builder
        tf: Transition formula
        state: State formula over post-state variables
        
    Returns:
        Preimage formula over pre-state variables
    """
    from .syntax import mk_and, mk_const, mk_symbol, substitute_const, typ_symbol, show_symbol
    from .memo import memo
    
    # Linearize the transition formula first
    tf = linearize(srk, tf)
    
    # Create fresh Skolem constants for pre-state variables
    fresh_skolem = memo(lambda sym: mk_const(srk, mk_symbol(
        srk,
        name=show_symbol(srk, sym),
        typ=typ_symbol(srk, sym)
    )))
    
    # Create mappings for substitution
    post_to_pre = {sym_prime: sym for sym, sym_prime in tf.symbols}
    pre_to_fresh = {sym: fresh_skolem(sym) for sym, _ in tf.symbols}
    
    # Substitute post-state vars in the transition formula with fresh Skolems
    def subst_tf(sym: Any) -> Any:
        """Substitution for transition formula."""
        if sym in post_to_pre:
            # Post-state var: map to corresponding fresh Skolem
            pre_sym = post_to_pre[sym]
            return pre_to_fresh[pre_sym]
        else:
            # Other symbol: leave as-is
            return mk_const(srk, sym)
    
    substituted_tf = substitute_const(srk, subst_tf, tf.formula)
    
    # Substitute post-state vars in the state formula
    def subst_state(sym: Any) -> Any:
        """Substitution for state formula."""
        if tf.exists(sym):
            # Symbol exists in tf
            if sym in post_to_pre:
                # It's a post-state var: map to fresh Skolem
                pre_sym = post_to_pre[sym]
                return pre_to_fresh[pre_sym]
            else:
                # It's another existential: leave as-is
                return mk_const(srk, sym)
        else:
            # It's a symbolic constant: create fresh Skolem
            return fresh_skolem(sym)
    
    substituted_state = substitute_const(srk, subst_state, state)
    
    # Combine: TF(fresh_pre) ∧ state(fresh_pre)
    result = mk_and(srk, [substituted_tf, substituted_state])
    
    return result


# Aliases for OCaml-style names
formula = lambda tf: tf.formula
symbols_of = lambda tf: tf.symbols
exists_of = lambda tf: tf.exists
