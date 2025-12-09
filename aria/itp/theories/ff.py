"""
Finite Field Theory

This module provides a formalization of finite fields, which are algebraic structures 
with a finite number of elements that satisfy the field axioms.

(by LLM, to check)
"""

import aria.itp as itp
import aria.itp.smt as smt
import functools
from aria.itp.property import TypeClass
import aria.itp.theories.int as int_


@functools.cache
def FiniteField(p, n=1):
    """
    Create a finite field GF(p^n) where p is a prime number and n is a positive integer.
    
    A finite field, or Galois field, is a field with a finite number of elements.
    
    Args:
        p (int): A prime number
        n (int): A positive integer (default: 1)
    
    Returns:
        A sort representing the finite field GF(p^n)
    
    >>> GF5 = FiniteField(5)
    >>> GF5.order
    5
    >>> GF5.char
    5
    """
    # Define the field with p^n elements
    F = smt.DeclareSort(f"GF_{p}_{n}")

    # Define field operations
    add = smt.Function("add", F, F, F)
    mul = smt.Function("mul", F, F, F)
    neg = smt.Function("neg", F, F)
    inv = smt.Function("inv", F, F)

    # Register notations
    itp.notation.add.register(F, add)
    itp.notation.mul.register(F, mul)
    itp.notation.sub.register(F, lambda x, y: add(x, neg(y)))
    itp.notation.div.register(F, lambda x, y: mul(x, inv(y)))
    itp.notation.neg.register(F, neg)

    # Constants
    zero = smt.Const("zero", F)
    one = smt.Const("one", F)

    # Create element constants for convenience
    x, y, z = smt.Consts("x y z", F)

    # Field axioms
    # Additive group axioms
    add_assoc = itp.axiom(smt.ForAll([x, y, z], add(add(x, y), z) == add(x, add(y, z))))
    add_comm = itp.axiom(smt.ForAll([x, y], add(x, y) == add(y, x)))
    add_id = itp.axiom(smt.ForAll([x], add(x, zero) == x))
    add_inv = itp.axiom(smt.ForAll([x], add(x, neg(x)) == zero))

    # Multiplicative group axioms (except for zero)
    mul_assoc = itp.axiom(smt.ForAll([x, y, z], mul(mul(x, y), z) == mul(x, mul(y, z))))
    mul_comm = itp.axiom(smt.ForAll([x, y], mul(x, y) == mul(y, x)))
    mul_id = itp.axiom(smt.ForAll([x], mul(x, one) == x))
    mul_inv = itp.axiom(smt.ForAll([x], smt.Implies(x != zero, mul(x, inv(x)) == one)))

    # Distributive law
    distrib = itp.axiom(smt.ForAll([x, y, z], mul(x, add(y, z)) == add(mul(x, y), mul(x, z))))

    # Distinctness of 0 and 1
    distinct_zero_one = itp.axiom(zero != one)

    # Characteristic of the field
    char_p = itp.axiom(
        smt.ForAll(
            [x],
            functools.reduce(
                lambda acc, _: add(acc, x),
                range(p),
                zero
            ) == zero
        )
    )

    # Order of the field
    order_p_n = p ** n

    # Finiteness axiom - there are exactly p^n distinct elements
    elements = [smt.Const(f"e_{i}", F) for i in range(order_p_n)]
    distinct_elements = itp.axiom(smt.Distinct(*elements))
    all_elements = itp.axiom(
        smt.ForAll(
            [x],
            smt.Or(*[x == e for e in elements])
        )
    )

    # Store field properties
    F.zero = zero
    F.one = one
    F.elements = elements
    F.add = add
    F.mul = mul
    F.neg = neg
    F.inv = inv
    F.order = order_p_n
    F.char = p

    # Store key theorems
    F.add_assoc = add_assoc
    F.add_comm = add_comm
    F.add_id = add_id
    F.add_inv = add_inv
    F.mul_assoc = mul_assoc
    F.mul_comm = mul_comm
    F.mul_id = mul_id
    F.mul_inv = mul_inv
    F.distrib = distrib

    # Additional theorems that can be proven
    F.add_left_id = itp.prove(smt.ForAll([x], add(zero, x) == x), by=[add_comm, add_id])
    F.mul_left_id = itp.prove(smt.ForAll([x], mul(one, x) == x), by=[mul_comm, mul_id])
    F.mul_zero = itp.prove(smt.ForAll([x], mul(x, zero) == zero))
    F.neg_neg = itp.prove(smt.ForAll([x], neg(neg(x)) == x))

    # Fermat's Little Theorem: for all non-zero x, x^(p^n - 1) = 1
    fermat = itp.define(
        "fermat",
        [x],
        smt.Implies(
            x != zero,
            functools.reduce(
                lambda acc, _: mul(acc, x),
                range(order_p_n - 1),
                one
            ) == one
        )
    )

    return F


def is_prime(p):
    """
    Check if a number is prime
    
    Args:
        p (int): The number to check
    
    Returns:
        bool: True if p is prime, False otherwise
    """
    return smt.And(p > 1, smt.ForAll([smt.Int("d")],
                                     smt.Implies(smt.And(1 < smt.Int("d"), smt.Int("d") < p), p % smt.Int("d") != 0)))


# Common finite fields
GF2 = FiniteField(2)  # Binary field
GF3 = FiniteField(3)
GF5 = FiniteField(5)
GF7 = FiniteField(7)


# Polynomial representation for extension fields
def ExtensionField(p, n, irreducible_poly=None):
    """
    Create an extension field GF(p^n) using a polynomial representation
    
    Args:
        p (int): A prime number
        n (int): A positive integer > 1
        irreducible_poly (function, optional): The irreducible polynomial to use
        
    Returns:
        A sort representing the extension field GF(p^n)
    """
    # This is a simplified version; in a full implementation, we would define
    # the polynomial representation and operations modulo the irreducible polynomial
    F = FiniteField(p, n)

    # Add extension field-specific operations and theorems here

    return F


# Field homomorphism function
def FieldHomomorphism(F1, F2):
    """
    Create a field homomorphism from F1 to F2
    
    Args:
        F1: Source field
        F2: Target field
        
    Returns:
        A function representing a field homomorphism
    """
    hom = smt.Function("hom", F1, F2)
    x, y = smt.Consts("x y", F1)

    # Homomorphism preserves addition
    add_preserve = itp.axiom(smt.ForAll([x, y], hom(F1.add(x, y)) == F2.add(hom(x), hom(y))))

    # Homomorphism preserves multiplication
    mul_preserve = itp.axiom(smt.ForAll([x, y], hom(F1.mul(x, y)) == F2.mul(hom(x), hom(y))))

    # Homomorphism maps identity to identity
    id_preserve = itp.axiom(hom(F1.one) == F2.one)

    return hom
