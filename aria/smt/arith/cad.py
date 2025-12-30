"""
Cylindrical algebraic decomposition in Python

# Implementation of cylindrical algebraic decomposition (CAD) in pure Python,
# based on SymPy as the computer algebra backend.
#
#
# CAD is a fundamental algorithm: it can test the consistency of a
# semialgebraic set, i.e., a system defined by Boolean combinations of
# multivariate polynomials. This makes it extremely general and powerful,
# but also means it scales poorly.
#
# Currently, there are few implementations of CAD available. There is one
# in Mathematica; there is also something called QEPCAD, developed by
# Christopher W. Brown, who was key in the development of CAD; as well,
# satisfiability modulo theories (SMT) solvers like Z3 by Microsoft also
# implement it as the theory solver for nonlinear real arithmetic (NRA).
#
# CAD decomposes ℝⁿ into *cells* over which each polynomial is
# sign-invariant, returning a *sample point* in each of these cells.
# Then, the consistency of the system can be checked over each of these
# cells just by plugging in the sample point representing it. In this way,
# a satisfying point for a semialgebraic set can be found, or none exists
# can be returned. That's what this CAD implementation does.
#
# There is a further step in CAD which goes beyond just satisfiability,
# and actually constructs a *solution formula* that describes the
# semialgebraic set; this is required to perform quantifier elimination
# over the reals. This has not been implemented yet. To current knowledge,
# only Mathematica and QEPCAD do. That step is quite complicated.

The main interface is `solve_poly_system_cad` which takes a list of
polynomials and a list of variables. The other functions are all background
to this. By default, it returns a single satisfying point, if it exists,
otherwise an empty list.

>>> import cad
>>> from sympy.abc import x,y,z
>>> from sympy import *
>>>
>>> cad.solve_poly_system_cad([x**2 + 1 > 0], [x])
[{x: 0}]
>>>
>>> cad.solve_poly_system_cad([x*y**2 - 3 < 1, x**5 + x**3 - x*y + 5 < 0], [x,y])
[{y: -2, x: -2}]
>>>
>>> cad.solve_poly_system_cad([x**3 + y**2 + z**5 - z**3 > 3, x+y+z<1], [x,y,z])
[{z: -3, y: -16, x: 0}]
>>>
>>> cad.solve_poly_system_cad([-x**2-1 > 0], [x])
[]
"""
from sympy.core import S, expand
from sympy.core.numbers import Integer, Rational
from sympy.core.function import diff
from sympy.core.relational import Relational
from sympy.polys.rootoftools import ComplexRootOf
from sympy.polys import Poly, real_roots, nroots
from sympy.polys.polytools import LT, LC, degree, subresultants, factor_list
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import PolynomialError
from sympy.functions.elementary.integers import floor, ceiling

def solve_poly_system_cad(seq, gens, return_one_sample=True):
    """
    Solves a system of polynomial inequalities/equalities via
    cylindrical algebraic decomposition. Returns a sample point
    in one (if return_one_sample=True) or all cells over which
    the system holds. If the system is invalid over all cells, then
    we return None.

    Parameters
    ==========

    seq: a list/tuple/set
        Listing all the (in)equalities that are needed to be solved
    gens: generators
        Generators of the (in)equalities in seq for which we want the
        solutions
    return_one_sample: bool
        If True, returns a single satisfying point. If False, returns
        a sample point from each CAD cell over which the system holds.


    Returns
    =======

    List[Dict]
        a list of dicts with the returned sample points. Each dict
        is a point, with the keys being the variables. If the system
        is unsatisfiable, then an empty list is returned.

    Examples
    ========

    >>> from sympy.abc import x,y
    >>> from sympy.solvers.polysys import solve_poly_system_cad

    >>> solve_poly_system_cad([-x**2-1 > 0], [x])
    []

    >>> solve_poly_system_cad([x**2-1 > 0], [x])
    [{x: -2}]

    >>> solve_poly_system_cad([x**2-1 > 0], [x], False)
    [{x: -2}, {x: 2}]

    >>> solve_poly_system_cad([y*x**2>0, x+y<1], [x,y], False)
    [{x: -1, y: 1/2}, {x: 1/4, y: 1/2}, {x: -1, y: 1}, {x: -2, y: 2}]
    """
    # prepare the atoms
    atoms = [Poly(p.lhs - p.rhs, gens) for p in seq]
    rels = [p.rel_op for p in seq]

    sample_points = cylindrical_algebraic_decomposition(atoms, gens)
    valid_samples = []

    for sample in sample_points:
        atoms_alg = [simplify_alg_sub(atom, sample) for atom in atoms]

        if all(Relational(atom, 0, rel) for atom, rel in zip(atoms_alg, rels)):
            valid_samples.append(sample)
            if return_one_sample:
                break

    return valid_samples



# HONG PROJECTOR OPERATOR AND OPERATIONS USED FOR IT


def red(f, mvar):
    """
    The reductum of a function f, as treated as a univariate function
    of the main variable (mvar), is red(f) = f - lt(f), where lt is
    the leading term.

    Parameters
    ==========

    f: Expr or Poly
        A polynomial
    mvar: a generator
        The "main variable".
        Polynomials are treated as univariate in the mvar.


    Returns
    =======

    Poly
        The reductum.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.solvers.polysys import red

    >>> red(x**3 + x**2 + 3*x, x)
    x**2 + 3*x
    """
    return f - LT(f, mvar)


def red_set(f, mvar):
    """
    The set of reducta of a function f is defined recursively.

    The ith level reducta of f, red^i(f), is defined recursively.
    red^0(f) = f
    red^i(f) = red(red^{i-1}(f))

    The reducta set RED(f) is defined as: {red^i(f) | 0 <= i <= deg(f)}.

    This function returns RED(f). Note the ith level reductum, if
    needed, can be accessed by indexing from the reducta set.

    Parameters
    ==========

    f: Expr or Poly
        A polynomial
    mvar: a generator
        The "main variable".
        Polynomials are treated as univariate in the mvar.


    Returns
    =======

    Poly
        The reducta set.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.solvers.polysys import red_set

    >>> red_set(x**3 + x**2 + 3*x, x)
    [x**3 + x**2 + 3*x, x**2 + 3*x, 3*x, 0]
    """

    reds = []

    # handle the constant case here
    # because otherwise sympy says deg= -infty
    try:
        if f.is_number:
            return []
    except AttributeError:  # if its not a sympy Basic object, then also return []
        return []

    for i in range(degree(f, mvar) + 1):
        if i == 0:
            reds.append(f)
        else:
            reds.append(red(reds[i-1], mvar))

    return reds


def subresultant_polynomials(f, g, mvar):
    """
    Computes the subresultant polynomials themselves. It uses the
    subresultant PRS which is already built into SymPy.

    Assume without loss of generality that the degree of g is
    no more than the degree of f (in this function, we gracefully
    handle the opposite case by swapping them). Then, we can compute
    the subresultant polynomials from the subresultant PRS.

    The remainder r_i is the deg(r_{i-1})-th subresultant polynomial.
    Additionally, if deg(r_i) < deg(r_{i-1}) - 1, then the deg(r_i)
    subresultant polynomial is r_i * LC(r_i) ^ c_i, where
    c_i = deg(r_{i-1})-deg(r_i)-1).

    Parameters
    ==========

    f: Expr or Poly
        A polynomial
    g: Expr or Poly
        A polynomial
    mvar: a generator
        The "main variable".
        Polynomials are treated as univariate in the mvar.


    Returns
    =======

    List
        The subresultants of f and g, in increasing degree.
        The 0th element is the resultant itself.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.solvers.polysys import subresultant_polynomials

    >>> subresultant_polynomials(x**2, x, x)
    [0, x]
    """
    # ensure deg(f) >= deg(g)
    if degree(f, mvar) < degree(g, mvar):
        f, g = g, f


    prs = subresultants(f, g, mvar)
    if len(prs) <=1:
        return []

    subres_polys = [0] * (degree(g, mvar) + 1)

    for i in reversed(range(2, len(prs))):

        subres_polys[ degree(prs[i-1], mvar) -1 ] = prs[i]

        if degree(prs[i], mvar) < degree(prs[i-1], mvar) - 1:
            degree_jump = degree(prs[i-1], mvar) - degree(prs[i], mvar) - 1
            subres_polys[ degree(prs[i], mvar) ]  =\
                prs[i] * LC(prs[i], mvar) ** degree_jump

    # get last one
    subres_polys[-1] = prs[1] * LC(g, mvar) ** (degree(f, mvar) - degree(g, mvar) - 1)

    # try to expand to simplify
    for i, poly in enumerate(subres_polys):
        try:
            subres_polys[i] = expand(poly)
        except (ValueError, TypeError, AttributeError):
            continue

    return subres_polys


def subresultant_coefficients(f, g, mvar):
    """
    Computes the principal subresultant coefficients (PSC). Given the
    subresultant polynomials, in increasing degree, the ith PSC is
    the coefficient of mvar^i in the ith subresultant.

    Parameters
    ==========

    f: Expr or Poly
        A polynomial
    g: Expr or Poly
        A polynomial
    mvar: a generator
        The "main variable".
        Polynomials are treated as univariate in the mvar.


    Returns
    =======

    List
        The principal subresultant coefficients (PSC) of f and g.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.solvers.polysys import subresultant_coefficients

    >>> subresultant_coefficients(x**2, x, x)
    [0, 1]
    """
    subres_polys = subresultant_polynomials(f, g, mvar)

    subres_coeffs = []

    for i in range(len(subres_polys)):
        curr_coeff = Poly(subres_polys[i], mvar).nth(i)
        subres_coeffs.append(curr_coeff)

    return subres_coeffs


# gets real roots, numeric if not algebraic
# returns them sorted
# not a public function!
def get_nice_roots(poly):
    """Get real roots of a polynomial, numeric if not algebraic."""
    # its a constant
    try:
        if poly.is_number:
            return []
    except (AttributeError, TypeError):
        return []

    factors = factor_list(poly)[1]  # the 0th is just a number

    roots_set = set()
    # get the roots of the factors that are polynomials
    for factor in factors:
        curr_factor = factor[0]

        if curr_factor.is_number:
            continue

        try:
            new_roots = real_roots(curr_factor)
            #for i, r in enumerate(new_roots):
                # want to avoid CRootOf
                #if r.has(ComplexRootOf):
                #    new_roots[i] = r.evalf()
        except NotImplementedError:
            new_roots = [Rational(root) for root in nroots(curr_factor) if root.is_real]
        except PolynomialError:
            new_roots = [Rational(root) for root in nroots(curr_factor)
                         if root.is_real]

        roots_set.update(new_roots)

    return sorted(roots_set, reverse=False)


def get_sample_point(left, right):
    """Get a sample point in the interval (left, right)."""
    left, right = S(left), S(right)
    # Edge case check
    if left == right:
        return left

    # Ensure left < right
    if left > right:
        left, right = right, left

    # If they lie on either side of 0 then just return 0
    if left < 0 < right:
        return 0

    # Ray cases
    # always +- 1 in case integer
    if left == S.NegativeInfinity:
        return Integer(floor(right)) - 1
    if right == S.Infinity:
        return Integer(ceiling(left)) + 1

    # Finite interval

    between = (left + right) / 2

    if between.is_Rational:
        return between

    current_precision = 1
    between_num = Rational(between.evalf(current_precision))

    while not left < between_num < right:
        current_precision += 1
        between_num = Rational(between.evalf(current_precision))

    # check if an integer is in the interval
    if left < floor(between_num) < right:
        return floor(between_num)
    if left < ceiling(between_num) < right:
        return ceiling(between_num)
    return between_num


# uses the new .lift() functionality to simplify substituting algebraic
# numbers in polynomials
def simplify_alg_sub(poly, point):
    """Simplify polynomial substitution with algebraic numbers."""
    alg_points = [p for p in point.values() if S(p).has(ComplexRootOf)]
    if not alg_points:
        return poly.subs(point)
    return poly.as_poly(domain=QQ.algebraic_field(*alg_points)).subs(point)


# HONG PROJECTION OPERATOR (1990)
# input: set F of k-variate polynomials
# output: set F' of (k-1)-variate polynomials such that a CAD of R^{k-1}
# can be lifted to R^k

def projone(poly_set, mvar):
    """
    Computes the PROJ1 operator as defined in Hong 1990.

    Let F be a set of polynomials with a given mvar. Then,

    PROJ1 = \\cup_{f \\in F, g \\in RED(f)} (ldcf(g) \\cup PSC(g, D(g)))

    where RED is the reducta set, ldcf is the leading coefficient,
    PSC is the principal subresultant coefficient set, and D is the
    derivative operator.

    Parameters
    ==========

    F: a list/tuple/set
        A list of polyomials
    mvar: a generator
        The "main variable".
        Polynomials are treated as univariate in the mvar.


    Returns
    =======

    Set
        A set of projection factors.

    Examples
    ========

    >>> from sympy.abc import x,y
    >>> from sympy.solvers.polysys import projone

    >>> projone([y*x**2, x+1], x)
    {0, 1, y, 2*y}
    """
    proj_set = set()
    for f in poly_set:
        for g in red_set(f, mvar):
            proj_set.add(LC(g, mvar))
            proj_set.update(
                subresultant_coefficients(g, diff(g,mvar), mvar)
                )

    return proj_set


def projtwo(poly_set, mvar):
    """
    Computes the PROJ2* operator as defined in Hong (1990).
    This is an updated version of the PROJ2 operator from Collins.
    We will just call it PROJ2 here.

    Let F be a set of polynomials with a given mvar. Then,

    PROJ2 = \\cup_{f,g \\in F, f < g} \\cup_{f' \\in RED(f)} PSC(f', g)

    where RED is the reducta set, < indicates an arbitray "linear
    ordering" to not loop over redundant pairs, and PSC is the
    principal subresultant coefficient set.

    Parameters
    ==========

    F: a list/tuple/set
        A list of polyomials
    mvar: a generator
        The "main variable".
        Polynomials are treated as univariate in the mvar.

    Returns
    =======

    Set
        A set of projection factors.

    Examples
    ========

    >>> from sympy.abc import x,y
    >>> from sympy.solvers.polysys import projtwo

    >>> projtwo([y*x**2, x+1], x)
    {1, y}
    """

    proj_set = set()
    for i, f in enumerate(poly_set):
        # impose "linear ordering"
        for g in poly_set[i+1:]:
            for f_red in red_set(f, mvar):
                proj_set.update(
                    subresultant_coefficients(f_red, g, mvar)
                )

    return proj_set


def hongproj(poly_set, mvar):
    """
    The Hong projection operator, as defined in Hong(1990).
    PROJH takes a set of k-variate polynomials F, with an mvar, and
    returns a set of (k-1)-variate polynomials F, with the mvar
    eliminated. These projection factors satisfy the property that a
    CAD of R^{k-1} can be lifted to a CAD of R^k.

    The Hong projector, PROJH, is defined as:

    PROJH(F) = PROJ1(F) \\cup PROJ2(F)

    PROJ1 = \\cup_{f \\in F, g \\in RED(f)} (ldcf(g) \\cup PSC(g, D(g)))

    PROJ2 = \\cup_{f,g \\in F, f < g} \\cup_{f' \\in RED(f)} PSC(f', g)

    where RED is the reducta set, ldcf is the leading coefficient,
    PSC is the principal subresultant coefficient set, < indicates
    an arbitray "linear ordering" to not loop over redundant pairs,
    and D is the derivative operator.

    We remove constants, and keep polynomials that are unique up to
    sign.

    Parameters
    ==========

    F: a list/tuple/set
        A list of polyomials
    mvar: a generator
        The "main variable".
        Polynomials are treated as univariate in the mvar.

    Returns
    =======

    Set
        The set of projection factors.


    Examples
    ========

    >>> from sympy.abc import x,y
    >>> from sympy.solvers.polysys import hongproj

    >>> hongproj([y*x**2, x+1], x)
    {y, 2*y}
    """

    proj_factors = projone(poly_set, mvar).union(projtwo(poly_set, mvar))
    proj_factors_clean = set()

    for p in proj_factors:
        if p.is_number:
            continue
        if -p not in proj_factors_clean:
            proj_factors_clean.add(p)

    return proj_factors_clean


def cylindrical_algebraic_decomposition(poly_set, gens):
    """
    Calculates a cylindrical algebraic decomposition adapted to F.
    Uses the Hong projection operator. Returns sample points which
    represent cells over which each f in F is sign-invariant. It
    projects iteratively down to lower-dimension spaces according to
    the list of generators given in gens, in their order.

    Parameters
    ==========

    F: a list/tuple/set
        A list of polyomials
    gens: a list of generators

    Returns
    =======

    List[Dict]
        a list of dicts with the returned sample points. Each dict
        is a point, with the keys being the variables. A sample point
        is returned from every cell made by the CAD algorithm.


    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.solvers.polysys import cylindrical_algebraic_decomposition

    >>> cylindrical_algebraic_decomposition([x**2-1], [x])
    [{x: -2}, {x: -1}, {x: 0}, {x: 1}, {x: 2}]
    """

    # Compute the projection sets
    projs_set = [poly_set]
    for i in range(len(gens) - 1):
        projs_set.append(list(hongproj(projs_set[-1], gens[i])))


    # Lifting
    sample_points = [{}]

    for i in reversed(range(len(gens))):
        projs = projs_set[i]
        gen = gens[i]

        new_sample_points = []

        for point in sample_points:
            roots_list = set()
            for proj in projs:
                subbed = simplify_alg_sub(proj, point)
                roots_list.update(get_nice_roots(subbed))
            # have to sort them overall now
            roots_list = sorted(roots_list, reverse=False)


            # Calculate sample points
            if not roots_list:
                samples = [0]
            elif len(roots_list) == 1:
                samples = [get_sample_point(S.NegativeInfinity, roots_list[0]),
                           roots_list[0],
                           get_sample_point(roots_list[0], S.Infinity)]
            else:
                samples = [get_sample_point(S.NegativeInfinity, roots_list[0])]
                for r1, r2 in zip(roots_list, roots_list[1:]):
                    samples.extend([r1,
                                    get_sample_point(r1, r2)])
                samples.extend([roots_list[-1],
                                get_sample_point(roots_list[-1], S.Infinity)])

            for value in samples:
                new_point = point.copy()
                new_point[gen] = value
                new_sample_points.append(new_point)

        sample_points = new_sample_points


    return sample_points
