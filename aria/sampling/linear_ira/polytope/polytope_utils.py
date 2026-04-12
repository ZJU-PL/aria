"""
Utilities for random walks over convex polytopes.
"""

from typing import Callable, Iterator, Optional, Tuple

import numpy as np
import z3

try:
    from scipy.optimize import linprog  # pylint: disable=import-error
except ImportError:
    linprog = None


ArrayLike = np.ndarray


def is_in_polytope(
    a: ArrayLike, b: ArrayLike, x: ArrayLike, tol: float = 1e-9
) -> bool:
    """Check whether ``x`` satisfies ``a @ x <= b`` up to tolerance."""
    return bool(np.all(a.dot(x) <= b + tol))


def sample_sphere_direction(
    dimension: int, rng: Optional[np.random.Generator] = None
) -> ArrayLike:
    """Sample a random unit vector."""
    generator = rng or np.random.default_rng()
    direction = generator.normal(size=dimension)
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0.0:
        return sample_sphere_direction(dimension, generator)
    return direction / direction_norm


def sample_unit_ball(
    dimension: int, rng: Optional[np.random.Generator] = None
) -> ArrayLike:
    """Sample uniformly from the unit Euclidean ball."""
    generator = rng or np.random.default_rng()
    point = sample_sphere_direction(dimension, generator)
    radius = generator.uniform() ** (1.0 / float(dimension))
    return radius * point


def chord_bounds(
    a: ArrayLike,
    b: ArrayLike,
    x: ArrayLike,
    direction: ArrayLike,
    tol: float = 1e-12,
) -> Tuple[float, float]:
    """Return the feasible interval ``[lower, upper]`` along a direction."""
    slack = b - a.dot(x)
    advance = a.dot(direction)

    lower = -np.inf
    upper = np.inf

    for slack_value, advance_value in zip(slack, advance):
        if abs(advance_value) <= tol:
            if slack_value < -tol:
                raise ValueError("Current point is outside the polytope")
            continue

        bound = slack_value / advance_value
        if advance_value > 0:
            upper = min(upper, float(bound))
        else:
            lower = max(lower, float(bound))

    if lower > upper + tol:
        raise ValueError("Direction does not intersect the feasible region")

    return lower, upper


def sample_chord_point(
    a: ArrayLike,
    b: ArrayLike,
    x: ArrayLike,
    direction: ArrayLike,
    rng: Optional[np.random.Generator] = None,
    tol: float = 1e-12,
) -> ArrayLike:
    """Sample uniformly from the feasible chord through ``x``."""
    generator = rng or np.random.default_rng()
    lower, upper = chord_bounds(a, b, x, direction, tol=tol)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Sampling requires a bounded polytope in the chosen direction")
    if upper - lower <= tol:
        return np.array(x, dtype=float, copy=True)
    step = generator.uniform(lower, upper)
    return x + step * direction


def chebyshev_center(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Return the Chebyshev center of ``{x | a @ x <= b}``."""
    if linprog is not None:
        norm_vector = np.reshape(np.linalg.norm(a, axis=1), (a.shape[0], 1))
        objective = np.zeros(a.shape[1] + 1)
        objective[-1] = -1
        constraints = np.hstack((a, norm_vector))
        result = linprog(objective, A_ub=constraints, b_ub=b, bounds=(None, None))
        if not result.success:
            raise ValueError("Unable to find Chebyshev center")
        return result.x[:-1]

    optimize = z3.Optimize()
    point_vars = [z3.Real(f"cc_{index}") for index in range(a.shape[1])]
    radius = z3.Real("cc_radius")
    optimize.maximize(radius)
    optimize.add(radius >= 0)

    for row, bound in zip(a, b):
        lhs = z3.RealVal("0")
        for coefficient, variable in zip(row, point_vars):
            if coefficient == 0:
                continue
            lhs += z3.RealVal(repr(float(coefficient))) * variable
        lhs += z3.RealVal(repr(float(np.linalg.norm(row)))) * radius
        optimize.add(lhs <= z3.RealVal(repr(float(bound))))

    if optimize.check() != z3.sat:
        raise ValueError("Unable to find Chebyshev center")

    model = optimize.model()
    center = []
    for variable in point_vars:
        value = model.eval(variable, model_completion=True)
        if z3.is_rational_value(value):
            center.append(value.numerator_as_long() / value.denominator_as_long())
        else:
            center.append(float(value.as_decimal(20).rstrip("?")))

    return np.array(center, dtype=float)


def collect_chain(
    sampler: Callable[..., Iterator[ArrayLike]],
    count: int,
    burn: int,
    thin: int,
    *args,
    **kwargs,
) -> ArrayLike:
    """Collect points from a sampler generator."""
    chain = sampler(*args, **kwargs)
    first_point = np.asarray(next(chain), dtype=float)
    points = np.empty((count, first_point.shape[0]))

    for _ in range(max(0, burn - 1)):
        next(chain)

    for index in range(count):
        points[index] = np.asarray(next(chain), dtype=float)
        for _ in range(max(0, thin - 1)):
            next(chain)

    return points
