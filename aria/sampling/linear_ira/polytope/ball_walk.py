"""
Ball walk sampler for convex polytopes.
"""

from typing import Iterator, Optional

import numpy as np

from .polytope_utils import is_in_polytope, sample_unit_ball


def ball_walk(
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    radius: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    tol: float = 1e-9,
) -> Iterator[np.ndarray]:
    """Generate points with a uniform ball walk."""
    if radius <= 0:
        raise ValueError("radius must be positive")

    generator = rng or np.random.default_rng()
    x = np.array(x0, dtype=float, copy=True)

    while True:
        if not is_in_polytope(a, b, x, tol=tol):
            raise ValueError(f"Invalid state: {x}")

        proposal = x + radius * sample_unit_ball(x.shape[0], generator)
        if is_in_polytope(a, b, proposal, tol=tol):
            x = proposal

        yield np.array(x, copy=True)
