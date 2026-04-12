"""
Coordinate hit-and-run sampler for convex polytopes.
"""

from typing import Iterator, Optional

import numpy as np

from .polytope_utils import is_in_polytope, sample_chord_point


def coordinate_hit_and_run(
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    tol: float = 1e-9,
) -> Iterator[np.ndarray]:
    """Generate points by sampling along random coordinate directions."""
    generator = rng or np.random.default_rng()
    x = np.array(x0, dtype=float, copy=True)
    dimension = a.shape[1]

    while True:
        if not is_in_polytope(a, b, x, tol=tol):
            raise ValueError(f"Invalid state: {x}")

        axis = int(generator.integers(0, dimension))
        direction = np.zeros(dimension)
        direction[axis] = 1.0 if generator.uniform() < 0.5 else -1.0
        x = sample_chord_point(a, b, x, direction, rng=generator)
        yield np.array(x, copy=True)
