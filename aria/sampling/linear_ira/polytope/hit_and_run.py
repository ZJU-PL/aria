"""
Hit-and-run sampler for convex polytopes.
"""

from typing import Iterator, Optional

import numpy as np

from .polytope_utils import is_in_polytope, sample_chord_point, sample_sphere_direction


def hit_and_run(
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    tol: float = 1e-9,
) -> Iterator[np.ndarray]:
    """Generate points with the classical hit-and-run walk."""
    generator = rng or np.random.default_rng()
    x = np.array(x0, dtype=float, copy=True)

    while True:
        if not is_in_polytope(a, b, x, tol=tol):
            raise ValueError(f"Invalid state: {x}")

        direction = sample_sphere_direction(a.shape[1], generator)
        x = sample_chord_point(a, b, x, direction, rng=generator)
        yield np.array(x, copy=True)
