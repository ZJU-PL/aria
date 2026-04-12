"""
Unit tests for low-level polytope samplers in linear_ira.
"""

import numpy as np

from aria.sampling.linear_ira import (
    ball_walk,
    chebyshev_center,
    collect_chain,
    coordinate_hit_and_run,
    hit_and_run,
)
from aria.sampling.linear_ira.polytope.polytope_utils import chord_bounds


def _unit_box_polytope():
    a = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )
    b = np.array([1.0, 0.0, 1.0, 0.0])
    x0 = np.array([0.5, 0.5])
    return a, b, x0


def test_chebyshev_center_for_unit_box():
    a, b, _ = _unit_box_polytope()
    center = chebyshev_center(a, b)
    assert np.allclose(center, np.array([0.5, 0.5]), atol=1e-7)


def test_chord_bounds_for_axis_direction():
    a, b, x0 = _unit_box_polytope()
    lower, upper = chord_bounds(a, b, x0, np.array([1.0, 0.0]))
    assert np.isclose(lower, -0.5)
    assert np.isclose(upper, 0.5)


def test_ball_walk_stays_feasible():
    a, b, x0 = _unit_box_polytope()
    points = collect_chain(
        ball_walk,
        count=20,
        burn=3,
        thin=1,
        a=a,
        b=b,
        x0=x0,
        radius=0.2,
        rng=np.random.default_rng(7),
    )
    assert np.all(a.dot(points.T) <= b[:, None] + 1e-9)
    assert np.unique(np.round(points, 6), axis=0).shape[0] > 1


def test_hit_and_run_stays_feasible():
    a, b, x0 = _unit_box_polytope()
    points = collect_chain(
        hit_and_run,
        count=20,
        burn=3,
        thin=1,
        a=a,
        b=b,
        x0=x0,
        rng=np.random.default_rng(11),
    )
    assert np.all(a.dot(points.T) <= b[:, None] + 1e-9)
    assert np.unique(np.round(points, 6), axis=0).shape[0] > 1


def test_coordinate_hit_and_run_stays_feasible():
    a, b, x0 = _unit_box_polytope()
    points = collect_chain(
        coordinate_hit_and_run,
        count=20,
        burn=3,
        thin=1,
        a=a,
        b=b,
        x0=x0,
        rng=np.random.default_rng(19),
    )
    assert np.all(a.dot(points.T) <= b[:, None] + 1e-9)
    assert np.unique(np.round(points, 6), axis=0).shape[0] > 1
