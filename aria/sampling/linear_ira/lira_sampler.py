"""
Linear Integer and Real Arithmetic sampler implementation.
"""

import random
import time
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, cast

import numpy as np
import z3

from aria.sampling.base import (
    Logic,
    Sampler,
    SamplingMethod,
    SamplingOptions,
    SamplingResult,
)
from aria.utils.z3.expr import get_variables, is_int_sort, is_real_sort

from .polytope.ball_walk import ball_walk
from .polytope.coordinate_hit_and_run import coordinate_hit_and_run
from .polytope.dikin_walk import dikin_walk
from .polytope.hit_and_run import hit_and_run
from .polytope.polytope_utils import chebyshev_center, collect_chain, is_in_polytope


def _z3_number_to_float(value: z3.ExprRef) -> float:
    """Convert a numeric Z3 value to a Python float."""
    if z3.is_int_value(value):
        return float(getattr(value, "as_long")())
    if z3.is_rational_value(value):
        numerator = float(getattr(value, "numerator_as_long")())
        denominator = float(getattr(value, "denominator_as_long")())
        return numerator / denominator
    if z3.is_algebraic_value(value):
        approximation = getattr(value, "approx")(20)
        decimal = getattr(approximation, "as_decimal")(20).rstrip("?")
        return float(decimal)
    return float(str(value))


def _zero_coefficients(variable_count: int) -> np.ndarray:
    return np.zeros(variable_count, dtype=float)


def _scale_linear_form(
    coefficients: np.ndarray, constant: float, scale: float
) -> Tuple[np.ndarray, float]:
    """Scale a linear form by a constant."""
    return coefficients * scale, constant * scale


def _add_linear_forms(
    left: Tuple[np.ndarray, float], right: Tuple[np.ndarray, float]
) -> Tuple[np.ndarray, float]:
    """Add two linear forms."""
    return left[0] + right[0], left[1] + right[1]


def _linearize_expression(
    expr: z3.ExprRef,
    variable_to_index: Dict[str, int],
    variable_count: int,
) -> Tuple[np.ndarray, float]:
    """Convert a Z3 linear arithmetic term to coefficients and a constant."""
    if (
        z3.is_int_value(expr)
        or z3.is_rational_value(expr)
        or z3.is_algebraic_value(expr)
    ):
        return _zero_coefficients(variable_count), _z3_number_to_float(expr)

    if is_int_sort(expr) or is_real_sort(expr):
        coefficients = _zero_coefficients(variable_count)
        coefficients[variable_to_index[str(expr)]] = 1.0
        return coefficients, 0.0

    decl_kind = expr.decl().kind()

    if decl_kind == z3.Z3_OP_TO_REAL:
        return _linearize_expression(expr.arg(0), variable_to_index, variable_count)

    if decl_kind == z3.Z3_OP_ADD:
        result = (_zero_coefficients(variable_count), 0.0)
        for child in expr.children():
            result = _add_linear_forms(
                result,
                _linearize_expression(child, variable_to_index, variable_count),
            )
        return result

    if decl_kind == z3.Z3_OP_SUB:
        children = expr.children()
        result = _linearize_expression(children[0], variable_to_index, variable_count)
        for child in children[1:]:
            result = _add_linear_forms(
                result,
                _scale_linear_form(
                    *_linearize_expression(child, variable_to_index, variable_count),
                    -1.0,
                ),
            )
        return result

    if decl_kind == z3.Z3_OP_UMINUS:
        return _scale_linear_form(
            *_linearize_expression(expr.arg(0), variable_to_index, variable_count),
            -1.0,
        )

    if decl_kind == z3.Z3_OP_MUL:
        scalar = 1.0
        symbolic_children: List[z3.ExprRef] = []
        for child in expr.children():
            if (
                z3.is_int_value(child)
                or z3.is_rational_value(child)
                or z3.is_algebraic_value(child)
            ):
                scalar *= _z3_number_to_float(child)
            else:
                symbolic_children.append(child)
        if not symbolic_children:
            return _zero_coefficients(variable_count), scalar
        if len(symbolic_children) != 1:
            raise ValueError(f"Non-linear term is not supported: {expr}")
        return _scale_linear_form(
            *_linearize_expression(
                symbolic_children[0], variable_to_index, variable_count
            ),
            scalar,
        )

    if decl_kind == z3.Z3_OP_DIV:
        numerator = _linearize_expression(expr.arg(0), variable_to_index, variable_count)
        denominator_expr = expr.arg(1)
        if not (
            z3.is_int_value(denominator_expr)
            or z3.is_rational_value(denominator_expr)
            or z3.is_algebraic_value(denominator_expr)
        ):
            raise ValueError(f"Non-linear division is not supported: {expr}")
        denominator = _z3_number_to_float(denominator_expr)
        if denominator == 0.0:
            raise ValueError(f"Division by zero in term: {expr}")
        return _scale_linear_form(*numerator, 1.0 / denominator)

    raise ValueError(f"Unsupported arithmetic term in walk sampler: {expr}")


def _append_linear_constraint(
    atom: z3.ExprRef,
    variable_to_index: Dict[str, int],
    variable_count: int,
    strict_epsilon: float,
    inequality_rows: List[np.ndarray],
    inequality_bounds: List[float],
    equality_rows: List[np.ndarray],
    equality_bounds: List[float],
) -> None:
    """Lower one atomic linear constraint to row form."""
    lhs = _linearize_expression(atom.arg(0), variable_to_index, variable_count)
    rhs = _linearize_expression(atom.arg(1), variable_to_index, variable_count)
    difference = _add_linear_forms(lhs, _scale_linear_form(*rhs, -1.0))
    row = difference[0]
    bound = -difference[1]
    kind = atom.decl().kind()

    if kind == z3.Z3_OP_LE:
        inequality_rows.append(row)
        inequality_bounds.append(bound)
        return

    if kind == z3.Z3_OP_LT:
        inequality_rows.append(row)
        inequality_bounds.append(bound - strict_epsilon)
        return

    if kind == z3.Z3_OP_GE:
        inequality_rows.append(-row)
        inequality_bounds.append(-bound)
        return

    if kind == z3.Z3_OP_GT:
        inequality_rows.append(-row)
        inequality_bounds.append(-bound - strict_epsilon)
        return

    if kind == z3.Z3_OP_EQ:
        equality_rows.append(row)
        equality_bounds.append(bound)
        return

    raise ValueError(f"Unsupported linear predicate in walk sampler: {atom}")


def _extract_polytope_constraints(
    formula: z3.ExprRef,
    variables: Sequence[z3.ExprRef],
    strict_epsilon: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract conjunctions of linear constraints into matrix form."""
    variable_to_index = {str(var): index for index, var in enumerate(variables)}
    variable_count = len(variables)
    inequality_rows: List[np.ndarray] = []
    inequality_bounds: List[float] = []
    equality_rows: List[np.ndarray] = []
    equality_bounds: List[float] = []

    def visit(expr: z3.ExprRef) -> None:
        expr = z3.simplify(expr)
        if z3.is_true(expr):
            return
        if z3.is_and(expr):
            for child in expr.children():
                visit(child)
            return
        if expr.decl().kind() in {
            z3.Z3_OP_LE,
            z3.Z3_OP_LT,
            z3.Z3_OP_GE,
            z3.Z3_OP_GT,
            z3.Z3_OP_EQ,
        }:
            _append_linear_constraint(
                expr,
                variable_to_index,
                variable_count,
                strict_epsilon,
                inequality_rows,
                inequality_bounds,
                equality_rows,
                equality_bounds,
            )
            return
        raise ValueError(
            "Walk-based LIRA sampling requires a conjunction of linear "
            f"equalities/inequalities; unsupported sub-formula: {expr}"
        )

    visit(formula)

    inequality_matrix = (
        np.vstack(inequality_rows)
        if inequality_rows
        else np.empty((0, variable_count), dtype=float)
    )
    inequality_vector = np.array(inequality_bounds, dtype=float)
    equality_matrix = (
        np.vstack(equality_rows)
        if equality_rows
        else np.empty((0, variable_count), dtype=float)
    )
    equality_vector = np.array(equality_bounds, dtype=float)
    return inequality_matrix, inequality_vector, equality_matrix, equality_vector


def _reduce_equalities(
    inequality_matrix: np.ndarray,
    inequality_vector: np.ndarray,
    equality_matrix: np.ndarray,
    equality_vector: np.ndarray,
    tolerance: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project linear equalities away via a nullspace parameterization."""
    variable_count = (
        equality_matrix.shape[1] if equality_matrix.size else inequality_matrix.shape[1]
    )
    offset = np.zeros(variable_count, dtype=float)
    basis = np.eye(variable_count, dtype=float)

    if equality_matrix.size:
        offset, _, _, _ = np.linalg.lstsq(equality_matrix, equality_vector, rcond=None)
        if not np.allclose(equality_matrix.dot(offset), equality_vector, atol=1e-7):
            raise ValueError("Unable to satisfy linear equalities for walk sampling")

        _, singular_values, vh = np.linalg.svd(equality_matrix)
        rank = int(np.sum(singular_values > tolerance))
        basis = vh[rank:].T

    if inequality_matrix.size == 0:
        reduced_matrix = np.empty((0, basis.shape[1]), dtype=float)
        reduced_vector = np.empty((0,), dtype=float)
    else:
        reduced_matrix = inequality_matrix.dot(basis)
        reduced_vector = inequality_vector - inequality_matrix.dot(offset)

    if reduced_matrix.size == 0:
        return reduced_matrix, reduced_vector, offset, basis

    filtered_rows: List[np.ndarray] = []
    filtered_bounds: List[float] = []
    for row, bound in zip(reduced_matrix, reduced_vector):
        if np.linalg.norm(row) <= tolerance:
            if bound < -1e-8:
                raise ValueError("Linear constraints are inconsistent")
            continue
        filtered_rows.append(row)
        filtered_bounds.append(float(bound))

    if filtered_rows:
        reduced_matrix = np.vstack(filtered_rows)
        reduced_vector = np.array(filtered_bounds, dtype=float)
    else:
        reduced_matrix = np.empty((0, basis.shape[1]), dtype=float)
        reduced_vector = np.empty((0,), dtype=float)

    return reduced_matrix, reduced_vector, offset, basis


class LIRASampler(Sampler):
    """Sampler for linear integer and real arithmetic formulas."""

    def __init__(self, **_kwargs):
        self.formula: Optional[z3.ExprRef] = None
        self.variables: List[z3.ExprRef] = []

    def supports_logic(self, logic: Logic) -> bool:
        return logic in [Logic.QF_LRA, Logic.QF_LIA, Logic.QF_LIRA]

    def init_from_formula(self, formula: z3.ExprRef) -> None:
        self.formula = formula
        self.variables = []
        for var in get_variables(formula):
            if is_int_sort(var) or is_real_sort(var):
                self.variables.append(var)
        self.variables.sort(key=str)

    def sample(self, options: SamplingOptions) -> SamplingResult:
        if self.formula is None:
            raise ValueError("Sampler not initialized with a formula")

        if options.method == SamplingMethod.DIKIN_WALK:
            return self._sample_via_walk(options)
        return self._sample_via_enumeration(options)

    def _sample_via_enumeration(self, options: SamplingOptions) -> SamplingResult:
        """Enumerate models with blocking clauses."""
        assert self.formula is not None
        if options.random_seed is not None:
            random.seed(options.random_seed)

        solver = z3.Solver()
        if options.random_seed is not None:
            solver.set("random_seed", options.random_seed)
            solver.set("seed", options.random_seed)
        solver.add(self.formula)

        samples = []
        stats = {"time_ms": 0, "iterations": 0}

        for _ in range(options.num_samples):
            if solver.check() != z3.sat:
                break

            model = solver.model()
            sample = {}
            for var in self.variables:
                value = model.evaluate(var, model_completion=True)
                if is_int_sort(var):
                    sample[str(var)] = int(str(value))
                else:
                    sample[str(var)] = _z3_number_to_float(value)
            samples.append(sample)

            block = []
            for var in self.variables:
                value = model.evaluate(var, model_completion=True)
                if is_int_sort(var):
                    block.append(var != value)
                else:
                    arith_var = cast(z3.ArithRef, var)
                    float_value = _z3_number_to_float(value)
                    delta = 0.001
                    lower = z3.RealVal(repr(float_value - delta))
                    upper = z3.RealVal(repr(float_value + delta))
                    block.append(z3.Or(arith_var < lower, arith_var > upper))

            solver.add(z3.Or(block))
            stats["iterations"] += 1

        return SamplingResult(samples, stats)

    def _sample_via_walk(self, options: SamplingOptions) -> SamplingResult:
        """Sample from a bounded real polytope using a random walk."""
        assert self.formula is not None
        int_variables = [var for var in self.variables if is_int_sort(var)]
        if int_variables:
            raise ValueError(
                "Walk-based sampling currently supports real-valued linear "
                f"formulas only; found integer variables: {', '.join(map(str, int_variables))}"
            )

        start_time = time.time()
        strict_epsilon = float(options.additional_options.get("strict_epsilon", 1e-6))
        walk_name = str(options.additional_options.get("walk", "dikin"))
        burn = int(options.additional_options.get("burn", 200))
        thin = int(options.additional_options.get("thin", 10))
        tolerance = float(options.additional_options.get("feasibility_tolerance", 1e-8))
        random_seed = options.random_seed
        rng = np.random.default_rng(random_seed)

        formula = cast(z3.ExprRef, self.formula)
        inequality_matrix, inequality_vector, equality_matrix, equality_vector = (
            _extract_polytope_constraints(
                formula,
                self.variables,
                strict_epsilon,
            )
        )
        reduced_matrix, reduced_vector, offset, basis = _reduce_equalities(
            inequality_matrix,
            inequality_vector,
            equality_matrix,
            equality_vector,
        )

        if basis.shape[1] == 0:
            point = offset
            if inequality_matrix.size and not is_in_polytope(
                inequality_matrix,
                inequality_vector,
                point,
                tol=tolerance,
            ):
                return SamplingResult([], {"time_ms": 0, "iterations": 0})
            sample = {
                str(var): float(point[index])
                for index, var in enumerate(self.variables)
            }
            return SamplingResult(
                [sample],
                {
                    "time_ms": int((time.time() - start_time) * 1000),
                    "iterations": 1,
                    "method": options.method.value,
                    "walk": "singleton",
                    "reduced_dimension": 0,
                },
            )

        if reduced_matrix.shape[0] == 0:
            raise ValueError("Walk-based sampling requires a bounded polytope")

        reduced_start = chebyshev_center(reduced_matrix, reduced_vector)

        walk_map: Dict[str, Callable[..., Iterator[np.ndarray]]] = {
            "dikin": dikin_walk,
            "hit_and_run": hit_and_run,
            "coordinate_hit_and_run": coordinate_hit_and_run,
            "ball_walk": ball_walk,
        }
        if walk_name not in walk_map:
            raise ValueError(
                "Unknown walk sampler. Expected one of: dikin, hit_and_run, "
                f"coordinate_hit_and_run, ball_walk; got {walk_name}"
            )

        sampler = walk_map[walk_name]
        if walk_name == "dikin":
            reduced_points = collect_chain(
                sampler,
                options.num_samples,
                burn,
                thin,
                a=reduced_matrix,
                b=reduced_vector,
                x0=reduced_start,
                r=float(options.additional_options.get("radius", 3.0 / 40.0)),
                rng=rng,
            )
        elif walk_name == "ball_walk":
            reduced_points = collect_chain(
                sampler,
                options.num_samples,
                burn,
                thin,
                a=reduced_matrix,
                b=reduced_vector,
                x0=reduced_start,
                radius=float(options.additional_options.get("radius", 0.5)),
                rng=rng,
            )
        else:
            reduced_points = collect_chain(
                sampler,
                options.num_samples,
                burn,
                thin,
                a=reduced_matrix,
                b=reduced_vector,
                x0=reduced_start,
                rng=rng,
            )
        original_points = offset + reduced_points.dot(basis.T)

        samples = []
        for point in original_points:
            sample = {}
            for index, var in enumerate(self.variables):
                sample[str(var)] = float(point[index])
            samples.append(sample)

        stats = {
            "time_ms": int((time.time() - start_time) * 1000),
            "iterations": int(options.num_samples),
            "method": options.method.value,
            "walk": walk_name,
            "burn": burn,
            "thin": thin,
            "reduced_dimension": int(basis.shape[1]),
            "inequality_count": int(inequality_matrix.shape[0]),
            "equality_count": int(equality_matrix.shape[0]),
        }
        return SamplingResult(samples, stats)

    def get_supported_methods(self) -> Set[SamplingMethod]:
        return {SamplingMethod.ENUMERATION, SamplingMethod.DIKIN_WALK}

    def get_supported_logics(self) -> Set[Logic]:
        return {Logic.QF_LRA, Logic.QF_LIA, Logic.QF_LIRA}


class LIASampler(Sampler):
    """Linear Integer Arithmetic sampler."""

    def __init__(self, **options):
        super().__init__()
        self.conjuntion_sampler = None
        self.number_samples = 0

    def supports_logic(self, logic: Logic) -> bool:
        """Check if this sampler supports the given logic."""
        return logic == Logic.QF_LIA

    def init_from_formula(self, formula: z3.ExprRef) -> None:
        """Initialize the sampler with a formula."""
        raise NotImplementedError

    def sample(self, options: SamplingOptions) -> SamplingResult:
        """
        External interface - generate samples.

        Args:
            options: Sampling options (num_samples will be used)
        """
        self.number_samples = options.num_samples
        return self.sample_via_enumeration()

    def sample_via_enumeration(self):
        """
        Call an SMT solver iteratively (block sampled models).
        """
        raise NotImplementedError

    def sample_via_smt_enumeration(self):
        """
        Call an SMT solver iteratively (block sampled models)
        """
        raise NotImplementedError

    def sample_via_smt_random_seed(self):
        """
        Call an SMT solver iteratively (no blocking, but give the solver different
        random seeds)
        """
        raise NotImplementedError
