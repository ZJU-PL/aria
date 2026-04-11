"""Common helpers for floating-point finite-domain samplers."""

from random import randrange
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import z3

from aria.utils.z3.expr import get_variables


def get_fp_variables(formula: z3.ExprRef) -> List[z3.ExprRef]:
    """Extract floating-point variables from a formula in deterministic order."""
    variables = [
        var
        for var in get_variables(formula)
        if var.sort().kind() == z3.Z3_FLOATING_POINT_SORT
    ]
    variables.sort(key=str)
    return variables


def fp_value_from_bits(bits: int, sort: z3.FPSortRef) -> z3.ExprRef:
    """Construct an exact FP value from its IEEE-754 bit pattern."""
    width = sort.ebits() + sort.sbits()
    return z3.fpBVToFP(z3.BitVecVal(bits, width), sort)


def fp_total_key_from_bits(bits: int, width: int) -> int:
    """Encode IEEE-754 totalOrder as an unsigned integer key."""
    sign_mask = 1 << (width - 1)
    if bits & sign_mask:
        return (~bits) & ((1 << width) - 1)
    return bits ^ sign_mask


def fp_model_value(model: z3.ModelRef, fp_expr: z3.ExprRef) -> z3.ExprRef:
    """Extract an exact FP value from a model without losing payload/sign bits."""
    bits = cast(z3.BitVecNumRef, model.eval(z3.fpToIEEEBV(fp_expr), model_completion=True))
    return fp_value_from_bits(bits.as_long(), cast(z3.FPSortRef, fp_expr.sort()))


def fp_value_bits(value: z3.ExprRef) -> int:
    """Extract the exact IEEE-754 bit pattern from an FP value term."""
    if value.num_args() == 1:
        return int(str(value.arg(0)))

    probe = z3.FP("__aria_fp_bits_probe", cast(z3.FPSortRef, value.sort()))
    solver = z3.Solver()
    solver.add(z3.fpToIEEEBV(probe) == z3.fpToIEEEBV(value))
    if solver.check() != z3.sat:
        raise ValueError("Unable to extract floating-point bit pattern")
    bits = cast(
        z3.BitVecNumRef,
        solver.model().eval(z3.fpToIEEEBV(probe), model_completion=True),
    )
    return bits.as_long()


def format_fp_value(value: z3.ExprRef) -> str:
    """Render an FP value together with its exact IEEE-754 bit pattern."""
    sort = cast(z3.FPSortRef, value.sort())
    width = sort.ebits() + sort.sbits()
    hex_width = (width + 3) // 4
    return f"{z3.simplify(value)} [bits=0x{fp_value_bits(value):0{hex_width}x}]"


def _pretty_fp_value(value: z3.ExprRef) -> str:
    return str(z3.simplify(value))


def _bits_fp_value(value: z3.ExprRef) -> str:
    sort = cast(z3.FPSortRef, value.sort())
    width = sort.ebits() + sort.sbits()
    hex_width = (width + 3) // 4
    return f"0x{fp_value_bits(value):0{hex_width}x}"


def get_fp_render_mode(options: Optional[Any] = None) -> str:
    """Resolve the requested floating-point rendering mode."""
    if options is None:
        return "pretty+bits"
    mode = options.additional_options.get("render_mode", "pretty+bits")
    if mode not in {"pretty", "bits", "pretty+bits"}:
        raise ValueError(
            "Unsupported FP render_mode. Expected one of: pretty, bits, pretty+bits"
        )
    return str(mode)


def render_fp_value(value: z3.ExprRef, render_mode: str = "pretty+bits") -> str:
    """Render an FP value according to the requested mode."""
    if render_mode == "pretty":
        return _pretty_fp_value(value)
    if render_mode == "bits":
        return _bits_fp_value(value)
    if render_mode == "pretty+bits":
        return format_fp_value(value)
    raise ValueError(
        "Unsupported FP render_mode. Expected one of: pretty, bits, pretty+bits"
    )


def render_fp_sample(
    model: z3.ModelRef,
    variables: Sequence[z3.ExprRef],
    render_mode: str = "pretty+bits",
) -> Dict[str, str]:
    """Render a floating-point model projection with exact bit patterns."""
    sample: Dict[str, str] = {}
    for var in variables:
        sample[str(var)] = render_fp_value(fp_model_value(model, var), render_mode)
    return sample


def fp_bit_equality(var: z3.ExprRef, value: z3.ExprRef) -> z3.BoolRef:
    """Compare two FP values by IEEE-754 bit pattern."""
    width = cast(z3.FPSortRef, var.sort()).ebits() + cast(z3.FPSortRef, var.sort()).sbits()
    return cast(
        z3.BoolRef,
        z3.fpToIEEEBV(var) == z3.BitVecVal(fp_value_bits(value), width),
    )


def get_fp_bit_atoms(fp_vars: Sequence[z3.ExprRef]) -> List[z3.ExprRef]:
    """Expose each floating-point variable as a list of bit predicates."""
    bit_atoms: List[z3.ExprRef] = []
    for var in fp_vars:
        bits = cast(z3.BitVecRef, z3.fpToIEEEBV(var))
        atom_list: List[z3.ExprRef] = []
        for i in range(bits.size()):
            atom_list.append(cast(z3.ExprRef, z3.Extract(i, i, bits) == 1))
        bit_atoms.extend(atom_list)
    return bit_atoms


def get_uniform_samples_with_fp_xor(
    fp_vars: Sequence[z3.ExprRef],
    formula: z3.ExprRef,
    num_samples: int,
    max_attempts_multiplier: int = 20,
) -> List[List[z3.ExprRef]]:
    """Sample FP assignments using random XOR constraints over IEEE bits."""
    if not fp_vars:
        raise ValueError("No floating-point variables found in formula")

    bit_atoms = get_fp_bit_atoms(fp_vars)
    if not bit_atoms:
        raise ValueError("No floating-point bit atoms available for hashing")

    solver = z3.Solver()
    solver.add(formula)
    results: List[List[z3.ExprRef]] = []
    max_attempts = max(num_samples * max_attempts_multiplier, num_samples)
    attempts = 0

    while len(results) < num_samples and attempts < max_attempts:
        attempts += 1
        solver.push()
        for _ in range(3):
            parity = z3.BoolVal(randrange(0, 2))
            for _ in range(10):
                parity = z3.Xor(parity, bit_atoms[randrange(0, len(bit_atoms))])
            solver.add(parity)

        if solver.check() == z3.sat:
            model = solver.model()
            values = [fp_model_value(model, var) for var in fp_vars]
            results.append(values)
            solver.pop()
            solver.add(z3.Or([z3.Not(fp_bit_equality(var, value)) for var, value in zip(fp_vars, values)]))
            continue

        solver.pop()

    return results


def enumerate_fp_assignments(
    formula: z3.ExprRef,
    variables: Sequence[z3.ExprRef],
    limit: int,
    random_seed: Optional[int] = None,
) -> List[List[z3.ExprRef]]:
    """Enumerate exact FP assignments with blocking over IEEE bit patterns."""
    solver = z3.Solver()
    if random_seed is not None:
        solver.set("random_seed", random_seed)
        solver.set("seed", random_seed)
    solver.add(formula)

    assignments: List[List[z3.ExprRef]] = []
    for _ in range(limit):
        if solver.check() != z3.sat:
            break
        model = solver.model()
        values = [fp_model_value(model, var) for var in variables]
        assignments.append(values)
        if not values:
            break
        solver.add(
            z3.Or(
                [
                    z3.Not(fp_bit_equality(var, value))
                    for var, value in zip(variables, values)
                ]
            )
        )
    return assignments


def fp_assignment_total_order_key(values: Sequence[z3.ExprRef]) -> Tuple[int, ...]:
    """Build a lexicographic total-order key tuple for an FP assignment."""
    keys: List[int] = []
    for value in values:
        sort = cast(z3.FPSortRef, value.sort())
        width = sort.ebits() + sort.sbits()
        keys.append(fp_total_key_from_bits(fp_value_bits(value), width))
    return tuple(keys)


def render_fp_assignment(
    variables: Sequence[z3.ExprRef],
    values: Sequence[z3.ExprRef],
    render_mode: str = "pretty+bits",
) -> Dict[str, str]:
    """Render an FP assignment from exact values."""
    sample: Dict[str, str] = {}
    for var, value in zip(variables, values):
        sample[str(var)] = render_fp_value(value, render_mode)
    return sample
