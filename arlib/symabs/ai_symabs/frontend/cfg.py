"""Control-flow graph based frontend and evaluator for ai_symabs.

This extends the straight-line two-operand notation with support for
branching and looping by representing code as a CFG and reusing the
existing abstract domains through symbolic abstraction (bilateral).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Union

import z3

from ..domains.algorithms import bilateral
from ..domains.core import ConjunctiveDomain
from ..domains.core.abstract import AbstractState


# === Expressions ===========================================================
# Bitwise operations are modeled via fixed-width bit-vectors. The choice of
# width is a practical compromise; domains remain integer-based.
BITWIDTH = 64


def _as_bv(value: z3.ArithRef) -> z3.BitVecRef:
    return z3.Int2BV(value, BITWIDTH)


def _bv_to_int(value: z3.BitVecRef) -> z3.ArithRef:
    return z3.BV2Int(value, is_signed=True)


@dataclass(frozen=True)
class Expr:
    """Base expression node."""


@dataclass(frozen=True)
class Var(Expr):
    name: str


@dataclass(frozen=True)
class Const(Expr):
    value: Union[int, bool]


@dataclass(frozen=True)
class UnaryOp(Expr):
    op: str  # "-", "+", "~", "not"
    operand: Expr


@dataclass(frozen=True)
class BinOp(Expr):
    op: str  # "+", "-", "*", "//", "%", "&", "|", "^", "<<", ">>"
    left: Expr
    right: Expr


@dataclass(frozen=True)
class BoolOp(Expr):
    op: str  # "and" or "or"
    values: List[Expr]


@dataclass(frozen=True)
class Compare(Expr):
    left: Expr
    op: str  # "<", "<=", ">", ">=", "==", "!="
    right: Expr


# === Statements and terminators ===========================================
@dataclass(frozen=True)
class AssignStmt:
    """Single assignment/aug-assignment."""

    target: str
    op: str  # "=", "+=", "-=", "*=", "//=", "%=", "&=", "|=", "^=", "<<=", ">>="
    expr: Expr


@dataclass(frozen=True)
class Branch:
    condition: Expr
    true_target: str
    false_target: str


@dataclass(frozen=True)
class Goto:
    target: str


Terminator = Union[Branch, Goto, None]


@dataclass
class BasicBlock:
    block_id: str
    statements: List[AssignStmt] = field(default_factory=list)
    terminator: Terminator = None

    def add_statement(self, stmt: AssignStmt) -> None:
        self.statements.append(stmt)


@dataclass
class CFG:
    """Minimal CFG representation."""

    entry: str
    blocks: Dict[str, BasicBlock]

    def successors(self, block_id: str) -> List[str]:
        block = self.blocks[block_id]
        term = block.terminator
        if isinstance(term, Branch):
            return [term.true_target, term.false_target]
        if isinstance(term, Goto):
            return [term.target]
        return []

    def is_exit(self, block_id: str) -> bool:
        return len(self.successors(block_id)) == 0


# === Z3 lowering ===========================================================
def expr_to_z3(expr: Expr, env: Dict[str, z3.ArithRef]) -> z3.ExprRef:
    """Translate an Expr into a Z3 expression using env for variables."""
    if isinstance(expr, Var):
        return env[expr.name]
    if isinstance(expr, Const):
        if isinstance(expr.value, bool):
            return z3.BoolVal(expr.value)
        return z3.IntVal(expr.value)
    if isinstance(expr, UnaryOp):
        inner = expr_to_z3(expr.operand, env)
        if expr.op == "-":
            return -inner
        if expr.op == "+":
            return inner
        if expr.op == "~":
            return _bv_to_int(~_as_bv(inner))
        if expr.op == "not":
            return z3.Not(inner)
        raise ValueError(f"Unsupported unary op: {expr.op}")
    if isinstance(expr, BinOp):
        lhs = expr_to_z3(expr.left, env)
        rhs = expr_to_z3(expr.right, env)
        if expr.op == "+":
            return lhs + rhs
        if expr.op == "-":
            return lhs - rhs
        if expr.op == "*":
            return lhs * rhs
        if expr.op == "//":
            return z3.IntDiv(lhs, rhs)
        if expr.op == "%":
            return z3.Mod(lhs, rhs)
        if expr.op == "&":
            return _bv_to_int(_as_bv(lhs) & _as_bv(rhs))
        if expr.op == "|":
            return _bv_to_int(_as_bv(lhs) | _as_bv(rhs))
        if expr.op == "^":
            return _bv_to_int(_as_bv(lhs) ^ _as_bv(rhs))
        if expr.op == "<<":
            return _bv_to_int(_as_bv(lhs) << _as_bv(rhs))
        if expr.op == ">>":
            return _bv_to_int(_as_bv(lhs) >> _as_bv(rhs))
        raise ValueError(f"Unsupported binary op: {expr.op}")
    if isinstance(expr, BoolOp):
        args = [expr_to_z3(v, env) for v in expr.values]
        if expr.op == "and":
            return z3.And(args)
        if expr.op == "or":
            return z3.Or(args)
        raise ValueError(f"Unsupported boolean op: {expr.op}")
    if isinstance(expr, Compare):
        lhs = expr_to_z3(expr.left, env)
        rhs = expr_to_z3(expr.right, env)
        if expr.op == "<":
            return lhs < rhs
        if expr.op == "<=":
            return lhs <= rhs
        if expr.op == ">":
            return lhs > rhs
        if expr.op == ">=":
            return lhs >= rhs
        if expr.op == "==":
            return lhs == rhs
        if expr.op == "!=":
            return lhs != rhs
        raise ValueError(f"Unsupported comparison op: {expr.op}")
    raise TypeError(f"Unhandled expression type: {type(expr).__name__}")


# === Transfer functions ====================================================
def _apply_assignment(domain: ConjunctiveDomain, state: AbstractState, stmt: AssignStmt) -> AbstractState:
    """Apply a single assignment to an abstract state."""
    translation = {name: f"{name}'" for name in domain.variables}  # type: ignore[attr-defined]
    post_domain = domain.translate(translation)

    pre_env = {name: domain.z3_variable(name) for name in domain.variables}  # type: ignore[attr-defined]
    post_env = {name: post_domain.z3_variable(translation[name]) for name in domain.variables}  # type: ignore[attr-defined]

    rhs_z3 = expr_to_z3(stmt.expr, pre_env)
    lhs = stmt.target
    lhs_pre = pre_env[lhs]
    lhs_post = post_env[lhs]

    if stmt.op == "=":
        assignment = lhs_post == rhs_z3
    elif stmt.op == "+=":
        assignment = lhs_post == lhs_pre + rhs_z3
    elif stmt.op == "-=":
        assignment = lhs_post == lhs_pre - rhs_z3
    elif stmt.op == "*=":
        assignment = lhs_post == lhs_pre * rhs_z3
    elif stmt.op == "//=":
        assignment = lhs_post == z3.IntDiv(lhs_pre, rhs_z3)
    elif stmt.op == "%=":
        assignment = lhs_post == z3.Mod(lhs_pre, rhs_z3)
    elif stmt.op == "&=":
        assignment = lhs_post == _bv_to_int(_as_bv(lhs_pre) & _as_bv(rhs_z3))
    elif stmt.op == "|=":
        assignment = lhs_post == _bv_to_int(_as_bv(lhs_pre) | _as_bv(rhs_z3))
    elif stmt.op == "^=":
        assignment = lhs_post == _bv_to_int(_as_bv(lhs_pre) ^ _as_bv(rhs_z3))
    elif stmt.op == "<<=":
        assignment = lhs_post == _bv_to_int(_as_bv(lhs_pre) << _as_bv(rhs_z3))
    elif stmt.op == ">>=":
        assignment = lhs_post == _bv_to_int(_as_bv(lhs_pre) >> _as_bv(rhs_z3))
    else:
        raise ValueError(f"Unsupported assignment op: {stmt.op}")

    frame_equalities = [
        (pre_env[name] == post_env[name])
        for name in domain.variables
        if name != lhs
    ]  # type: ignore[attr-defined]

    phi = domain.logic_and([domain.gamma_hat(state), assignment, *frame_equalities])
    post_state = bilateral(post_domain, phi)
    return post_state.translate({translation[name]: name for name in domain.variables})  # type: ignore[attr-defined]


def _apply_block(domain: ConjunctiveDomain, state: AbstractState, block: BasicBlock) -> AbstractState:
    new_state = state
    for stmt in block.statements:
        new_state = _apply_assignment(domain, new_state, stmt)
    return new_state


def _refine(domain: ConjunctiveDomain, state: AbstractState, constraint: z3.ExprRef) -> AbstractState:
    phi = domain.logic_and([domain.gamma_hat(state), constraint])
    return bilateral(domain, phi)


def analyze_cfg(cfg: CFG, domain: ConjunctiveDomain, input_state: AbstractState) -> AbstractState:
    """Run abstract interpretation over the CFG and return join of exit states."""
    states: Dict[str, AbstractState] = {cfg.entry: input_state}
    worklist: deque[str] = deque([cfg.entry])
    exit_states: List[AbstractState] = []

    while worklist:
        block_id = worklist.popleft()
        in_state = states[block_id]
        block = cfg.blocks[block_id]

        post_state = _apply_block(domain, in_state, block)
        term = block.terminator

        if isinstance(term, Branch):
            env = {name: domain.z3_variable(name) for name in domain.variables}  # type: ignore[attr-defined]
            cond = expr_to_z3(term.condition, env)
            true_state = _refine(domain, post_state, cond)
            false_state = _refine(domain, post_state, domain.logic_not(cond))
            _propagate(domain, states, worklist, term.true_target, true_state)
            _propagate(domain, states, worklist, term.false_target, false_state)
        elif isinstance(term, Goto):
            _propagate(domain, states, worklist, term.target, post_state)
        else:
            exit_states.append(post_state)

    if not exit_states:
        return domain.bottom  # type: ignore[attr-defined]
    return domain.join(exit_states)


def _propagate(
    domain: ConjunctiveDomain,
    states: Dict[str, AbstractState],
    worklist: deque[str],
    target: str,
    candidate: AbstractState,
) -> None:
    if target not in states:
        states[target] = candidate
        worklist.append(target)
        return
    joined = domain.join([states[target], candidate])
    if not (joined <= states[target]):  # type: ignore[operator]
        states[target] = joined
        worklist.append(target)
