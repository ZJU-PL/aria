"""Python-to-CFG lowering for ai_symabs."""
from __future__ import annotations

import ast
from typing import List, Optional, Set, Tuple

from .cfg import (
    AssignStmt,
    BasicBlock,
    BinOp,
    BoolOp,
    Branch,
    CFG,
    Compare,
    Const,
    Expr,
    Goto,
    UnaryOp,
    Var,
)


class _CFGBuilder:
    def __init__(self) -> None:
        self.blocks: dict[str, BasicBlock] = {}
        self.counter = 0
        entry = self._new_block()
        self.entry = entry.block_id
        self.variables: Set[str] = set()

    def build(self, module: ast.Module) -> Tuple[CFG, List[str]]:
        open_blocks = [self.entry]
        open_blocks = self._build_body(module.body, open_blocks)
        # Nothing to do with dangling exits; analysis treats missing terminator as exit.
        cfg = CFG(self.entry, self.blocks)
        return cfg, sorted(self.variables)

    # --- block helpers -------------------------------------------------
    def _new_block(self) -> BasicBlock:
        block_id = f"b{self.counter}"
        self.counter += 1
        block = BasicBlock(block_id)
        self.blocks[block_id] = block
        return block

    def _ensure_single_open(self, open_blocks: List[str]) -> BasicBlock:
        if len(open_blocks) == 1:
            return self.blocks[open_blocks[0]]
        join = self._new_block()
        for block_id in open_blocks:
            self._set_goto(block_id, join.block_id)
        return join

    def _set_goto(self, block_id: str, target: str) -> None:
        block = self.blocks[block_id]
        if block.terminator is not None:
            raise ValueError(f"Block {block_id} already terminated")
        block.terminator = Goto(target)

    # --- expression parsing ---------------------------------------------
    def _parse_expr(self, node: ast.AST) -> Expr:
        if isinstance(node, ast.Name):
            self.variables.add(node.id)
            return Var(node.id)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, bool)):
                return Const(node.value)
            raise ValueError(f"Unsupported constant: {node.value}")
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return UnaryOp("-", self._parse_expr(node.operand))
            if isinstance(node.op, ast.UAdd):
                return UnaryOp("+", self._parse_expr(node.operand))
            if isinstance(node.op, ast.Not):
                return UnaryOp("not", self._parse_expr(node.operand))
            if isinstance(node.op, ast.Invert):
                return UnaryOp("~", self._parse_expr(node.operand))
            raise ValueError(f"Unsupported unary op: {node.op}")
        if isinstance(node, ast.BinOp):
            op = node.op
            if isinstance(op, ast.Add):
                op_str = "+"
            elif isinstance(op, ast.Sub):
                op_str = "-"
            elif isinstance(op, ast.Mult):
                op_str = "*"
            elif isinstance(op, ast.FloorDiv):
                op_str = "//"
            elif isinstance(op, ast.Mod):
                op_str = "%"
            elif isinstance(op, ast.BitAnd):
                op_str = "&"
            elif isinstance(op, ast.BitOr):
                op_str = "|"
            elif isinstance(op, ast.BitXor):
                op_str = "^"
            elif isinstance(op, ast.LShift):
                op_str = "<<"
            elif isinstance(op, ast.RShift):
                op_str = ">>"
            else:
                raise ValueError(f"Unsupported binary op: {op}")
            return BinOp(
                op_str, self._parse_expr(node.left), self._parse_expr(node.right)
            )
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                op_str = "and"
            elif isinstance(node.op, ast.Or):
                op_str = "or"
            else:
                raise ValueError(f"Unsupported boolean op: {node.op}")
            return BoolOp(op_str, [self._parse_expr(value) for value in node.values])
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                # Chain comparisons desugar to an and-chain.
                left_expr = self._parse_expr(node.left)
                parts = []
                for op, right in zip(node.ops, node.comparators):
                    op_str = _cmp_to_str(op)
                    right_expr = self._parse_expr(right)
                    parts.append(Compare(left_expr, op_str, right_expr))
                    left_expr = right_expr
                return BoolOp("and", parts)
            op_str = _cmp_to_str(node.ops[0])
            return Compare(
                self._parse_expr(node.left), op_str,
                self._parse_expr(node.comparators[0])
            )
        raise ValueError(f"Unsupported expression node: {ast.dump(node)}")

    # --- statement lowering ---------------------------------------------
    def _build_body(self, body: List[ast.stmt], open_blocks: List[str]) -> List[str]:
        current = open_blocks
        for stmt in body:
            current = self._lower_stmt(stmt, current)
        return current

    def _lower_stmt(self, stmt: ast.stmt, open_blocks: List[str]) -> List[str]:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise ValueError("Only simple assignments to names are supported")
            block = self._ensure_single_open(open_blocks)
            target = stmt.targets[0].id
            self.variables.add(target)
            block.add_statement(AssignStmt(target, "=", self._parse_expr(stmt.value)))
            return [block.block_id]

        if isinstance(stmt, ast.AugAssign):
            if not isinstance(stmt.target, ast.Name):
                raise ValueError("AugAssign target must be a variable name")
            block = self._ensure_single_open(open_blocks)
            target = stmt.target.id
            self.variables.add(target)
            op = stmt.op
            op_str = _aug_to_str(op)
            block.add_statement(
                AssignStmt(target, op_str, self._parse_expr(stmt.value))
            )
            return [block.block_id]

        if isinstance(stmt, ast.If):
            cond_block = self._ensure_single_open(open_blocks)
            then_entry = self._new_block()
            else_entry = self._new_block()
            cond_block.terminator = Branch(
                self._parse_expr(stmt.test), then_entry.block_id,
                else_entry.block_id)

            then_exits = self._build_body(stmt.body, [then_entry.block_id])
            else_body = stmt.orelse if stmt.orelse else []
            else_exits = self._build_body(else_body, [else_entry.block_id])

            join = self._new_block()
            for b in then_exits + else_exits:
                self._set_goto(b, join.block_id)
            return [join.block_id]

        if isinstance(stmt, ast.While):
            cond_block = self._ensure_single_open(open_blocks)
            body_entry = self._new_block()
            after_loop = self._new_block()

            cond_block.terminator = Branch(
                self._parse_expr(stmt.test), body_entry.block_id,
                after_loop.block_id)

            body_exits = self._build_body(stmt.body, [body_entry.block_id])
            for b in body_exits:
                self._set_goto(b, cond_block.block_id)
            return [after_loop.block_id]

        if isinstance(stmt, ast.For):
            return self._lower_for(stmt, open_blocks)

        if isinstance(stmt, ast.Pass):
            return open_blocks

        raise ValueError(f"Unsupported statement: {ast.dump(stmt)}")

    def _lower_for(self, stmt: ast.For, open_blocks: List[str]) -> List[str]:
        if not isinstance(stmt.target, ast.Name):
            raise ValueError("For loop target must be a variable name")
        if (not isinstance(stmt.iter, ast.Call) or
                not isinstance(stmt.iter.func, ast.Name) or
                stmt.iter.func.id != "range"):
            raise ValueError("Only for-loops over range(...) are supported")

        args = stmt.iter.args
        if len(args) == 1:
            start_expr = Const(0)
            stop_expr = self._parse_expr(args[0])
            step_expr = Const(1)
        elif len(args) == 2:
            start_expr = self._parse_expr(args[0])
            stop_expr = self._parse_expr(args[1])
            step_expr = Const(1)
        elif len(args) == 3:
            start_expr = self._parse_expr(args[0])
            stop_expr = self._parse_expr(args[1])
            step_expr = self._parse_expr(args[2])
        else:
            raise ValueError("range() in for-loops must have 1-3 arguments")

        step_const = _maybe_const_int(step_expr)
        if step_const == 0:
            raise ValueError("range() step cannot be zero")

        # Initialize iterator variable.
        block = self._ensure_single_open(open_blocks)
        target_name = stmt.target.id
        self.variables.add(target_name)
        block.add_statement(AssignStmt(target_name, "=", start_expr))

        cond_block = self._new_block()
        self._set_goto(block.block_id, cond_block.block_id)

        cond_expr = _range_condition(target_name, stop_expr, step_expr, step_const)
        body_entry = self._new_block()
        after_loop = self._new_block()
        cond_block.terminator = Branch(
            cond_expr, body_entry.block_id, after_loop.block_id
        )

        body_exits = self._build_body(stmt.body, [body_entry.block_id])

        # Increment iterator at end of body.
        increment_op = "+=" if (step_const is None or step_const > 0) else "-="
        increment_expr = step_expr if increment_op == "+=" else Const(abs(step_const))
        for b in body_exits:
            self.blocks[b].add_statement(
                AssignStmt(target_name, increment_op, increment_expr)
            )
            self._set_goto(b, cond_block.block_id)

        return [after_loop.block_id]


def _aug_to_str(op: ast.AST) -> str:
    if isinstance(op, ast.Add):
        return "+="
    if isinstance(op, ast.Sub):
        return "-="
    if isinstance(op, ast.Mult):
        return "*="
    if isinstance(op, ast.FloorDiv):
        return "//="
    if isinstance(op, ast.Mod):
        return "%="
    if isinstance(op, ast.BitAnd):
        return "&="
    if isinstance(op, ast.BitOr):
        return "|="
    if isinstance(op, ast.BitXor):
        return "^="
    if isinstance(op, ast.LShift):
        return "<<="
    if isinstance(op, ast.RShift):
        return ">>="
    raise ValueError(f"Unsupported AugAssign op: {op}")


def _cmp_to_str(op: ast.cmpop) -> str:
    if isinstance(op, ast.Lt):
        return "<"
    if isinstance(op, ast.LtE):
        return "<="
    if isinstance(op, ast.Gt):
        return ">"
    if isinstance(op, ast.GtE):
        return ">="
    if isinstance(op, ast.Eq):
        return "=="
    if isinstance(op, ast.NotEq):
        return "!="
    raise ValueError(f"Unsupported comparison op: {op}")


def _maybe_const_int(expr: Expr) -> Optional[int]:
    if isinstance(expr, Const) and isinstance(expr.value, int):
        return expr.value
    return None


def _range_condition(
    var: str, stop: Expr, step: Expr, step_const: Optional[int]
) -> Expr:
    var_expr = Var(var)
    stop_cmp_lt = Compare(var_expr, "<", stop)
    stop_cmp_gt = Compare(var_expr, ">", stop)
    positive = Compare(step, ">", Const(0))
    negative = Compare(step, "<", Const(0))

    if step_const is not None:
        return stop_cmp_lt if step_const > 0 else stop_cmp_gt

    return BoolOp(
        "or",
        [
            BoolOp("and", [positive, stop_cmp_lt]),
            BoolOp("and", [negative, stop_cmp_gt]),
        ],
    )


def build_python_cfg(source: str) -> Tuple[CFG, List[str]]:
    """Parse Python source (integer-only subset) into a CFG."""
    module = ast.parse(source)
    builder = _CFGBuilder()
    return builder.build(module)
