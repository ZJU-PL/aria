"""C to EFMC transition system converter.

This frontend parses a (loop-centric) C program with ``pycparser`` and
builds an EFMC :class:`~efmc.sts.TransitionSystem`. The translation is
lightweight and currently focuses on integer variables and the first loop
encountered in the selected function (default: ``main``).
"""

import logging
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import z3
from aria.efmc.sts import TransitionSystem
from aria.utils.pycparser import c_ast, parse_file

logger = logging.getLogger(__name__)


class CToEFMCConverter:
    """Convert simple C programs into EFMC transition systems."""

    def __init__(self) -> None:
        self.logger = logger
        self.variable_mapping: Dict[str, z3.ExprRef] = {}
        self.prime_variable_mapping: Dict[str, z3.ExprRef] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def parse_c_file(
        self,
        filename: str,
        use_cpp: bool = False,
        cpp_args: Optional[List[str]] = None,
    ) -> c_ast.FileAST:
        """Parse a C file into a ``pycparser`` AST."""
        cpp_args = cpp_args or []
        return parse_file(filename, use_cpp=use_cpp, cpp_args=cpp_args)

    def convert_file_to_transition_system(
        self,
        filename: str,
        function: Optional[str] = None,
        use_cpp: bool = False,
        cpp_args: Optional[List[str]] = None,
    ) -> TransitionSystem:
        """Main entry point for converting a C file."""
        file_ast = self.parse_c_file(filename, use_cpp=use_cpp, cpp_args=cpp_args)
        func_def = self._pick_function(file_ast, function)
        pre_stmts, loop_node, post_stmts = self._split_around_first_loop(func_def)

        if loop_node is None:
            raise ValueError("No loop found in target function")

        variables = self._collect_variables(func_def)
        if not variables:
            raise ValueError("No variables detected in target function")

        self._create_z3_variables(variables)

        init = self._build_init_condition(pre_stmts)
        trans = self._build_transition(loop_node)

        # Safety after the loop should only be enforced once the loop exits.
        loop_guard = self._as_bool(
            self._c_expr_to_z3(loop_node.cond, self.variable_mapping)
        )
        post = self._build_post_condition(post_stmts, exit_guard=z3.Not(loop_guard))

        ts = TransitionSystem(
            variables=[self.variable_mapping[v] for v in variables],
            prime_variables=[self.prime_variable_mapping[v] for v in variables],
            init=init,
            trans=trans,
            post=post,
        )

        self.logger.info("Created transition system with %d variables", len(variables))
        return ts

    # ------------------------------------------------------------------ #
    # AST utilities                                                      #
    # ------------------------------------------------------------------ #
    def _pick_function(
        self, ast: c_ast.FileAST, target: Optional[str]
    ) -> c_ast.FuncDef:
        """Return the requested function definition or the first one."""
        for ext in ast.ext:
            if isinstance(ext, c_ast.FuncDef):
                name = ext.decl.name
                if target is None or target == name:
                    return ext
        raise ValueError(f"Function '{target or '<first>'}' not found")

    def _split_around_first_loop(
        self, func_def: c_ast.FuncDef
    ) -> Tuple[List[c_ast.Node], Optional[c_ast.Node], List[c_ast.Node]]:
        """Split the function body into pre-loop, loop, and post-loop parts."""
        pre: List[c_ast.Node] = []
        post: List[c_ast.Node] = []
        loop_node: Optional[c_ast.Node] = None

        body_items = func_def.body.block_items or []
        for idx, stmt in enumerate(body_items):
            if isinstance(stmt, (c_ast.While, c_ast.For)):
                loop_node = self._normalize_loop(stmt, pre)
                post = body_items[idx + 1 :]
                break
            pre.append(stmt)

        return pre, loop_node, post

    def _normalize_loop(self, node: c_ast.Node, pre: List[c_ast.Node]) -> c_ast.While:
        """Convert ``for`` loops to an equivalent ``while`` shape."""
        if isinstance(node, c_ast.While):
            return node

        assert isinstance(node, c_ast.For)
        # Handle for-loop initialization as pre-loop statements
        if node.init is not None:
            pre.append(node.init)

        cond = node.cond or c_ast.Constant(type="int", value="1")

        body_items: List[c_ast.Node] = []
        if isinstance(node.stmt, c_ast.Compound) and node.stmt.block_items:
            body_items.extend(node.stmt.block_items)
        elif node.stmt is not None:
            body_items.append(node.stmt)

        # The increment executes after the body
        if node.next is not None:
            body_items.append(node.next)

        new_body = c_ast.Compound(block_items=body_items)
        return c_ast.While(cond=cond, stmt=new_body)

    def _collect_variables(self, func_def: c_ast.FuncDef) -> List[str]:
        """Collect variable names (parameters + local declarations)."""
        names: List[str] = []

        # Parameters
        params = getattr(func_def.decl.type, "args", None)
        if params and params.params:
            for p in params.params:
                if isinstance(p, c_ast.Decl) and p.name:
                    names.append(p.name)

        class DeclVisitor(c_ast.NodeVisitor):
            def __init__(self) -> None:
                self.names: List[str] = []

            def visit_Decl(self, node: c_ast.Decl) -> None:
                if node.name:
                    self.names.append(node.name)

        dv = DeclVisitor()
        dv.visit(func_def.body)
        names.extend(dv.names)

        # Preserve deterministic ordering
        unique_names = sorted(set(names))
        return unique_names

    def _create_z3_variables(self, names: List[str]) -> None:
        """Create Z3 variables and their prime counterparts."""
        for name in names:
            var = z3.Int(name)
            prime = z3.Int(f"{name}!")
            self.variable_mapping[name] = var
            self.prime_variable_mapping[name] = prime

    # ------------------------------------------------------------------ #
    # Translation helpers                                                #
    # ------------------------------------------------------------------ #
    def _build_init_condition(self, stmts: List[c_ast.Node]) -> z3.ExprRef:
        """Translate pre-loop statements into an initial condition."""
        paths = self._enumerate_paths(stmts, self.variable_mapping)
        disjuncts = []
        for guard, env in paths:
            eqs = [
                self.variable_mapping[name] == value
                for name, value in env.items()
                if str(self.variable_mapping[name]) != str(value)
            ]
            pieces = [guard] + eqs if eqs else [guard]
            disjuncts.append(z3.And(pieces))

        if not disjuncts:
            return z3.BoolVal(True)
        return disjuncts[0] if len(disjuncts) == 1 else z3.Or(disjuncts)

    def _build_transition(self, loop_node: c_ast.While) -> z3.ExprRef:
        """Translate the loop body into a transition relation."""
        guard = self._as_bool(self._c_expr_to_z3(loop_node.cond, self.variable_mapping))
        body_items = self._body_items(loop_node.stmt)
        paths = self._enumerate_paths(body_items, self.variable_mapping)

        transitions = []
        for path_guard, env in paths:
            eqs = [
                self.prime_variable_mapping[name] == env[name]
                for name in self.variable_mapping
            ]
            transitions.append(z3.And(guard, path_guard, z3.And(eqs)))

        if not transitions:
            return z3.BoolVal(False)
        return transitions[0] if len(transitions) == 1 else z3.Or(transitions)

    def _build_post_condition(
        self, stmts: List[c_ast.Node], exit_guard: z3.ExprRef = z3.BoolVal(True)
    ) -> z3.ExprRef:
        """Translate post-loop assertions into a safety property.

        The guard captures the loop-exit condition so assertions are only
        required to hold after leaving the loop.
        """
        assertions = self._collect_assertions_with_guards(
            stmts, deepcopy(self.variable_mapping), initial_guard=exit_guard
        )
        return z3.And(assertions) if assertions else z3.BoolVal(True)

    # ------------------------------------------------------------------ #
    # Statement/path handling                                            #
    # ------------------------------------------------------------------ #
    def _enumerate_paths(
        self, stmts: List[c_ast.Node], env: Dict[str, z3.ExprRef]
    ) -> List[Tuple[z3.ExprRef, Dict[str, z3.ExprRef]]]:
        """Symbolically enumerate straight-line paths through a statement list."""
        paths: List[Tuple[z3.ExprRef, Dict[str, z3.ExprRef]]] = [
            (z3.BoolVal(True), deepcopy(env))
        ]

        for stmt in stmts or []:
            new_paths: List[Tuple[z3.ExprRef, Dict[str, z3.ExprRef]]] = []
            for guard, cur_env in paths:
                handled = self._process_statement(stmt, guard, cur_env)
                new_paths.extend(handled)
            paths = new_paths

        return paths

    def _process_statement(
        self, stmt: c_ast.Node, guard: z3.ExprRef, env: Dict[str, z3.ExprRef]
    ) -> List[Tuple[z3.ExprRef, Dict[str, z3.ExprRef]]]:
        """Process a single statement under a given guard and environment."""
        if isinstance(stmt, c_ast.Compound):
            sub_paths = self._enumerate_paths(stmt.block_items or [], env)
            return [(z3.And(guard, g), e) for g, e in sub_paths]

        if isinstance(stmt, c_ast.If):
            cond = self._as_bool(self._c_expr_to_z3(stmt.cond, env))
            then_items = self._body_items(stmt.iftrue)
            else_items = self._body_items(stmt.iffalse)

            then_paths = self._enumerate_paths(then_items, deepcopy(env))
            else_paths = self._enumerate_paths(else_items, deepcopy(env))

            results: List[Tuple[z3.ExprRef, Dict[str, z3.ExprRef]]] = []
            for g, e in then_paths:
                results.append((z3.And(guard, cond, g), e))
            for g, e in else_paths:
                results.append((z3.And(guard, z3.Not(cond), g), e))
            return results

        if isinstance(stmt, c_ast.Assignment):
            if stmt.op not in ("=", "+=", "-="):
                raise NotImplementedError(f"Assignment op '{stmt.op}' not supported")
            target = stmt.lvalue.name if isinstance(stmt.lvalue, c_ast.ID) else None
            if target is None or target not in env:
                raise ValueError(f"Unsupported assignment target: {stmt.lvalue}")

            rhs = self._c_expr_to_z3(stmt.rvalue, env)
            if stmt.op == "+=":
                rhs = env[target] + rhs
            elif stmt.op == "-=":
                rhs = env[target] - rhs

            new_env = deepcopy(env)
            new_env[target] = rhs
            return [(guard, new_env)]

        if isinstance(stmt, c_ast.Decl):
            if stmt.name in env and stmt.init is not None:
                new_env = deepcopy(env)
                new_env[stmt.name] = self._c_expr_to_z3(stmt.init, env)
                return [(guard, new_env)]
            return [(guard, env)]

        if isinstance(stmt, c_ast.FuncCall):
            name = self._func_name(stmt)
            if name in ("__VERIFIER_assume", "assume"):
                if stmt.args and stmt.args.exprs:
                    assume_expr = self._as_bool(
                        self._c_expr_to_z3(stmt.args.exprs[0], env)
                    )
                    return [(z3.And(guard, assume_expr), env)]
                return [(guard, env)]
            # Other calls are treated as no-ops here
            return [(guard, env)]

        # Unsupported or no-op statements keep the environment unchanged
        return [(guard, env)]

    def _body_items(self, node: Optional[c_ast.Node]) -> List[c_ast.Node]:
        """Return the block items list for a node if present."""
        if node is None:
            return []
        if isinstance(node, c_ast.Compound):
            return node.block_items or []
        return [node]

    # ------------------------------------------------------------------ #
    # Expression translation                                             #
    # ------------------------------------------------------------------ #
    def _c_expr_to_z3(self, expr: c_ast.Node, env: Dict[str, z3.ExprRef]) -> z3.ExprRef:
        """Translate a ``pycparser`` expression node to a Z3 expression."""
        if isinstance(expr, c_ast.Constant):
            if expr.type == "int":
                return z3.IntVal(int(expr.value))
            raise NotImplementedError(f"Unsupported constant type: {expr.type}")

        if isinstance(expr, c_ast.ID):
            if expr.name in env:
                return env[expr.name]
            raise ValueError(f"Unknown identifier: {expr.name}")

        if isinstance(expr, c_ast.BinaryOp):
            lhs = self._c_expr_to_z3(expr.left, env)
            rhs = self._c_expr_to_z3(expr.right, env)
            op = expr.op
            if op == "+":
                return lhs + rhs
            if op == "-":
                return lhs - rhs
            if op == "*":
                return lhs * rhs
            if op == "/":
                return z3.IntDiv(lhs, rhs)
            if op == "%":
                return lhs % rhs
            if op == "==":
                return lhs == rhs
            if op == "!=":
                return lhs != rhs
            if op == "<":
                return lhs < rhs
            if op == "<=":
                return lhs <= rhs
            if op == ">":
                return lhs > rhs
            if op == ">=":
                return lhs >= rhs
            if op == "&&":
                return z3.And(self._as_bool(lhs), self._as_bool(rhs))
            if op == "||":
                return z3.Or(self._as_bool(lhs), self._as_bool(rhs))
            raise NotImplementedError(f"Binary op '{op}' not supported")

        if isinstance(expr, c_ast.UnaryOp):
            operand = self._c_expr_to_z3(expr.expr, env)
            if expr.op == "-":
                return -operand
            if expr.op == "+":
                return operand
            if expr.op == "!":
                return z3.Not(self._as_bool(operand))
            if expr.op in ("p++", "p--", "++", "--"):
                raise NotImplementedError("Increment/decrement not supported")
            raise NotImplementedError(f"Unary op '{expr.op}' not supported")

        if isinstance(expr, c_ast.TernaryOp):
            cond = self._as_bool(self._c_expr_to_z3(expr.cond, env))
            iftrue = self._c_expr_to_z3(expr.iftrue, env)
            iffalse = self._c_expr_to_z3(expr.iffalse, env)
            return z3.If(cond, iftrue, iffalse)

        if isinstance(expr, c_ast.Cast):
            # Ignore the cast for now; assume integer-like sorts.
            return self._c_expr_to_z3(expr.expr, env)

        raise NotImplementedError(f"Expression '{type(expr)}' not supported")

    # ------------------------------------------------------------------ #
    # Assertions and helpers                                            #
    # ------------------------------------------------------------------ #
    def _collect_assertions_with_guards(
        self,
        stmts: List[c_ast.Node],
        env: Dict[str, z3.ExprRef],
        initial_guard: z3.ExprRef = z3.BoolVal(True),
    ) -> List[z3.ExprRef]:
        """Collect guarded assertions, including those nested in conditionals."""

        collected: List[z3.ExprRef] = []

        def walk(
            stmt_list: List[c_ast.Node],
            guard: z3.ExprRef,
            cur_env: Dict[str, z3.ExprRef],
        ) -> List[Tuple[z3.ExprRef, Dict[str, z3.ExprRef]]]:
            paths: List[Tuple[z3.ExprRef, Dict[str, z3.ExprRef]]] = [(guard, cur_env)]

            for stmt in stmt_list or []:
                new_paths: List[Tuple[z3.ExprRef, Dict[str, z3.ExprRef]]] = []
                for g, env_at_stmt in paths:
                    if isinstance(stmt, c_ast.FuncCall) and self._is_assert_call(stmt):
                        expr = self._as_bool(
                            self._c_expr_to_z3(stmt.args.exprs[0], env_at_stmt)
                        )
                        collected.append(z3.Implies(g, expr))
                        new_paths.append((g, env_at_stmt))
                        continue

                    if isinstance(stmt, c_ast.If):
                        cond = self._as_bool(self._c_expr_to_z3(stmt.cond, env_at_stmt))
                        then_paths = walk(
                            self._body_items(stmt.iftrue),
                            z3.And(g, cond),
                            deepcopy(env_at_stmt),
                        )
                        else_paths = walk(
                            self._body_items(stmt.iffalse),
                            z3.And(g, z3.Not(cond)),
                            deepcopy(env_at_stmt),
                        )
                        new_paths.extend(then_paths)
                        new_paths.extend(else_paths)
                        continue

                    new_paths.extend(self._process_statement(stmt, g, env_at_stmt))

                paths = new_paths

            return paths

        walk(stmts, initial_guard, env)
        return collected

    def _as_bool(self, expr: z3.ExprRef) -> z3.ExprRef:
        """Ensure integer expressions are treated as booleans."""
        if z3.is_bool(expr):
            return expr
        return expr != 0

    def _func_name(self, call: c_ast.FuncCall) -> Optional[str]:
        """Get the function name from a call node."""
        if isinstance(call.name, c_ast.ID):
            return call.name.name
        return None

    def _is_assert_call(self, call: c_ast.FuncCall) -> bool:
        """Check whether a function call represents an assertion."""
        name = self._func_name(call)
        return bool(
            name in ("__VERIFIER_assert", "assert") and call.args and call.args.exprs
        )


def c_to_efmc(filename: str) -> TransitionSystem:
    """Convenience wrapper mirroring :func:`boogie_to_efmc`."""
    converter = CToEFMCConverter()
    return converter.convert_file_to_transition_system(filename)
